"""
Replays a calibration recorded with `eyetrax_server.py --record DIR` and
compares gaze models on it, to see what actually limits accuracy for a real
user, camera and room.

    .venv/Scripts/python analyze_recording.py recordings/calibration-*.npz

A recording holds every calibration sample: the face landmarks MediaPipe found
(not the image), the features eyetrax_server trained on, the target on screen,
whether it was taken during head movement, and which point it belonged to.

Two tests:

- Unseen targets: each adaptive round is predicted by a model trained on
  everything before it, as the server does when it reports confidence.
- Head movement: each head-movement part is predicted by a model trained
  without it (the user looked at the centre dot while moving). Only
  recordings made before the head-movement step was removed have these.

Errors are in CSS px. "settled" is the error of the mean prediction over each
target, which is what the smoothed cursor shows; "jitter" is the spread of
single frames around that mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
import gaze_model
from eyetrax.models import create_model
from face_features import pose_scaling

EYETRAX_DIMS = (len(LEFT_EYE_INDICES) + len(RIGHT_EYE_INDICES) + len(MUTUAL_INDICES)) * 3 + 3


def eyetrax_landmarks(lms: np.ndarray):
    """
    eyetrax's GazeEstimator.extract_features on recorded landmarks: rotated
    into the head's frame and scaled by the eye-corner distance. Returns the
    normalised points (n, 478, 3) and eyetrax's three head angles (n, 3)
    """
    n = len(lms)
    points = np.empty_like(lms)
    angles = np.empty((n, 3), np.float32)
    for i, p in enumerate(lms):
        left, right, top = p[33], p[263], p[10]
        centre = (left + right) / 2
        x = right - left
        x /= np.linalg.norm(x) + 1e-9
        y = top - centre
        y /= np.linalg.norm(y) + 1e-9
        y = y - np.dot(y, x) * x
        y /= np.linalg.norm(y) + 1e-9
        z = np.cross(x, y)
        z /= np.linalg.norm(z) + 1e-9
        R = np.column_stack((x, y, z))
        rotated = (R.T @ (p - centre).T).T
        d = np.linalg.norm(R.T @ (right - centre) - R.T @ (left - centre))
        if d > 1e-7:
            rotated /= d
        points[i] = rotated
        angles[i] = (
            np.arctan2(R[1, 0], R[0, 0]),
            np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)),
            np.arctan2(R[2, 1], R[2, 2]),
        )
    return points, angles


# --- feature sets: (points, angles, pose) -> features ending in pose -------

SUBSET = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
# Irises, eye corners and lids only: LEFT/RIGHT_EYE_INDICES[10:31]
EYE_CORE = LEFT_EYE_INDICES[10:31] + RIGHT_EYE_INDICES[10:31]


def eyetrax_feats(points, angles, pose):
    return np.column_stack([points[:, SUBSET].reshape(len(points), -1), angles])


def with_pose(points, angles, pose):
    return np.column_stack([eyetrax_feats(points, angles, pose), pose])


def xy_with_pose(points, angles, pose):
    """MediaPipe's depth (z) is its noisiest coordinate: leave it out"""
    return np.column_stack([points[:, SUBSET, :2].reshape(len(points), -1), angles, pose])


def core_with_pose(points, angles, pose):
    return np.column_stack([points[:, EYE_CORE].reshape(len(points), -1), angles, pose])


def core_xy_with_pose(points, angles, pose):
    return np.column_stack([points[:, EYE_CORE, :2].reshape(len(points), -1), angles, pose])


def as_recorded(points, angles, pose):
    """What the server trained on (filled in by main from the recording)"""
    return RECORDED


RECORDED = None


class Ridge:
    """eyetrax's ridge model; alpha fixed, or `per_sample` times the samples"""

    def __init__(self, alpha=1.0, per_sample=None, fixed_units=True):
        self.alpha, self.per_sample, self.fixed_units = alpha, per_sample, fixed_units

    def train(self, X, y):
        alpha = self.per_sample * len(X) if self.per_sample else self.alpha
        self.m = create_model("ridge", alpha=alpha)
        self.m.train(X, y, pose_scaling(X) if self.fixed_units else None)

    def predict(self, X):
        return self.m.predict(X)


# name: (features, model, train on head movement)
VARIANTS = {
    "server (gaze_model)": (as_recorded, lambda: gaze_model.make_model("ridge"), True),
    "eyetrax alone": (eyetrax_feats, lambda: Ridge(fixed_units=False), False),
    "eyetrax + head step data": (eyetrax_feats, lambda: Ridge(fixed_units=False), True),
    "+ pose, alpha 1 (before blending)": (with_pose, lambda: Ridge(), True),
    "+ pose, precise only": (with_pose, lambda: Ridge(per_sample=gaze_model.PRECISE), True),
    "+ pose, robust only": (with_pose, lambda: Ridge(per_sample=gaze_model.ROBUST), True),
    "x,y only (no depth) + pose": (xy_with_pose, lambda: Ridge(), True),
    "eye core only + pose": (core_with_pose, lambda: Ridge(), True),
}


def fit_predict(X, y, train, test, make):
    m = make()
    m.train(X[train], y[train])
    return m.predict(X[test])


def errors(pred, y, groups):
    """per-frame mean, settled (per-target mean prediction) and jitter, px"""
    frame = np.linalg.norm(pred - y, axis=1)
    settled, jitter = [], []
    for g in np.unique(groups):
        idx = groups == g
        mean = pred[idx].mean(axis=0)
        settled.append(np.linalg.norm(mean - y[idx][0]))
        jitter.append(np.sqrt(((pred[idx] - mean) ** 2).sum(axis=1).mean()))
    return frame.mean(), np.mean(settled), np.mean(jitter), np.array(settled)


def segments(finishes, n):
    bounds = [0, *finishes]
    if bounds[-1] < n:
        bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    r = np.load(args.recording)
    lms, feats, y = r["landmarks"], r["features"], r["targets"].astype(float)
    head, point = r["head"].astype(bool), r["point"]
    w, h = r["viewport"]
    n = len(y)

    print(f"{args.recording.name}: {n} samples, {len(np.unique(point))} captures, "
          f"{head.sum()} during head movement")
    print(f"camera {r['frame_size'][0]}x{r['frame_size'][1]} at {float(r['fps']):.0f} fps, "
          f"screen {w}x{h}, trainings after samples {list(r['finishes'])}")

    global RECORDED
    RECORDED = feats.astype(float)
    points, angles = eyetrax_landmarks(lms)
    pose = feats[:, EYETRAX_DIMS:]
    same = np.abs(eyetrax_feats(points, angles, pose) - feats[:, :EYETRAX_DIMS]).max()
    print(f"recomputed eyetrax features match the recording to {same:.1e}")

    # How much the head moved, as the head-movement step measures it
    if head.any():
        span = np.percentile(pose[head], 95, axis=0) - np.percentile(pose[head], 5, axis=0)
        still = np.percentile(pose[~head], 95, axis=0) - np.percentile(pose[~head], 5, axis=0)
        names = ["x", "y", "turn", "tilt", "roll", "size"]
        print("head range (5-95%), head step vs rest: " + ", ".join(
            f"{k} {a:.3f}/{b:.3f}" for k, a, b in zip(names, span, still)))

    segs = segments(list(r["finishes"]), n)
    rounds = [(a, b) for a, b in segs[1:] if not head[a:b].any()]
    parts = [p for p in np.unique(point[head])]
    border = (np.minimum(y[:, 0], w - y[:, 0]) < 0.12 * w) | (np.minimum(y[:, 1], h - y[:, 1]) < 0.12 * h)

    print()
    print(f"{'':32s} {'unseen targets (' + str(len(rounds)) + ' rounds)':>42s}   {'head movement (' + str(len(parts)) + ' parts)':>28s}")
    print(f"{'model':32s} {'frame':>7s} {'settled':>8s} {'centre':>7s} {'border':>7s} {'jitter':>7s}   {'frame':>7s} {'settled':>8s} {'jitter':>7s}")
    for name, (build, make, use_head) in VARIANTS.items():
        X = build(points, angles, pose)
        out = []
        if rounds:
            preds, ys, gs, bs = [], [], [], []
            for a, b in rounds:
                train = np.arange(n) < a
                if not use_head:
                    train &= ~head
                test = np.arange(a, b)
                preds.append(fit_predict(X, y, train, test, make))
                ys.append(y[test]); gs.append(point[test]); bs.append(border[test])
            pred, yy, g, b = map(np.concatenate, (preds, ys, gs, bs))
            frame, settled, jitter, per = errors(pred, yy, g)
            on_border = np.array([b[g == k][0] for k in np.unique(g)])
            centre_err = per[~on_border].mean() if (~on_border).any() else np.nan
            border_err = per[on_border].mean() if on_border.any() else np.nan
            out += [frame, settled, centre_err, border_err, jitter]
        else:
            out += [np.nan] * 5
        if parts:
            preds, ys, gs = [], [], []
            for p in parts:
                test = np.flatnonzero(point == p)
                train = point != p
                if not use_head:
                    train &= ~head
                preds.append(fit_predict(X, y, train, test, make))
                ys.append(y[test]); gs.append(point[test])
            frame, settled, jitter, _ = errors(*map(np.concatenate, (preds, ys, gs)))
            out += [frame, settled, jitter]
        else:
            out += [np.nan] * 3
        print(f"{name:32s} " + " ".join(f"{v:7.0f}" if i != 1 else f"{v:8.0f}" for i, v in enumerate(out[:5]))
              + "   " + " ".join(f"{v:7.0f}" if i != 1 else f"{v:8.0f}" for i, v in enumerate(out[5:])))

    # What the user saw: the live model's predictions on each adaptive round
    live = r["prediction"].astype(float)
    if rounds:
        idx = np.concatenate([np.arange(a, b) for a, b in rounds])
        ok = ~np.isnan(live[idx, 0])
        if ok.any():
            frame, settled, jitter, _ = errors(live[idx][ok], y[idx][ok], point[idx][ok])
            print(f"\nlive model during the adaptive rounds: frame {frame:.0f}, settled {settled:.0f}, jitter {jitter:.0f}")


if __name__ == "__main__":
    main()
