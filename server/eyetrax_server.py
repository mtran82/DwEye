"""
WebSocket bridge between eyetrax and the DwEye front end.

eyetrax's own calibration routines draw into an OpenCV window. Here the
browser draws the calibration targets instead, and this process owns the
webcam and eyetrax's GazeEstimator: it samples eye features while the front
end holds a target on screen, trains the model, then streams gaze back.

After the initial grid, calibration can be refined in rounds, as in eyetrax's
adaptive calibration (calibration/adaptive.py): extra targets come from
eyetrax's BlueNoiseSampler and the model is retrained after each round. Each
round is predicted by the model before it is trained on, which gives an
honest accuracy measure on points the model has never seen.

eyetrax's features ignore where the head is (see face_features.py); head
pose features are added to them. After the head moves, the front end
re-centers: a few targets from the new position, weighted RECENTER_WEIGHT.

All coordinates are CSS pixels relative to the browser viewport.

Protocol (JSON text messages):

  client -> server
    {"type": "calibration_start", "width": w, "height": h}
    {"type": "point_start", "x": x, "y": y, "recenter": bool}
                                              start sampling for a target;
                                              recenter: see RECENTER_WEIGHT
    {"type": "point_end"}                     stop; replies "point_captured"
    {"type": "calibration_finish"}            train; replies "calibration_done"
    {"type": "next_points"}                   replies "next_points"
    {"type": "calibration_cancel"}            drop samples not yet trained on

  Only one page calibrates at a time: calibration_start makes a page the
  owner, the previous owner gets {"type": "calibration_taken"}, and other
  pages' calibration messages get {"type": "error", "error": str}.

  server -> client
    {"type": "hello", "calibrated": bool}
    {"type": "frame", "face": bool, "blink": bool, "gaze": [x, y] | null,
     "head": [x, y, yaw, pitch, size] | null}
        head_pose() as seen from the screen: nose tip position (face widths,
        x right, y down), turn and tilt (right, down), log face size
    {"type": "point_captured", "samples": n}
    {"type": "calibration_done", "ok": bool, "samples": n, "points": n,
     "accuracy": {...} | null, "remaining": n, "next_round": n, "error": str}
    {"type": "next_points", "points": [[x, y], ...]}   next adaptive round

  "accuracy" describes the targets just added, measured before training on
  them:
    {"error_px": mean distance from target, "hit_rate": share of samples
     within "tolerance_px" of the target, "points": n, "samples": n}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from eyetrax.calibration.adaptive import BlueNoiseSampler
from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.filters import (
    KDESmoother,
    KalmanEMASmoother,
    KalmanSmoother,
    NoSmoother,
    make_kalman,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera
import gaze_model
from face_features import (
    BlinkDetector,
    LandmarkTap,
    eye_aspect_ratio,
    head_pose,
)

# Only pages served from this machine may connect (browsers always send an
# Origin header; non-browser clients such as test scripts send none)
LOCAL_ORIGINS = [re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"), None]

# A 2D mapping needs at least three non-collinear targets
MIN_POINTS = 3

# Re-centering after the head has moved captures a few targets from the new
# position. Each of their samples counts this many times in training, so they
# outweigh the calibration taken from where the head was. On real recordings
# this beat correcting an offset or an affine map (87 px against 117 and 97)
RECENTER_WEIGHT = 5

# eyetrax's features (eye-region landmarks and three head angles), then
# head_pose()
FEATURES = (len(LEFT_EYE_INDICES) + len(RIGHT_EYE_INDICES) + len(MUTUAL_INDICES)) * 3 + 3 + 6

# Gaze within this share of the viewport diagonal counts as a hit when
# measuring accuracy: about 110 px on a 1920x1080 screen, roughly half a key
# on the DwEye keyboard
HIT_RADIUS_RATIO = 0.05


def tune_kalman(kf, preds: np.ndarray, targets: np.ndarray) -> None:
    """
    Same idea as eyetrax's KalmanSmoother.tune, which opens its own OpenCV
    window: set the measurement noise to the spread of gaze predictions while
    the user fixates a known point. The calibration samples provide that here.
    """
    resid = []
    for t in np.unique(targets, axis=0):
        p = preds[(targets == t).all(axis=1)]
        if len(p) >= 2:
            resid.append(p - p.mean(axis=0))
    if not resid:
        return
    var = np.concatenate(resid).var(axis=0)
    var[var == 0] = 1e-4
    kf.measurementNoiseCov = np.diag(var).astype(np.float32)


class Tracker:
    """
    Runs the camera loop on a background thread and holds calibration state
    """

    def __init__(self, estimator, cap, args):
        self.estimator = estimator  # for its features; see gaze_model for the model
        self.model = gaze_model.make_model(args.model)
        self.tap = LandmarkTap.install(estimator)
        self.blink = BlinkDetector()
        self.cap = cap
        self.args = args

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._viewport = None
        self._target = None
        self._point_weight = 1
        self._point_feats = []
        self._feats, self._targs = [], []
        # How many times each sample counts in training
        self._weight = []
        # With --record: per sample (landmarks, time, live prediction, point
        # number), saved after each training; see analyze_recording.py
        self._point_extra = []
        self._extra = []
        self._point_id = 0
        self._finishes = []
        self._record_path = None
        self.fps = None
        # The page calibrating; see handle()
        self.owner = None
        # Samples the current model was trained on; later ones are unseen
        self._trained_n = 0
        # Adaptive targets for this calibration and how many have been handed out
        self._plan = None
        self._plan_pos = 0
        self._plan_trained = 0

        self.calibrated = False
        self.smoother = NoSmoother()
        self.frame = {"face": False, "blink": False, "gaze": None}
        self.seq = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            feats, _ = self.estimator.extract_features(frame)
            face = feats is not None
            blink = False
            head = None
            if face:
                landmarks = self.tap.landmarks
                pose = head_pose(landmarks, self.tap.size)
                feats = np.concatenate([feats, pose])
                blink = self.blink.update(eye_aspect_ratio(landmarks))
                # The camera image is unmirrored: moving or turning right
                # moves the nose left in it
                x, y, yaw, pitch, _, size = pose
                head = [round(float(v), 4) for v in (-x, y, -yaw, pitch, size)]
            usable = face and not blink
            gaze = None
            with self._lock:
                raw = None
                if usable and self.calibrated:
                    raw = self.model.predict(np.array([feats]))[0]
                if usable and self._target is not None:
                    self._point_feats.append(feats)
                    if self.args.record:
                        lms = np.array([(p.x, p.y, p.z) for p in landmarks], np.float32)
                        self._point_extra.append((lms, time.time(), raw))
                if raw is not None:
                    x, y = raw
                    if self._viewport:
                        # Keep the cursor on screen however far off a guess is
                        w, h = self._viewport
                        x, y = min(max(x, 0), w), min(max(y, 0), h)
                    gaze = self.smoother.step(int(x), int(y))
                self.frame = {"face": face, "blink": blink, "gaze": gaze, "head": head}
                self.seq += 1

    def snapshot(self):
        with self._lock:
            return self.seq, self.frame

    # --- calibration, driven by the front end -------------------------------

    def start_calibration(self, width: int, height: int):
        with self._lock:
            self._viewport = (int(width), int(height))
            self._feats, self._targs, self._weight = [], [], []
            self._extra, self._point_extra, self._finishes = [], [], []
            if self.args.record:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self._record_path = self.args.record / f"calibration-{stamp}.npz"
            self._trained_n = 0
            self._plan, self._plan_pos, self._plan_trained = None, 0, 0
            self._target = None
            self._point_feats = []

    def next_points(self) -> list:
        """
        The next round of adaptive targets, like run_adaptive_calibration:
        eyetrax's blue-noise points, `retrain_every` at a time
        """
        with self._lock:
            if self._viewport is None:
                return []
            if self._plan is None:
                w, h = self._viewport
                self._plan = BlueNoiseSampler(w, h).sample(self.args.random_points)
            pts = self._plan[self._plan_pos : self._plan_pos + self.args.retrain_every]
            self._plan_pos += len(pts)
            return [list(p) for p in pts]

    def _remaining(self) -> int:
        return max(self.args.random_points - self._plan_pos, 0)

    def start_point(self, x: float, y: float, recenter: bool = False):
        with self._lock:
            # Rounded so re-measuring the same target counts as one point
            self._target = (round(float(x), 1), round(float(y), 1))
            self._point_weight = RECENTER_WEIGHT if recenter else 1
            self._point_feats = []
            self._point_extra = []
            self._point_id += 1

    def end_point(self) -> int:
        with self._lock:
            n = len(self._point_feats)
            if self._target is not None:
                self._feats.extend(self._point_feats)
                self._targs.extend([self._target] * n)
                self._weight.extend([self._point_weight] * n)
                self._extra.extend(e + (self._point_id,) for e in self._point_extra)
            self._target = None
            self._point_feats = []
            self._point_extra = []
            return n

    def cancel(self):
        """
        Abandon whatever is being captured; what the model was already
        trained on is kept, so later rounds still build on it
        """
        with self._lock:
            self._target = None
            self._point_feats = []
            del self._feats[self._trained_n :]
            del self._targs[self._trained_n :]
            del self._weight[self._trained_n :]
            del self._extra[self._trained_n :]
            self._point_extra = []
            self._plan_pos = self._plan_trained

    def finish_calibration(self) -> dict:
        with self._lock:
            points = len(set(self._targs))
            samples = len(self._feats)
            if points < MIN_POINTS:
                return {
                    "ok": False,
                    "samples": samples,
                    "points": points,
                    "error": "Not enough usable samples; keep your face in view",
                }
            X, y = np.array(self._feats), np.array(self._targs)
            # Score the new samples before the model sees them. Unlike
            # eyetrax's adaptive calibration, which retrains on the random
            # points alone, keep training on everything collected so far
            accuracy = None
            if 0 < self._trained_n < samples:
                accuracy = self._accuracy(X[self._trained_n :], y[self._trained_n :])
            counted = np.repeat(np.arange(samples), self._weight)
            X, y = X[counted], y[counted]
            self.model.train(X, y)
            self._trained_n = samples
            self._finishes.append(samples)
            recording = self._recording() if self._record_path else None
            self._plan_trained = self._plan_pos
            self.smoother = self._make_smoother(X, y)
            self.calibrated = True
            if self.args.model_file:
                self.model.save(self.args.model_file)
            remaining = self._remaining()
        if recording:
            np.savez_compressed(self._record_path, **recording)
        msg = f"[eyetrax-server] Calibrated with {samples} samples from {points} points"
        if accuracy:
            msg += (
                f"; on unseen points: {accuracy['error_px']:.0f} px mean error, "
                f"{accuracy['hit_rate']:.0%} within {accuracy['tolerance_px']:.0f} px"
            )
        print(msg)
        return {
            "ok": True,
            "samples": samples,
            "points": points,
            "accuracy": accuracy,
            "remaining": remaining,
            "next_round": min(self.args.retrain_every, remaining),
        }

    def _recording(self) -> dict:
        """This calibration's samples so far, for --record"""
        lms, times, preds, points = zip(*self._extra)
        nan = (np.nan, np.nan)
        return {
            "landmarks": np.stack(lms),
            "features": np.array(self._feats, np.float32),
            "targets": np.array(self._targs, np.float32),
            "head": np.zeros(len(self._feats), bool),  # no head-movement step any more
            "weight": np.array(self._weight),
            "point": np.array(points),
            "time": np.array(times),
            "prediction": np.array([nan if p is None else p for p in preds], np.float32),
            "finishes": np.array(self._finishes),
            "viewport": np.array(self._viewport),
            "frame_size": np.array(self.tap.size),
            "fps": np.array(self.fps or 0.0),
        }

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> dict:
        dist = np.linalg.norm(self.model.predict(X) - y, axis=1)
        tolerance = HIT_RADIUS_RATIO * math.hypot(*self._viewport)
        return {
            "error_px": float(dist.mean()),
            "hit_rate": float((dist <= tolerance).mean()),
            "tolerance_px": float(tolerance),
            "points": len(np.unique(y, axis=0)),
            "samples": len(y),
        }

    def _make_smoother(self, X=None, y=None):
        name = self.args.filter
        if name == "kde":
            w, h = self._viewport or get_screen_size()
            return KDESmoother(w, h, confidence=self.args.confidence)
        if name in ("kalman", "kalman_ema"):
            kf = make_kalman()
            if X is not None:
                tune_kalman(kf, self.model.predict(X), y)
            if name == "kalman":
                return KalmanSmoother(kf)
            return KalmanEMASmoother(kf, ema_alpha=self.args.ema_alpha)
        return NoSmoother()

    def load_model(self, path: Path) -> bool:
        model = gaze_model.load(path)
        if getattr(model, "n_features", None) != FEATURES:
            return False  # saved by an older version
        self.model = model
        self.smoother = self._make_smoother()
        self.calibrated = True
        return True


def send_json(ws, msg: dict):
    return ws.send(json.dumps(msg))


async def stream_frames(ws, tracker: Tracker):
    last = -1
    try:
        while True:
            seq, frame = tracker.snapshot()
            if seq != last:
                last = seq
                await send_json(ws, {"type": "frame", **frame})
            await asyncio.sleep(1 / 60)
    except ConnectionClosed:
        pass


async def take_ownership(ws, tracker: Tracker):
    previous, tracker.owner = tracker.owner, ws
    if previous is not None and previous is not ws:
        try:
            await send_json(previous, {"type": "calibration_taken"})
        except ConnectionClosed:
            pass


async def handle(ws, tracker: Tracker):
    await send_json(ws, {"type": "hello", "calibrated": tracker.calibrated})
    streamer = asyncio.create_task(stream_frames(ws, tracker))
    try:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "calibration_start":
                await take_ownership(ws, tracker)
                tracker.start_calibration(msg["width"], msg["height"])
            elif ws is not tracker.owner:
                # Another page is calibrating; don't let this one mix in
                await send_json(ws, {"type": "error", "error": "Another window is calibrating"})
            elif kind == "point_start":
                tracker.start_point(msg["x"], msg["y"], msg.get("recenter", False))
            elif kind == "point_end":
                n = tracker.end_point()
                await send_json(ws, {"type": "point_captured", "samples": n})
            elif kind == "calibration_finish":
                result = await asyncio.to_thread(tracker.finish_calibration)
                await send_json(ws, {"type": "calibration_done", **result})
            elif kind == "next_points":
                points = tracker.next_points()
                await send_json(ws, {"type": "next_points", "points": points})
            elif kind == "calibration_cancel":
                tracker.cancel()
    except ConnectionClosed:
        pass
    finally:
        streamer.cancel()
        if ws is tracker.owner:
            tracker.owner = None
            # Don't keep sampling for a target nobody is showing any more
            tracker.cancel()


def parse_args():
    parser = argparse.ArgumentParser(description="eyetrax WebSocket server for DwEye")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Camera resolution to ask for, WIDTHxHEIGHT, or 'default' for the "
        "camera's own. More pixels on the eyes give steadier iris landmarks",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Save each calibration's samples (face landmarks, not images) to "
        "this folder, for analyze_recording.py",
    )
    parser.add_argument(
        "--model", default="ridge", help="eyetrax gaze model (ridge, svr, ...)"
    )
    parser.add_argument(
        "--filter",
        choices=["kalman", "kalman_ema", "kde", "none"],
        default="kalman_ema",
        help="Smoothing applied to gaze after calibration",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.25, help="EMA strength for kalman_ema"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Contour probability for kde"
    )
    parser.add_argument(
        "--random-points",
        type=int,
        default=60,
        help="Most adaptive (blue-noise) points per calibration, as in "
        "eyetrax-build-model --random",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=10,
        help="Adaptive points per round; the model retrains after each round",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Load this trained model at startup if it exists, and save to it "
        "after each calibration",
    )
    parser.add_argument(
        "--exit-with-parent",
        action="store_true",
        help="Exit when stdin closes, i.e. when the process that started this "
        "one exits (used by the Vite dev server)",
    )
    return parser.parse_args()


def on_stdin_closed(callback):
    def watch():
        sys.stdin.read()  # returns at EOF
        callback()

    threading.Thread(target=watch, daemon=True).start()


def measure_camera(cap, timeout: float = 5.0):
    """The frame size and rate `cap` delivers, or None if it delivers nothing"""
    frames, start, deadline = 0, None, time.time() + timeout
    while frames < 20 and time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        if start is None:
            start = time.time()  # the first frame can take a while to arrive
            size = frame.shape[1], frame.shape[0]
        frames += 1
    if frames < 2:
        return None
    return size, (frames - 1) / max(time.time() - start, 1e-6)


def open_configured_camera(index: int, resolution: str):
    """
    Opens the webcam at `resolution` (WIDTHxHEIGHT), falling back to its own
    default if it can't deliver that. Returns the camera, size and frame rate
    """
    cap = open_camera(index)
    if resolution != "default":
        w, h = (int(v) for v in resolution.lower().split("x"))
        # MJPG lets most webcams send 720p and up at full frame rate
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        measured = measure_camera(cap)
        if measured:
            return cap, *measured
        print(f"[eyetrax-server] Camera gave no frames at {resolution}; using its default")
        cap.release()
        cap = open_camera(index)
    measured = measure_camera(cap)
    if not measured:
        raise RuntimeError(f"Camera {index} gives no frames")
    return cap, *measured


async def main():
    args = parse_args()

    estimator = GazeEstimator(model_name=args.model)
    cap, size, fps = open_configured_camera(args.camera, args.resolution)
    print(f"[eyetrax-server] Camera {size[0]}x{size[1]} at {fps:.0f} fps")
    if fps < 15:
        print("[eyetrax-server] Low frame rate: try --resolution default")
    tracker = Tracker(estimator, cap, args)
    tracker.fps = fps
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        print(f"[eyetrax-server] Recording calibrations to {args.record.resolve()}")
    if args.model_file and args.model_file.is_file():
        if tracker.load_model(args.model_file):
            print(f"[eyetrax-server] Loaded gaze model from {args.model_file}")
        else:
            print(f"[eyetrax-server] {args.model_file} is from an older version; recalibrate")
    tracker.start()

    loop = asyncio.get_running_loop()
    stopped = loop.create_future()
    if args.exit_with_parent:
        on_stdin_closed(
            lambda: loop.call_soon_threadsafe(
                lambda: stopped.done() or stopped.set_result(None)
            )
        )

    try:
        async with serve(
            lambda ws: handle(ws, tracker),
            args.host,
            args.port,
            origins=LOCAL_ORIGINS,
        ):
            print(f"[eyetrax-server] Listening on ws://{args.host}:{args.port}")
            await stopped
    finally:
        tracker.stop()
        print("[eyetrax-server] Stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
