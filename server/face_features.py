"""
Face measurements that eyetrax computes but doesn't return.

eyetrax's GazeEstimator.extract_features normalises the face landmarks into a
head-fixed frame: centred between the eyes (in early 2025 eyetrax centred them
on the nose tip), turned upright and scaled by the eye-corner distance, plus
three head angles. The eye features then stay the same wherever the head is.
But where the head is decides where a given eye angle lands on the screen, so
leaning sideways or towards the screen moved the cursor.

head_pose() adds the head's pose back, from the same FaceLandmarker result:
where the nose tip is, how the nose sits between the eyes and the chin (turn
and nod), the tilt of the eyes, and the face's size (distance). The
calibration includes a step where the user moves their head while looking at
one target, which teaches eyetrax's model to cancel head movement with them.
(That step has since been removed; see the README.)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# MediaPipe face mesh landmarks
NOSE_TIP = 1
CHIN = 152
EYE_OUTER = (33, 263)

# A meaningful movement in each head_pose() value: x, y (face widths), yaw,
# pitch (nose shift), roll (radians), size (log). See pose_scaling()
POSE_UNITS = np.array([0.6, 0.6, 0.08, 0.08, 0.2, 0.2])


class LandmarkTap:
    """
    Stands in for eyetrax's FaceLandmarker and keeps the landmarks it finds,
    so they are reused here rather than running the face model twice
    """

    def __init__(self, landmarker):
        self._landmarker = landmarker
        self.landmarks = None
        self.size = None

    @classmethod
    def install(cls, estimator) -> "LandmarkTap":
        # eyetrax 0.4's GazeEstimator keeps its FaceLandmarker here and calls
        # its detect_for_video() from extract_features()
        if not hasattr(estimator, "_face_landmarker"):
            raise RuntimeError("Unsupported eyetrax version: see requirements.txt")
        tap = cls(estimator._face_landmarker)
        estimator._face_landmarker = tap
        return tap

    def detect_for_video(self, image, timestamp_ms):
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self.landmarks = result.face_landmarks[0] if result.face_landmarks else None
        self.size = (image.width, image.height)
        return result

    def close(self):
        self._landmarker.close()


def head_pose(landmarks, size) -> np.ndarray:
    """
    Head pose from the nose tip, chin and outer eye corners, in units that
    don't depend on the camera:

      x, y   nose tip offset from the image centre, in face widths
      yaw    nose tip offset from between the eyes, along the eye line, in
             eye widths: the nose sticks out, so it swings as the head turns
      pitch  nose tip's drop below the eye line as a share of the chin's:
             nodding changes it
      roll   tilt of the eye line, radians
      size   log of the face's size (eyes-to-chin by eye width, in 3D, so
             turning barely changes it) in image widths: grows as the head
             comes closer
    """
    w, h = size

    def px(i):
        # MediaPipe x, y are fractions of the image size; z is on x's scale
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h, lm.z * w])

    left, right, nose, chin = px(EYE_OUTER[0]), px(EYE_OUTER[1]), px(NOSE_TIP), px(CHIN)
    mid = (left + right) / 2
    eye_line = (right - left)[:2]
    roll = math.atan2(eye_line[1], eye_line[0])
    c, s = math.cos(roll), math.sin(roll)

    def upright(p):
        dx, dy = (p - mid)[:2]
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    n, ch = upright(nose), upright(chin)
    eye_width = np.linalg.norm(eye_line) + 1e-9
    width3 = np.linalg.norm(right - left) + 1e-9
    height3 = np.linalg.norm(chin - mid) + 1e-9
    return np.array([
        (nose[0] - w / 2) / width3,
        (nose[1] - h / 2) / width3,
        n[0] / eye_width,
        n[1] / (ch[1] + 1e-9),
        roll,
        math.log(math.sqrt(width3 * height3) / w),
    ])


def pose_scaling(X: np.ndarray) -> np.ndarray:
    """
    eyetrax's `variable_scaling` for features ending in head_pose(): eyetrax
    divides every feature by its spread during calibration, so a head value
    that barely changed then (someone who can't move, or didn't in some way)
    became huge when it changed later, throwing the gaze to the screen edge.
    This undoes that for the head-pose values and measures them in POSE_UNITS
    """
    std = X[:, -len(POSE_UNITS):].std(axis=0)
    std[std == 0] = 1.0  # as eyetrax's StandardScaler does
    scaling = np.ones(X.shape[1])
    scaling[-len(POSE_UNITS):] = std / POSE_UNITS
    return scaling


def eye_aspect_ratio(landmarks) -> float:
    """eyetrax's eye aspect ratio, averaged over both eyes"""

    def ratio(inner, outer, top, bottom):
        def pt(i):
            return np.array([landmarks[i].x, landmarks[i].y])

        width = np.linalg.norm(pt(outer) - pt(inner))
        return np.linalg.norm(pt(top) - pt(bottom)) / (width + 1e-9)

    return (ratio(133, 33, 159, 145) + ratio(362, 263, 386, 374)) / 2


class BlinkDetector:
    """
    eyetrax counts a blink when the eye aspect ratio drops below 80% of its
    recent mean. Looking down narrows the eyes about that much, so gaze at
    the bottom of the screen could read as blinking: the cursor froze there
    and bottom calibration targets lost samples. Here the reference is the
    open eye (a high percentile of the last 1.5 s), and a blink has to close
    the eyes below 60% of it, which a real blink does easily.
    """

    def __init__(self, history: int = 45, ratio: float = 0.6, min_history: int = 15):
        self._history = deque(maxlen=history)
        self._ratio = ratio
        self._min_history = min_history

    def update(self, ear: float) -> bool:
        self._history.append(ear)
        if len(self._history) < self._min_history:
            return bool(ear < 0.2)  # eyetrax's threshold until there is a history
        return bool(ear < np.percentile(self._history, 90) * self._ratio)
