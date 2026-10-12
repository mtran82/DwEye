"""
The gaze model: eyetrax's ridge model twice, blended by head pose.

eyetrax trains ridge regression with alpha=1 on ~490 features. On a real
recording (see analyze_recording.py) that fit the calibration closely, but
when the head moved in a way calibration hadn't covered, it read the changed
landmarks as gaze and the cursor drifted off: 140 px settled error on a held-out
head movement, against 87 px on new targets with the head roughly still.
Much stronger regularisation stopped the drift (about 40 px) but lost some
precision where the head stayed within its calibrated poses.

So both are trained, on the same features:

- precise: alpha = PRECISE x samples, close to eyetrax's
- robust: alpha = ROBUST x samples

Within the head poses seen during calibration the gaze is a mix of the two
(BASE of the robust one); as the head moves beyond them it shifts to the
robust one, fully at FULL_AT of the calibrated range beyond. On the recording
this gave 83 px on new targets and 45 px on held-out head movement. Alpha
scales with the number of samples because ridge's penalty is fixed while the
fit's error grows with them.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from eyetrax.models import create_model
from face_features import POSE_UNITS, pose_scaling

PRECISE = 0.003
ROBUST = 0.3
BASE = 0.5
FULL_AT = 0.5

POSE = slice(-len(POSE_UNITS), None)  # head_pose() ends the features


class Saved:
    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)


def load(path: str | Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def make_model(name: str = "ridge"):
    """The blended model for eyetrax's ridge; any other eyetrax model alone"""
    return PoseBlendedModel() if name == "ridge" else SingleModel(name)


class SingleModel(Saved):
    """An eyetrax model (--model), trained as eyetrax does but with pose_scaling"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.n_features = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = create_model(self.name)
        self.model.train(X, y, pose_scaling(X))
        self.n_features = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PoseBlendedModel(Saved):
    def __init__(self):
        self.precise = None
        self.robust = None
        self.n_features = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        scaling = pose_scaling(X)
        self.precise = create_model("ridge", alpha=PRECISE * n)
        self.precise.train(X, y, scaling)
        self.robust = create_model("ridge", alpha=ROBUST * n)
        self.robust.train(X, y, scaling)
        poses = X[:, POSE]
        self.lo = np.percentile(poses, 2, axis=0)
        self.hi = np.percentile(poses, 98, axis=0)
        self.span = np.maximum(self.hi - self.lo, 1e-6)
        self.n_features = X.shape[1]

    def outside(self, X: np.ndarray) -> np.ndarray:
        """How far each head pose is beyond the calibrated ones, in calibrated ranges"""
        poses = X[:, POSE]
        beyond = np.maximum(self.lo - poses, poses - self.hi) / self.span
        return beyond.max(axis=1).clip(0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        k = BASE + (1 - BASE) * (self.outside(X) / FULL_AT).clip(0, 1)
        return (1 - k[:, None]) * self.precise.predict(X) + k[:, None] * self.robust.predict(X)
