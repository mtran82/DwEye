"""
WebSocket bridge between eyetrax and the DwEye front end.

eyetrax's own calibration routines draw into an OpenCV window. Here the
browser draws the calibration targets instead, and this process owns the
webcam and eyetrax's GazeEstimator: it samples eye features while the front
end holds a target on screen, trains the model, then streams gaze back.

All coordinates are CSS pixels relative to the browser viewport.

Protocol (JSON text messages):

  client -> server
    {"type": "calibration_start", "width": w, "height": h}
    {"type": "point_start", "x": x, "y": y}   start sampling for a target
    {"type": "point_end"}                     stop; replies "point_captured"
    {"type": "calibration_finish"}            train; replies "calibration_done"
    {"type": "calibration_cancel"}

  server -> client
    {"type": "hello", "calibrated": bool}
    {"type": "frame", "face": bool, "blink": bool, "gaze": [x, y] | null}
    {"type": "point_captured", "samples": n}
    {"type": "calibration_done", "ok": bool, "samples": n, "points": n, "error": str}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

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

# Only pages served from this machine may connect (browsers always send an
# Origin header; non-browser clients such as test scripts send none)
LOCAL_ORIGINS = [re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"), None]

# A 2D mapping needs at least three non-collinear targets
MIN_POINTS = 3


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
        self.estimator = estimator
        self.cap = cap
        self.args = args

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._viewport = None
        self._target = None
        self._point_feats = []
        self._feats, self._targs = [], []

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
            feats, blink = self.estimator.extract_features(frame)
            face = feats is not None
            usable = face and not blink
            gaze = None
            with self._lock:
                if usable and self._target is not None:
                    self._point_feats.append(feats)
                if usable and self.calibrated:
                    x, y = self.estimator.predict(np.array([feats]))[0]
                    gaze = self.smoother.step(int(x), int(y))
                self.frame = {"face": face, "blink": bool(blink), "gaze": gaze}
                self.seq += 1

    def snapshot(self):
        with self._lock:
            return self.seq, self.frame

    # --- calibration, driven by the front end -------------------------------

    def start_calibration(self, width: int, height: int):
        with self._lock:
            self._viewport = (int(width), int(height))
            self._feats, self._targs = [], []
            self._target = None
            self._point_feats = []

    def start_point(self, x: float, y: float):
        with self._lock:
            self._target = (float(x), float(y))
            self._point_feats = []

    def end_point(self) -> int:
        with self._lock:
            n = len(self._point_feats)
            if self._target is not None:
                self._feats.extend(self._point_feats)
                self._targs.extend([self._target] * n)
            self._target = None
            self._point_feats = []
            return n

    def cancel(self):
        with self._lock:
            self._target = None
            self._point_feats = []
            self._feats, self._targs = [], []

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
            self.estimator.train(X, y)
            self.smoother = self._make_smoother(X, y)
            self.calibrated = True
            if self.args.model_file:
                self.estimator.save_model(self.args.model_file)
        print(f"[eyetrax-server] Calibrated with {samples} samples from {points} points")
        return {"ok": True, "samples": samples, "points": points}

    def _make_smoother(self, X=None, y=None):
        name = self.args.filter
        if name == "kde":
            w, h = self._viewport or get_screen_size()
            return KDESmoother(w, h, confidence=self.args.confidence)
        if name in ("kalman", "kalman_ema"):
            kf = make_kalman()
            if X is not None:
                tune_kalman(kf, self.estimator.predict(X), y)
            if name == "kalman":
                return KalmanSmoother(kf)
            return KalmanEMASmoother(kf, ema_alpha=self.args.ema_alpha)
        return NoSmoother()

    def load_model(self, path: Path):
        self.estimator.load_model(path)
        self.smoother = self._make_smoother()
        self.calibrated = True


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


async def handle(ws, tracker: Tracker):
    await send_json(ws, {"type": "hello", "calibrated": tracker.calibrated})
    streamer = asyncio.create_task(stream_frames(ws, tracker))
    try:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "calibration_start":
                tracker.start_calibration(msg["width"], msg["height"])
            elif kind == "point_start":
                tracker.start_point(msg["x"], msg["y"])
            elif kind == "point_end":
                n = tracker.end_point()
                await send_json(ws, {"type": "point_captured", "samples": n})
            elif kind == "calibration_finish":
                result = await asyncio.to_thread(tracker.finish_calibration)
                await send_json(ws, {"type": "calibration_done", **result})
            elif kind == "calibration_cancel":
                tracker.cancel()
    except ConnectionClosed:
        pass
    finally:
        streamer.cancel()
        # Don't keep sampling for a target nobody is showing any more
        tracker.cancel()


def parse_args():
    parser = argparse.ArgumentParser(description="eyetrax WebSocket server for DwEye")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
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


async def main():
    args = parse_args()

    estimator = GazeEstimator(model_name=args.model)
    tracker = Tracker(estimator, open_camera(args.camera), args)
    if args.model_file and args.model_file.is_file():
        tracker.load_model(args.model_file)
        print(f"[eyetrax-server] Loaded gaze model from {args.model_file}")
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
