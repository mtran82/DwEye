# eyetrax server

Connects [eyetrax](https://github.com/ck-zhang/eyetrax) to the React front end in `../template`.
The server owns the webcam and eyetrax's `GazeEstimator`. The browser draws the
calibration targets and tells the server where they are. The server samples eye
features for each target, trains the eyetrax model, then streams gaze back over a
WebSocket (`ws://localhost:8765`). The message format is documented at the top of
`eyetrax_server.py`.

## Setup

eyetrax pins `numpy<2`, so use Python 3.12 or older.

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

To use a local eyetrax checkout instead of the PyPI release, run
`uv pip install -e path/to/eyetrax`.

## Run

`npm run dev` in `../template` starts this server too, and Ctrl+C stops both. To
run the server on its own, for example with the flags below, use
`.venv/Scripts/python eyetrax_server.py` and stop it with Ctrl+C.

Open the page full screen (F11). Calibration starts automatically once your face is in view. Targets are
sampled in page coordinates, so the page restarts calibration when the window is
resized. Press R to recalibrate and Esc to cancel.

| Flag | Default | |
|------|---------|-|
| `--camera` | `0` | Webcam index |
| `--filter` | `kalman_ema` | Gaze smoothing: `kalman`, `kalman_ema`, `kde` or `none` |
| `--ema-alpha` | `0.25` | EMA strength for `kalman_ema` |
| `--confidence` | `0.5` | Contour probability for `kde` |
| `--model` | `ridge` | eyetrax model: `ridge`, `elastic_net`, `svr` or `tiny_mlp` |
| `--model-file` | none | Load this model at startup if it exists, and save to it after each calibration |
| `--port` | `8765` | Change `VITE_EYETRAX_URL` in the front end to match |
