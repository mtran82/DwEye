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

`npm run dev` in `../template` starts this server too, and Ctrl+C stops both.
`EYETRAX_ARGS` passes it flags from the table below, e.g. in PowerShell:
`$env:EYETRAX_ARGS='--record recordings'; npm run dev`. To run the server on its
own instead, start the front end with `EYETRAX_SERVER=off`, run
`.venv/Scripts/python eyetrax_server.py`, and stop it with Ctrl+C.

Open the page full screen (F11). Calibration starts automatically once your face is in view. Targets are
sampled in page coordinates, so the page restarts calibration when the window is
resized.

## Calibration

1. **Grid**: eyetrax's 9-point calibration, then the model trains.
2. **Adaptive rounds**: rounds of 10 more targets from eyetrax's
   `BlueNoiseSampler`, as in `eyetrax-build-model` (`calibration/adaptive.py`);
   the model retrains after each on every point so far (eyetrax's own version
   retrains on the random points alone). No gaze cursor is shown, as it pulls
   the eyes off the targets. After two rounds (29 points, about a minute),
   calibration only ends once the last round's confidence is above 80%
   (`TARGET_CONFIDENCE` in `template/src/calibration.js`); below that it adds
   another round, until `--random-points` (60 extra) are used up.
3. **Result**: tracking confidence, then a choice of *Add 10 points* (if any
   are left) or *Start typing*. Pick one by looking at it for 1.5 s.

### Re-centering and the blink gesture

Blinking 5 times quickly (each within 1.2 s of the last;
`template/src/blinkGesture.js`) is a hands-free command:

| When | Five blinks |
|---|---|
| After calibrating (result screen or keyboard) | Re-center: 5 dots (center, then near each corner) from where the head is now, about 10 s, then back to typing |
| Within 8 s of a re-center | Recalibrate fully |
| Calibration failed, cancelled, or taken by another window | Calibrate again |

A blink counter shows at the top of the screen while blinking. Re-centering adds
its samples to the calibration with 5 times the weight (`RECENTER_WEIGHT`) and
retrains, so the model learns the new head position and keeps what it knew.
On three real recordings, re-centering with 5 dots after the head had moved
brought the error on that round's other targets from 270 to 87 px, against 117
for correcting an offset and 97 for an affine map.

### Head movement

eyetrax normalises the face landmarks into a head-fixed frame: centred between
the eyes, turned upright and scaled to the eye-corner distance. (In early 2025
it centred them on the nose tip instead.) The eye features then ignore where
the head is, but leaning sideways or towards the screen still moves where the
eyes land on it. `face_features.head_pose` measures the head from the same face
landmarks, without running the face model twice: the nose tip's position, how
the nose sits between the eyes and the chin (turning and nodding), the tilt of
the eyes, and the face's size, from eye width and eyes-to-chin (its distance).
These six numbers are added to eyetrax's features, in fixed units
(`POSE_UNITS`, via eyetrax's `variable_scaling`) rather than eyetrax's usual
division by how much each varied during calibration, which turned any later
movement into a huge value for someone who had barely moved.

An earlier version also had a calibration step where the user moved their head
in five guided ways while looking at one dot. It was removed: people with ALS
often can't move their head that way or find it tiring, and on real recordings
a webcam still couldn't keep the gaze accurate through head movement (about
95 px with the head still, 140 to 175 px while it moved). Re-centering after a
change of position does better.

The server also uses its own blink detector (`BlinkDetector`): eyetrax's
flags a blink when the eyes narrow to 80% of normal, which looking down at the
bottom of the screen can do, freezing the cursor there.

### Gaze model

`gaze_model.py`. eyetrax trains one ridge model with alpha = 1. On a real
recording it was accurate while the head stayed in poses seen during
calibration, but drifted when the head moved otherwise: with ~490 features it
had fitted small landmark changes that move with the head. Much stronger
regularisation stopped the drift but cost some precision. So the server trains
both (alpha 0.003 and 0.3 times the number of samples) and blends them: half
each while the head is within its calibrated poses, moving to the strong one
as the head goes beyond them.

On that recording (6 rounds of new targets; 5 parts of the since-removed
head-movement step, each held out of training):

| Settled error, px | New targets | Head movement | Head movement, per frame |
|---|---|---|---|
| eyetrax's alpha = 1, with head pose | 87 | 140 | 258 |
| Blended | 83 | 45 | 105 |

This was tuned on one person's recording; `analyze_recording.py` shows how it
does on others.

Only one page calibrates at a time. A page that starts calibrating takes over;
the other shows *Calibrating in another window* (R takes it back). Before this,
two open pages (a second tab, or a leftover preview) mixed their targets into
one calibration.

Tracking confidence is the share of gaze samples that land within 5% of the
screen diagonal of a target (about 110 px at 1920x1080). It is measured on each
adaptive round **before** the model trains on it, so it reflects targets the
model has not seen. It uses raw gaze; the on-screen cursor is smoothed. High is
above 80% (what calibration aims for), Medium 50% or more.

Keys, for a helper: Esc cancels (adding points or re-centering is skipped and
the previous model kept), R recalibrates, A adds points, Enter starts typing.

| Flag | Default | |
|------|---------|-|
| `--camera` | `0` | Webcam index |
| `--filter` | `kalman_ema` | Gaze smoothing: `kalman`, `kalman_ema`, `kde` or `none` |
| `--ema-alpha` | `0.25` | EMA strength for `kalman_ema` |
| `--confidence` | `0.5` | Contour probability for `kde` |
| `--model` | `ridge` | eyetrax model: `ridge`, `elastic_net`, `svr` or `tiny_mlp` |
| `--random-points` | `60` | Most adaptive points per calibration (`eyetrax-build-model --random`) |
| `--retrain-every` | `10` | Adaptive points per round |
| `--model-file` | none | Load this model at startup if it exists, and save to it after each calibration |
| `--port` | `8765` | Change `VITE_EYETRAX_URL` in the front end to match |
| `--resolution` | `1280x720` | Camera resolution to ask for, or `default`. More pixels on the eyes give steadier iris landmarks; falls back to the camera's default if it can't |
| `--record` | none | Save each calibration to this folder (see below) |

## Recording and analysing a calibration

With `--record recordings`, each calibration is saved as
`recordings/calibration-<time>.npz`, updated after every training: the face
landmarks MediaPipe found for each sample (no images), the features trained on,
the targets, and which samples were head movement. `recordings/` is ignored by
git: it describes the face of whoever calibrated.

```bash
.venv/Scripts/python analyze_recording.py recordings/calibration-<time>.npz
```

replays it and compares models on two tests: targets the model had not seen
(each adaptive round, predicted from everything before it), and head movement
(each head-movement part, predicted by a model trained without it). It reports
per-frame error, the settled error a smoothed cursor shows, centre vs border,
and jitter, for eyetrax alone, the current model and variations of it.
