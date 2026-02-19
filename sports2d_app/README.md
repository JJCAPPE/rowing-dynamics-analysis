# Sports2D App

Primary video processing app for this repository.

It runs a 7-stage pipeline:

1. Sports2D pose/tracking
2. Export TRC/MOT to CSV/NPZ
3. MotionBERT input prep
4. MotionBERT 3D lift
5. Optional handle-machine stroke tracking
6. 3D overlay + angle plots
7. ZIP packaging

Both interfaces below write results to `sports2d_app/runs/<video_stem>_<timestamp>/`.

## Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r sports2d_app/requirements.txt
```

Notes:

- Sports2D code is expected at `sports2d_app/third_party/Sports2D`.
- MotionBERT code is expected at `sports2d_app/third_party/MotionBERT`.
- MotionBERT checkpoints are downloaded on demand into `sports2d_app/models/motionbert/`.

## Run (Streamlit UI)

```bash
.venv/bin/python -m streamlit run sports2d_app/app.py
```

The UI lets you choose model mode, detected person count/index, and optional stroke tracking settings.

## Run (Terminal wizard)

```bash
.venv/bin/python sports2d_app/app_cli.py
```

`app_cli.py` is interactive (no required CLI flags) and prompts for:

- source video
- Sports2D mode/model/device
- person index
- debug video policy
- optional handle/machine stroke tracking options

By default, the wizard lists videos from:

- `/Users/giacomo/dev/rowing-video-analysis/source-videos`
- `/Volumes/T9/rowing-research`

## Standalone Stroke Tracking

If you want to run stroke tracking independently:

```bash
.venv/bin/python sports2d_app/stroke_signal.py \
  --video /absolute/path/to/video.mp4 \
  --out-dir /absolute/path/to/output_dir \
  --angles-csv /absolute/path/to/angles_h36m.csv \
  --handle-source pose \
  --pose-npz /absolute/path/to/exports/video_points.npz \
  --annotate \
  --debug-video
```

If not using `--annotate`, provide machine geometry explicitly:

- `--machine-bbox x,y,w,h`
- `--machine-cable-point x,y`

## Output Layout

For each run directory:

- `input/`: copied input video used by pipeline.
- `sports2d/`: raw Sports2D outputs (`logs`, TRC, MOT, annotated video).
- `exports/`: parsed points/angles CSV + NPZ exports.
- `motionbert/`: `pose3d.npz`, `angles_h36m.csv`, `metrics.json`.
- `stroke/` (if enabled): `stroke_signal.csv`, `stroke_signal.npz`, merged angle/stroke CSV and plot, optional `stroke_tracking_debug.mp4`.
- `overlay/`: `pose3d_overlay.mp4` (when debug videos are enabled).
- `results.zip`: packaged outputs.

## Next Step

After generating a run, use `inference/inference_cli.py` (documented in `../inference/README.md`) to infer drive events and optionally match RP3 clean CSV data.
