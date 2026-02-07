# Sports2D App

Standalone Streamlit UI that runs **Sports2D only** (no manual annotation) and always produces a MotionBERT 3D overlay for **Person 0**. All Sports2D outputs are preserved.

## Setup
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r sports2d_app/requirements.txt
```

Sports2D uses a local clone at:

```
sports2d_app/third_party/Sports2D
```

MotionBERT uses the local copy at:

```
sports2d_app/third_party/MotionBERT
```

If Sports2D fails with a Pose2Sim import error, ensure Pose2Sim is upgraded:

```bash
pip install --upgrade "Pose2Sim>=0.10.40"
```

## Run

```bash
.venv/bin/python -m streamlit run sports2d_app/app.py
```

CLI version (no Streamlit UI):

```bash
.venv/bin/python sports2d_app/app_cli.py
```

`app_cli.py` now includes an integrated stroke-tracking stage (machine + handle), and
produces:
- stroke signals aligned to frame/time
- a merged angles plot with a bottom handle-domain subplot
- overlay video with 3D + handle/machine tracking

## Stroke Phase / Handle-Machine Tracking (Standalone)
If you want to run stroke tracking separately from the full CLI pipeline:

```bash
.venv/bin/python sports2d_app/stroke_signal.py \
  --video /absolute/path/to/video.mp4 \
  --out-dir /absolute/path/to/output_dir \
  --angles-csv /absolute/path/to/angles_h36m.csv \
  --annotate \
  --debug-video
```

What it outputs:
- `stroke_signal.csv`: per-frame handle/machine centers, relative distance, velocity, stroke phase, catch/finish flags
- `stroke_signal.npz`: same data + tracked boxes for reproducibility
- `angles_h36m_with_stroke.csv` (if `--angles-csv` is provided)
- `angles_h36m_with_stroke_plot.png` (combined angles + stroke signal plot)
- `stroke_tracking_debug.mp4` (if `--debug-video` is enabled)

## Outputs
Each run writes to:

```
sports2d_app/runs/<video_stem>_<timestamp>/
```

Contents:
- `sports2d/` (raw Sports2D folder, including annotated video, TRC, MOT)
- `exports/` (CSV + NPZ exports of all points/angles)
- `motionbert/` (`pose3d.npz`, `angles_h36m.csv`, `metrics.json`)
- `overlay/pose3d_overlay.mp4`
- `results.zip` (all outputs above)
