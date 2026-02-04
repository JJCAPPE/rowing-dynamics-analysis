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
pose-extraction-test/third_party/Sports2D
```

MotionBERT uses the existing submodule at:

```
pose-extraction-test/third_party/MotionBERT
```

If Sports2D fails with a Pose2Sim import error, ensure Pose2Sim is upgraded:

```bash
pip install --upgrade "Pose2Sim>=0.10.40"
```

## Run

```bash
.venv/bin/python -m streamlit run sports2d_app/app.py
```

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
