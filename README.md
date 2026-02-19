# rowing-video-analysis

Video-first rowing biomechanics research pipeline.

The current working path is:

1. Extract pose and stroke signals from video (`sports2d_app/`).
2. Infer catch/finish drive events and optionally align to RP3 clean exports (`inference/`).
3. Build matched pose/force rows for force-curve modeling.

## Repository Modules

- `sports2d_app/`: main Sports2D + MotionBERT pipeline (Streamlit app and terminal wizard).
- `inference/`: drive-event detection, RP3 stroke matching, and matched segment export.
- `rp3-extraction/`: RP3 CSV cleanup/expansion utilities.
- `pose-extraction-test/`: annotation-first experimental pipeline kept for side-by-side testing.

## Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r sports2d_app/requirements.txt
```

## Typical Workflow

### 1) Run Sports2D pipeline

```bash
.venv/bin/python sports2d_app/app_cli.py
```

This creates a run folder under `sports2d_app/runs/<video_stem>_<timestamp>/`.

### 2) Infer drive events from stroke signal

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir sports2d_app/runs/<run_name> \
  --no-match-rp3 \
  --overlay-video
```

### 3) Optional RP3 matching + force/pose segment export

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir sports2d_app/runs/<run_name> \
  --match-rp3 \
  --rp3-clean-csv rp3-extraction/workouts/clean/<workout>-clean.csv \
  --anchor-rp3-stroke-number <stroke_number> \
  --active-side right
```

## Key Run Artifacts

Within `sports2d_app/runs/<run_name>/`:

- `stroke/stroke_signal.csv`: handle/machine stroke signal with catch/finish flags.
- `motionbert/angles_h36m.csv`: per-frame joint angles.
- `inference/drive_events.csv`: per-stroke catch/finish timing inferred from stroke signal.
- `inference/rp3_match_manifest.csv`: video stroke to RP3 row mapping (if RP3 matching enabled).
- `inference/rp3_pose_force_matched_segments.csv`: canonical pose features aligned to RP3 force bins.

## Notes

- If `--run-dir` or RP3 arguments are omitted in a TTY session, inference scripts can prompt interactively.
- Generated artifacts (videos, NPZ/CSV outputs) are intentionally not committed.
