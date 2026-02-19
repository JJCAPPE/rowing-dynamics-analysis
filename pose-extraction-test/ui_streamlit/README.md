# Streamlit UI (pose-extraction-test)

Browser UI for the `rowing_pose` pipeline in `pose-extraction-test/`.

It covers:

- video selection/upload
- annotation steps (anchor, rigger bbox, athlete bbox, scale points)
- one-click pipeline run
- 3D overlay and artifact downloads

## Install

From `pose-extraction-test/`:

```bash
source .venv/bin/activate
pip install -r ui_streamlit/requirements-ui.txt
```

For full 2D+3D execution, also install the pipeline package dependencies:

```bash
pip install -e .
```

Optional strict-ID tracking extras:

```bash
pip install -e ".[tracking]"
```

## Run

From `pose-extraction-test/`:

```bash
.venv/bin/python -m streamlit run ui_streamlit/app.py
```

## Outputs

Each run writes to an `out_<video_stem>/` folder in `pose-extraction-test/`.
If that folder already exists, the UI appends a timestamp suffix.

Typical contents:

- `run.json`
- `stabilization.npz`
- `pose2d.npz`
- `pose3d.npz` (when 3D is enabled)
- `angles.csv`
- `metrics.json`
- `debug/source_video.<ext>`
- `debug/angles_overlay.mp4`
- `debug/pose3d_overlay.mp4` (when 3D is enabled)

## MotionBERT assets

The UI checks for MotionBERT config in:

- `pose-extraction-test/third_party/MotionBERT/`

Default checkpoints are cached in:

- `pose-extraction-test/models/motionbert/`

If missing, the UI can download required model weights automatically.
