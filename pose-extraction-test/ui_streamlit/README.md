# Streamlit UI (pose-extraction-test)

This folder contains a **Streamlit wrapper UI** for the existing `rowing_pose` pipeline.

## Install

From `pose-extraction-test/`:

```bash
source .venv/bin/activate
pip install -r ui_streamlit/requirements-ui.txt
```

## Run

From `pose-extraction-test/`:

```bash
.venv/bin/python -m streamlit run ui_streamlit/app.py
```

## Streamlit Cloud

- App file: `pose-extraction-test/ui_streamlit/app.py`
- Dependencies: root `requirements.txt` installs the UI stack for deployment.
- Full pipeline (2D/3D) requires extra ML deps (torch, mmpose, mmcv, mmdet, pyyaml, easydict).
  The UI will disable those stages if missing.
- Optional strict ID tracking uses extra deps: `pip install -e ".[tracking]"`.

## Annotation notes

The UI now collects:

- **Anchor point** (reference)
- **Rigger bbox** (tight box around oarlock hardware, used for stabilization)
- **Athlete bbox** (initial crop)
- **Scale points** + known distance

## MotionBERT checkpoint (one-time)

The 3D overlay requires MotionBERT weights. The app will look for (and reuse):

- `pose-extraction-test/third_party/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`

If it’s missing, the UI will offer a **one-time download** and save it locally under `third_party/MotionBERT/checkpoint/`.

## Outputs

The UI writes results to:

- `pose-extraction-test/out_<video_name>/`

Including:

- the same artifacts the CLI produces (`run.json`, `pose2d.npz`, `pose3d.npz`, `angles.csv`, etc.)
- plus `debug/source_video.<ext>`
- plus `debug/pose3d_overlay.mp4` (final UI-rendered 3D overlay)
- plus `debug/rigger_track.mp4` (if rigger bbox provided)
- plus `debug/person_track.mp4` (if strict ID tracking enabled)

