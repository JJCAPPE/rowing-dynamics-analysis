# rowing-pose (pose-extraction-test)

This folder contains an **offline, modular pipeline** to process a rowing video into:

- boat-relative stabilized frames
- smooth athlete crops
- **2D keypoints** (MMPose)
- optional **3D lift** (MotionBERT)
- per-frame **angles/metrics**

It follows the design/outputs described in `planning.MD`.

## Install

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

> Notes:
>
> - `mmpose/mmcv/mmdet` can be heavy and may require platform-specific wheels.
> - MotionBERT is not on PyPI; the pipeline supports using a local MotionBERT checkout (see CLI flags).
> - Optional strict ID tracking uses extra deps: `pip install -e ".[tracking]"`.

## CLI

### Annotate (creates `run.json`)

```bash
python -m rowing_pose.cli annotate --video trimmed.mov --out out/
```

The annotator asks for a **rigger bbox** (tight box around oarlock hardware) and an
**athlete bbox**. The rigger box is used for stabilization and unambiguous tracking.

### Run pipeline

```bash
python -m rowing_pose.cli run --video trimmed.mov --out out/
```

### Debug renders from saved artifacts

```bash
python -m rowing_pose.cli debug --run out/run.json
```

## Outputs (in `--out`)

- `run.json`
- `rigger_track.npz` (if rigger bbox provided)
- `stabilization.npz`
- `crop_boxes.npy`
- `person_track.npz` (optional, if strict ID tracking enabled)
- `pose2d.npz`
- `pose3d.npz` (optional)
- `angles.csv`
- `metrics.json`
- `debug/` videos:
  - `rigger_track.mp4` (if rigger bbox provided)
  - `stabilized.mp4`
  - `crop_boxes.mp4`
  - `person_track.mp4` (optional, if strict ID tracking enabled)
  - `pose2d_overlay.mp4`
  - `angles_overlay.mp4` (keypoints + computed angles overlaid on the original frames)
