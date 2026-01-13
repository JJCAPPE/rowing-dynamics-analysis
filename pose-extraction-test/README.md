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

## CLI

### Annotate (creates `run.json`)

```bash
python -m rowing_pose.cli annotate --video trimmed.mov --out out/
```

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
- `stabilization.npz`
- `crop_boxes.npy`
- `pose2d.npz`
- `pose3d.npz` (optional)
- `angles.csv`
- `metrics.json`
- `debug/` videos:
  - `stabilized.mp4`
  - `crop_boxes.mp4`
  - `pose2d_overlay.mp4`
  - `angles_overlay.mp4` (keypoints + computed angles overlaid on the original frames)
