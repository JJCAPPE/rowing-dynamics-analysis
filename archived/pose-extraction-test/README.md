# pose-extraction-test (rowing-pose)

Annotation-first experimental pipeline for rowing video analysis.

It supports:

- reference-point stabilization
- optional rigger-based stabilization
- crop/person tracking
- 2D pose (MMPose)
- optional MotionBERT 3D lift
- angle/metric export and debug overlays

## Install

From `pose-extraction-test/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional strict-ID tracking extras:

```bash
pip install -e ".[tracking]"
```

## CLI

### 1) Create or update annotations (`run.json`)

```bash
python -m rowing_pose.cli annotate --video /path/to/video.mp4 --out out_run/
```

### 2) Run pipeline

```bash
python -m rowing_pose.cli run --video /path/to/video.mp4 --out out_run/
```

Useful flags:

- `--skip-2d`: reuse an existing `pose2d.npz`
- `--skip-3d`: skip MotionBERT lift
- `--pose-tracking-smooth-alpha <float>`: additional 2D keypoint EMA smoothing
- `--motionbert-root`, `--motionbert-config`, `--motionbert-ckpt`: custom MotionBERT paths

### 3) Regenerate debug videos from saved artifacts

```bash
python -m rowing_pose.cli debug --run out_run/run.json
```

## Outputs (in `--out`)

- `run.json`
- `stabilization.npz`
- `crop_boxes.npy`
- `pose2d.npz`
- `angles.csv`
- `metrics.json`
- `pose3d.npz` (if 3D enabled)
- `rigger_track.npz` (if rigger bbox is used)
- `person_track.npz` (if strict ID tracking is enabled)
- `debug/stabilized.mp4`
- `debug/crop_boxes.mp4`
- `debug/pose2d_overlay.mp4`
- `debug/angles_overlay.mp4`
- `debug/rigger_track.mp4` (if rigger bbox is used)
- `debug/person_track.mp4` (if strict ID tracking is enabled)

## Notes

- MotionBERT code is expected under `pose-extraction-test/third_party/MotionBERT`.
- Default model assets are cached under `pose-extraction-test/models/`.
- See `planning.MD` for pipeline design context.
