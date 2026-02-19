# inference

Post-processing CLIs that run after `sports2d_app` and produce stroke-level inference outputs.

## Prerequisites

- A Sports2D run directory under `sports2d_app/runs/<run_name>/`.
- At minimum: `stroke/stroke_signal.csv`.
- For overlay video output: `input/<video_file>` must exist in the run.
- For RP3 matching: a clean RP3 CSV in `rp3-extraction/workouts/clean/`.

## `inference_cli.py`

Computes drive events from `stroke_signal.csv` (catch = local minima, finish = local maxima on handle distance), and can optionally match video strokes to RP3 rows and export per-force-bin pose/force rows.

### Drive events only

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir sports2d_app/runs/<run_name> \
  --no-match-rp3 \
  --overlay-video
```

### Drive events + RP3 matching + matched segment export

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir sports2d_app/runs/<run_name> \
  --match-rp3 \
  --rp3-clean-csv rp3-extraction/workouts/clean/<workout>-clean.csv \
  --anchor-rp3-stroke-number <stroke_number> \
  --active-side right
```

`--anchor-rp3-stroke-number` is the recommended anchor input for the first matched video stroke (default video anchor is `--anchor-video-stroke-idx 0`).

### Outputs (`<run_dir>/inference/`)

- `drive_events.csv`
- `stroke_signal_with_drive_events.csv`
- `drive_events_summary.json`
- `drive_phase_overlay.mp4` (when overlay is enabled)
- `rp3_match_manifest.csv` (when RP3 matching is enabled)
- `rp3_pose_force_matched_segments.csv` (when RP3 matching is enabled)
- `rp3_match_summary.json` (when RP3 matching is enabled)

## `match_rp3_cli.py`

Standalone matcher if `inference/drive_events.csv` already exists and you only want stroke-to-RP3 alignment.

```bash
.venv/bin/python inference/match_rp3_cli.py \
  --run-dir sports2d_app/runs/<run_name> \
  --rp3-clean-csv rp3-extraction/workouts/clean/<workout>-clean.csv \
  --anchor-rp3-stroke-number <stroke_number>
```

### Outputs (`<run_dir>/inference/`)

- `rp3_match_manifest.csv`
- `rp3_video_aligned_strokes.csv`
- `rp3_match_summary.json`

## RP3 CSV Requirements

For matching (`match_rp3_cli.py` and `inference_cli.py --match-rp3`):

- `stroke_number`
- `time`
- `drive_time`
- `recover_time`

For matched segment export (`rp3_pose_force_matched_segments.csv`), RP3 CSV must also contain:

- `stroke_length`
- force bins named `force_at_<distance>cm` (for example `force_at_2.2cm`)

Use `rp3-extraction/expand_rp3_curve_data.py` to expand raw RP3 exports into this format.
