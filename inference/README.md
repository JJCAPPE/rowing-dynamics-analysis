# inference

Post-processing CLIs that run after `sports2d_app` and produce stroke-level inference outputs.

## Prerequisites

- A Sports2D run directory under `runs/<run_name>/`.
- At minimum: `stroke/stroke_signal.csv`.
- For overlay video output: either `input/<video_file>` exists in the run, or `input_video_source.txt` points to an accessible source video path.
- For RP3 matching in `inference_cli.py`: put dirty RP3 CSV files in `<run_name>/rp3/`.

## `inference_cli.py`

Computes drive events from `stroke_signal.csv` (catch = local minima, finish = local maxima on handle distance), and can match video strokes to RP3 rows and export per-force-bin pose/force rows.

When RP3 matching is enabled, it now:

1. Selects a dirty RP3 CSV from `<run>/rp3/`.
2. Generates `<dirty_stem>-clean.csv` in the same `<run>/rp3/` folder.
3. Runs matching and segment export from that clean file.

### Drive events only

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir runs/<run_name> \
  --no-match-rp3 \
  --overlay-video
```

### Drive events + RP3 matching + matched segment export

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir runs/<run_name> \
  --anchor-rp3-stroke-number <stroke_number> \
  --active-side right
```

Notes:

- If dirty RP3 CSVs exist in `<run>/rp3/`, matching auto-runs unless you pass `--no-match-rp3`.
- If multiple dirty files exist, interactive mode prompts for selection; non-interactive mode requires `--rp3-dirty-csv <run>/rp3/<file>.csv`.
- `--anchor-rp3-stroke-number` is the recommended anchor input for the first matched video stroke (default video anchor is `--anchor-video-stroke-idx 0`).

### Outputs (`<run_dir>/inference/`)

- `drive_events.csv`
- `stroke_signal_with_drive_events.csv`
- `drive_events_summary.json`
- `drive_phase_overlay.mp4` (when overlay is enabled)
- `rp3_match_manifest.csv` (when RP3 matching is enabled)
- `rp3_pose_force_matched_segments.csv` (when RP3 matching is enabled)
- `rp3_pose_force_export_status.csv` (when RP3 matching is enabled)
- `rp3_match_summary.json` (when RP3 matching is enabled)

## `match_rp3_cli.py`

Standalone matcher if `inference/drive_events.csv` already exists and you only want stroke-to-RP3 alignment.

```bash
.venv/bin/python inference/match_rp3_cli.py \
  --run-dir runs/<run_name> \
  --rp3-clean-csv runs/<run_name>/rp3/<workout>-clean.csv \
  --anchor-rp3-stroke-number <stroke_number>
```

If `--rp3-clean-csv` is omitted, it auto-selects from `<run>/rp3/*-clean.csv` (single file auto, multiple interactive prompt, non-interactive requires explicit flag).

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

Dirty RP3 exports are converted to this format automatically by `inference_cli.py` using `rp3-extraction/expand_rp3_curve_data.py`.

## Matched Segment Export Semantics

In `rp3_pose_force_matched_segments.csv`:

- `force_n` is a per-stroke normalized PDF density over `s_force` (not absolute force units).
- `force_raw` stores the original RP3 bin value before normalization.
- `match_seq_idx` is the stable 0-based sequence index from `rp3_match_manifest.csv`.

`rp3_pose_force_export_status.csv` contains one row per matched stroke and explicitly reports:

- whether the stroke was exported to segment rows (`segment_exported`)
- row counts (`segment_rows_written`)
- drop reasons (`drop_reason`) when not exported
- normalization diagnostics (`raw_area_trapz`, `normalized_area_trapz`)
