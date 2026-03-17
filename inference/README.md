# inference

Post-processing CLIs that run after `sports2d_app` and produce stroke-level inference outputs, matched pose/force segments, and training dataset artifacts.

## Prerequisites

- A Sports2D run directory under `runs/<run_name>/`.
- At minimum: `stroke/stroke_signal.csv` and `motionbert/angles_h36m.csv`.
- For overlay video output: either `input/<video_file>` exists in the run, or `input_video_source.txt` points to an accessible source video path.
- For RP3 matching: put dirty RP3 CSV files in `<run_name>/rp3/`.

---

## `inference_cli.py`

Full end-to-end CLI. Runs drive-event detection from `stroke_signal.csv` using velocity-based catch/finish thresholds. When RP3 data is present, performs a two-pass calibration:

1. **Coarse detection** — initial catch/finish boundaries using `velocity_threshold`.
2. **RP3 match** — align video strokes to RP3 rows to obtain per-stroke RP3 drive durations.
3. **Calibrate** — sweep catch/finish velocity fractions to minimize drive-duration MAE vs RP3.
4. **Re-detect** — re-run detection with the calibrated fractions (`velocity_calibrated`).
5. **Final RP3 match** — rematching with calibrated events.
6. **Segment export** — export `rp3_pose_force_matched_segments.csv` with aligned pose and force.
7. **Training dataset** — automatically builds `training_dataset/` artifacts (PCA, fixed-grid sequences).

### Drive events only (no RP3)

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir runs/<run_name> \
  --no-match-rp3 \
  --overlay-video
```

Use `--finish-method` to control detection strategy:
- `velocity_calibrated` (default when RP3 is available) — calibrated per-run velocity fractions.
- `velocity_threshold` — fixed fraction of peak handle velocity.
- `position_max` — handle position maximum (original method; tends to over-extend the drive).

### Drive events + RP3 matching + segment export + training dataset

```bash
.venv/bin/python inference/inference_cli.py \
  --run-dir runs/<run_name> \
  --anchor-rp3-stroke-number <stroke_number> \
  --active-side right
```

Notes:

- If dirty RP3 CSVs exist in `<run>/rp3/`, matching auto-runs unless you pass `--no-match-rp3`.
- If multiple dirty files exist, interactive mode prompts for selection; non-interactive requires `--rp3-dirty-csv <run>/rp3/<file>.csv`.
- `--anchor-rp3-stroke-number` is the recommended anchor (default video anchor is `--anchor-video-stroke-idx 1`).
- The training dataset is built automatically after segment export. Pass `--no-build-dataset` to skip.

### Outputs (`<run_dir>/inference/`)

- `drive_events.csv`
- `stroke_signal_with_drive_events.csv`
- `drive_events_summary.json`
- `drive_phase_overlay.mp4` (when `--overlay-video`)
- `rp3_match_manifest.csv` (when RP3 matching is enabled)
- `rp3_pose_force_matched_segments.csv` (when RP3 matching is enabled)
- `rp3_pose_force_export_status.csv` (when RP3 matching is enabled)
- `rp3_match_summary.json` (when RP3 matching is enabled)
- `training_dataset/` (when RP3 matching succeeds, unless `--no-build-dataset`)

### Key CLI flags

**Detection:**

| Flag | Default | Purpose |
|---|---|---|
| `--finish-method` | `velocity_calibrated` | `velocity_calibrated`, `velocity_threshold`, or `position_max` |
| `--catch-velocity-frac` | calibrated | override calibrated catch threshold |
| `--finish-velocity-frac` | calibrated | override calibrated finish threshold |
| `--smooth-window-s` | 0.04 | handle signal smoothing window |

**RP3 matching:**

| Flag | Default | Purpose |
|---|---|---|
| `--anchor-rp3-stroke-number` | — | RP3 stroke number for first anchor (recommended) |
| `--anchor-rp3-row-idx` | — | alternative: 0-based row index in RP3 CSV |
| `--anchor-video-stroke-idx` | 1 | video stroke index to align with RP3 anchor |
| `--use-rp3-finish` / `--no-use-rp3-finish` | on | override video finish with RP3 drive time for segment export |
| `--active-side` | — | `left` or `right`; required for segment export |
| `--max-abs-cum-error-s` | 4.0 | hard cap on cumulative timing drift |

**Segment export:**

| Flag | Default | Purpose |
|---|---|---|
| `--include-second-derivatives` | off | add `d2theta/ds2` columns for all five angles |

**Training dataset:**

| Flag | Default | Purpose |
|---|---|---|
| `--no-build-dataset` | off | skip training dataset build after segment export |
| `--dataset-output-dir` | `<output>/training_dataset/` | override artifact output location |
| `--dataset-qc-mode` | `soft` | `soft`: keep all strokes, mark `qc_excluded`; `hard`: drop flagged strokes |
| `--dataset-n-grid` | 64 | fixed-grid resolution for resampled sequences |
| `--dataset-n-pca-components` | 20 | PCA components for force curve shape decomposition |
| `--dataset-force-col` | `force_raw` | `force_raw` (Newtons) or `force_n` (PDF-normalized) as target |

---

## `build_training_dataset.py`

Standalone script for building training artifacts. Use this when you want to aggregate strokes across multiple runs or rebuild the dataset with different parameters without re-running detection and matching.

```bash
.venv/bin/python inference/build_training_dataset.py \
  --segment-csv runs/*/inference/rp3_pose_force_matched_segments.csv \
  --output-dir training_dataset_all_runs/ \
  --qc-mode hard \
  --n-pca-components 20
```

Supports shell glob expansion — all matched CSVs are concatenated before building.

### Outputs (`<output-dir>/`)

| File | Shape / Content |
|---|---|
| `strokes.csv` | One row per stroke: metadata + scalar kinematics + PCA coefficients |
| `force_curves_resampled.npy` | (N, 64) raw Newton curves on fixed grid |
| `force_curves_peak_norm.npy` | (N, 64) peak-normalized curves used for PCA |
| `force_curves_padded.npy` | (N, 77) native-bin padded curves (NaN where bin > stroke length) |
| `force_mask.npy` | (N, 77) boolean validity mask for padded bins |
| `kinematic_sequences.npy` | (N, 64, 12) all kinematic channels on fixed grid |
| `kinematic_padded.npy` | (N, 77, 12) kinematic channels on native bin grid |
| `s_grid.npy` | (64,) fixed-grid s values |
| `pca_model.joblib` | Fitted sklearn PCA for reconstruction |
| `pca_explained_variance.csv` | Component-wise explained variance ratio |
| `feature_names.json` | Ordered list of 12 kinematic channel names |
| `dataset_summary.json` | QC counts, grid params, runs included |

The 12 kinematic channels are: 5 joint angles (`knee_active`, `hip_active`, `elbow_active`, `trunk_vs_horizontal`, `spine_flexion`) + 5 first derivatives (`*_ddeg_ds`) + `handle_velocity_px_s` + `handle_accel_px_s2`.

---

## `match_rp3_cli.py`

Standalone matcher when `inference/drive_events.csv` already exists and you only want stroke-to-RP3 alignment.

```bash
.venv/bin/python inference/match_rp3_cli.py \
  --run-dir runs/<run_name> \
  --rp3-clean-csv runs/<run_name>/rp3/<workout>-clean.csv \
  --anchor-rp3-stroke-number <stroke_number>
```

If `--rp3-clean-csv` is omitted, auto-selects from `<run>/rp3/*-clean.csv`.

### Outputs (`<run_dir>/inference/`)

- `rp3_match_manifest.csv`
- `rp3_video_aligned_strokes.csv`
- `rp3_match_summary.json`

---

## RP3 CSV Requirements

For matching (`match_rp3_cli.py` and `inference_cli.py --match-rp3`):

- `stroke_number`, `time`, `drive_time`, `recover_time`

For matched segment export, RP3 CSV must also contain:

- `stroke_length`
- force bins named `force_at_<distance>cm` (e.g. `force_at_2.2cm` through `force_at_169.4cm`)

Dirty RP3 exports are converted to this format automatically by `inference_cli.py` using `rp3-extraction/expand_rp3_curve_data.py`.

---

## Matched Segment Export Semantics

In `rp3_pose_force_matched_segments.csv`:

- `force_raw` — original RP3 bin value in Newtons.
- `force_n` — per-stroke PDF-normalized density over `s_force` (integral = 1; not absolute force units).
- `match_seq_idx` — stable 0-based sequence index from `rp3_match_manifest.csv`.
- `s_force` — normalized drive progress of the bin, in [0, 1].
- `qc_flags` — comma-separated per-stroke quality flags (empty string if none):
  - `qc_tracking_sparse`: >30% NaN angle values in the drive window.
  - `qc_nonphysio_deriv`: max |dθ/dt| exceeds 600 deg/s (tracking spike).
  - `qc_ds_dt_stall`: handle progress rate drops to <5% of its median (stall or tracking failure).
- `handle_velocity_px_s`, `handle_accel_px_s2` — handle kinematics resampled to each force bin.
- `*_ddeg_ds` — chain-rule progress-domain derivatives for all five angles (computed as `dθ/dt ÷ ds/dt` in time domain, then interpolated). Includes `spine_flexion_ddeg_ds`.
- `*_d2deg_ds2` — second derivatives (only when `--include-second-derivatives` is passed).

`rp3_pose_force_export_status.csv` contains one row per matched stroke and reports:

- `segment_exported`, `segment_rows_written`, `drop_reason`
- `raw_area_trapz`, `normalized_area_trapz`
- `nan_frac_angles`, `max_deriv_deg_s`, `ds_dt_min`, `qc_flags`
