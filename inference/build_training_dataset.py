#!/usr/bin/env python3
"""Build training dataset artifacts from one or more rp3_pose_force_matched_segments.csv files.

Produces both target representations from Section 10 of the inference plan:
  1. Native padded bins  (N x MAX_BINS) with boolean validity mask.
  2. Fixed-grid resampled (N x N_GRID) for PCA and sequence modeling.

Also outputs stroke-level scalar kinematic summary features and PCA coefficients
ready for Stage 0/A/B modeling.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RP3_STEP_CM = 2.2
RP3_MAX_CM = 170.0
MAX_BINS: int = int(RP3_MAX_CM / RP3_STEP_CM)          # 77

N_GRID_DEFAULT = 64
N_PCA_DEFAULT = 20

ANGLE_COLS = [
    "knee_active_deg",
    "hip_active_deg",
    "elbow_active_deg",
    "trunk_vs_horizontal_deg",
    "spine_flexion_deg",
]
DERIV_COLS = [c.replace("_deg", "_ddeg_ds") for c in ANGLE_COLS]
SUPPORT_COLS = ["handle_velocity_px_s", "handle_accel_px_s2"]
KINEMATIC_FEATURE_COLS: list[str] = ANGLE_COLS + DERIV_COLS + SUPPORT_COLS

SCALAR_METADATA_COLS = [
    "run_name",
    "rp3_clean_csv",
    "active_side",
    "match_seq_idx",
    "video_stroke_idx",
    "rp3_row_idx",
    "rp3_stroke_number",
    "video_catch_time_s",
    "video_finish_time_s",
    "video_drive_s",
    "video_recover_s",
    "video_cycle_s",
    "rp3_drive_s",
    "rp3_recover_s",
    "rp3_cycle_s",
    "cum_catch_err_s",
    "interval_err_s",
    "rp3_rows_skipped_since_prev",
    "stroke_length_cm",
    "qc_flags",
]

QC_HARD_DROP_FLAGS = {"qc_tracking_sparse", "qc_nonphysio_deriv"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_segment_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if "run_name" not in df.columns:
            df["run_name"] = p.parent.parent.name
        frames.append(df)
    if not frames:
        raise ValueError("No segment CSVs could be loaded.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} bin rows from {len(paths)} file(s).")
    return combined


# ---------------------------------------------------------------------------
# Pivot to stroke-level
# ---------------------------------------------------------------------------

def _pivot_to_strokes(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group the long-format dataframe into one record per stroke.

    Each record contains scalar metadata and per-bin arrays sorted by s_force.
    """
    strokes: list[dict[str, Any]] = []
    group_cols = ["run_name", "match_seq_idx"]
    for (run_name, seq_idx), grp in df.groupby(group_cols, sort=False):
        grp = grp.sort_values("s_force").reset_index(drop=True)
        first = grp.iloc[0]

        record: dict[str, Any] = {
            "stroke_key": f"{run_name}__{int(seq_idx)}",
        }
        for col in SCALAR_METADATA_COLS:
            record[col] = first[col] if col in grp.columns else float("nan")

        record["stroke_rate_spm"] = (
            60.0 / float(first["rp3_cycle_s"])
            if "rp3_cycle_s" in grp.columns and np.isfinite(float(first["rp3_cycle_s"]))
            else float("nan")
        )

        record["s_force_arr"] = grp["s_force"].to_numpy(dtype=np.float64)
        record["force_raw_arr"] = grp["force_raw"].to_numpy(dtype=np.float64)
        record["distance_cm_arr"] = grp["distance_cm"].to_numpy(dtype=np.float64) if "distance_cm" in grp.columns else np.full(len(grp), np.nan)
        record["n_bins"] = len(grp)

        for col in KINEMATIC_FEATURE_COLS:
            if col in grp.columns:
                record[f"{col}_arr"] = grp[col].to_numpy(dtype=np.float64)
            else:
                record[f"{col}_arr"] = np.full(len(grp), np.nan)

        strokes.append(record)

    print(f"Pivoted to {len(strokes):,} strokes.")
    return strokes


# ---------------------------------------------------------------------------
# QC filtering
# ---------------------------------------------------------------------------

def _apply_qc(strokes: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Mark or drop strokes based on qc_flags.

    soft: add qc_excluded bool; keep all strokes.
    hard: drop strokes with hard-drop flags; still mark soft-flagged ones.
    """
    n_before = len(strokes)
    kept: list[dict[str, Any]] = []
    n_hard_dropped = 0
    for s in strokes:
        flags_str = str(s.get("qc_flags", "") or "")
        flags = {f.strip() for f in flags_str.split(",") if f.strip()}
        hard_hit = flags & QC_HARD_DROP_FLAGS
        s["qc_excluded"] = bool(hard_hit)
        s["qc_flags_set"] = flags
        if mode == "hard" and hard_hit:
            n_hard_dropped += 1
            continue
        kept.append(s)

    print(
        f"QC ({mode} mode): {n_before} strokes -> {len(kept)} kept "
        f"({n_hard_dropped} hard-dropped, "
        f"{sum(1 for s in kept if s['qc_excluded'])} soft-flagged)."
    )
    return kept


# ---------------------------------------------------------------------------
# Representation 1: native padded bins
# ---------------------------------------------------------------------------

def _build_native_padded(
    strokes: list[dict[str, Any]],
    force_col: str = "force_raw_arr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (force_padded, force_mask, kinematic_padded).

    force_padded   : (N, MAX_BINS)  float64, NaN where invalid.
    force_mask     : (N, MAX_BINS)  bool, True where bin is valid.
    kinematic_padded: (N, MAX_BINS, K) float64, NaN where invalid.
    """
    N = len(strokes)
    K = len(KINEMATIC_FEATURE_COLS)
    force_padded = np.full((N, MAX_BINS), np.nan, dtype=np.float64)
    force_mask = np.zeros((N, MAX_BINS), dtype=bool)
    kinematic_padded = np.full((N, MAX_BINS, K), np.nan, dtype=np.float64)

    for i, s in enumerate(strokes):
        f_arr = s[force_col]
        d_arr = s["distance_cm_arr"]

        for j, (d_cm, f_val) in enumerate(zip(d_arr, f_arr)):
            bin_idx = int(round(d_cm / RP3_STEP_CM)) - 1
            if 0 <= bin_idx < MAX_BINS and np.isfinite(f_val):
                force_padded[i, bin_idx] = f_val
                force_mask[i, bin_idx] = True

        for k, feat_col in enumerate(KINEMATIC_FEATURE_COLS):
            arr = s[f"{feat_col}_arr"]
            for j, (d_cm, v) in enumerate(zip(d_arr, arr)):
                bin_idx = int(round(d_cm / RP3_STEP_CM)) - 1
                if 0 <= bin_idx < MAX_BINS:
                    kinematic_padded[i, bin_idx, k] = v

    return force_padded, force_mask, kinematic_padded


# ---------------------------------------------------------------------------
# Representation 2: fixed-grid resampling
# ---------------------------------------------------------------------------

def _resample_to_grid(
    arr_s: np.ndarray,
    arr_v: np.ndarray,
    s_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate arr_v (defined at arr_s) onto s_grid using np.interp."""
    valid = np.isfinite(arr_s) & np.isfinite(arr_v)
    if valid.sum() < 2:
        return np.full(len(s_grid), np.nan)
    return np.interp(s_grid, arr_s[valid], arr_v[valid])


def _build_fixed_grid(
    strokes: list[dict[str, Any]],
    s_grid: np.ndarray,
    force_col: str = "force_raw_arr",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (force_resampled, kinematic_sequences).

    force_resampled     : (N, N_GRID) float64
    kinematic_sequences : (N, N_GRID, K) float64
    """
    N = len(strokes)
    N_GRID = len(s_grid)
    K = len(KINEMATIC_FEATURE_COLS)
    force_resampled = np.full((N, N_GRID), np.nan, dtype=np.float64)
    kinematic_sequences = np.full((N, N_GRID, K), np.nan, dtype=np.float64)

    for i, s in enumerate(strokes):
        s_arr = s["s_force_arr"]
        f_arr = s[force_col]
        force_resampled[i] = _resample_to_grid(s_arr, f_arr, s_grid)
        for k, feat_col in enumerate(KINEMATIC_FEATURE_COLS):
            kinematic_sequences[i, :, k] = _resample_to_grid(
                s_arr, s[f"{feat_col}_arr"], s_grid
            )

    return force_resampled, kinematic_sequences


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def _fit_pca(
    force_resampled: np.ndarray,
    n_components: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on peak-normalized force curves.

    Returns (pca_model, pca_coeffs, force_peak_norm).
    Rows with any NaN are excluded from fitting but still transformed
    (their coefficients are set to NaN).
    """
    N, N_GRID = force_resampled.shape
    finite_rows = np.all(np.isfinite(force_resampled), axis=1)
    row_max = np.nanmax(force_resampled, axis=1, keepdims=True)
    valid_max = (row_max > 0).ravel() & finite_rows

    force_peak_norm = np.full_like(force_resampled, np.nan)
    force_peak_norm[valid_max] = (
        force_resampled[valid_max] / row_max[valid_max]
    )

    n_components = min(n_components, valid_max.sum(), N_GRID)
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(force_peak_norm[valid_max])

    pca_coeffs = np.full((N, n_components), np.nan, dtype=np.float64)
    pca_coeffs[valid_max] = pca.transform(force_peak_norm[valid_max])

    var_explained = pca.explained_variance_ratio_.cumsum()
    print(
        f"PCA: {n_components} components on {valid_max.sum()} strokes. "
        f"Cumulative variance: "
        + ", ".join(f"PC{j+1}={v:.3f}" for j, v in enumerate(var_explained[:5]))
        + (" ..." if n_components > 5 else "")
    )
    return pca, pca_coeffs, force_peak_norm


# ---------------------------------------------------------------------------
# Scalar kinematic summary
# ---------------------------------------------------------------------------

def _compute_scalar_summary(
    strokes: list[dict[str, Any]],
    s_grid: np.ndarray,
    kinematic_sequences: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame with one row per stroke, containing kinematic summary scalars.

    For each angle (not derivative or support) compute:
      {angle}_min, {angle}_max, {angle}_range, {angle}_mean, {angle}_s_at_max
    """
    records: list[dict[str, Any]] = []
    for i, s in enumerate(strokes):
        row: dict[str, Any] = {"stroke_key": s["stroke_key"]}
        for k, feat_col in enumerate(KINEMATIC_FEATURE_COLS):
            if feat_col not in ANGLE_COLS:
                continue
            vals = kinematic_sequences[i, :, k]
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                row[f"{feat_col}_min"] = float("nan")
                row[f"{feat_col}_max"] = float("nan")
                row[f"{feat_col}_range"] = float("nan")
                row[f"{feat_col}_mean"] = float("nan")
                row[f"{feat_col}_s_at_max"] = float("nan")
            else:
                vmin = float(np.nanmin(vals))
                vmax = float(np.nanmax(vals))
                row[f"{feat_col}_min"] = vmin
                row[f"{feat_col}_max"] = vmax
                row[f"{feat_col}_range"] = vmax - vmin
                row[f"{feat_col}_mean"] = float(np.nanmean(vals))
                idx_max = int(np.nanargmax(vals))
                row[f"{feat_col}_s_at_max"] = float(s_grid[idx_max])
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Assemble strokes.csv
# ---------------------------------------------------------------------------

def _build_strokes_csv(
    strokes: list[dict[str, Any]],
    pca_coeffs: np.ndarray,
    scalar_summary_df: pd.DataFrame,
    n_pca: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(strokes):
        row: dict[str, Any] = {"stroke_key": s["stroke_key"]}
        for col in SCALAR_METADATA_COLS:
            row[col] = s.get(col, float("nan"))
        row["stroke_rate_spm"] = s.get("stroke_rate_spm", float("nan"))
        row["qc_excluded"] = s.get("qc_excluded", False)
        row["n_bins"] = s["n_bins"]
        for c in range(n_pca):
            row[f"pca_{c}"] = float(pca_coeffs[i, c]) if np.isfinite(pca_coeffs[i, c]) else float("nan")
        rows.append(row)

    base_df = pd.DataFrame(rows)
    summary_cols = [c for c in scalar_summary_df.columns if c != "stroke_key"]
    merged = base_df.merge(scalar_summary_df[["stroke_key"] + summary_cols], on="stroke_key", how="left")
    return merged


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_training_dataset(
    segment_csvs: list[Path],
    output_dir: Path,
    qc_mode: str = "soft",
    n_grid: int = N_GRID_DEFAULT,
    n_pca_components: int = N_PCA_DEFAULT,
    force_col: str = "force_raw",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and pivot
    df = _load_segment_csvs(segment_csvs)
    strokes = _pivot_to_strokes(df)

    # 2. QC filter
    strokes = _apply_qc(strokes, qc_mode)
    N = len(strokes)
    if N == 0:
        raise RuntimeError("No strokes remain after QC filtering.")

    force_arr_col = f"{force_col}_arr"
    missing_force = [s["stroke_key"] for s in strokes if force_arr_col not in s]
    if missing_force:
        raise ValueError(
            f"Column '{force_col}' not found in segment data. "
            f"Use --force-col force_raw or force_n."
        )

    # 3. Fixed s-grid
    s_grid = np.linspace(0.0, 1.0, n_grid)

    # 4. Representation 1: native padded bins
    print("Building Representation 1 (native padded bins)...")
    force_padded, force_mask, kinematic_padded = _build_native_padded(strokes, force_col=force_arr_col)

    # 5. Representation 2: fixed-grid resampling
    print("Building Representation 2 (fixed-grid resampling)...")
    force_resampled, kinematic_sequences = _build_fixed_grid(strokes, s_grid, force_col=force_arr_col)

    # 6. PCA
    print("Fitting PCA...")
    pca_model, pca_coeffs, force_peak_norm = _fit_pca(force_resampled, n_pca_components)
    actual_n_pca = pca_coeffs.shape[1]

    # 7. Scalar kinematic summary
    print("Computing scalar kinematic summary...")
    scalar_summary_df = _compute_scalar_summary(strokes, s_grid, kinematic_sequences)

    # 8. Assemble strokes.csv
    print("Assembling strokes.csv...")
    strokes_df = _build_strokes_csv(strokes, pca_coeffs, scalar_summary_df, actual_n_pca)

    # 9. Write all artifacts
    print("Writing artifacts...")
    _write_artifacts(
        output_dir=output_dir,
        strokes_df=strokes_df,
        force_resampled=force_resampled,
        force_peak_norm=force_peak_norm,
        force_padded=force_padded,
        force_mask=force_mask,
        kinematic_sequences=kinematic_sequences,
        kinematic_padded=kinematic_padded,
        s_grid=s_grid,
        pca_model=pca_model,
        segment_csvs=segment_csvs,
        qc_mode=qc_mode,
        n_grid=n_grid,
        n_pca_components=actual_n_pca,
        force_col=force_col,
        n_strokes_before_qc=len(df.groupby(["run_name", "match_seq_idx"])),
        n_strokes_after_qc=N,
        strokes=strokes,
    )
    print(f"Done. Dataset written to: {output_dir}")


# ---------------------------------------------------------------------------
# Write artifacts
# ---------------------------------------------------------------------------

def _write_artifacts(
    *,
    output_dir: Path,
    strokes_df: pd.DataFrame,
    force_resampled: np.ndarray,
    force_peak_norm: np.ndarray,
    force_padded: np.ndarray,
    force_mask: np.ndarray,
    kinematic_sequences: np.ndarray,
    kinematic_padded: np.ndarray,
    s_grid: np.ndarray,
    pca_model: PCA,
    segment_csvs: list[Path],
    qc_mode: str,
    n_grid: int,
    n_pca_components: int,
    force_col: str,
    n_strokes_before_qc: int,
    n_strokes_after_qc: int,
    strokes: list[dict[str, Any]],
) -> None:
    strokes_df.to_csv(output_dir / "strokes.csv", index=False)

    np.save(output_dir / "force_curves_resampled.npy", force_resampled)
    np.save(output_dir / "force_curves_peak_norm.npy", force_peak_norm)
    np.save(output_dir / "force_curves_padded.npy", force_padded)
    np.save(output_dir / "force_mask.npy", force_mask)
    np.save(output_dir / "kinematic_sequences.npy", kinematic_sequences)
    np.save(output_dir / "kinematic_padded.npy", kinematic_padded)
    np.save(output_dir / "s_grid.npy", s_grid)

    feature_names_path = output_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(KINEMATIC_FEATURE_COLS, f, indent=2)

    joblib.dump(pca_model, output_dir / "pca_model.joblib")

    pca_ev_df = pd.DataFrame({
        "component": range(1, n_pca_components + 1),
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
        "cumulative_explained_variance": pca_model.explained_variance_ratio_.cumsum(),
    })
    pca_ev_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    n_soft_flagged = sum(1 for s in strokes if s.get("qc_excluded", False))
    n_hard_dropped = n_strokes_before_qc - n_strokes_after_qc if qc_mode == "hard" else 0
    summary = {
        "n_strokes_before_qc": n_strokes_before_qc,
        "n_strokes_after_qc": n_strokes_after_qc,
        "n_hard_dropped": n_hard_dropped,
        "n_soft_flagged": n_soft_flagged,
        "qc_mode": qc_mode,
        "force_col": force_col,
        "n_grid": n_grid,
        "max_bins": MAX_BINS,
        "rp3_step_cm": RP3_STEP_CM,
        "rp3_max_cm": RP3_MAX_CM,
        "n_pca_components": n_pca_components,
        "pca_total_explained_variance": float(pca_model.explained_variance_ratio_.sum()),
        "kinematic_feature_cols": KINEMATIC_FEATURE_COLS,
        "n_kinematic_features": len(KINEMATIC_FEATURE_COLS),
        "source_files": [str(p) for p in segment_csvs],
        "runs_included": sorted({s["run_name"] for s in strokes}),
    }
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  strokes.csv            : {len(strokes_df)} rows x {len(strokes_df.columns)} cols")
    print(f"  force_curves_resampled : {force_resampled.shape}")
    print(f"  force_curves_padded    : {force_padded.shape}")
    print(f"  force_mask             : {force_mask.shape}  ({force_mask.sum()} valid bins)")
    print(f"  kinematic_sequences    : {kinematic_sequences.shape}")
    print(f"  kinematic_padded       : {kinematic_padded.shape}")
    print(f"  pca_model              : {n_pca_components} components, "
          f"{pca_model.explained_variance_ratio_.sum():.3f} total var explained")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Section 10 training dataset artifacts from "
            "rp3_pose_force_matched_segments.csv file(s)."
        )
    )
    parser.add_argument(
        "--segment-csv",
        nargs="+",
        required=True,
        metavar="CSV",
        help=(
            "Path(s) to rp3_pose_force_matched_segments.csv. "
            "Supports shell globs, e.g. 'runs/*/inference/rp3_pose_force_matched_segments.csv'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="training_dataset",
        metavar="DIR",
        help="Directory to write all output artifacts (default: training_dataset/).",
    )
    parser.add_argument(
        "--qc-mode",
        choices=["soft", "hard"],
        default="soft",
        help=(
            "soft: keep all strokes, mark qc_excluded=True for flagged ones. "
            "hard: drop strokes with qc_tracking_sparse or qc_nonphysio_deriv flags."
        ),
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=N_GRID_DEFAULT,
        help=f"Number of evenly-spaced grid points over s in [0,1] (default: {N_GRID_DEFAULT}).",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=N_PCA_DEFAULT,
        help=f"Maximum number of PCA components to fit (default: {N_PCA_DEFAULT}).",
    )
    parser.add_argument(
        "--force-col",
        choices=["force_raw", "force_n"],
        default="force_raw",
        help=(
            "Force column to use for target representations. "
            "force_raw: raw Newtons (default). "
            "force_n: PDF-normalized density (integral=1 over s)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    raw_paths: list[str] = args.segment_csv
    resolved: list[Path] = []
    for pattern in raw_paths:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            resolved.extend(Path(p) for p in expanded)
        else:
            p = Path(pattern)
            if p.exists():
                resolved.append(p)
            else:
                print(f"Warning: no files matched pattern '{pattern}'.")

    resolved = sorted(set(resolved))
    if not resolved:
        print("Error: no segment CSV files found.")
        return 1

    print(f"Found {len(resolved)} segment CSV file(s):")
    for p in resolved:
        print(f"  {p}")

    output_dir = Path(args.output_dir)

    try:
        build_training_dataset(
            segment_csvs=resolved,
            output_dir=output_dir,
            qc_mode=args.qc_mode,
            n_grid=args.n_grid,
            n_pca_components=args.n_pca_components,
            force_col=args.force_col,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
