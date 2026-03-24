"""Shared evaluation metrics, CV splitters, and reporting for Section 11/12 modeling.

Provides:
  - Curve-level metrics: RMSE, MAE, Pearson correlation per stroke.
  - Rowing-relevant metrics: peak force error, peak position error,
    impulse error, phase-specific errors.
  - Aggregation helper that computes all metrics in one call.
  - Cross-validation splitters: time-block and session-held-out.
  - PCA reconstruction helper for PCA-coefficient models.
"""
from __future__ import annotations

from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Curve-level metrics  (operate on (N, G) arrays)
# ---------------------------------------------------------------------------


def rmse_per_stroke(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-stroke RMSE across grid points.  Returns (N,)."""
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=1))


def mae_per_stroke(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-stroke MAE across grid points.  Returns (N,)."""
    return np.nanmean(np.abs(y_true - y_pred), axis=1)


def curve_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-stroke Pearson r between true and predicted curves.  Returns (N,).

    Strokes with zero variance in either signal get NaN.
    """
    N = y_true.shape[0]
    r = np.full(N, np.nan)
    for i in range(N):
        yt = y_true[i]
        yp = y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 3:
            continue
        yt_m, yp_m = yt[mask], yp[mask]
        std_t = yt_m.std()
        std_p = yp_m.std()
        if std_t < 1e-12 or std_p < 1e-12:
            continue
        r[i] = np.corrcoef(yt_m, yp_m)[0, 1]
    return r


# ---------------------------------------------------------------------------
# Rowing-relevant metrics
# ---------------------------------------------------------------------------


def peak_force_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute error in peak force magnitude.  Returns (N,)."""
    return np.abs(np.nanmax(y_true, axis=1) - np.nanmax(y_pred, axis=1))


def peak_position_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_grid: np.ndarray,
) -> np.ndarray:
    """Absolute error in peak-force position (fraction of drive).  Returns (N,)."""
    N = y_true.shape[0]
    err = np.full(N, np.nan)
    for i in range(N):
        yt, yp = y_true[i], y_pred[i]
        ft = np.isfinite(yt)
        fp = np.isfinite(yp)
        if ft.sum() == 0 or fp.sum() == 0:
            continue
        s_true = s_grid[np.nanargmax(yt)]
        s_pred = s_grid[np.nanargmax(yp)]
        err[i] = abs(s_true - s_pred)
    return err


def impulse_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_grid: np.ndarray,
) -> np.ndarray:
    """Absolute error in impulse (area under curve via trapezoid).  Returns (N,)."""
    imp_true = np.trapz(np.nan_to_num(y_true, nan=0.0), s_grid, axis=1)
    imp_pred = np.trapz(np.nan_to_num(y_pred, nan=0.0), s_grid, axis=1)
    return np.abs(imp_true - imp_pred)


def phase_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_grid: np.ndarray,
    n_phases: int = 3,
) -> dict[str, np.ndarray]:
    """Per-stroke MAE within equal-width drive phases.

    Returns dict with keys like ``phase_0_mae``, ``phase_1_mae``, etc.,
    each an (N,) array.  With ``n_phases=3`` the phases correspond to
    early / mid / late drive.
    """
    boundaries = np.linspace(s_grid[0], s_grid[-1], n_phases + 1)
    result: dict[str, np.ndarray] = {}
    for p in range(n_phases):
        lo, hi = boundaries[p], boundaries[p + 1]
        mask = (s_grid >= lo) & (s_grid < hi) if p < n_phases - 1 else (s_grid >= lo) & (s_grid <= hi)
        if mask.sum() == 0:
            result[f"phase_{p}_mae"] = np.full(y_true.shape[0], np.nan)
            continue
        result[f"phase_{p}_mae"] = np.nanmean(
            np.abs(y_true[:, mask] - y_pred[:, mask]), axis=1
        )
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_PHASE_LABELS = {0: "early", 1: "mid", 2: "late"}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_grid: np.ndarray,
    n_phases: int = 3,
) -> dict[str, float]:
    """Compute all Section 12 metrics and return aggregated statistics.

    Keys use the pattern ``{metric}_{agg}`` where *agg* is one of
    ``median``, ``mean``, ``std``.
    """
    metrics: dict[str, np.ndarray] = {
        "rmse": rmse_per_stroke(y_true, y_pred),
        "mae": mae_per_stroke(y_true, y_pred),
        "correlation": curve_correlation(y_true, y_pred),
        "peak_force_err": peak_force_error(y_true, y_pred),
        "peak_pos_err": peak_position_error(y_true, y_pred, s_grid),
        "impulse_err": impulse_error(y_true, y_pred, s_grid),
    }
    for k, arr in phase_errors(y_true, y_pred, s_grid, n_phases).items():
        phase_idx = int(k.split("_")[1])
        label = _PHASE_LABELS.get(phase_idx, str(phase_idx))
        metrics[f"{label}_drive_mae"] = arr

    out: dict[str, float] = {}
    for name, arr in metrics.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out[f"{name}_median"] = float("nan")
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_std"] = float("nan")
        else:
            out[f"{name}_median"] = float(np.median(finite))
            out[f"{name}_mean"] = float(np.mean(finite))
            out[f"{name}_std"] = float(np.std(finite))
    out["n_strokes"] = int(y_true.shape[0])
    return out


def format_metrics_table(
    results: list[dict[str, Any]],
    *,
    label_key: str = "model",
    highlight_keys: tuple[str, ...] = (
        "rmse_median",
        "peak_force_err_median",
        "impulse_err_median",
        "correlation_median",
    ),
) -> str:
    """Format a list of metric dicts into an aligned text table."""
    if not results:
        return "(no results)"
    cols = [label_key] + list(highlight_keys)
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in cols]
    widths = [max(w, 8) for w in widths]

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    lines = [header, sep]
    for r in results:
        vals: list[str] = []
        for c, w in zip(cols, widths):
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}".ljust(w))
            else:
                vals.append(str(v).ljust(w))
        lines.append("  ".join(vals))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-validation splitters
# ---------------------------------------------------------------------------


def time_block_split(
    strokes_df: pd.DataFrame,
    n_splits: int = 5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) based on temporal ordering.

    Strokes are ordered by ``(run_name, match_seq_idx)`` and split into
    *n_splits* contiguous blocks.  Each fold holds out one block.
    """
    order_cols = []
    if "run_name" in strokes_df.columns:
        order_cols.append("run_name")
    if "match_seq_idx" in strokes_df.columns:
        order_cols.append("match_seq_idx")
    elif "video_stroke_idx" in strokes_df.columns:
        order_cols.append("video_stroke_idx")

    if order_cols:
        sorted_idx = strokes_df.sort_values(order_cols).index.to_numpy()
    else:
        sorted_idx = strokes_df.index.to_numpy()

    fold_indices = np.array_split(sorted_idx, n_splits)
    for i in range(n_splits):
        test = fold_indices[i]
        train = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])
        yield train, test


def session_held_out_split(
    strokes_df: pd.DataFrame,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) leaving one run_name out per fold."""
    if "run_name" not in strokes_df.columns:
        raise ValueError("strokes_df must contain a 'run_name' column for session splits.")
    runs = strokes_df["run_name"].unique()
    for held_out in runs:
        test = strokes_df.index[strokes_df["run_name"] == held_out].to_numpy()
        train = strokes_df.index[strokes_df["run_name"] != held_out].to_numpy()
        if train.size == 0 or test.size == 0:
            continue
        yield train, test


def athlete_held_out_split(
    strokes_df: pd.DataFrame,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) leaving one athlete_id out per fold.

    Requires an ``athlete_id`` column with at least two distinct non-"unknown"
    values; raises ValueError otherwise.
    """
    if "athlete_id" not in strokes_df.columns:
        raise ValueError("strokes_df must contain an 'athlete_id' column for athlete splits.")
    athletes = [
        a for a in strokes_df["athlete_id"].unique()
        if str(a) != "unknown"
    ]
    if len(athletes) < 2:
        raise ValueError(
            f"Need >= 2 distinct athletes for athlete-held-out splits, "
            f"found {len(athletes)}: {athletes}"
        )
    for held_out in athletes:
        test = strokes_df.index[strokes_df["athlete_id"] == held_out].to_numpy()
        train = strokes_df.index[strokes_df["athlete_id"] != held_out].to_numpy()
        if train.size == 0 or test.size == 0:
            continue
        yield train, test


# ---------------------------------------------------------------------------
# Alignment integrity metrics (Section 12 pairing checks)
# ---------------------------------------------------------------------------

_ALIGNMENT_DRIFT_THRESHOLD_S = 1.5
_ALIGNMENT_RATE_THRESHOLD_SPM = 2.0


def compute_alignment_metrics(
    strokes_df: pd.DataFrame,
    drift_threshold_s: float = _ALIGNMENT_DRIFT_THRESHOLD_S,
    rate_threshold_spm: float = _ALIGNMENT_RATE_THRESHOLD_SPM,
) -> dict[str, float]:
    """Compute pairing/alignment integrity statistics from matched strokes.

    Uses ``cum_catch_err_s``, ``interval_err_s``, ``video_cycle_s``, and
    ``rp3_cycle_s`` columns when present.
    """
    out: dict[str, float] = {}

    if "cum_catch_err_s" in strokes_df.columns:
        cum = pd.to_numeric(strokes_df["cum_catch_err_s"], errors="coerce")
        abs_cum = cum.abs()
        out["mean_abs_cum_catch_err_s"] = float(abs_cum.mean()) if abs_cum.notna().any() else float("nan")
        out["max_abs_cum_catch_err_s"] = float(abs_cum.max()) if abs_cum.notna().any() else float("nan")
        out["n_above_drift_threshold"] = int((abs_cum > drift_threshold_s).sum())
    else:
        out["mean_abs_cum_catch_err_s"] = float("nan")
        out["max_abs_cum_catch_err_s"] = float("nan")
        out["n_above_drift_threshold"] = 0

    if "interval_err_s" in strokes_df.columns:
        ie = pd.to_numeric(strokes_df["interval_err_s"], errors="coerce")
        out["mean_abs_interval_err_s"] = float(ie.abs().mean()) if ie.notna().any() else float("nan")
    else:
        out["mean_abs_interval_err_s"] = float("nan")

    has_cycles = (
        "video_cycle_s" in strokes_df.columns
        and "rp3_cycle_s" in strokes_df.columns
    )
    if has_cycles:
        vc = pd.to_numeric(strokes_df["video_cycle_s"], errors="coerce")
        rc = pd.to_numeric(strokes_df["rp3_cycle_s"], errors="coerce")
        valid = (vc > 0) & (rc > 0) & vc.notna() & rc.notna()
        if valid.any():
            video_spm = 60.0 / vc[valid]
            rp3_spm = 60.0 / rc[valid]
            rate_diff = (video_spm - rp3_spm).abs()
            out["mean_rate_diff_spm"] = float(rate_diff.mean())
            out["max_rate_diff_spm"] = float(rate_diff.max())
            out["n_above_rate_threshold"] = int((rate_diff > rate_threshold_spm).sum())

            both_finite = video_spm.replace([np.inf, -np.inf], np.nan).dropna()
            rp3_finite = rp3_spm.loc[both_finite.index]
            if len(both_finite) >= 3:
                out["stroke_rate_pearson_r"] = float(
                    np.corrcoef(both_finite.to_numpy(), rp3_finite.to_numpy())[0, 1]
                )
            else:
                out["stroke_rate_pearson_r"] = float("nan")
        else:
            out["mean_rate_diff_spm"] = float("nan")
            out["max_rate_diff_spm"] = float("nan")
            out["n_above_rate_threshold"] = 0
            out["stroke_rate_pearson_r"] = float("nan")
    else:
        out["mean_rate_diff_spm"] = float("nan")
        out["max_rate_diff_spm"] = float("nan")
        out["n_above_rate_threshold"] = 0
        out["stroke_rate_pearson_r"] = float("nan")

    out["n_strokes_evaluated"] = len(strokes_df)
    return out


# ---------------------------------------------------------------------------
# Evaluation regime detection (Section 12)
# ---------------------------------------------------------------------------

def detect_evaluation_regime(
    strokes_df: pd.DataFrame,
    cv_method: str,
) -> dict[str, str]:
    """Determine whether results are provisional or generalization-grade.

    Returns a dict with ``regime`` (one of "provisional_within_athlete",
    "provisional_mixed", "generalization") and a human-readable ``disclaimer``.
    """
    has_athlete = (
        "athlete_id" in strokes_df.columns
        and strokes_df["athlete_id"].nunique() > 1
        and not (strokes_df["athlete_id"] == "unknown").all()
    )
    n_athletes = (
        strokes_df["athlete_id"].nunique()
        if "athlete_id" in strokes_df.columns
        else 0
    )

    if cv_method == "athlete_held_out" and has_athlete:
        regime = "generalization"
        disclaimer = (
            f"Athlete-held-out evaluation across {n_athletes} athletes. "
            "Results reflect cross-athlete generalization."
        )
    elif has_athlete:
        regime = "provisional_mixed"
        disclaimer = (
            f"Within-session or time-block splits over {n_athletes} athletes. "
            "Results are provisional: the same athlete may appear in train and test. "
            "Use --cv-method athlete_held_out for generalization claims."
        )
    else:
        regime = "provisional_within_athlete"
        disclaimer = (
            "All data comes from a single athlete (or athlete_id is unknown). "
            "Results are provisional and do not support generalization claims."
        )

    return {"regime": regime, "disclaimer": disclaimer, "n_athletes": str(n_athletes)}


# ---------------------------------------------------------------------------
# PCA reconstruction
# ---------------------------------------------------------------------------


def reconstruct_from_pca(
    pca_model: PCA,
    coeffs: np.ndarray,
    peak_forces: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct force curves from PCA coefficients.

    Parameters
    ----------
    pca_model : fitted PCA
    coeffs : (N, n_components) predicted PCA coefficients
    peak_forces : optional (N,) peak force values.  If provided the
        reconstructed peak-normalised curves are rescaled back to Newtons.

    Returns
    -------
    (N, G) reconstructed force curves.
    """
    n_components = pca_model.n_components_
    if coeffs.shape[1] < n_components:
        padded = np.zeros((coeffs.shape[0], n_components), dtype=np.float64)
        padded[:, : coeffs.shape[1]] = coeffs
        coeffs = padded
    elif coeffs.shape[1] > n_components:
        coeffs = coeffs[:, :n_components]

    reconstructed = pca_model.inverse_transform(coeffs)

    if peak_forces is not None:
        pf = np.asarray(peak_forces, dtype=np.float64).reshape(-1, 1)
        reconstructed = reconstructed * pf

    return reconstructed
