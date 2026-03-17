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
