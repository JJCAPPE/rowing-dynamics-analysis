"""Matplotlib plot helpers shared by HTML reports.

Every helper writes a PNG to disk and returns the path it wrote to. We use the
``Agg`` backend so the reports can be generated headlessly during inference or
via the menu's "View report" action.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = [
    "plot_drive_durations",
    "plot_match_drift",
    "plot_match_pair_table_image",
    "plot_force_grid",
    "plot_qc_drop_reasons",
    "plot_pca_explained_variance",
    "plot_metric_bar_group",
    "plot_residual_histogram",
    "plot_true_vs_pred_overlay",
    "plot_cohort_metric_bars",
]


# Friendly default styling — match the editor and diagnostics viewer.
_PRIMARY = "#1f77b4"
_ACCENT = "#d62728"
_NEUTRAL = "#7f7f7f"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Detection panels
# ---------------------------------------------------------------------------


def plot_drive_durations(events: pd.DataFrame, out: Path) -> Path:
    """Drive / recover duration scatter (per stroke) — sanity check on detection."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if events.empty:
        ax.text(0.5, 0.5, "No drive events.", transform=ax.transAxes,
                ha="center", va="center", color="red")
        return _save(fig, out)

    x = events["stroke_idx"].astype(int).to_numpy()
    drive = events["drive_duration_s"].astype(float).to_numpy()
    recover = events["recover_duration_s"].astype(float).to_numpy()

    ax.plot(x, drive, marker="o", color=_PRIMARY, label="drive (s)", linewidth=1.0)
    ax.plot(x, recover, marker="o", color=_ACCENT, label="recover (s)", linewidth=1.0)
    ax.axhline(np.mean(drive), color=_PRIMARY, linestyle=":", alpha=0.5)
    ax.axhline(np.mean(recover), color=_ACCENT, linestyle=":", alpha=0.5)

    ax.set_xlabel("stroke_idx")
    ax.set_ylabel("seconds")
    ax.set_title("Drive / recover durations per stroke")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Match panels
# ---------------------------------------------------------------------------


def plot_match_drift(manifest: pd.DataFrame, out: Path) -> Path:
    """Cumulative catch-time drift across the matched stroke sequence."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if manifest.empty:
        ax.text(0.5, 0.5, "No match manifest.", transform=ax.transAxes,
                ha="center", va="center", color="red")
        return _save(fig, out)

    x = manifest["video_stroke_idx"].astype(int).to_numpy()
    cum_err = manifest["cum_catch_err_s"].astype(float).to_numpy()
    ax.plot(x, cum_err, marker="o", color=_PRIMARY, linewidth=1.1)
    ax.fill_between(x, 0.0, cum_err, alpha=0.15, color=_PRIMARY)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)

    mean_abs = float(np.mean(np.abs(cum_err)))
    ax.set_xlabel("video_stroke_idx")
    ax.set_ylabel("cum catch err (s)")
    ax.set_title(f"Cumulative match drift  —  mean |cum err| = {mean_abs:.3f}s")
    ax.grid(True, alpha=0.25)
    return _save(fig, out)


def plot_match_pair_table_image(
    manifest: pd.DataFrame, out: Path, *, max_rows: int = 30,
) -> Path:
    """Wide pair-table snapshot rendered as a PNG (works in any browser)."""
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.18 * min(len(manifest), max_rows) + 1.4)))
    ax.axis("off")
    if manifest.empty:
        ax.text(0.5, 0.5, "No match manifest.", transform=ax.transAxes,
                ha="center", va="center", color="red")
        return _save(fig, out)

    cols = [
        "video_stroke_idx",
        "rp3_stroke_number",
        "rp3_rows_skipped_since_prev",
        "video_drive_s",
        "rp3_drive_s",
        "drive_err_s",
        "interval_err_s",
        "cum_catch_err_s",
    ]
    available = [c for c in cols if c in manifest.columns]
    df = manifest[available].copy()
    if len(df) > max_rows:
        df = df.head(max_rows)
    fmt = df.copy()
    for col in available:
        if col in {"video_stroke_idx", "rp3_stroke_number", "rp3_rows_skipped_since_prev"}:
            fmt[col] = fmt[col].astype(int).astype(str)
        else:
            fmt[col] = fmt[col].astype(float).map(lambda v: f"{v:+.3f}")
    table = ax.table(
        cellText=fmt.values,
        colLabels=[c.replace("_", " ") for c in available],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.1)
    if len(manifest) > max_rows:
        ax.set_title(f"Match pair table (first {max_rows} of {len(manifest)} strokes)")
    else:
        ax.set_title("Match pair table")
    return _save(fig, out)


def plot_force_grid(
    segments: pd.DataFrame, manifest: pd.DataFrame, out: Path,
    *, max_strokes: int = 12,
) -> Path:
    """Small-multiples force curves for the first *max_strokes* matched strokes."""
    if segments.empty or manifest.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No segments to plot.", transform=ax.transAxes,
                ha="center", va="center", color="red")
        return _save(fig, out)

    stroke_ids = manifest["video_stroke_idx"].astype(int).head(max_strokes).tolist()
    n = len(stroke_ids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.9 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, stroke_idx in zip(axes, stroke_ids):
        seg = segments[segments["video_stroke_idx"].astype(int) == stroke_idx]
        if seg.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=8)
            ax.set_title(f"v{stroke_idx}", fontsize=8)
            continue
        s = seg["s_force"].astype(float).to_numpy()
        f = seg["force_raw"].astype(float).to_numpy()
        ax.plot(s, f, color=_PRIMARY, linewidth=1.1)
        ax.fill_between(s, 0.0, f, alpha=0.18, color=_PRIMARY)
        rp3_no_row = manifest[manifest["video_stroke_idx"].astype(int) == stroke_idx]
        rp3_no = int(rp3_no_row.iloc[0]["rp3_stroke_number"]) if not rp3_no_row.empty else "?"
        ax.set_title(f"v{stroke_idx} / r{rp3_no}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused subplot frames if any.
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Per-stroke force curves (matched segments)", fontsize=10, fontweight="bold")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Segments / QC panel
# ---------------------------------------------------------------------------


def plot_qc_drop_reasons(status: pd.DataFrame, out: Path) -> Path:
    """Bar chart of QC drop reasons recorded in ``rp3_pose_force_export_status.csv``."""
    fig, ax = plt.subplots(figsize=(7, 3.0))
    if status.empty or "drop_reason" not in status.columns:
        ax.text(0.5, 0.5, "No segment status data.", transform=ax.transAxes,
                ha="center", va="center", color=_NEUTRAL)
        return _save(fig, out)

    failed = status[~status["segment_exported"].astype(bool)]
    if failed.empty:
        ax.text(0.5, 0.5, "All segments exported successfully.",
                transform=ax.transAxes, ha="center", va="center", color="green")
        return _save(fig, out)

    counts = (
        failed["drop_reason"]
        .astype(str)
        .replace("", "(empty)")
        .value_counts()
        .sort_values(ascending=True)
    )
    ax.barh(counts.index.tolist(), counts.values, color=_ACCENT, alpha=0.7)
    for i, (label, val) in enumerate(zip(counts.index.tolist(), counts.values)):
        ax.text(val, i, f" {int(val)}", va="center", fontsize=8)
    ax.set_xlabel("dropped strokes")
    ax.set_title("Segment-export drop reasons")
    ax.grid(True, axis="x", alpha=0.25)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Dataset panel
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training-report panels
# ---------------------------------------------------------------------------


def plot_metric_bar_group(
    metrics: list[dict[str, float | str]],
    out: Path,
    *,
    metric_keys: Iterable[str] = ("rmse_median", "peak_force_err_median", "correlation_median"),
    label_key: str = "model_name",
    title: str = "Stage metrics comparison",
) -> Path:
    """Grouped bar chart of multiple metrics across model variants."""
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * max(1, len(metrics))), 3.4))
    if not metrics:
        ax.text(0.5, 0.5, "No metrics available.", transform=ax.transAxes,
                ha="center", va="center", color=_NEUTRAL)
        return _save(fig, out)

    metric_keys = list(metric_keys)
    labels = [str(m.get(label_key, f"#{i}")) for i, m in enumerate(metrics)]
    x = np.arange(len(metrics), dtype=float)
    width = 0.85 / max(1, len(metric_keys))

    for i, key in enumerate(metric_keys):
        values = [
            float(m.get(key, np.nan)) if isinstance(m.get(key), (int, float)) else np.nan
            for m in metrics
        ]
        offset = (i - (len(metric_keys) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=min(len(metric_keys), 3))
    return _save(fig, out)


def plot_residual_histogram(
    residuals: np.ndarray,
    out: Path,
    *,
    title: str = "Stroke-level RMSE distribution",
    units: str = "N",
) -> Path:
    """Histogram of per-stroke RMSE residuals."""
    fig, ax = plt.subplots(figsize=(7, 3.0))
    if residuals.size == 0:
        ax.text(0.5, 0.5, "No residual data.", transform=ax.transAxes,
                ha="center", va="center", color=_NEUTRAL)
        return _save(fig, out)
    ax.hist(residuals, bins=30, color=_PRIMARY, alpha=0.75, edgecolor="white")
    median = float(np.median(residuals))
    mean = float(np.mean(residuals))
    ax.axvline(median, color=_ACCENT, linestyle="--", linewidth=1.0,
               label=f"median={median:.2f}{units}")
    ax.axvline(mean, color="black", linestyle=":", linewidth=0.8,
               label=f"mean={mean:.2f}{units}")
    ax.set_xlabel(f"RMSE per stroke ({units})")
    ax.set_ylabel("strokes")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out)


def plot_true_vs_pred_overlay(
    s_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out: Path,
    *,
    title: str = "True vs predicted force curves",
    sample_indices: Iterable[int] | None = None,
    max_curves: int = 6,
) -> Path:
    """Overlay a sample of true vs predicted force curves (one row per stroke)."""
    if y_true.shape != y_pred.shape:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Shape mismatch between true & pred.",
                transform=ax.transAxes, ha="center", va="center", color="red")
        return _save(fig, out)
    n_strokes = y_true.shape[0]
    if n_strokes == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No predictions available.",
                transform=ax.transAxes, ha="center", va="center", color="red")
        return _save(fig, out)

    if sample_indices is None:
        # Take evenly-spaced samples to span the test set.
        indices = list(np.linspace(0, n_strokes - 1, num=min(max_curves, n_strokes)).astype(int))
    else:
        indices = list(sample_indices)[:max_curves]

    cols = min(3, len(indices))
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 1.9 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        if idx < 0 or idx >= n_strokes:
            ax.axis("off")
            continue
        t = y_true[idx]
        p = y_pred[idx]
        ax.plot(s_grid, t, color="black", linewidth=1.2, label="true")
        ax.plot(s_grid, p, color=_ACCENT, linewidth=1.2, alpha=0.85, label="pred")
        ax.fill_between(s_grid, t, p, alpha=0.12, color=_ACCENT)
        ax.set_title(f"stroke {int(idx)}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xlim(float(s_grid[0]), float(s_grid[-1]))
        ax.grid(True, alpha=0.2)

    for ax in axes[len(indices):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    return _save(fig, out)


def plot_cohort_metric_bars(
    cohort_metrics: dict[str, float],
    out: Path,
    *,
    title: str = "Per-athlete RMSE",
    metric_label: str = "RMSE",
) -> Path:
    """Horizontal bar chart of a per-cohort scalar metric (e.g. RMSE per athlete)."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * max(1, len(cohort_metrics)))))
    if not cohort_metrics:
        ax.text(0.5, 0.5, "No cohort metrics available.",
                transform=ax.transAxes, ha="center", va="center", color=_NEUTRAL)
        return _save(fig, out)
    labels = list(cohort_metrics.keys())
    values = [float(v) for v in cohort_metrics.values()]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    ax.barh(labels, values, color=_PRIMARY, alpha=0.7)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    return _save(fig, out)


def plot_pca_explained_variance(pca_ev: pd.DataFrame, out: Path) -> Path:
    """Bar chart of per-component PCA explained variance with cumulative line."""
    fig, ax = plt.subplots(figsize=(7, 3.0))
    if pca_ev.empty:
        ax.text(0.5, 0.5, "No PCA explained-variance data.",
                transform=ax.transAxes, ha="center", va="center", color=_NEUTRAL)
        return _save(fig, out)
    components = pca_ev["component"].astype(int).to_numpy()
    ev = pca_ev["explained_variance_ratio"].astype(float).to_numpy()
    cum = pca_ev["cumulative_explained_variance"].astype(float).to_numpy()

    ax.bar(components, ev, color=_PRIMARY, alpha=0.7, label="per component")
    ax2 = ax.twinx()
    ax2.plot(components, cum, color=_ACCENT, marker="o", label="cumulative")
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance")
    ax2.set_ylabel("Cumulative")
    ax.set_title("Force-curve PCA explained variance")
    ax.grid(True, axis="y", alpha=0.25)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    return _save(fig, out)
