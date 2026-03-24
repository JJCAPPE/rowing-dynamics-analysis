#!/usr/bin/env python3
"""Section 11 modeling pipeline: Stage 0 sanity baselines, Stage A kinematic
baselines, and Stage B sequence models for force-curve prediction.

Usage
-----
    python -m inference.modeling \\
        --dataset-dir runs/<run>/inference/training_dataset/ \\
        --stages 0 A B \\
        --output-dir runs/<run>/inference/modeling_results/
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Generator

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from eval_metrics import (
    athlete_held_out_split,
    compute_alignment_metrics,
    compute_all_metrics,
    detect_evaluation_regime,
    format_metrics_table,
    reconstruct_from_pca,
    rmse_per_stroke,
    session_held_out_split,
    time_block_split,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_FEATURE_COLS = ["stroke_rate_spm", "stroke_length_cm", "rp3_drive_s"]

ANGLE_COLS = [
    "knee_active_deg",
    "hip_active_deg",
    "elbow_active_deg",
    "trunk_vs_horizontal_deg",
    "spine_flexion_deg",
]
SUMMARY_SUFFIXES = ["_min", "_max", "_range", "_mean", "_s_at_max"]
KINEMATIC_SUMMARY_COLS = [
    f"{a}{s}" for a in ANGLE_COLS for s in SUMMARY_SUFFIXES
]


def _get_cv_splitter(
    strokes_df: pd.DataFrame,
    method: str,
    n_splits: int,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    if method == "athlete_held_out":
        return athlete_held_out_split(strokes_df)
    if method == "session_held_out":
        return session_held_out_split(strokes_df)
    return time_block_split(strokes_df, n_splits=n_splits)


def _load_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Load all training-dataset artifacts into a dict."""
    d: dict[str, Any] = {}
    d["strokes_df"] = pd.read_csv(dataset_dir / "strokes.csv")
    d["force_curves"] = np.load(dataset_dir / "force_curves_resampled.npy")
    d["force_peak_norm"] = np.load(dataset_dir / "force_curves_peak_norm.npy")
    d["kinematic_sequences"] = np.load(dataset_dir / "kinematic_sequences.npy")
    d["s_grid"] = np.load(dataset_dir / "s_grid.npy")
    d["pca_model"] = joblib.load(dataset_dir / "pca_model.joblib")
    with open(dataset_dir / "feature_names.json") as f:
        d["feature_names"] = json.load(f)
    with open(dataset_dir / "dataset_summary.json") as f:
        d["dataset_summary"] = json.load(f)
    return d


# ===================================================================
# Stage 0.1  --  Force Reproducibility Floor
# ===================================================================


def stage0_reproducibility(
    strokes_df: pd.DataFrame,
    force_curves: np.ndarray,
    s_grid: np.ndarray,
    *,
    rate_bin_width: float = 2.0,
    length_bin_width: float = 5.0,
) -> dict[str, Any]:
    """Quantify within-condition force-curve variability.

    Bins strokes by (run_name, stroke_rate_spm, stroke_length_cm) and
    computes pairwise curve distances within each bin.
    """
    df = strokes_df.copy()
    df["rate_bin"] = (df["stroke_rate_spm"] / rate_bin_width).round() * rate_bin_width
    df["len_bin"] = (df["stroke_length_cm"] / length_bin_width).round() * length_bin_width

    group_cols = ["run_name", "rate_bin", "len_bin"]
    available = [c for c in group_cols if c in df.columns]
    if not available:
        available = ["rate_bin"]

    bin_stats: list[dict[str, Any]] = []
    all_pairwise_rmse: list[float] = []

    for key, grp in df.groupby(available, sort=False):
        indices = grp.index.to_numpy()
        if len(indices) < 2:
            continue

        curves = force_curves[indices]
        valid = np.all(np.isfinite(curves), axis=1)
        curves_valid = curves[valid]
        if curves_valid.shape[0] < 2:
            continue

        pairwise: list[float] = []
        for i, j in combinations(range(curves_valid.shape[0]), 2):
            d = float(np.sqrt(np.mean((curves_valid[i] - curves_valid[j]) ** 2)))
            pairwise.append(d)
        all_pairwise_rmse.extend(pairwise)

        peaks = np.max(curves_valid, axis=1)
        impulses = np.trapz(curves_valid, s_grid, axis=1)
        peak_cv = float(peaks.std() / peaks.mean()) if peaks.mean() > 1e-9 else float("nan")
        impulse_cv = float(impulses.std() / impulses.mean()) if impulses.mean() > 1e-9 else float("nan")

        bin_stats.append({
            "bin_key": str(key),
            "n_strokes": int(curves_valid.shape[0]),
            "n_pairs": len(pairwise),
            "median_pairwise_rmse": float(np.median(pairwise)),
            "mean_pairwise_rmse": float(np.mean(pairwise)),
            "peak_force_cv": peak_cv,
            "impulse_cv": impulse_cv,
        })

    summary: dict[str, Any] = {
        "n_bins_with_pairs": len(bin_stats),
        "total_pairs": len(all_pairwise_rmse),
    }
    if all_pairwise_rmse:
        arr = np.asarray(all_pairwise_rmse)
        summary["overall_median_pairwise_rmse"] = float(np.median(arr))
        summary["overall_mean_pairwise_rmse"] = float(np.mean(arr))
        summary["overall_std_pairwise_rmse"] = float(np.std(arr))
    else:
        summary["overall_median_pairwise_rmse"] = float("nan")
        summary["overall_mean_pairwise_rmse"] = float("nan")
        summary["overall_std_pairwise_rmse"] = float("nan")

    if bin_stats:
        summary["median_peak_force_cv"] = float(
            np.nanmedian([b["peak_force_cv"] for b in bin_stats])
        )
        summary["median_impulse_cv"] = float(
            np.nanmedian([b["impulse_cv"] for b in bin_stats])
        )
    else:
        summary["median_peak_force_cv"] = float("nan")
        summary["median_impulse_cv"] = float("nan")

    summary["per_bin"] = bin_stats
    return summary


# ===================================================================
# Stage 0.2  --  Metadata-Only Baseline
# ===================================================================


def _extract_pca_targets(
    strokes_df: pd.DataFrame,
    n_pca_targets: int,
) -> np.ndarray:
    cols = [f"pca_{i}" for i in range(n_pca_targets)]
    missing = [c for c in cols if c not in strokes_df.columns]
    if missing:
        raise ValueError(f"Missing PCA target columns: {missing}")
    return strokes_df[cols].to_numpy(dtype=np.float64)


def _extract_features(
    strokes_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    present = [c for c in feature_cols if c in strokes_df.columns]
    if not present:
        raise ValueError(f"No feature columns found. Expected: {feature_cols}")
    X = strokes_df[present].to_numpy(dtype=np.float64)
    return X


def _get_peak_forces(force_curves: np.ndarray) -> np.ndarray:
    return np.nanmax(force_curves, axis=1)


def _run_pca_regression_cv(
    *,
    strokes_df: pd.DataFrame,
    force_curves: np.ndarray,
    pca_model: PCA,
    s_grid: np.ndarray,
    feature_cols: list[str],
    n_pca_targets: int,
    cv_splitter: Generator[tuple[np.ndarray, np.ndarray], None, None],
    model_factory: Any,
    model_name: str,
) -> dict[str, Any]:
    """Run a CV loop for a PCA-coefficient regression model.

    Returns per-fold and aggregated metrics, plus the model refitted on all data.
    """
    X = _extract_features(strokes_df, feature_cols)
    Y_pca = _extract_pca_targets(strokes_df, n_pca_targets)
    peak_forces = _get_peak_forces(force_curves)

    N = X.shape[0]
    y_pred_all = np.full_like(force_curves, np.nan)
    fold_metrics: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv_splitter):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y_pca[train_idx], Y_pca[test_idx]

        finite_train = np.all(np.isfinite(X_train), axis=1) & np.all(
            np.isfinite(Y_train), axis=1
        )
        X_train_f = X_train[finite_train]
        Y_train_f = Y_train[finite_train]
        if X_train_f.shape[0] < 3:
            continue

        scaler = StandardScaler().fit(X_train_f)
        X_train_s = scaler.transform(X_train_f)
        X_test_s = scaler.transform(X_test)

        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_s, Y_train_f)

        Y_pred_pca = model.predict(X_test_s)
        curves_pred = reconstruct_from_pca(
            pca_model, Y_pred_pca, peak_forces=peak_forces[test_idx]
        )
        y_pred_all[test_idx] = curves_pred

        fm = compute_all_metrics(force_curves[test_idx], curves_pred, s_grid)
        fm["fold"] = fold_i
        fm["train_size"] = int(X_train_f.shape[0])
        fm["test_size"] = int(X_test.shape[0])
        fold_metrics.append(fm)

    has_pred = np.any(np.isfinite(y_pred_all), axis=1)
    if has_pred.sum() > 0:
        overall = compute_all_metrics(
            force_curves[has_pred], y_pred_all[has_pred], s_grid
        )
    else:
        overall = {"rmse_median": float("nan")}
    overall["model"] = model_name

    finite_all = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y_pca), axis=1)
    scaler_full = StandardScaler().fit(X[finite_all])
    final_model = model_factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(scaler_full.transform(X[finite_all]), Y_pca[finite_all])

    return {
        "model_name": model_name,
        "overall_metrics": overall,
        "fold_metrics": fold_metrics,
        "final_model": final_model,
        "final_scaler": scaler_full,
        "feature_cols": feature_cols,
        "cv_predictions": y_pred_all,
    }


def stage0_metadata_baseline(
    strokes_df: pd.DataFrame,
    force_curves: np.ndarray,
    pca_model: PCA,
    s_grid: np.ndarray,
    cv_method: str = "time_block",
    n_splits: int = 5,
    n_pca_targets: int = 5,
) -> dict[str, Any]:
    """Stage 0.2: metadata-only baseline (Ridge on stroke_rate, stroke_length, drive_time)."""
    feature_cols = [c for c in METADATA_FEATURE_COLS if c in strokes_df.columns]
    if not feature_cols:
        raise ValueError("No metadata feature columns found in strokes.csv.")

    def _make_ridge() -> Ridge:
        return Ridge(alpha=1.0)

    cv = _get_cv_splitter(strokes_df, cv_method, n_splits)
    return _run_pca_regression_cv(
        strokes_df=strokes_df,
        force_curves=force_curves,
        pca_model=pca_model,
        s_grid=s_grid,
        feature_cols=feature_cols,
        n_pca_targets=n_pca_targets,
        cv_splitter=cv,
        model_factory=_make_ridge,
        model_name="metadata_ridge",
    )


# ===================================================================
# Stage 0.3  --  Baseline Gate
# ===================================================================


def check_baseline_gate(
    metadata_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> bool:
    """Return True if *candidate* beats metadata-only on both curve RMSE and
    at least one rowing-relevant metric (peak force error, impulse error)."""
    rmse_better = candidate_metrics.get("rmse_median", float("inf")) < metadata_metrics.get(
        "rmse_median", float("inf")
    )
    pf_better = candidate_metrics.get("peak_force_err_median", float("inf")) < metadata_metrics.get(
        "peak_force_err_median", float("inf")
    )
    imp_better = candidate_metrics.get("impulse_err_median", float("inf")) < metadata_metrics.get(
        "impulse_err_median", float("inf")
    )
    return rmse_better and (pf_better or imp_better)


# ===================================================================
# Stage A  --  Interpretable Kinematic Baselines
# ===================================================================


def stageA_kinematic_baselines(
    strokes_df: pd.DataFrame,
    force_curves: np.ndarray,
    pca_model: PCA,
    s_grid: np.ndarray,
    cv_method: str = "time_block",
    n_splits: int = 5,
    n_pca_targets: int = 5,
) -> dict[str, Any]:
    """Stage A: kinematic summary features -> PCA coefficients.

    Runs Ridge, Lasso, and GBR; returns results for all three plus
    feature importance rankings.
    """
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in strokes_df.columns]
    kin_cols = [c for c in KINEMATIC_SUMMARY_COLS if c in strokes_df.columns]
    feature_cols = meta_cols + kin_cols
    if len(feature_cols) < 3:
        raise ValueError(f"Too few feature columns found ({len(feature_cols)}). "
                         "Ensure strokes.csv has scalar kinematic summaries.")

    models: dict[str, Any] = {
        "ridge": lambda: Ridge(alpha=1.0),
        "lasso": lambda: Lasso(alpha=0.01, max_iter=5000),
        "gbr": lambda: MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        ),
    }

    results: dict[str, Any] = {"feature_cols": feature_cols, "models": {}}
    for name, factory in models.items():
        cv = _get_cv_splitter(strokes_df, cv_method, n_splits)
        res = _run_pca_regression_cv(
            strokes_df=strokes_df,
            force_curves=force_curves,
            pca_model=pca_model,
            s_grid=s_grid,
            feature_cols=feature_cols,
            n_pca_targets=n_pca_targets,
            cv_splitter=cv,
            model_factory=factory,
            model_name=f"stageA_{name}",
        )
        importance = _extract_importance(res["final_model"], feature_cols, name)
        res["feature_importance"] = importance
        results["models"][name] = res

    return results


def _extract_importance(
    model: Any,
    feature_cols: list[str],
    model_type: str,
) -> list[dict[str, Any]]:
    """Extract feature importance from a fitted model."""
    n_features = len(feature_cols)
    importances = np.zeros(n_features)

    if model_type in ("ridge", "lasso"):
        coefs = np.asarray(model.coef_)
        if coefs.ndim == 2:
            importances = np.mean(np.abs(coefs), axis=0)
        else:
            importances = np.abs(coefs)
    elif model_type == "gbr":
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                fi = getattr(est, "feature_importances_", None)
                if fi is not None:
                    importances += fi
            importances /= max(len(model.estimators_), 1)

    if importances.shape[0] != n_features:
        importances = importances[:n_features]

    ranked = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": v} for f, v in ranked]


# ===================================================================
# Stage B  --  Sequence Models (TCN)
# ===================================================================


def _import_torch():
    """Lazily import torch to avoid hard dependency for Stage 0/A."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for Stage B sequence models. "
            "Install with: pip install torch"
        ) from e


class _ForceCurveTCN:
    """Wrapper that builds and holds the TCN model.

    Defined as a regular class so the module can be imported without torch.
    The actual nn.Module is created inside ``build()``.
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model = None

    def build(self):
        torch, nn = _import_torch()

        class TemporalBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
                super().__init__()
                padding = (kernel_size - 1) * dilation
                self.conv1 = nn.Conv1d(
                    in_ch, out_ch, kernel_size,
                    padding=padding, dilation=dilation,
                )
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.conv2 = nn.Conv1d(
                    out_ch, out_ch, kernel_size,
                    padding=padding, dilation=dilation,
                )
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.drop = nn.Dropout(dropout)
                self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
                self.padding = padding
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.conv1(x)
                out = out[:, :, :x.size(2)]
                out = self.relu(self.bn1(out))
                out = self.drop(out)
                out = self.conv2(out)
                out = out[:, :, :x.size(2)]
                out = self.relu(self.bn2(out))
                out = self.drop(out)
                return self.relu(out + self.residual(x))

        class TCN(nn.Module):
            def __init__(self, in_ch, hidden, n_blocks, ks, dropout):
                super().__init__()
                layers = []
                for i in range(n_blocks):
                    dilation = 2 ** i
                    ic = in_ch if i == 0 else hidden
                    layers.append(TemporalBlock(ic, hidden, ks, dilation, dropout))
                self.network = nn.Sequential(*layers)
                self.head = nn.Conv1d(hidden, 1, 1)

            def forward(self, x):
                # x: (batch, seq_len, in_channels) -> transpose to (batch, in_channels, seq_len)
                x = x.transpose(1, 2)
                x = self.network(x)
                x = self.head(x)
                return x.squeeze(1)  # (batch, seq_len)

        self.model = TCN(
            self.in_channels,
            self.hidden_channels,
            self.n_blocks,
            self.kernel_size,
            self.dropout,
        )
        return self.model


def stageB_sequence_models(
    kinematic_sequences: np.ndarray,
    force_curves: np.ndarray,
    s_grid: np.ndarray,
    strokes_df: pd.DataFrame,
    cv_method: str = "time_block",
    n_splits: int = 5,
    *,
    hidden_channels: int = 64,
    n_blocks: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Stage B: TCN sequence model with CV evaluation."""
    torch, nn = _import_torch()

    N, G, K = kinematic_sequences.shape
    y_pred_all = np.full_like(force_curves, np.nan)
    fold_metrics: list[dict[str, Any]] = []
    best_state_dict = None
    best_fold_rmse = float("inf")

    cv = _get_cv_splitter(strokes_df, cv_method, n_splits)
    for fold_i, (train_idx, test_idx) in enumerate(cv):
        X_train = kinematic_sequences[train_idx]
        y_train = force_curves[train_idx]
        X_test = kinematic_sequences[test_idx]
        y_test = force_curves[test_idx]

        finite_train = (
            np.all(np.isfinite(X_train.reshape(X_train.shape[0], -1)), axis=1)
            & np.all(np.isfinite(y_train), axis=1)
        )
        X_train = X_train[finite_train]
        y_train = y_train[finite_train]
        if X_train.shape[0] < 5:
            continue

        val_size = max(1, int(0.15 * X_train.shape[0]))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        if X_train.shape[0] < 3:
            continue

        feat_mean = np.nanmean(X_train.reshape(-1, K), axis=0)
        feat_std = np.nanstd(X_train.reshape(-1, K), axis=0)
        feat_std = np.where(feat_std < 1e-9, 1.0, feat_std)
        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std
        X_test_norm = (X_test - feat_mean) / feat_std

        force_mean = np.nanmean(y_train)
        force_std = np.nanstd(y_train)
        if force_std < 1e-9:
            force_std = 1.0
        y_train_n = (y_train - force_mean) / force_std
        y_val_n = (y_val - force_mean) / force_std

        np.nan_to_num(X_train, copy=False, nan=0.0)
        np.nan_to_num(X_val, copy=False, nan=0.0)
        np.nan_to_num(X_test_norm, copy=False, nan=0.0)
        np.nan_to_num(y_train_n, copy=False, nan=0.0)
        np.nan_to_num(y_val_n, copy=False, nan=0.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tcn_wrapper = _ForceCurveTCN(
            in_channels=K,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        model = tcn_wrapper.build().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        loss_fn = nn.MSELoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train_n, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_n, dtype=torch.float32, device=device)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_epoch_state = None

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(X_train_t.size(0), device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, X_train_t.size(0), batch_size):
                idx = perm[start : start + batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_epoch_state is not None:
            model.load_state_dict(best_epoch_state)
        model.eval()

        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
        with torch.no_grad():
            test_pred_n = model(X_test_t).cpu().numpy()
        test_pred = test_pred_n * force_std + force_mean
        y_pred_all[test_idx] = test_pred

        fm = compute_all_metrics(y_test, test_pred, s_grid)
        fm["fold"] = fold_i
        fm["train_size"] = int(X_train.shape[0])
        fm["test_size"] = int(X_test.shape[0])
        fm["best_val_loss"] = float(best_val_loss)
        fold_metrics.append(fm)

        fold_rmse = fm.get("rmse_median", float("inf"))
        if fold_rmse < best_fold_rmse:
            best_fold_rmse = fold_rmse
            best_state_dict = best_epoch_state

    has_pred = np.any(np.isfinite(y_pred_all), axis=1)
    if has_pred.sum() > 0:
        overall = compute_all_metrics(
            force_curves[has_pred], y_pred_all[has_pred], s_grid
        )
    else:
        overall = {"rmse_median": float("nan")}
    overall["model"] = "stageB_tcn"

    return {
        "model_name": "stageB_tcn",
        "overall_metrics": overall,
        "fold_metrics": fold_metrics,
        "best_state_dict": best_state_dict,
        "cv_predictions": y_pred_all,
        "architecture": {
            "hidden_channels": hidden_channels,
            "n_blocks": n_blocks,
            "kernel_size": kernel_size,
            "dropout": dropout,
        },
    }


# ===================================================================
# Unified Evaluation Report (Section 12)
# ===================================================================


def write_evaluation_report(
    output_dir: Path,
    *,
    regime_info: dict[str, str],
    alignment_metrics: dict[str, float],
    all_results: list[dict[str, Any]],
    cv_method: str,
    n_splits: int,
) -> None:
    """Write a structured evaluation report as JSON and human-readable text."""
    report: dict[str, Any] = {
        "evaluation_regime": regime_info,
        "cv_method": cv_method,
        "n_splits": n_splits,
        "alignment_integrity": alignment_metrics,
        "model_results": all_results,
    }

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(_serializable(report), f, indent=2)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  EVALUATION REPORT (Section 12)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Regime:     {regime_info['regime']}")
    lines.append(f"  CV method:  {cv_method} (n_splits={n_splits})")
    lines.append(f"  Athletes:   {regime_info.get('n_athletes', '?')}")
    lines.append("")
    lines.append(f"  NOTE: {regime_info['disclaimer']}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  ALIGNMENT INTEGRITY")
    lines.append("-" * 72)
    for k, v in alignment_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:40s} {v:.4f}")
        else:
            lines.append(f"  {k:40s} {v}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  MODEL COMPARISON")
    lines.append("-" * 72)
    if all_results:
        lines.append(format_metrics_table(all_results))
    else:
        lines.append("  (no model results)")
    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(text + "\n")

    print(text)


# ===================================================================
# CLI
# ===================================================================


def _serializable(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Section 11 modeling: Stage 0 baselines, Stage A kinematic models, Stage B sequence models."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to training dataset directory from build_training_dataset.py.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["0", "A", "B"],
        choices=["0", "A", "B"],
        help="Which stages to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for modeling results (default: <dataset-dir>/../modeling_results/).",
    )
    parser.add_argument(
        "--cv-method",
        choices=["time_block", "session_held_out", "athlete_held_out"],
        default="time_block",
        help="Cross-validation method (default: time_block).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds for time_block (default: 5).",
    )
    parser.add_argument(
        "--n-pca-targets",
        type=int,
        default=5,
        help="Number of PCA components to predict (default: 5).",
    )
    parser.add_argument(
        "--skip-gate-check",
        action="store_true",
        help="Run all stages even if baseline gate fails.",
    )
    parser.add_argument(
        "--tcn-hidden",
        type=int,
        default=64,
        help="TCN hidden channels (default: 64).",
    )
    parser.add_argument(
        "--tcn-blocks",
        type=int,
        default=4,
        help="TCN number of temporal blocks (default: 4).",
    )
    parser.add_argument(
        "--tcn-epochs",
        type=int,
        default=300,
        help="TCN max training epochs (default: 300).",
    )
    parser.add_argument(
        "--tcn-patience",
        type=int,
        default=20,
        help="TCN early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--tcn-batch-size",
        type=int,
        default=32,
        help="TCN batch size (default: 32).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        return 1

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_dir.parent / "modeling_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_pred_dir = output_dir / "cv_predictions"
    cv_pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_dir}")
    ds = _load_dataset(dataset_dir)
    strokes_df = ds["strokes_df"]
    force_curves = ds["force_curves"]
    pca_model = ds["pca_model"]
    s_grid = ds["s_grid"]
    kinematic_sequences = ds["kinematic_sequences"]

    valid_mask = (
        np.all(np.isfinite(force_curves), axis=1)
        & (~strokes_df.get("qc_excluded", pd.Series(False)).astype(bool).to_numpy())
    )
    print(f"Dataset: {force_curves.shape[0]} strokes, {int(valid_mask.sum())} valid.")

    regime_info = detect_evaluation_regime(strokes_df, args.cv_method)
    print(f"\nEvaluation regime: {regime_info['regime']}")
    print(f"  {regime_info['disclaimer']}")

    alignment_metrics = compute_alignment_metrics(strokes_df)
    print(f"\nAlignment integrity:")
    print(f"  mean |cum_catch_err|: {alignment_metrics.get('mean_abs_cum_catch_err_s', float('nan')):.4f} s")
    print(f"  max  |cum_catch_err|: {alignment_metrics.get('max_abs_cum_catch_err_s', float('nan')):.4f} s")
    print(f"  mean rate diff:       {alignment_metrics.get('mean_rate_diff_spm', float('nan')):.4f} spm")
    print(f"  stroke-rate r:        {alignment_metrics.get('stroke_rate_pearson_r', float('nan')):.4f}")

    all_results: list[dict[str, Any]] = []
    gate_metrics: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Stage 0
    # ------------------------------------------------------------------
    if "0" in args.stages:
        print("\n=== Stage 0.1: Force Reproducibility Floor ===")
        repro = stage0_reproducibility(strokes_df, force_curves, s_grid)
        print(f"  Bins with >= 2 strokes: {repro['n_bins_with_pairs']}")
        print(f"  Total pairwise comparisons: {repro['total_pairs']}")
        print(f"  Overall median pairwise RMSE: {repro['overall_median_pairwise_rmse']:.4f}")
        print(f"  Median peak-force CV: {repro['median_peak_force_cv']:.4f}")
        print(f"  Median impulse CV: {repro['median_impulse_cv']:.4f}")

        with open(output_dir / "stage0_reproducibility.json", "w") as f:
            json.dump(_serializable(repro), f, indent=2)

        print("\n=== Stage 0.2: Metadata-Only Baseline ===")
        s0_result = stage0_metadata_baseline(
            strokes_df,
            force_curves,
            pca_model,
            s_grid,
            cv_method=args.cv_method,
            n_splits=args.n_splits,
            n_pca_targets=args.n_pca_targets,
        )
        gate_metrics = s0_result["overall_metrics"]
        print(f"  RMSE median: {gate_metrics.get('rmse_median', float('nan')):.4f}")
        print(f"  Peak force err median: {gate_metrics.get('peak_force_err_median', float('nan')):.4f}")
        print(f"  Impulse err median: {gate_metrics.get('impulse_err_median', float('nan')):.4f}")
        print(f"  Correlation median: {gate_metrics.get('correlation_median', float('nan')):.4f}")

        s0_out = {
            "evaluation_regime": regime_info["regime"],
            "overall_metrics": s0_result["overall_metrics"],
            "fold_metrics": s0_result["fold_metrics"],
            "feature_cols": s0_result["feature_cols"],
        }
        with open(output_dir / "stage0_metadata_baseline.json", "w") as f:
            json.dump(_serializable(s0_out), f, indent=2)

        np.save(cv_pred_dir / "stage0_metadata_predictions.npy", s0_result["cv_predictions"])
        all_results.append(s0_result["overall_metrics"])

        joblib.dump(
            {"model": s0_result["final_model"], "scaler": s0_result["final_scaler"]},
            output_dir / "stage0_metadata_model.joblib",
        )

    # ------------------------------------------------------------------
    # Stage A
    # ------------------------------------------------------------------
    if "A" in args.stages:
        print("\n=== Stage A: Interpretable Kinematic Baselines ===")
        sa_result = stageA_kinematic_baselines(
            strokes_df,
            force_curves,
            pca_model,
            s_grid,
            cv_method=args.cv_method,
            n_splits=args.n_splits,
            n_pca_targets=args.n_pca_targets,
        )

        sa_out: dict[str, Any] = {"evaluation_regime": regime_info["regime"], "feature_cols": sa_result["feature_cols"], "models": {}}
        best_sa_name = None
        best_sa_rmse = float("inf")

        for name, res in sa_result["models"].items():
            m = res["overall_metrics"]
            passed = "N/A"
            if gate_metrics is not None:
                passed = "PASS" if check_baseline_gate(gate_metrics, m) else "FAIL"
            rmse_val = m.get("rmse_median", float("nan"))
            print(f"  {name}: RMSE={rmse_val:.4f}  gate={passed}")
            all_results.append(m)

            if rmse_val < best_sa_rmse:
                best_sa_rmse = rmse_val
                best_sa_name = name

            sa_out["models"][name] = {
                "overall_metrics": res["overall_metrics"],
                "fold_metrics": res["fold_metrics"],
                "feature_importance": res["feature_importance"],
                "gate_passed": passed,
            }
            np.save(
                cv_pred_dir / f"stageA_{name}_predictions.npy",
                res["cv_predictions"],
            )

        if best_sa_name:
            best_res = sa_result["models"][best_sa_name]
            joblib.dump(
                {"model": best_res["final_model"], "scaler": best_res["final_scaler"]},
                output_dir / "stageA_best_model.joblib",
            )
            sa_out["best_model"] = best_sa_name

            top_features = best_res["feature_importance"][:10]
            print(f"\n  Top features ({best_sa_name}):")
            for fi in top_features:
                print(f"    {fi['feature']}: {fi['importance']:.4f}")

        with open(output_dir / "stageA_results.json", "w") as f:
            json.dump(_serializable(sa_out), f, indent=2)

        if gate_metrics is not None and best_sa_name:
            best_m = sa_result["models"][best_sa_name]["overall_metrics"]
            if not check_baseline_gate(gate_metrics, best_m) and not args.skip_gate_check:
                print("\n  WARNING: Best Stage A model does not beat the metadata-only gate.")
                print("  Proceeding to Stage B anyway (use --skip-gate-check to suppress).")

    # ------------------------------------------------------------------
    # Stage B
    # ------------------------------------------------------------------
    if "B" in args.stages:
        print("\n=== Stage B: Sequence Models (TCN) ===")
        try:
            sb_result = stageB_sequence_models(
                kinematic_sequences,
                force_curves,
                s_grid,
                strokes_df,
                cv_method=args.cv_method,
                n_splits=args.n_splits,
                hidden_channels=args.tcn_hidden,
                n_blocks=args.tcn_blocks,
                max_epochs=args.tcn_epochs,
                patience=args.tcn_patience,
                batch_size=args.tcn_batch_size,
            )
            m = sb_result["overall_metrics"]
            passed = "N/A"
            if gate_metrics is not None:
                passed = "PASS" if check_baseline_gate(gate_metrics, m) else "FAIL"
            print(f"  TCN: RMSE={m.get('rmse_median', float('nan')):.4f}  gate={passed}")
            print(f"  Peak force err: {m.get('peak_force_err_median', float('nan')):.4f}")
            print(f"  Correlation: {m.get('correlation_median', float('nan')):.4f}")
            all_results.append(m)

            sb_out = {
                "evaluation_regime": regime_info["regime"],
                "overall_metrics": sb_result["overall_metrics"],
                "fold_metrics": sb_result["fold_metrics"],
                "architecture": sb_result["architecture"],
                "gate_passed": passed,
            }
            with open(output_dir / "stageB_results.json", "w") as f:
                json.dump(_serializable(sb_out), f, indent=2)

            np.save(cv_pred_dir / "stageB_tcn_predictions.npy", sb_result["cv_predictions"])

            if sb_result["best_state_dict"] is not None:
                torch, _ = _import_torch()
                torch.save(sb_result["best_state_dict"], output_dir / "stageB_tcn_state.pt")

        except ImportError:
            print("  SKIPPED: PyTorch not available. Install torch to run Stage B.")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    # ------------------------------------------------------------------
    # Comparison summary + unified evaluation report
    # ------------------------------------------------------------------
    if all_results:
        with open(output_dir / "comparison_summary.json", "w") as f:
            json.dump(_serializable(all_results), f, indent=2)

    write_evaluation_report(
        output_dir,
        regime_info=regime_info,
        alignment_metrics=alignment_metrics,
        all_results=all_results,
        cv_method=args.cv_method,
        n_splits=args.n_splits,
    )

    print(f"\nResults written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
