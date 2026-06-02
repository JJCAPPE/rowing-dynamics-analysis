#!/usr/bin/env python3
"""Create executable Jupyter labs for the rowing force-curve study guide."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


BASE_SETUP = r'''
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise RuntimeError("These labs require scikit-learn.") from exc


def make_synthetic_dataset(n=90, g=64, k=14, seed=11):
    """Return arrays with the same broad contract as the real pipeline."""
    rng = np.random.default_rng(seed)
    s_grid = np.linspace(0.0, 1.0, g)
    feature_names = [
        "knee_active_deg", "hip_active_deg", "elbow_active_deg",
        "trunk_vs_horizontal_deg", "spine_flexion_deg", "head_vs_trunk_deg",
        "knee_active_ddeg_ds", "hip_active_ddeg_ds", "elbow_active_ddeg_ds",
        "trunk_vs_horizontal_ddeg_ds", "spine_flexion_ddeg_ds",
        "head_vs_trunk_ddeg_ds", "handle_velocity_px_s", "handle_accel_px_s2",
    ][:k]

    seq = np.zeros((n, g, len(feature_names)), dtype=float)
    rate = rng.normal(27.0, 2.4, size=n)
    stroke_len = rng.normal(142.0, 8.0, size=n)
    drive_s = rng.normal(0.86, 0.08, size=n)
    athlete = np.where(np.arange(n) < n / 2, "athlete_A", "athlete_B")
    run = np.where(np.arange(n) % 3 == 0, "run_0", np.where(np.arange(n) % 3 == 1, "run_1", "run_2"))

    for i in range(n):
        phase = rng.normal(0, 0.025)
        knee = 55 + 92 * np.clip(s_grid + phase, 0, 1) + rng.normal(0, 2.0, g)
        hip = 42 + 38 * np.sin(0.5 * np.pi * np.clip(s_grid + 0.05 + phase, 0, 1)) + rng.normal(0, 1.5, g)
        elbow = 158 - 82 / (1 + np.exp(-14 * (s_grid - 0.67 + phase))) + rng.normal(0, 2.5, g)
        trunk = 72 + 28 * s_grid + rng.normal(0, 1.3, g)
        spine = 18 + 7 * np.sin(np.pi * s_grid) + rng.normal(0, 0.8, g)
        head = 9 + 2 * np.cos(2 * np.pi * s_grid) + rng.normal(0, 0.6, g)
        angles = [knee, hip, elbow, trunk, spine, head]
        cols = []
        for a in angles:
            cols.append(a)
        for a in angles:
            cols.append(np.gradient(a, s_grid))
        cols.append(460 * np.sin(np.pi * s_grid) + rng.normal(0, 10, g))
        cols.append(460 * np.pi * np.cos(np.pi * s_grid) + rng.normal(0, 25, g))
        seq[i] = np.stack(cols[:len(feature_names)], axis=1)

    curves = []
    for i in range(n):
        a = 2.45 + 0.05 * (rate[i] - rate.mean()) + rng.normal(0, 0.12)
        b = 3.65 + rng.normal(0, 0.18)
        shape = (s_grid ** a) * ((1 - s_grid) ** b)
        shape = shape / shape.max()
        coordination_bump = 0.04 * (seq[i, :, 0] - seq[i, :, 0].mean()) / max(seq[i, :, 0].std(), 1)
        peak = 505 + 4.0 * (stroke_len[i] - 140) + 7.0 * (rate[i] - 27) + rng.normal(0, 24)
        curve = peak * np.clip(shape + coordination_bump, 0, None) + rng.normal(0, 8, g)
        curves.append(np.maximum(0, curve))
    force_curves = np.asarray(curves)
    force_mask = np.ones_like(force_curves, dtype=bool)

    strokes = pd.DataFrame({
        "stroke_key": [f"synthetic__{i}" for i in range(n)],
        "run_name": run,
        "athlete_id": athlete,
        "match_seq_idx": np.arange(n),
        "stroke_rate_spm": rate,
        "stroke_length_cm": stroke_len,
        "rp3_drive_s": drive_s,
        "rp3_cycle_s": 60.0 / rate,
        "qc_excluded": False,
    })
    return {
        "strokes_df": strokes,
        "force_curves": force_curves,
        "force_mask": force_mask,
        "kinematic_sequences": seq,
        "s_grid": s_grid,
        "feature_names": feature_names,
    }


def load_or_synthetic_dataset():
    dataset_env = os.environ.get("DATASET_DIR", "").strip()
    dataset_dir = Path(dataset_env).expanduser() if dataset_env else None
    required = [
        "strokes.csv",
        "force_curves_resampled.npy",
        "kinematic_sequences.npy",
        "s_grid.npy",
        "feature_names.json",
    ]
    if dataset_dir is not None and dataset_dir.exists() and all((dataset_dir / name).exists() for name in required):
        print(f"Loading real dataset from {dataset_dir}")
        return {
            "strokes_df": pd.read_csv(dataset_dir / "strokes.csv"),
            "force_curves": np.load(dataset_dir / "force_curves_resampled.npy"),
            "force_mask": np.load(dataset_dir / "force_mask.npy") if (dataset_dir / "force_mask.npy").exists() else None,
            "kinematic_sequences": np.load(dataset_dir / "kinematic_sequences.npy"),
            "s_grid": np.load(dataset_dir / "s_grid.npy"),
            "feature_names": json.load(open(dataset_dir / "feature_names.json")),
        }
    if dataset_env:
        print(f"DATASET_DIR={dataset_dir} is missing required artifacts; using synthetic fallback data.")
    else:
        print("DATASET_DIR is not set; using synthetic fallback data.")
    return make_synthetic_dataset()


ds = load_or_synthetic_dataset()
strokes_df = ds["strokes_df"]
force_curves = ds["force_curves"]
kinematic_sequences = ds["kinematic_sequences"]
s_grid = ds["s_grid"]
feature_names = ds["feature_names"]
print("strokes_df:", strokes_df.shape)
print("force_curves:", force_curves.shape)
print("kinematic_sequences:", kinematic_sequences.shape)
'''


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_dataset_contract.ipynb": [
        ("markdown", "# Lab 01 - Dataset Contract\n\nInspect the core arrays and tables consumed by the training code. Set `DATASET_DIR=/path/to/training_dataset` to use real artifacts."),
        ("code", BASE_SETUP),
        ("code", "display(strokes_df.head())\nprint('feature_names:', feature_names)\nprint('force finite fraction:', np.isfinite(force_curves).mean())"),
        ("code", "fig, ax = plt.subplots(figsize=(8, 4))\nfor y in force_curves[:12]:\n    ax.plot(s_grid, y, alpha=0.45)\nax.set(title='First force curves', xlabel='drive progress s', ylabel='force')\nax.grid(True, alpha=0.25)\nplt.show()"),
        ("code", "print('One sample:')\ni = 0\nprint('X_i shape:', kinematic_sequences[i].shape)\nprint('Y_i shape:', force_curves[i].shape)\nprint(strokes_df.iloc[i][['stroke_key', 'stroke_rate_spm', 'stroke_length_cm']])"),
    ],
    "02_progress_features.ipynb": [
        ("markdown", "# Lab 02 - Progress Features\n\nVisualize interpolation and the chain-rule derivative idea used by the training/inference feature contract."),
        ("code", BASE_SETUP),
        ("code", "i = 0\nfeature = 'knee_active_deg' if 'knee_active_deg' in feature_names else feature_names[0]\nk = feature_names.index(feature)\nangle = kinematic_sequences[i, :, k]\ndtheta_ds = np.gradient(angle, s_grid)\nfig, axes = plt.subplots(1, 2, figsize=(11, 4))\naxes[0].plot(s_grid, angle, lw=2)\naxes[0].set(title=f'{feature} on s-grid', xlabel='s', ylabel='degrees')\naxes[1].plot(s_grid, dtheta_ds, lw=2, color='tab:red')\naxes[1].set(title='Numerical dtheta/ds', xlabel='s', ylabel='deg per progress')\nfor ax in axes: ax.grid(True, alpha=0.25)\nplt.show()"),
        ("code", "raw_s = np.sort(np.random.default_rng(4).uniform(0, 1, size=18))\nraw_y = np.interp(raw_s, s_grid, angle) + np.random.default_rng(5).normal(0, 2, size=18)\ninterp_y = np.interp(s_grid, raw_s, raw_y)\nplt.figure(figsize=(8, 4))\nplt.scatter(raw_s, raw_y, label='irregular video samples')\nplt.plot(s_grid, interp_y, label='interpolated to model grid')\nplt.xlabel('s'); plt.ylabel('angle'); plt.title('Interpolation onto a shared progress grid')\nplt.legend(frameon=False); plt.grid(True, alpha=0.25); plt.show()"),
    ],
    "03_pca_force_curves.ipynb": [
        ("markdown", "# Lab 03 - PCA Force Curves\n\nFit PCA on peak-normalized force curves and reconstruct curves from a small number of scores."),
        ("code", BASE_SETUP),
        ("code", "row_max = np.nanmax(force_curves, axis=1, keepdims=True)\nvalid = np.isfinite(force_curves).all(axis=1) & (row_max[:, 0] > 0)\nY_norm = force_curves[valid] / row_max[valid]\npca = PCA(n_components=min(6, Y_norm.shape[0], Y_norm.shape[1])).fit(Y_norm)\nprint('explained variance:', np.round(pca.explained_variance_ratio_, 3))\nprint('cumulative:', np.round(pca.explained_variance_ratio_.cumsum(), 3))"),
        ("code", "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\naxes[0].bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)\naxes[0].plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker='o')\naxes[0].set(title='PCA explained variance', xlabel='component', ylim=(0, 1.05))\nfor j in range(min(4, pca.n_components_)):\n    axes[1].plot(s_grid, pca.components_[j], label=f'PC{j+1}')\naxes[1].set(title='PCA component directions', xlabel='s')\naxes[1].legend(frameon=False)\nfor ax in axes: ax.grid(True, alpha=0.25)\nplt.show()"),
        ("code", "scores = pca.transform(Y_norm[:5])\nrecon = pca.inverse_transform(scores)\nplt.figure(figsize=(8, 4))\nfor y, r in zip(Y_norm[:5], recon):\n    plt.plot(s_grid, y, color='tab:blue', alpha=0.45)\n    plt.plot(s_grid, r, color='tab:red', alpha=0.7, linestyle='--')\nplt.title('True normalized curves (blue) vs PCA reconstructions (red dashed)')\nplt.xlabel('s'); plt.ylabel('normalized force'); plt.grid(True, alpha=0.25); plt.show()"),
    ],
    "04_cv_and_metrics.ipynb": [
        ("markdown", "# Lab 04 - Cross-Validation And Metrics\n\nCompute the main evaluation metrics and compare split logic."),
        ("code", BASE_SETUP),
        ("code", "def rmse_per_stroke(y_true, y_pred):\n    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=1))\n\ndef curve_corr(y_true, y_pred):\n    out = []\n    for yt, yp in zip(y_true, y_pred):\n        mask = np.isfinite(yt) & np.isfinite(yp)\n        out.append(np.corrcoef(yt[mask], yp[mask])[0, 1] if mask.sum() >= 3 else np.nan)\n    return np.asarray(out)\n\ndef impulse(y):\n    return np.trapz(np.nan_to_num(y, nan=0.0), s_grid, axis=1)\n\npred = 0.94 * np.roll(force_curves, 2, axis=1) + 15\nprint('median RMSE:', np.nanmedian(rmse_per_stroke(force_curves, pred)))\nprint('median corr:', np.nanmedian(curve_corr(force_curves, pred)))\nprint('median impulse error:', np.nanmedian(np.abs(impulse(force_curves) - impulse(pred))))"),
        ("code", "order = strokes_df.sort_values(['run_name', 'match_seq_idx'] if 'match_seq_idx' in strokes_df.columns else ['run_name']).index.to_numpy()\nfolds = np.array_split(order, 5)\nfor i, test in enumerate(folds):\n    train = np.setdiff1d(order, test)\n    print(f'fold {i}: train={len(train):3d}, test={len(test):3d}, test range={test.min()}..{test.max()}')"),
        ("code", "if 'athlete_id' in strokes_df.columns:\n    for athlete in strokes_df['athlete_id'].dropna().unique():\n        test = strokes_df.index[strokes_df['athlete_id'] == athlete]\n        train = strokes_df.index[strokes_df['athlete_id'] != athlete]\n        print(f'held-out athlete={athlete}: train={len(train)}, test={len(test)}')"),
    ],
    "05_stage0_stageA.ipynb": [
        ("markdown", "# Lab 05 - Stage 0 And Stage A\n\nTrain small metadata and kinematic PCA-regression baselines."),
        ("code", BASE_SETUP),
        ("code", "row_max = np.nanmax(force_curves, axis=1, keepdims=True)\nvalid = np.isfinite(force_curves).all(axis=1) & (row_max[:, 0] > 0)\nY_norm = force_curves[valid] / row_max[valid]\npca = PCA(n_components=min(5, Y_norm.shape[0], Y_norm.shape[1])).fit(Y_norm)\nY_scores = pca.transform(Y_norm)\nwork = strokes_df.loc[valid].reset_index(drop=True)\nprint('target scores shape:', Y_scores.shape)"),
        ("code", "meta_cols = [c for c in ['stroke_rate_spm', 'stroke_length_cm', 'rp3_drive_s'] if c in work.columns]\nX_meta = work[meta_cols].to_numpy(float)\nscaler = StandardScaler().fit(X_meta)\nmodel = Ridge(alpha=1.0).fit(scaler.transform(X_meta), Y_scores)\nscore_pred = model.predict(scaler.transform(X_meta))\ncurve_pred = pca.inverse_transform(score_pred) * row_max[valid]\nrmse = np.sqrt(np.mean((force_curves[valid] - curve_pred) ** 2, axis=1))\nprint('Stage 0 metadata columns:', meta_cols)\nprint('in-sample median RMSE:', np.median(rmse))"),
        ("code", "summary = {}\nangle_cols = [c for c in feature_names if c.endswith('_deg')][:6]\nfor col in angle_cols:\n    k = feature_names.index(col)\n    vals = kinematic_sequences[valid, :, k]\n    summary[f'{col}_min'] = np.nanmin(vals, axis=1)\n    summary[f'{col}_max'] = np.nanmax(vals, axis=1)\n    summary[f'{col}_range'] = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)\n    summary[f'{col}_mean'] = np.nanmean(vals, axis=1)\nX_stageA_df = pd.concat([work[meta_cols].reset_index(drop=True), pd.DataFrame(summary)], axis=1)\nX = X_stageA_df.to_numpy(float)\nscaler_a = StandardScaler().fit(X)\nmodels = {'ridge': Ridge(alpha=1.0), 'lasso': Lasso(alpha=0.001, max_iter=5000), 'gbr': MultiOutputRegressor(GradientBoostingRegressor(random_state=0))}\nfor name, m in models.items():\n    m.fit(scaler_a.transform(X), Y_scores)\n    pred = pca.inverse_transform(m.predict(scaler_a.transform(X))) * row_max[valid]\n    print(name, 'median RMSE:', np.median(np.sqrt(np.mean((force_curves[valid] - pred) ** 2, axis=1))))"),
    ],
    "06_stageB_toy_model.ipynb": [
        ("markdown", "# Lab 06 - Stage B Toy Model\n\nDemonstrate masked sequence loss and, if PyTorch is installed, fit a tiny 1D convolutional model."),
        ("code", BASE_SETUP),
        ("code", "mask = np.ones_like(force_curves, dtype=bool)\nmask[:, -8:] = np.arange(force_curves.shape[1])[-8:][None, :] < (force_curves.shape[1] - np.arange(force_curves.shape[0])[:, None] % 8)\npred = force_curves + np.random.default_rng(9).normal(0, 25, size=force_curves.shape)\nmasked_mse = (((pred - force_curves) ** 2) * mask).sum() / mask.sum()\nplain_mse = np.mean((pred - force_curves) ** 2)\nprint('plain MSE:', plain_mse)\nprint('masked MSE:', masked_mse)"),
        ("code", "try:\n    import torch\n    import torch.nn as nn\n    torch.manual_seed(0)\n    X = torch.tensor(kinematic_sequences[:48], dtype=torch.float32).transpose(1, 2)\n    y = torch.tensor(force_curves[:48], dtype=torch.float32)\n    model = nn.Sequential(nn.Conv1d(X.shape[1], 24, kernel_size=5, padding=2), nn.ReLU(), nn.Conv1d(24, 1, kernel_size=1))\n    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)\n    for epoch in range(20):\n        opt.zero_grad()\n        out = model(X).squeeze(1)\n        loss = ((out - y) ** 2).mean()\n        loss.backward(); opt.step()\n    print('tiny conv final loss:', float(loss.detach()))\nexcept Exception as exc:\n    print('PyTorch unavailable or failed; masked-loss NumPy demo above is still valid.')\n    print(type(exc).__name__, exc)"),
    ],
}


def make_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for index, (cell_type, source) in enumerate(cells):
        if cell_type == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": f"markdown-{index}", "metadata": {}, "source": source.strip() + "\n"})
        elif cell_type == "code":
            nb_cells.append({
                "cell_type": "code",
                "id": f"code-{index}",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.strip() + "\n",
            })
        else:
            raise ValueError(cell_type)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = NB_DIR / name
        path.write_text(json.dumps(make_notebook(cells), indent=2), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
