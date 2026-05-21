"""Tests for :mod:`rowing.reports.training_report` (Phase 6)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CI must stay headless

import numpy as np
import pandas as pd
import pytest

from rowing.reports.training_report import generate_training_report


def _make_modeling_dir(
    tmp_path: Path,
    *,
    n_strokes: int = 16,
    n_grid: int = 32,
    include_dataset: bool = True,
    include_stage_b: bool = True,
) -> tuple[Path, Path | None]:
    rng = np.random.default_rng(0)
    modeling_dir = tmp_path / "modeling_results"
    cv_pred_dir = modeling_dir / "cv_predictions"
    cv_pred_dir.mkdir(parents=True)

    # Synthetic curves + predictions.
    s_grid = np.linspace(0, 1, n_grid)
    base = 500 * np.sin(np.pi * s_grid)
    force_curves = base[None, :] + rng.normal(0, 5.0, size=(n_strokes, n_grid))
    pred = force_curves + rng.normal(0, 8.0, size=(n_strokes, n_grid))

    np.save(cv_pred_dir / "stage0_metadata_predictions.npy", pred)
    np.save(cv_pred_dir / "stageA_ridge_predictions.npy", pred)
    if include_stage_b:
        np.save(cv_pred_dir / "stageB_tcn_predictions.npy", pred)

    # JSON artefacts that the report reads.
    (modeling_dir / "evaluation_report.json").write_text(json.dumps({
        "evaluation_regime": {
            "regime": "intra_athlete",
            "n_athletes": 1,
            "disclaimer": "Held-out splits within a single athlete.",
        },
        "cv_method": "kfold",
        "n_splits": 5,
        "alignment_integrity": {
            "mean_abs_cum_catch_err_s": 0.05,
            "max_abs_cum_catch_err_s": 0.20,
        },
        "target_representation": "pca",
        "model_results": [],
    }))

    (modeling_dir / "stage0_metadata_baseline.json").write_text(json.dumps({
        "overall_metrics": {
            "rmse_median": 18.0,
            "peak_force_err_median": 30.0,
            "impulse_err_median": 25.0,
            "correlation_median": 0.92,
        },
        "fold_metrics": [],
        "feature_cols": [],
    }))

    (modeling_dir / "stageA_results.json").write_text(json.dumps({
        "feature_cols": ["a", "b"],
        "models": {
            "ridge": {
                "overall_metrics": {
                    "rmse_median": 16.0,
                    "peak_force_err_median": 28.0,
                    "impulse_err_median": 22.0,
                    "correlation_median": 0.94,
                },
                "fold_metrics": [],
                "feature_importance": [],
                "gate_passed": "PASS",
            },
        },
    }))

    if include_stage_b:
        (modeling_dir / "stageB_results.json").write_text(json.dumps({
            "overall_metrics": {
                "rmse_median": 12.0,
                "peak_force_err_median": 20.0,
                "impulse_err_median": 18.0,
                "correlation_median": 0.97,
            },
            "fold_metrics": [],
            "architecture": {"name": "tcn", "type": "tcn"},
            "gate_passed": "PASS",
        }))

    dataset_dir: Path | None = None
    if include_dataset:
        dataset_dir = tmp_path / "training_dataset"
        dataset_dir.mkdir()
        np.save(dataset_dir / "force_curves_resampled.npy", force_curves)
        np.save(dataset_dir / "s_grid.npy", s_grid)
        athletes = ["alice"] * (n_strokes // 2) + ["bob"] * (n_strokes - n_strokes // 2)
        pd.DataFrame({
            "athlete_id": athletes,
            "session_id": ["s1"] * n_strokes,
            "qc_excluded": [False] * n_strokes,
        }).to_csv(dataset_dir / "strokes.csv", index=False)
        (dataset_dir / "dataset_summary.json").write_text(json.dumps({
            "n_strokes_before_qc": n_strokes,
            "n_strokes_after_qc": n_strokes,
            "qc_mode": "soft",
            "force_col": "force_raw",
            "n_grid": n_grid,
            "n_pca_components": 8,
            "n_athletes": 2,
            "runs_included": ["alpha", "beta"],
        }))

    return modeling_dir, dataset_dir


def test_generate_training_report_full(tmp_path: Path) -> None:
    modeling_dir, dataset_dir = _make_modeling_dir(tmp_path)
    out = generate_training_report(modeling_dir, dataset_dir=dataset_dir)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Training report" in html
    for anchor in ("overview", "regime", "stages", "cohorts", "provenance"):
        assert f'id="{anchor}"' in html

    plots_dir = out.parent / "plots"
    assert (plots_dir / "metrics_bar.png").exists()
    assert (plots_dir / "cohort_athletes.png").exists()
    # Each stage produces residual + true-vs-pred plots.
    residual_pngs = list(plots_dir.glob("residual_*.png"))
    overlay_pngs = list(plots_dir.glob("true_vs_pred_*.png"))
    assert len(residual_pngs) >= 1
    assert len(overlay_pngs) >= 1


def test_generate_training_report_auto_detects_dataset(tmp_path: Path) -> None:
    modeling_dir, dataset_dir = _make_modeling_dir(tmp_path)
    assert dataset_dir is not None
    out = generate_training_report(modeling_dir)  # no dataset_dir kwarg
    html = out.read_text(encoding="utf-8")
    assert "alpha" in html or "beta" in html  # runs_included rendered


def test_generate_training_report_without_dataset(tmp_path: Path) -> None:
    modeling_dir, _ = _make_modeling_dir(tmp_path, include_dataset=False)
    out = generate_training_report(modeling_dir)
    html = out.read_text(encoding="utf-8")
    assert "Training report" in html
    assert "No per-athlete RMSE data." in html


def test_generate_training_report_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_training_report(tmp_path / "does-not-exist")


def test_generate_training_report_no_stage_b(tmp_path: Path) -> None:
    modeling_dir, dataset_dir = _make_modeling_dir(tmp_path, include_stage_b=False)
    out = generate_training_report(modeling_dir, dataset_dir=dataset_dir)
    html = out.read_text(encoding="utf-8")
    # Stage 0 + A still rendered, Stage B section absent.
    assert "Stage 0 — Metadata baseline" in html
    assert "Stage A — Kinematic baselines" in html
    assert "Stage B" not in html
