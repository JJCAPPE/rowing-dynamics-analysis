"""Tests for :mod:`rowing.reports.run_report` (Phase 5 per-run HTML report)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ensure CI never opens a GUI window

import numpy as np
import pandas as pd
import pytest

from rowing.reports.run_report import generate_run_report


# ---------------------------------------------------------------------------
# Synthetic run-dir factory
# ---------------------------------------------------------------------------


def _make_run_dir(
    tmp_path: Path,
    *,
    n_strokes: int = 6,
    include_segments: bool = True,
    include_status: bool = True,
    include_dataset: bool = False,
) -> Path:
    run_dir = tmp_path / "run-fake_20990101_000000"
    inf = run_dir / "inference"
    inf.mkdir(parents=True)

    # drive_events_summary.json
    (inf / "drive_events_summary.json").write_text(json.dumps({
        "frame_count": 1200,
        "fps_estimate": 120.0,
        "catch_candidates_raw": n_strokes + 2,
        "catches_filtered": n_strokes + 1,
        "complete_drives": n_strokes,
        "active_side": "left",
        "parameters": {
            "finish_method": "velocity_calibrated",
            "catch_velocity_frac": 0.43,
            "finish_velocity_frac": 0.74,
            "calibration": {
                "source": "rp3_optimized",
                "mae_ms": 12.0,
                "me_ms": -1.0,
                "std_ms": 8.0,
                "n_strokes": n_strokes,
            },
        },
    }))

    # drive_events.csv
    catch = np.cumsum(np.full(n_strokes, 2.0))
    drive = np.full(n_strokes, 0.7)
    pd.DataFrame({
        "stroke_idx": np.arange(n_strokes),
        "catch_time_s": catch,
        "finish_time_s": catch + drive,
        "drive_duration_s": drive,
        "recover_duration_s": np.full(n_strokes, 1.3),
        "cycle_duration_s": np.full(n_strokes, 2.0),
    }).to_csv(inf / "drive_events.csv", index=False)

    # rp3_match_manifest.csv
    manifest = pd.DataFrame({
        "video_stroke_idx": np.arange(n_strokes, dtype=int),
        "rp3_stroke_number": np.arange(1, n_strokes + 1, dtype=int),
        "rp3_row_idx": np.arange(n_strokes, dtype=int),
        "rp3_rows_skipped_since_prev": np.zeros(n_strokes, dtype=int),
        "video_drive_s": drive,
        "rp3_drive_s": drive + 0.01,
        "drive_err_s": np.full(n_strokes, 0.01),
        "interval_err_s": np.zeros(n_strokes),
        "cum_catch_err_s": np.linspace(0, 0.05, n_strokes),
    })
    manifest.to_csv(inf / "rp3_match_manifest.csv", index=False)

    # rp3_match_summary.json
    (inf / "rp3_match_summary.json").write_text(json.dumps({
        "anchor_video_stroke_idx": 0,
        "anchor_rp3_stroke_number": 1,
        "anchor_rp3_row_idx": 0,
        "active_side": "left",
        "matched_video_strokes": n_strokes,
        "total_skipped_rp3_rows": 0,
        "total_score": 1.234,
        "mean_abs_cum_catch_err_s": 0.025,
        "mean_abs_interval_err_s": 0.0,
        "mean_abs_drive_err_s": 0.01,
        "mean_abs_recover_err_s": 0.02,
        "segment_rows": 60 * n_strokes,
        "segment_exported_strokes": n_strokes if include_segments else 0,
        "segment_dropped_strokes": 0 if include_segments else n_strokes,
        "segment_drop_reason_counts": {} if include_segments else {"missing_force": n_strokes},
        "outputs": {},
    }))

    # rp3_pose_force_export_status.csv
    if include_status:
        pd.DataFrame({
            "match_seq_idx": np.arange(n_strokes),
            "video_stroke_idx": np.arange(n_strokes),
            "rp3_row_idx": np.arange(n_strokes),
            "rp3_stroke_number": np.arange(1, n_strokes + 1),
            "segment_exported": [include_segments] * n_strokes,
            "segment_rows_written": np.full(n_strokes, 60 if include_segments else 0),
            "drop_reason": ["" if include_segments else "missing_force"] * n_strokes,
        }).to_csv(inf / "rp3_pose_force_export_status.csv", index=False)

    # rp3_pose_force_matched_segments.csv
    if include_segments:
        rows = []
        for v in range(n_strokes):
            s = np.linspace(0, 1, 60)
            rows.append(pd.DataFrame({
                "video_stroke_idx": np.full(60, v, dtype=int),
                "s_force": s,
                "force_raw": 500 * np.sin(np.pi * s),
            }))
        pd.concat(rows).to_csv(inf / "rp3_pose_force_matched_segments.csv", index=False)

    # Optional dataset
    if include_dataset:
        ds_dir = inf / "training_dataset"
        ds_dir.mkdir()
        (ds_dir / "dataset_summary.json").write_text(json.dumps({
            "n_strokes_before_qc": n_strokes,
            "n_strokes_after_qc": n_strokes,
            "qc_mode": "soft",
            "n_pca_components": 5,
            "pca_total_explained_variance": 0.95,
            "n_athletes": 1,
            "runs_included": [run_dir.name],
        }))
        pd.DataFrame({
            "component": np.arange(1, 6),
            "explained_variance_ratio": [0.5, 0.2, 0.1, 0.1, 0.05],
            "cumulative_explained_variance": np.cumsum([0.5, 0.2, 0.1, 0.1, 0.05]),
        }).to_csv(ds_dir / "pca_explained_variance.csv", index=False)

    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_run_report_full(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, include_dataset=True)
    out = generate_run_report(run_dir)
    assert out.exists()
    assert out.name == "index.html"
    html = out.read_text(encoding="utf-8")
    assert "Run report" in html
    assert run_dir.name in html
    # Section anchors all present
    for anchor in ("overview", "detection", "match", "segments", "dataset"):
        assert f'id="{anchor}"' in html
    # Plots written
    plots_dir = out.parent / "plots"
    assert (plots_dir / "drive_durations.png").exists()
    assert (plots_dir / "match_drift.png").exists()
    assert (plots_dir / "match_pair_table.png").exists()
    assert (plots_dir / "force_grid.png").exists()
    assert (plots_dir / "qc_drop_reasons.png").exists()
    assert (plots_dir / "pca_explained_variance.png").exists()


def test_generate_run_report_minimal_no_segments(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, include_segments=False)
    out = generate_run_report(run_dir)
    html = out.read_text(encoding="utf-8")
    assert "Run report" in html
    # Without dataset, the dataset section degrades to a placeholder.
    assert "No training-dataset summary on disk." in html
    plots_dir = out.parent / "plots"
    assert (plots_dir / "drive_durations.png").exists()
    assert (plots_dir / "qc_drop_reasons.png").exists()
    # No force grid because segments were not written.
    assert not (plots_dir / "force_grid.png").exists()


def test_generate_run_report_missing_inference_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        generate_run_report(run_dir)


def test_generate_run_report_ignores_malformed_summary(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, include_dataset=False)
    # Corrupt drive_events_summary.json — section should degrade gracefully.
    (run_dir / "inference" / "drive_events_summary.json").write_text("{not valid json")
    out = generate_run_report(run_dir)
    html = out.read_text(encoding="utf-8")
    assert "No detection summary on disk." in html
