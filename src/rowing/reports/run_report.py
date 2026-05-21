"""Per-run HTML report generator.

Reads everything from the run's ``inference/`` directory and writes a
self-contained ``inference/report/index.html`` plus a ``plots/`` subfolder of
PNGs. Designed to be safe when individual artefacts are missing — each
section degrades to a placeholder rather than crashing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.selectors import (
    discover_run_dirs,
    pick_run_with_curses,
    pick_run_with_prompt,
)
from rowing.reports import plots as report_plots


__all__ = ["generate_run_report", "main"]


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _resolve_dataset_dir(run_dir: Path, summary: dict[str, Any] | None) -> Path | None:
    """Locate the per-run training dataset directory if present."""
    if summary:
        rp3_outputs = summary.get("outputs") or {}
        candidate = rp3_outputs.get("training_dataset_dir")
        if isinstance(candidate, str):
            p = Path(candidate)
            if p.exists():
                return p
    fallback = run_dir / "inference" / "training_dataset"
    return fallback if fallback.exists() else None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _detection_context(detection_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if detection_summary is None:
        return None
    params = detection_summary.get("parameters") or {}
    return {
        "frame_count": int(detection_summary.get("frame_count", 0)),
        "fps_estimate": float(detection_summary.get("fps_estimate", 0.0)),
        "catch_candidates_raw": int(detection_summary.get("catch_candidates_raw", 0)),
        "catches_filtered": int(detection_summary.get("catches_filtered", 0)),
        "complete_drives": int(detection_summary.get("complete_drives", 0)),
        "finish_method": params.get("finish_method", "—"),
        "catch_velocity_frac": float(params.get("catch_velocity_frac", 0.0)),
        "finish_velocity_frac": float(params.get("finish_velocity_frac", 0.0)),
        "calibration": params.get("calibration"),
    }


def _match_context(match_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if match_summary is None:
        return None
    return {
        "anchor_video_stroke_idx": match_summary.get("anchor_video_stroke_idx"),
        "anchor_rp3_stroke_number": match_summary.get("anchor_rp3_stroke_number"),
        "anchor_rp3_row_idx": match_summary.get("anchor_rp3_row_idx"),
        "active_side": match_summary.get("active_side"),
        "matched_video_strokes": int(match_summary.get("matched_video_strokes", 0)),
        "total_skipped_rp3_rows": int(match_summary.get("total_skipped_rp3_rows", 0)),
        "total_score": float(match_summary.get("total_score", 0.0)),
        "mean_abs_cum_catch_err_s": float(match_summary.get("mean_abs_cum_catch_err_s", 0.0)),
        "mean_abs_interval_err_s": float(match_summary.get("mean_abs_interval_err_s", 0.0)),
        "mean_abs_drive_err_s": float(match_summary.get("mean_abs_drive_err_s", 0.0)),
        "mean_abs_recover_err_s": float(match_summary.get("mean_abs_recover_err_s", 0.0)),
    }


def _segments_context(
    match_summary: dict[str, Any] | None,
    status_df: pd.DataFrame,
) -> dict[str, Any] | None:
    if match_summary is None and status_df.empty:
        return None
    drop_reasons: dict[str, int] = {}
    if match_summary:
        drop_reasons = dict(match_summary.get("segment_drop_reason_counts") or {})
    if not drop_reasons and not status_df.empty and "drop_reason" in status_df.columns:
        failed = status_df[~status_df["segment_exported"].astype(bool)]
        if not failed.empty:
            counts = failed["drop_reason"].astype(str).value_counts()
            drop_reasons = {str(k): int(v) for k, v in counts.items()}

    if match_summary:
        exported = int(match_summary.get("segment_exported_strokes", 0))
        dropped = int(match_summary.get("segment_dropped_strokes", 0))
        segment_rows = int(match_summary.get("segment_rows", 0))
        total = exported + dropped
    else:
        if status_df.empty:
            exported = dropped = segment_rows = total = 0
        else:
            exported = int(status_df["segment_exported"].astype(bool).sum())
            total = len(status_df)
            dropped = total - exported
            segment_rows = int(status_df.get("segment_rows_written", pd.Series(dtype=int)).sum())
    return {
        "exported": exported,
        "dropped": dropped,
        "total": total,
        "segment_rows": segment_rows,
        "drop_reason_counts": drop_reasons,
    }


def _dataset_context(
    dataset_dir: Path | None,
    dataset_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if dataset_summary is None:
        return None
    return {
        "dir": str(dataset_dir) if dataset_dir else "—",
        "n_strokes_before_qc": int(dataset_summary.get("n_strokes_before_qc", 0)),
        "n_strokes_after_qc": int(dataset_summary.get("n_strokes_after_qc", 0)),
        "qc_mode": dataset_summary.get("qc_mode", "—"),
        "n_pca_components": int(dataset_summary.get("n_pca_components", 0)),
        "pca_total_explained_variance": float(dataset_summary.get("pca_total_explained_variance", 0.0)),
        "n_athletes": int(dataset_summary.get("n_athletes", 0)),
        "runs_included": list(dataset_summary.get("runs_included", []) or []),
    }


def _overview_cards(
    detection: dict[str, Any] | None,
    match: dict[str, Any] | None,
    segments: dict[str, Any] | None,
    dataset: dict[str, Any] | None,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    if detection:
        cards.append({"label": "Drives detected", "value": str(detection["complete_drives"])})
        cards.append({"label": "FPS", "value": f"{detection['fps_estimate']:.1f}"})
    if match:
        cards.append({"label": "Matched strokes", "value": str(match["matched_video_strokes"])})
        cards.append({
            "label": "Mean |cum err|",
            "value": f"{match['mean_abs_cum_catch_err_s']:.3f}s",
        })
    if segments:
        cards.append({
            "label": "Segments exported",
            "value": f"{segments['exported']}/{segments['total']}",
        })
    if dataset:
        cards.append({"label": "Dataset strokes (after QC)", "value": str(dataset["n_strokes_after_qc"])})
        cards.append({
            "label": "PCA cum. variance",
            "value": f"{dataset['pca_total_explained_variance']:.2f}",
        })
    return cards


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_run_report(run_dir: Path) -> Path:
    """Build the per-run HTML report under ``<run>/inference/report/``.

    Returns the path to the generated ``index.html``.
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    inf_dir = run_dir / "inference"
    if not inf_dir.is_dir():
        raise FileNotFoundError(f"No inference directory under: {run_dir}")

    report_dir = inf_dir / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    detection_summary = _read_json(inf_dir / "drive_events_summary.json")
    match_summary = _read_json(inf_dir / "rp3_match_summary.json")
    events_df = _read_csv(inf_dir / "drive_events.csv")
    manifest_df = _read_csv(inf_dir / "rp3_match_manifest.csv")
    segments_df = _read_csv(inf_dir / "rp3_pose_force_matched_segments.csv")
    status_df = _read_csv(inf_dir / "rp3_pose_force_export_status.csv")

    dataset_dir = _resolve_dataset_dir(run_dir, match_summary)
    dataset_summary = _read_json(dataset_dir / "dataset_summary.json") if dataset_dir else None
    pca_ev_df = _read_csv(dataset_dir / "pca_explained_variance.csv") if dataset_dir else pd.DataFrame()

    plots: dict[str, str] = {}
    if not events_df.empty:
        report_plots.plot_drive_durations(events_df, plots_dir / "drive_durations.png")
        plots["drive_durations"] = "plots/drive_durations.png"

    if not manifest_df.empty:
        report_plots.plot_match_drift(manifest_df, plots_dir / "match_drift.png")
        plots["match_drift"] = "plots/match_drift.png"

        report_plots.plot_match_pair_table_image(manifest_df, plots_dir / "match_pair_table.png")
        plots["match_pair_table"] = "plots/match_pair_table.png"

        if not segments_df.empty:
            report_plots.plot_force_grid(
                segments_df, manifest_df, plots_dir / "force_grid.png",
            )
            plots["force_grid"] = "plots/force_grid.png"

    if not status_df.empty:
        report_plots.plot_qc_drop_reasons(status_df, plots_dir / "qc_drop_reasons.png")
        plots["qc_drop_reasons"] = "plots/qc_drop_reasons.png"

    if not pca_ev_df.empty:
        report_plots.plot_pca_explained_variance(pca_ev_df, plots_dir / "pca_explained_variance.png")
        plots["pca_variance"] = "plots/pca_explained_variance.png"

    detection_ctx = _detection_context(detection_summary)
    match_ctx = _match_context(match_summary)
    segments_ctx = _segments_context(match_summary, status_df)
    dataset_ctx = _dataset_context(dataset_dir, dataset_summary)
    overview = _overview_cards(detection_ctx, match_ctx, segments_ctx, dataset_ctx)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("run_report.html.j2")
    rendered = template.render(
        run_name=run_dir.name,
        run_dir=str(run_dir),
        generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        overview=overview,
        detection=detection_ctx,
        match=match_ctx,
        segments=segments_ctx,
        dataset=dataset_ctx,
        plots=plots,
    )

    out_path = report_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-run HTML inference report.",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help=f"Sports2D runs root (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory (skip interactive selection).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the generated report in the default browser.",
    )
    return parser.parse_args()


def _select_run(runs_root: Path) -> Path:
    options = [
        r for r in discover_run_dirs(runs_root)
        if (r / "inference").is_dir()
    ]
    if not options:
        raise FileNotFoundError(f"No runs with inference/ found under {runs_root}")
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return pick_run_with_curses(options)
        except Exception:
            pass
    return pick_run_with_prompt(options)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args() if argv is None else _parse_args()  # argparse reads sys.argv
    if args.run_dir is None:
        run_dir = _select_run(args.runs_root)
    else:
        run_dir = args.run_dir.expanduser().resolve()

    out = generate_run_report(run_dir)
    print(f"Report → {out}")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
