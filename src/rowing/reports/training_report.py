"""Per-training HTML report generator (Phase 6).

Reads the artefacts left by ``rowing.modeling.train`` (typically
``modeling_results/`` or ``runs/_training/<ts>/``) and produces a
self-contained ``report/index.html`` next to them.

Sections:
    - Overview cards (best RMSE per stage, athletes, regime)
    - Evaluation regime + alignment integrity (from ``evaluation_report.json``)
    - Per-stage metrics tables (Stage 0 / A / B), residual histograms, true-vs-pred overlays
    - Cohort breakdown (per-athlete held-out RMSE)
    - Provenance (dataset hash, code SHA, runs included)

The generator degrades gracefully when individual artefacts are missing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from rowing import REPO_ROOT
from rowing.reports import plots as report_plots


__all__ = ["generate_training_report", "main"]


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
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_json_any(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except (OSError, ValueError):
        return None


def _safe_load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _git_sha() -> tuple[str | None, bool]:
    """Return (short SHA, dirty?) for the repo containing this code."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    try:
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = False
    return sha or None, dirty


# ---------------------------------------------------------------------------
# Stage extraction
# ---------------------------------------------------------------------------


_STAGE_FILES = {
    "0": ("Stage 0 — Metadata baseline", "stage0_metadata_baseline.json", "stage0_metadata_predictions.npy"),
    "A": ("Stage A — Kinematic baselines", "stageA_results.json", None),  # Stage A: per-model preds
    "B": ("Stage B — Sequence models", "stageB_results.json", None),
}


def _stage0_metrics(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    overall = payload.get("overall_metrics") or {}
    metrics = dict(overall)
    metrics.setdefault("model_name", "metadata_baseline")
    return [metrics]


def _stageA_metrics(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    out: list[dict[str, Any]] = []
    models = payload.get("models") or {}
    for name, res in models.items():
        m = dict(res.get("overall_metrics") or {})
        m["model_name"] = name
        m["gate_passed"] = res.get("gate_passed")
        out.append(m)
    return out


def _stageB_metrics(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    overall = payload.get("overall_metrics") or {}
    metrics = dict(overall)
    arch = payload.get("architecture") or {}
    metrics.setdefault("model_name", arch.get("name") or arch.get("type") or "stageB")
    metrics["gate_passed"] = payload.get("gate_passed")
    return [metrics]


# ---------------------------------------------------------------------------
# Residual / overlay computation
# ---------------------------------------------------------------------------


def _per_stroke_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        return np.array([])
    diff = y_true - y_pred
    return np.sqrt(np.nanmean(diff ** 2, axis=1))


def _per_athlete_mean_rmse(
    rmse: np.ndarray, athletes: pd.Series | None,
) -> dict[str, float]:
    if rmse.size == 0 or athletes is None:
        return {}
    out: dict[str, float] = {}
    for athlete in pd.unique(athletes.dropna()):
        mask = (athletes == athlete).to_numpy()
        if mask.size != rmse.size:
            continue
        vals = rmse[mask]
        finite = vals[np.isfinite(vals)]
        if finite.size:
            out[str(athlete)] = float(np.mean(finite))
    return out


def _detect_leakage_warnings(strokes_df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    n_athletes = (
        int(strokes_df["athlete_id"].nunique()) if "athlete_id" in strokes_df.columns else 0
    )
    n_sessions = (
        int(strokes_df["session_id"].nunique()) if "session_id" in strokes_df.columns else 0
    )
    if n_athletes <= 1:
        warnings.append(
            f"Only {n_athletes} unique athlete(s) — within-athlete CV cannot test cross-athlete generalisation."
        )
    if n_sessions <= 1:
        warnings.append(
            f"Only {n_sessions} unique session(s) — within-session CV may overfit per-session conditions."
        )
    return warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_training_report(
    modeling_dir: Path,
    *,
    dataset_dir: Path | None = None,
) -> Path:
    """Generate the training report under ``<modeling_dir>/report/``.

    Returns the path to the generated ``index.html``.
    """
    modeling_dir = modeling_dir.expanduser().resolve()
    if not modeling_dir.is_dir():
        raise FileNotFoundError(f"Modeling results directory not found: {modeling_dir}")

    if dataset_dir is None:
        dataset_dir = _infer_dataset_dir(modeling_dir)
    elif dataset_dir is not None:
        dataset_dir = dataset_dir.expanduser().resolve()
        if not dataset_dir.is_dir():
            print(f"Warning: dataset_dir not found: {dataset_dir}; degrading report.")
            dataset_dir = None

    report_dir = modeling_dir / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    eval_report = _read_json(modeling_dir / "evaluation_report.json") or {}
    stage0 = _read_json(modeling_dir / "stage0_metadata_baseline.json")
    stageA = _read_json(modeling_dir / "stageA_results.json")
    stageB = _read_json(modeling_dir / "stageB_results.json")

    stages_metrics = {
        "0": _stage0_metrics(stage0),
        "A": _stageA_metrics(stageA),
        "B": _stageB_metrics(stageB),
    }

    # Aggregate metrics across stages for the comparison bar chart.
    flat_metrics: list[dict[str, Any]] = []
    for stage_id, items in stages_metrics.items():
        for m in items:
            entry = dict(m)
            entry["model_name"] = f"{stage_id}: {m.get('model_name', '?')}"
            flat_metrics.append(entry)

    plots: dict[str, str] = {}
    if flat_metrics:
        report_plots.plot_metric_bar_group(
            flat_metrics,
            plots_dir / "metrics_bar.png",
            metric_keys=("rmse_median", "peak_force_err_median", "correlation_median"),
            title="Stage / model metric comparison",
        )
        plots["metrics_bar"] = "plots/metrics_bar.png"

    # Dataset / cohort context
    dataset_summary: dict[str, Any] | None = None
    strokes_df = pd.DataFrame()
    force_curves: np.ndarray | None = None
    s_grid: np.ndarray | None = None
    if dataset_dir is not None:
        dataset_summary = _read_json(dataset_dir / "dataset_summary.json")
        strokes_df = _safe_load_csv(dataset_dir / "strokes.csv")
        force_curves = _safe_load_npy(dataset_dir / "force_curves_resampled.npy")
        s_grid = _safe_load_npy(dataset_dir / "s_grid.npy")

    # Per-stage panels: residuals + true-vs-pred overlays, when predictions exist.
    cv_pred_dir = modeling_dir / "cv_predictions"
    stage_blocks: list[dict[str, Any]] = []
    cohort_athletes: dict[str, float] = {}

    def _build_stage_block(
        title: str, metrics: list[dict[str, Any]], pred_files: list[Path],
    ) -> dict[str, Any]:
        block: dict[str, Any] = {"title": title, "metrics": metrics}
        # Pick the best preds (lowest rmse_median) for residual + overlay panels.
        if not pred_files or force_curves is None or s_grid is None:
            return block
        best_pred = pred_files[0]
        if len(pred_files) > 1:
            best_idx = int(np.argmin([
                float(m.get("rmse_median", np.inf)) if isinstance(m.get("rmse_median"), (int, float)) else np.inf
                for m in metrics
            ])) if metrics else 0
            if 0 <= best_idx < len(pred_files):
                best_pred = pred_files[best_idx]
        preds = _safe_load_npy(best_pred)
        if preds is None or preds.shape != force_curves.shape:
            return block

        rmse = _per_stroke_rmse(force_curves, preds)
        finite = rmse[np.isfinite(rmse)]
        if finite.size:
            resid_plot = plots_dir / f"residual_{best_pred.stem}.png"
            report_plots.plot_residual_histogram(
                finite, resid_plot,
                title=f"{title} — per-stroke RMSE",
            )
            block["residual_plot"] = f"plots/{resid_plot.name}"

        overlay_plot = plots_dir / f"true_vs_pred_{best_pred.stem}.png"
        report_plots.plot_true_vs_pred_overlay(
            s_grid, force_curves, preds, overlay_plot,
            title=f"{title} — true vs predicted (sample)",
        )
        block["true_vs_pred_plot"] = f"plots/{overlay_plot.name}"

        nonlocal cohort_athletes
        if not strokes_df.empty and "athlete_id" in strokes_df.columns and not cohort_athletes:
            cohort_athletes = _per_athlete_mean_rmse(rmse, strokes_df["athlete_id"])
        return block

    if stage0 and stages_metrics["0"]:
        stage_blocks.append(_build_stage_block(
            "Stage 0 — Metadata baseline",
            stages_metrics["0"],
            sorted(cv_pred_dir.glob("stage0_*_predictions.npy")),
        ))
    if stageA and stages_metrics["A"]:
        stage_blocks.append(_build_stage_block(
            "Stage A — Kinematic baselines",
            stages_metrics["A"],
            sorted(cv_pred_dir.glob("stageA_*_predictions.npy")),
        ))
    if stageB and stages_metrics["B"]:
        stage_blocks.append(_build_stage_block(
            "Stage B — Sequence models",
            stages_metrics["B"],
            sorted(cv_pred_dir.glob("stageB_*_predictions.npy")),
        ))

    if cohort_athletes:
        report_plots.plot_cohort_metric_bars(
            cohort_athletes,
            plots_dir / "cohort_athletes.png",
            title="Held-out RMSE per athlete",
            metric_label="RMSE",
        )
        plots["cohort_athletes"] = "plots/cohort_athletes.png"

    cohort_warnings = _detect_leakage_warnings(strokes_df) if not strokes_df.empty else []

    # Regime block
    regime: dict[str, Any] | None = None
    if eval_report:
        regime_info = eval_report.get("evaluation_regime") or {}
        regime = {
            "regime": regime_info.get("regime", "?"),
            "n_athletes": regime_info.get("n_athletes", "?"),
            "disclaimer": regime_info.get("disclaimer", ""),
            "cv_method": eval_report.get("cv_method", "?"),
            "n_splits": eval_report.get("n_splits", "?"),
            "target_representation": eval_report.get("target_representation", "?"),
            "alignment_metrics": eval_report.get("alignment_integrity") or {},
        }

    # Provenance
    sha, dirty = _git_sha()
    provenance: dict[str, Any] = {
        "code_sha": sha,
        "code_dirty": dirty,
        "dataset_summary": dataset_summary,
        "dataset_summary_path": (
            str(dataset_dir / "dataset_summary.json") if dataset_dir else None
        ),
        "runs_included": (dataset_summary or {}).get("runs_included") or [],
    }

    # Overview cards
    overview = _overview_cards(stages_metrics, dataset_summary, regime)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("training_report.html.j2")
    rendered = template.render(
        training_name=modeling_dir.name,
        modeling_dir=str(modeling_dir),
        dataset_dir=str(dataset_dir) if dataset_dir else None,
        generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        overview=overview,
        regime=regime,
        stages=stage_blocks,
        plots=plots,
        cohort_warnings=cohort_warnings,
        provenance=provenance,
    )

    out_path = report_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def _infer_dataset_dir(modeling_dir: Path) -> Path | None:
    """Best-effort: look for a sibling ``training_dataset*`` directory."""
    parent = modeling_dir.parent
    for candidate in parent.iterdir():
        if not candidate.is_dir():
            continue
        if not candidate.name.startswith("training_dataset"):
            continue
        if (candidate / "dataset_summary.json").exists():
            return candidate.resolve()
    # Also accept the modeling_dir's own training_dataset/
    nested = modeling_dir / "training_dataset"
    if nested.exists() and (nested / "dataset_summary.json").exists():
        return nested.resolve()
    return None


def _overview_cards(
    stages_metrics: dict[str, list[dict[str, Any]]],
    dataset_summary: dict[str, Any] | None,
    regime: dict[str, Any] | None,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for stage_id, items in stages_metrics.items():
        if not items:
            continue
        rmses = [
            float(m.get("rmse_median")) for m in items
            if isinstance(m.get("rmse_median"), (int, float))
        ]
        if rmses:
            cards.append({
                "label": f"Stage {stage_id} best RMSE",
                "value": f"{min(rmses):.4f}",
            })
    if dataset_summary:
        cards.append({
            "label": "Dataset strokes (after QC)",
            "value": str(dataset_summary.get("n_strokes_after_qc", "—")),
        })
        cards.append({
            "label": "Athletes / runs",
            "value": (
                f"{dataset_summary.get('n_athletes', '?')} / "
                f"{len(dataset_summary.get('runs_included') or [])}"
            ),
        })
    if regime:
        cards.append({
            "label": "Regime",
            "value": str(regime.get("regime", "?")),
        })
    return cards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-training HTML report.",
    )
    parser.add_argument(
        "--modeling-dir", type=Path, required=True,
        help="Modeling results directory (where evaluation_report.json lives).",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Dataset directory (auto-detected from sibling 'training_dataset*' if omitted).",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the generated report in the default browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = generate_training_report(
        args.modeling_dir,
        dataset_dir=args.dataset_dir,
    )
    print(f"Training report → {out}")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
