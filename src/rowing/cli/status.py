"""Compute per-run status badges for the Rich TUI selectors.

Reads only the on-disk artifacts under each run directory; never invokes the
heavy pipeline. Cheap enough to be called every time the user opens the run
picker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunStatus:
    """Snapshot of which pipeline stages have completed for a single run."""

    run_dir: Path
    pose: bool = False
    stroke: bool = False
    rp3_dirty: bool = False
    rp3_clean: bool = False
    drive_events: bool = False
    match: bool = False
    segments: bool = False
    dataset: bool = False
    report: bool = False
    match_qc_score: float | None = None
    match_qc_label: str | None = None
    matched_strokes: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.run_dir.name


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _label_for(score: float) -> str:
    if score < 0.05:
        return "excellent"
    if score < 0.15:
        return "good"
    if score < 0.35:
        return "acceptable"
    return "poor"


def compute_status(run_dir: Path) -> RunStatus:
    """Compute on-disk status flags for a single run directory."""
    pose = (run_dir / "motionbert" / "angles_h36m.csv").exists()
    stroke = (run_dir / "stroke" / "stroke_signal.csv").exists()

    rp3_dir = run_dir / "rp3"
    rp3_dirty = False
    rp3_clean = False
    if rp3_dir.exists() and rp3_dir.is_dir():
        for p in rp3_dir.glob("*.csv"):
            if p.name.startswith("."):
                continue
            if p.name.lower().endswith("-clean.csv"):
                rp3_clean = True
            else:
                rp3_dirty = True

    inference_dir = run_dir / "inference"
    drive_events = (inference_dir / "drive_events.csv").exists()
    match = (inference_dir / "rp3_match_manifest.csv").exists()
    segments = (inference_dir / "rp3_pose_force_matched_segments.csv").exists()
    dataset = (inference_dir / "training_dataset").is_dir()
    report = (inference_dir / "report" / "index.html").exists()

    match_qc_score: float | None = None
    match_qc_label: str | None = None
    matched_strokes: int | None = None
    summary = _read_json(inference_dir / "rp3_match_summary.json")
    if summary is not None:
        try:
            match_qc_score = float(summary.get("mean_abs_cum_catch_err_s"))
            match_qc_label = _label_for(match_qc_score)
        except (TypeError, ValueError):
            match_qc_score = None
        try:
            matched_strokes = int(summary.get("matched_video_strokes"))
        except (TypeError, ValueError):
            matched_strokes = None

    return RunStatus(
        run_dir=run_dir,
        pose=pose,
        stroke=stroke,
        rp3_dirty=rp3_dirty,
        rp3_clean=rp3_clean,
        drive_events=drive_events,
        match=match,
        segments=segments,
        dataset=dataset,
        report=report,
        match_qc_score=match_qc_score,
        match_qc_label=match_qc_label,
        matched_strokes=matched_strokes,
    )


def discover_run_statuses(runs_root: Path) -> list[RunStatus]:
    """Return a list of :class:`RunStatus` objects for every run under *runs_root*.

    Sorted newest-first by mtime to match the legacy curses picker.
    """
    if not runs_root.exists() or not runs_root.is_dir():
        return []
    candidates = [
        p for p in runs_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [compute_status(p) for p in candidates]
