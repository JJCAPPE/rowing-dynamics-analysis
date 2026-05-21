"""Inference pipeline orchestrator.

The Phase 1 refactor pulled the body of the legacy ``inference_cli.main()``
function into a typed orchestrator that the new TUI menu can drive directly,
without going through ``argparse``. The public entry point is
:func:`run_inference`; ``rowing.cli.inference`` retains a thin argparse
wrapper around it.

Behaviour and on-disk artifacts are preserved exactly. The pipeline can be
invoked with either an :class:`argparse.Namespace` (legacy) or a
:class:`PipelineOptions` dataclass (TUI / programmatic callers).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.selectors import (
    discover_run_dirs,
    pick_file_with_curses,
    pick_file_with_prompt,
    ensure_path_in_dir,
    prompt_choice,
    resolve_run_dir,
    select_yes_no,
)
from rowing.dataset.build import build_training_dataset as _build_training_dataset
from rowing.matching.detect import (
    CalibrationResult,
    DetectionResult,
    DEFAULT_CATCH_VELOCITY_FRAC,
    DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED,
    FINISH_METHOD_VELOCITY_CALIBRATED,
    FINISH_METHOD_VELOCITY_THRESHOLD,
    _infer_time_s,
    calibrate_velocity_fracs,
    detect_drive_events,
)
from rowing.matching.match import (
    MatchConfig as Rp3MatchConfig,
    _build_match_manifest as _build_rp3_match_manifest,
    _load_rp3 as _load_rp3_clean_csv,
    _resolve_anchor_rp3_idx as _resolve_rp3_anchor_idx,
)
from rowing.matching.overrides import (
    MatchOverrides,
    load_overrides,
    resolve_pin_to_row_idx,
    validate_overrides,
)
from rowing.matching.segments import (
    build_force_pose_segments,
    build_frame_level_recomputed_columns,
    events_to_dataframe,
    validate_force_segment_exports,
)


__all__ = [
    "PipelineOptions",
    "PipelineResult",
    "RP3_CLEAN_MAX_STROKE_LENGTH_CM",
    "RP3_CLEAN_STEP_CM",
    "VIDEO_SUFFIXES",
    "INPUT_VIDEO_SOURCE_PATH_FILE",
    "discover_run_rp3_dirty_csvs",
    "resolve_rp3_dirty_csv",
    "clean_rp3_dirty_csv",
    "find_input_video",
    "options_from_argparse",
    "run_inference",
    "write_drive_overlay_video",
]


RP3_CLEAN_MAX_STROKE_LENGTH_CM = 170.0
RP3_CLEAN_STEP_CM = 2.2
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
    ".mpg",
    ".mpeg",
    ".mts",
    ".m2ts",
    ".wmv",
}
INPUT_VIDEO_SOURCE_PATH_FILE = "input_video_source.txt"


# ---------------------------------------------------------------------------
# Option / result containers
# ---------------------------------------------------------------------------


@dataclass
class PipelineOptions:
    """Typed options for the inference pipeline.

    Mirrors every field of the legacy argparse CLI so that
    :func:`run_inference` is callable from both the legacy script and the
    upcoming Rich TUI.
    """

    # Inputs
    runs_root: Path = field(default_factory=lambda: DEFAULT_RUNS_ROOT)
    run_dir: Path | None = None
    output_dir: Path | None = None
    interactive: bool | None = None  # None = auto-detect from TTY

    # Drive detection
    smooth_window_s: float | None = None
    min_cycle_s: float = 0.8
    min_drive_s: float = 0.2
    min_recover_s: float = 0.2
    min_drive_disp_frac: float = 0.05
    slope_tol_frac: float = 0.05
    finish_method: str = FINISH_METHOD_VELOCITY_CALIBRATED
    finish_velocity_frac: float | None = None
    catch_velocity_frac: float | None = None

    # Overlay video
    overlay_video: bool = False
    no_overlay_video: bool = False
    overlay_opacity: float = 0.10

    # RP3 matching
    match_rp3: bool = False
    no_match_rp3: bool = False
    rp3_dirty_csv: Path | None = None
    anchor_video_stroke_idx: int = 1
    anchor_rp3_row_idx: int | None = None
    anchor_rp3_stroke_number: int | None = None
    active_side: str | None = None
    use_rp3_finish: bool = True
    no_use_rp3_finish: bool = False
    max_jump_rows: int = 10
    max_interval_error_s: float = 2.0
    max_cumulative_error_base_s: float = 1.5
    max_cumulative_error_per_s: float = 0.15
    max_abs_cum_error_s: float = 4.0
    w_drive: float = 0.4
    w_recover: float = 0.4
    w_interval: float = 1.0
    w_cumulative: float = 1.0
    w_skip: float = 0.08

    # Segment export
    rower_facing: str = "auto"
    include_second_derivatives: bool = False

    # Training dataset build (after segment export)
    no_build_dataset: bool = False
    dataset_output_dir: Path | None = None
    dataset_qc_mode: str = "soft"
    dataset_n_grid: int = 64
    dataset_n_pca_components: int = 20
    dataset_force_col: str = "force_raw"
    dataset_onset_frac: float = 0.15


@dataclass
class PipelineResult:
    """Summary of a single :func:`run_inference` execution."""

    exit_code: int
    run_dir: Path | None = None
    summary: dict[str, Any] | None = None
    rp3_summary: dict[str, Any] | None = None
    detection: DetectionResult | None = None
    calibration: CalibrationResult | None = None
    events_csv: Path | None = None
    frame_csv: Path | None = None
    summary_json: Path | None = None
    overlay_video: Path | None = None
    rp3_manifest_csv: Path | None = None
    rp3_segments_csv: Path | None = None
    rp3_segment_status_csv: Path | None = None
    rp3_summary_json: Path | None = None
    training_dataset_dir: Path | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Option conversion
# ---------------------------------------------------------------------------


def options_from_argparse(args: argparse.Namespace) -> PipelineOptions:
    """Convert an argparse :class:`Namespace` to :class:`PipelineOptions`.

    Mutually-exclusive flag pairs are validated downstream by
    :func:`run_inference`.
    """
    field_names = {f.name for f in dataclasses.fields(PipelineOptions)}
    payload = {k: getattr(args, k) for k in field_names if hasattr(args, k)}
    return PipelineOptions(**payload)


# ---------------------------------------------------------------------------
# RP3 dirty-CSV discovery + cleaning
# ---------------------------------------------------------------------------


def discover_run_rp3_dirty_csvs(run_dir: Path) -> list[Path]:
    rp3_dir = (run_dir / "rp3").resolve()
    if not rp3_dir.exists() or not rp3_dir.is_dir():
        return []
    return [
        p.resolve()
        for p in sorted(rp3_dir.glob("*.csv"))
        if p.is_file() and not p.name.startswith(".") and not p.name.lower().endswith("-clean.csv")
    ]


def resolve_rp3_dirty_csv(
    *,
    run_dir: Path,
    rp3_dirty_csv: Path | None,
    interactive: bool,
) -> Path:
    rp3_dir = (run_dir / "rp3").resolve()
    if not rp3_dir.exists() or not rp3_dir.is_dir():
        raise FileNotFoundError(f"Run RP3 directory not found: {rp3_dir}")

    if rp3_dirty_csv is not None:
        csv_path = rp3_dirty_csv.expanduser().resolve()
        if not csv_path.exists() or not csv_path.is_file():
            raise FileNotFoundError(f"RP3 dirty CSV not found: {csv_path}")
        ensure_path_in_dir(csv_path, rp3_dir, label="--rp3-dirty-csv")
        if csv_path.name.lower().endswith("-clean.csv"):
            raise ValueError(f"Expected dirty RP3 CSV, got clean CSV: {csv_path.name}")
        return csv_path

    options = discover_run_rp3_dirty_csvs(run_dir)
    if not options:
        raise FileNotFoundError(
            f"No RP3 dirty CSV files found in {rp3_dir}. Add one or run with --no-match-rp3."
        )
    if len(options) == 1:
        return options[0]

    if interactive:
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                return pick_file_with_curses(options, "Select RP3 dirty CSV")
            except Exception:
                pass
        return pick_file_with_prompt(options, "Select RP3 dirty CSV")

    raise ValueError(
        f"Multiple RP3 dirty CSV files found in {rp3_dir}. "
        "Specify one with --rp3-dirty-csv."
    )


def clean_rp3_dirty_csv(dirty_csv: Path) -> Path:
    if dirty_csv.name.lower().endswith("-clean.csv"):
        raise ValueError(f"Expected dirty RP3 CSV, got clean CSV: {dirty_csv}")

    clean_csv = dirty_csv.with_name(f"{dirty_csv.stem}-clean.csv")
    from rowing.rp3.clean import process_file

    process_file(
        input_csv=dirty_csv,
        output_csv=clean_csv,
        max_stroke_length_cm=RP3_CLEAN_MAX_STROKE_LENGTH_CM,
        step_cm=RP3_CLEAN_STEP_CM,
        drop_curve_data=False,
        truncate=False,
    )
    return clean_csv.resolve()


def find_input_video(run_dir: Path) -> Path:
    input_dir = run_dir / "input"
    if input_dir.exists() and input_dir.is_dir():
        candidates = [
            path
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        if candidates:
            return candidates[0]

    pointer_path = run_dir / INPUT_VIDEO_SOURCE_PATH_FILE
    if pointer_path.exists() and pointer_path.is_file():
        source_text = pointer_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not source_text:
            raise FileNotFoundError(f"Input video source pointer is empty: {pointer_path}")
        source_video = Path(source_text).expanduser().resolve()
        if not source_video.exists() or not source_video.is_file():
            raise FileNotFoundError(
                f"Input video source path from {pointer_path} does not exist: {source_video}"
            )
        return source_video

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input video directory not found: {input_dir}. "
            f"Expected copied input video or {INPUT_VIDEO_SOURCE_PATH_FILE}."
        )
    raise FileNotFoundError(
        f"No input video found in: {input_dir}. "
        f"Expected copied input video or {INPUT_VIDEO_SOURCE_PATH_FILE}."
    )


# ---------------------------------------------------------------------------
# Drive overlay video
# ---------------------------------------------------------------------------


def write_drive_overlay_video(
    *,
    input_video: Path,
    is_drive_flags: np.ndarray,
    out_video: Path,
    alpha: float = 0.10,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {out_video}")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    drive_color = np.asarray([80, 255, 80], dtype=np.uint8)  # BGR light green
    recover_color = np.asarray([80, 80, 255], dtype=np.uint8)  # BGR light red

    frame_idx = 0
    drive_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            is_drive = bool(is_drive_flags[frame_idx]) if frame_idx < is_drive_flags.size else False
            if is_drive:
                drive_frames += 1
            color = drive_color if is_drive else recover_color

            overlay = np.empty_like(frame, dtype=np.uint8)
            overlay[:, :] = color
            blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
            writer.write(blended)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return frame_idx, drive_frames


# ---------------------------------------------------------------------------
# Internal phases
# ---------------------------------------------------------------------------


def _resolve_active_side(
    opts: PipelineOptions,
    *,
    run_rp3_matching: bool,
    selected_run_via_selector: bool,
) -> str:
    if opts.active_side is not None:
        return str(opts.active_side)
    if run_rp3_matching and selected_run_via_selector:
        return prompt_choice(
            "Active side for unilateral features",
            options=["right", "left"],
            default="right",
        )
    return "right"


def _resolve_overlay_video(
    opts: PipelineOptions,
    *,
    selected_run_via_selector: bool,
) -> bool:
    if opts.overlay_video:
        return True
    if opts.no_overlay_video:
        return False
    if selected_run_via_selector:
        return select_yes_no("Write drive-phase overlay video?", default_no=True)
    return False


def _resolve_run_rp3_matching(opts: PipelineOptions, *, has_dirty_rp3: bool) -> bool:
    if opts.match_rp3:
        return True
    if opts.no_match_rp3:
        return False
    if (
        opts.rp3_dirty_csv is not None
        or opts.anchor_rp3_row_idx is not None
        or opts.anchor_rp3_stroke_number is not None
    ):
        return True
    return has_dirty_rp3


def _resolve_detection_params(opts: PipelineOptions) -> tuple[float, float, float, dict[str, Any]]:
    is_calibrated = opts.finish_method == FINISH_METHOD_VELOCITY_CALIBRATED
    smooth_window_s = (
        float(opts.smooth_window_s) if opts.smooth_window_s is not None
        else (0.04 if is_calibrated else 0.08)
    )
    finish_velocity_frac = (
        float(opts.finish_velocity_frac) if opts.finish_velocity_frac is not None
        else (DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED if is_calibrated else 0.85)
    )
    catch_velocity_frac = (
        float(opts.catch_velocity_frac) if opts.catch_velocity_frac is not None
        else DEFAULT_CATCH_VELOCITY_FRAC
    )

    detect_kwargs: dict[str, Any] = dict(
        smooth_window_s=smooth_window_s,
        min_cycle_s=float(opts.min_cycle_s),
        min_drive_s=float(opts.min_drive_s),
        min_recover_s=float(opts.min_recover_s),
        min_drive_disp_frac=float(opts.min_drive_disp_frac),
        slope_tol_frac=float(opts.slope_tol_frac),
        finish_velocity_frac=finish_velocity_frac,
        catch_velocity_frac=catch_velocity_frac,
    )
    return smooth_window_s, finish_velocity_frac, catch_velocity_frac, detect_kwargs


def _build_match_config(opts: PipelineOptions, *, calibrated: bool) -> Rp3MatchConfig:
    cum_err_base = float(opts.max_cumulative_error_base_s)
    if calibrated:
        cum_err_base = max(cum_err_base, 2.0)
    return Rp3MatchConfig(
        max_jump_rows=int(opts.max_jump_rows),
        max_interval_error_s=float(opts.max_interval_error_s),
        max_cumulative_error_base_s=cum_err_base,
        max_cumulative_error_per_s=float(opts.max_cumulative_error_per_s),
        max_abs_cum_error_s=float(opts.max_abs_cum_error_s),
        w_drive=float(opts.w_drive),
        w_recover=float(opts.w_recover),
        w_interval=float(opts.w_interval),
        w_cumulative=float(opts.w_cumulative),
        w_skip=float(opts.w_skip),
    )


class CoarseDetectionEmpty(RuntimeError):
    """Raised when the calibration pre-pass finds zero drives.

    The legacy CLI aborts with exit-code 2 in this case; :func:`run_inference`
    catches this exception explicitly to keep that contract.
    """


def _run_calibration(
    opts: PipelineOptions,
    *,
    df: pd.DataFrame,
    detect_kwargs: dict[str, Any],
    interactive: bool,
    smooth_window_s: float,
) -> tuple[CalibrationResult | None, float, float]:
    """Two-pass coarse-detect → match → calibrate velocity fractions.

    Returns ``(calibration_or_None, catch_frac, finish_frac)``. Raises
    :class:`CoarseDetectionEmpty` when the coarse pass yields zero drives so
    the caller can short-circuit with a fatal exit code.
    """
    catch_velocity_frac = detect_kwargs["catch_velocity_frac"]
    finish_velocity_frac = detect_kwargs["finish_velocity_frac"]
    try:
        rp3_dirty_csv_path_cal = resolve_rp3_dirty_csv(
            run_dir=opts.run_dir,  # type: ignore[arg-type]
            rp3_dirty_csv=opts.rp3_dirty_csv,
            interactive=interactive,
        )
        rp3_clean_csv_path_cal = clean_rp3_dirty_csv(rp3_dirty_csv_path_cal)
        rp3_df_cal = _load_rp3_clean_csv(rp3_clean_csv_path_cal)
        anchor_rp3_idx_cal = _resolve_rp3_anchor_idx(
            rp3_df_cal,
            anchor_rp3_row_idx=opts.anchor_rp3_row_idx,
            anchor_rp3_stroke_number=opts.anchor_rp3_stroke_number,
            interactive=interactive,
        )

        coarse_detect = detect_drive_events(
            df,
            **{**detect_kwargs, "finish_method": FINISH_METHOD_VELOCITY_THRESHOLD},
        )
        if not coarse_detect.events:
            raise CoarseDetectionEmpty(
                "Pass-1 coarse detection found 0 drives; cannot calibrate."
            )

        coarse_events_df = events_to_dataframe(coarse_detect.events)
        match_cfg_cal = _build_match_config(opts, calibrated=False)
        coarse_match = _build_rp3_match_manifest(
            video_df=coarse_events_df,
            rp3_df=rp3_df_cal,
            anchor_video_idx=int(opts.anchor_video_stroke_idx),
            anchor_rp3_idx=int(anchor_rp3_idx_cal),
            cfg=match_cfg_cal,
        )
        coarse_manifest = coarse_match.manifest

        rp3_drive_durations: dict[int, float] = {}
        for _, row in coarse_manifest.iterrows():
            vi = int(row["video_stroke_idx"])
            rp3_drive_durations[vi] = float(row["rp3_drive_s"])

        calibration = calibrate_velocity_fracs(
            df,
            rp3_drive_durations,
            smooth_window_s=smooth_window_s,
            min_cycle_s=float(opts.min_cycle_s),
            slope_tol_frac=float(opts.slope_tol_frac),
        )

        if opts.catch_velocity_frac is None:
            catch_velocity_frac = calibration.catch_velocity_frac
        if opts.finish_velocity_frac is None:
            finish_velocity_frac = calibration.finish_velocity_frac

        print(
            f"Calibration: catch_frac={catch_velocity_frac:.3f} "
            f"finish_frac={finish_velocity_frac:.3f} "
            f"(MAE={calibration.mae_s * 1000:.1f}ms, "
            f"ME={calibration.me_s * 1000:.1f}ms, "
            f"n={calibration.n_strokes})"
        )
        return calibration, catch_velocity_frac, finish_velocity_frac
    except CoarseDetectionEmpty:
        raise
    except Exception as exc:  # noqa: BLE001 — preserve legacy "warn and proceed" behaviour
        print(f"Calibration failed ({exc}); using default fracs.")
        return None, catch_velocity_frac, finish_velocity_frac


def _run_match_export(
    opts: PipelineOptions,
    *,
    run_dir: Path,
    output_dir: Path,
    frame_df: pd.DataFrame,
    events_df: pd.DataFrame,
    active_side: str,
    use_rp3_finish: bool,
    interactive: bool,
    calibration: CalibrationResult | None,
    rp3_manifest_csv: Path,
    rp3_segments_csv: Path,
    rp3_segment_status_csv: Path,
    rp3_summary_json: Path,
) -> tuple[dict[str, Any] | None, Path | None, Path | None, Path | None]:
    """Resolve dirty CSV → clean → match → export segments → optional dataset."""
    if events_df.empty:
        print("RP3 match failed: no detected drive events available for matching.")
        return None, None, None, None

    rp3_dirty_csv_path = resolve_rp3_dirty_csv(
        run_dir=run_dir,
        rp3_dirty_csv=opts.rp3_dirty_csv,
        interactive=interactive,
    )
    rp3_clean_csv_path = clean_rp3_dirty_csv(rp3_dirty_csv_path)
    rp3_df = _load_rp3_clean_csv(rp3_clean_csv_path)

    overrides = load_overrides(run_dir)
    if not overrides.is_empty:
        validate_overrides(
            overrides,
            video_stroke_indices=events_df["stroke_idx"].astype(int).tolist(),
            rp3_stroke_numbers=rp3_df["stroke_number"].astype(int).tolist(),
        )

    # CLI-supplied anchor wins over the sidecar; sidecar is only used to fill
    # gaps. Same precedence policy applies to active_side / rower_facing.
    anchor_video_idx = int(opts.anchor_video_stroke_idx)
    if (
        opts.anchor_rp3_row_idx is None
        and opts.anchor_rp3_stroke_number is None
        and overrides.anchor_rp3_stroke_number is not None
    ):
        anchor_rp3_idx = resolve_pin_to_row_idx(rp3_df, overrides.anchor_rp3_stroke_number)
    else:
        anchor_rp3_idx = _resolve_rp3_anchor_idx(
            rp3_df,
            anchor_rp3_row_idx=opts.anchor_rp3_row_idx,
            anchor_rp3_stroke_number=opts.anchor_rp3_stroke_number,
            interactive=interactive,
        )
    if (
        overrides.anchor_video_stroke_idx is not None
        and opts.anchor_video_stroke_idx == PipelineOptions.__dataclass_fields__["anchor_video_stroke_idx"].default
    ):
        anchor_video_idx = int(overrides.anchor_video_stroke_idx)

    pinned_rp3_row_by_relative_idx: dict[int, int] = {}
    for pin in overrides.pinned:
        rel_i = int(pin.video_stroke_idx) - anchor_video_idx
        if rel_i < 0:
            raise ValueError(
                f"Pinned video_stroke_idx={pin.video_stroke_idx} precedes anchor "
                f"({anchor_video_idx}); move the anchor or remove the pin."
            )
        pinned_rp3_row_by_relative_idx[rel_i] = resolve_pin_to_row_idx(
            rp3_df, pin.rp3_stroke_number,
        )

    excluded_relative_indices = {
        int(idx) - anchor_video_idx
        for idx in overrides.excluded_video_stroke_idx
    }
    excluded_relative_indices = {i for i in excluded_relative_indices if i > 0}

    match_cfg = _build_match_config(opts, calibrated=calibration is not None)
    match_result = _build_rp3_match_manifest(
        video_df=events_df,
        rp3_df=rp3_df,
        anchor_video_idx=anchor_video_idx,
        anchor_rp3_idx=int(anchor_rp3_idx),
        cfg=match_cfg,
        pinned_rp3_row_by_relative_idx=pinned_rp3_row_by_relative_idx or None,
        excluded_relative_indices=excluded_relative_indices or None,
    )
    manifest_df = match_result.manifest
    manifest_df.to_csv(rp3_manifest_csv, index=False)

    segments_df, segment_status_df = build_force_pose_segments(
        run_dir=run_dir,
        frame_df=frame_df,
        events_df=events_df,
        manifest_df=manifest_df,
        rp3_df=rp3_df,
        active_side=active_side,
        rp3_clean_csv=rp3_clean_csv_path,
        use_rp3_finish=use_rp3_finish,
        include_second_derivatives=opts.include_second_derivatives,
        rower_facing=opts.rower_facing,
    )
    validate_force_segment_exports(
        manifest_df=manifest_df,
        segments_df=segments_df,
        status_df=segment_status_df,
    )
    segments_df.to_csv(rp3_segments_csv, index=False)
    segment_status_df.to_csv(rp3_segment_status_csv, index=False)

    training_dataset_dir: Path | None = None
    if not opts.no_build_dataset and not segments_df.empty:
        dataset_dir = (
            opts.dataset_output_dir.expanduser().resolve()
            if opts.dataset_output_dir is not None
            else output_dir / "training_dataset"
        )
        try:
            _build_training_dataset(
                segment_csvs=[rp3_segments_csv],
                output_dir=dataset_dir,
                qc_mode=opts.dataset_qc_mode,
                n_grid=opts.dataset_n_grid,
                n_pca_components=opts.dataset_n_pca_components,
                force_col=opts.dataset_force_col,
                onset_frac=opts.dataset_onset_frac,
            )
            training_dataset_dir = dataset_dir
        except Exception as exc:  # noqa: BLE001 — non-fatal
            print(f"Training dataset build failed (non-fatal): {exc}")

    exported_strokes = int(segment_status_df["segment_exported"].astype(bool).sum())
    dropped_strokes = int(len(segment_status_df) - exported_strokes)
    drop_counts_series = (
        segment_status_df.loc[
            (~segment_status_df["segment_exported"].astype(bool))
            & (segment_status_df["drop_reason"].astype(str).str.len() > 0),
            "drop_reason",
        ]
        .value_counts()
        .sort_index()
    )
    drop_counts = {str(k): int(v) for k, v in drop_counts_series.items()}

    rp3_summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "rp3_dirty_csv": str(rp3_dirty_csv_path),
        "rp3_clean_csv": str(rp3_clean_csv_path),
        "active_side": active_side,
        "anchor_video_stroke_idx": int(anchor_video_idx),
        "anchor_video_stroke_label": int(manifest_df.iloc[0]["video_stroke_idx"]),
        "anchor_rp3_row_idx": int(manifest_df.iloc[0]["rp3_row_idx"]),
        "anchor_rp3_stroke_number": int(manifest_df.iloc[0]["rp3_stroke_number"]),
        "matched_video_strokes": int(len(manifest_df)),
        "total_skipped_rp3_rows": int(manifest_df["rp3_rows_skipped_since_prev"].sum()),
        "total_score": float(match_result.total_score),
        "mean_abs_cum_catch_err_s": float(manifest_df["cum_catch_err_s"].abs().mean()),
        "mean_abs_interval_err_s": float(manifest_df["interval_err_s"].abs().mean()),
        "mean_abs_drive_err_s": float(manifest_df["drive_err_s"].abs().mean()),
        "mean_abs_recover_err_s": float(manifest_df["recover_err_s"].abs().mean()),
        "segment_rows": int(len(segments_df)),
        "segment_exported_strokes": exported_strokes,
        "segment_dropped_strokes": dropped_strokes,
        "segment_drop_reason_counts": drop_counts,
        "segment_export_status_csv": str(rp3_segment_status_csv),
        "outputs": {
            "rp3_match_manifest_csv": str(rp3_manifest_csv),
            "rp3_pose_force_segments_csv": str(rp3_segments_csv),
            "segment_export_status_csv": str(rp3_segment_status_csv),
            "training_dataset_dir": str(training_dataset_dir) if training_dataset_dir is not None else None,
        },
    }
    with rp3_summary_json.open("w", encoding="utf-8") as f:
        json.dump(rp3_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return rp3_summary, rp3_dirty_csv_path, rp3_clean_csv_path, training_dataset_dir


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_inference(opts: PipelineOptions) -> PipelineResult:
    """Execute the full inference pipeline.

    Mirrors the legacy ``inference_cli.main()`` exactly: same on-disk artifacts,
    same console output, same exit codes.
    """
    if opts.overlay_video and opts.no_overlay_video:
        print("Input error: use only one of --overlay-video or --no-overlay-video.")
        return PipelineResult(exit_code=1, error="conflicting overlay-video flags")
    if opts.match_rp3 and opts.no_match_rp3:
        print("Input error: use only one of --match-rp3 or --no-match-rp3.")
        return PipelineResult(exit_code=1, error="conflicting match-rp3 flags")
    if opts.anchor_rp3_row_idx is not None and opts.anchor_rp3_stroke_number is not None:
        print("Input error: use only one of --anchor-rp3-row-idx or --anchor-rp3-stroke-number.")
        return PipelineResult(exit_code=1, error="conflicting anchor flags")

    use_rp3_finish = opts.use_rp3_finish and not opts.no_use_rp3_finish

    interactive = (
        opts.interactive
        if opts.interactive is not None
        else (sys.stdin.isatty() and sys.stdout.isatty())
    )
    selected_run_via_selector = opts.run_dir is None
    try:
        run_dir = resolve_run_dir(opts.run_dir, opts.runs_root)
    except Exception as exc:
        print(f"Run selection failed: {exc}")
        return PipelineResult(exit_code=1, error=str(exc))

    # Mutate opts.run_dir so calibration helpers can read it back without
    # threading the path through every call.
    opts.run_dir = run_dir
    has_dirty_rp3 = bool(discover_run_rp3_dirty_csvs(run_dir))

    # Merge non-conflicting overrides from <run>/inference/match_overrides.json
    # so resolution of active_side / rower_facing / anchor honours editor edits.
    sidecar = load_overrides(run_dir)
    if not sidecar.is_empty:
        if opts.active_side is None and sidecar.active_side is not None:
            opts.active_side = sidecar.active_side
        if opts.rower_facing == "auto" and sidecar.rower_facing is not None:
            opts.rower_facing = sidecar.rower_facing

    write_overlay_video = _resolve_overlay_video(opts, selected_run_via_selector=selected_run_via_selector)
    run_rp3_matching = _resolve_run_rp3_matching(opts, has_dirty_rp3=has_dirty_rp3)
    active_side = _resolve_active_side(
        opts,
        run_rp3_matching=run_rp3_matching,
        selected_run_via_selector=selected_run_via_selector,
    )

    smooth_window_s, finish_velocity_frac, catch_velocity_frac, detect_kwargs = (
        _resolve_detection_params(opts)
    )
    finish_method = str(opts.finish_method)
    is_calibrated = finish_method == FINISH_METHOD_VELOCITY_CALIBRATED

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    try:
        df = pd.read_csv(stroke_csv)
    except Exception as exc:
        print(f"Failed to read {stroke_csv}: {exc}")
        return PipelineResult(exit_code=2, run_dir=run_dir, error=str(exc))

    calibration: CalibrationResult | None = None
    if is_calibrated and run_rp3_matching:
        try:
            calibration, catch_velocity_frac, finish_velocity_frac = _run_calibration(
                opts,
                df=df,
                detect_kwargs=detect_kwargs,
                interactive=interactive,
                smooth_window_s=smooth_window_s,
            )
        except CoarseDetectionEmpty as exc:
            print(str(exc))
            return PipelineResult(exit_code=2, run_dir=run_dir, error=str(exc))
        if calibration is not None:
            detect_kwargs["catch_velocity_frac"] = catch_velocity_frac
            detect_kwargs["finish_velocity_frac"] = finish_velocity_frac

    try:
        detection = detect_drive_events(
            df,
            **{**detect_kwargs, "finish_method": finish_method},
        )
    except Exception as exc:
        print(f"Failed to process {stroke_csv}: {exc}")
        return PipelineResult(exit_code=2, run_dir=run_dir, error=str(exc))

    output_dir = (
        opts.output_dir.expanduser().resolve()
        if opts.output_dir is not None
        else (run_dir / "inference").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df = events_to_dataframe(detection.events)
    frame_df = build_frame_level_recomputed_columns(df, detection)

    events_csv = output_dir / "drive_events.csv"
    frame_csv = output_dir / "stroke_signal_with_drive_events.csv"
    summary_json = output_dir / "drive_events_summary.json"
    overlay_video_path = output_dir / "drive_phase_overlay.mp4"
    rp3_manifest_csv = output_dir / "rp3_match_manifest.csv"
    rp3_summary_json = output_dir / "rp3_match_summary.json"
    rp3_segments_csv = output_dir / "rp3_pose_force_matched_segments.csv"
    rp3_segment_status_csv = output_dir / "rp3_pose_force_export_status.csv"

    events_df.to_csv(events_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)

    overlay_frames_written = 0
    overlay_drive_frames = 0
    input_video_path: str | None = None
    if write_overlay_video:
        input_video = find_input_video(run_dir)
        input_video_path = str(input_video)
        is_drive_flags = frame_df["is_drive_recomputed"].to_numpy(dtype=np.uint8)
        overlay_frames_written, overlay_drive_frames = write_drive_overlay_video(
            input_video=input_video,
            is_drive_flags=is_drive_flags,
            out_video=overlay_video_path,
            alpha=float(opts.overlay_opacity),
        )

    rp3_summary: dict[str, Any] | None = None
    rp3_dirty_csv_path: Path | None = None
    rp3_clean_csv_path: Path | None = None
    training_dataset_dir: Path | None = None
    if run_rp3_matching:
        try:
            rp3_summary, rp3_dirty_csv_path, rp3_clean_csv_path, training_dataset_dir = _run_match_export(
                opts,
                run_dir=run_dir,
                output_dir=output_dir,
                frame_df=frame_df,
                events_df=events_df,
                active_side=active_side,
                use_rp3_finish=use_rp3_finish,
                interactive=interactive,
                calibration=calibration,
                rp3_manifest_csv=rp3_manifest_csv,
                rp3_segments_csv=rp3_segments_csv,
                rp3_segment_status_csv=rp3_segment_status_csv,
                rp3_summary_json=rp3_summary_json,
            )
            if rp3_summary is None:
                # events_df was empty
                return PipelineResult(
                    exit_code=3,
                    run_dir=run_dir,
                    detection=detection,
                    calibration=calibration,
                    error="no detected drive events available for matching",
                )
        except Exception as exc:
            print(f"RP3 match failed: {exc}")
            return PipelineResult(
                exit_code=3,
                run_dir=run_dir,
                detection=detection,
                calibration=calibration,
                error=str(exc),
            )

    inferred_time = _infer_time_s(df)
    if inferred_time.size:
        t0 = float(inferred_time[0])
        t1 = float(inferred_time[-1])
    else:
        t0 = 0.0
        t1 = 0.0

    cal_info: dict[str, Any] | None = None
    if calibration is not None:
        cal_info = {
            "source": "rp3_optimized",
            "mae_ms": round(calibration.mae_s * 1000, 2),
            "me_ms": round(calibration.me_s * 1000, 2),
            "std_ms": round(calibration.std_s * 1000, 2),
            "n_strokes": calibration.n_strokes,
        }

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "stroke_signal_csv": str(stroke_csv),
        "frame_count": int(len(df)),
        "time_start_s": t0,
        "time_end_s": t1,
        "fps_estimate": detection.fps_estimate,
        "catch_candidates_raw": int(detection.catch_candidates_raw.size),
        "catches_filtered": int(detection.catches_filtered.size),
        "complete_drives": int(len(detection.events)),
        "min_drive_displacement_px": float(detection.min_drive_disp_px),
        "parameters": {
            "smooth_window_s": smooth_window_s,
            "min_cycle_s": float(opts.min_cycle_s),
            "min_drive_s": float(opts.min_drive_s),
            "min_recover_s": float(opts.min_recover_s),
            "min_drive_disp_frac": float(opts.min_drive_disp_frac),
            "slope_tol_frac": float(opts.slope_tol_frac),
            "catch_velocity_frac": catch_velocity_frac,
            "finish_velocity_frac": finish_velocity_frac,
            "finish_method": finish_method,
            "use_rp3_finish": use_rp3_finish if run_rp3_matching else None,
            "calibration": cal_info,
        },
        "outputs": {
            "drive_events_csv": str(events_csv),
            "stroke_signal_with_drive_events_csv": str(frame_csv),
            "drive_phase_overlay_video": str(overlay_video_path) if write_overlay_video else None,
            "rp3_match_manifest_csv": str(rp3_manifest_csv) if rp3_summary is not None else None,
            "rp3_pose_force_segments_csv": str(rp3_segments_csv) if rp3_summary is not None else None,
            "rp3_pose_force_export_status_csv": str(rp3_segment_status_csv) if rp3_summary is not None else None,
            "rp3_match_summary_json": str(rp3_summary_json) if rp3_summary is not None else None,
            "training_dataset_dir": str(rp3_summary["outputs"]["training_dataset_dir"]) if rp3_summary is not None and rp3_summary["outputs"].get("training_dataset_dir") else None,
        },
        "input_video": input_video_path,
        "overlay_frames_written": int(overlay_frames_written),
        "overlay_drive_frames": int(overlay_drive_frames),
        "active_side": active_side if rp3_summary is not None else None,
        "rp3_dirty_csv": str(rp3_dirty_csv_path) if rp3_dirty_csv_path is not None else None,
        "rp3_clean_csv": str(rp3_clean_csv_path) if rp3_clean_csv_path is not None else None,
    }

    if detection.events:
        drive_durations = [event.drive_duration_s for event in detection.events]
        recover_durations = [event.recover_duration_s for event in detection.events]
        summary["drive_duration_mean_s"] = float(np.mean(drive_durations))
        summary["recover_duration_mean_s"] = float(np.mean(recover_durations))
    else:
        summary["drive_duration_mean_s"] = math.nan
        summary["recover_duration_mean_s"] = math.nan

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Run: {run_dir.name}")
    print(f"Input: {stroke_csv}")
    print(
        "Detected events: "
        f"{len(detection.events)} complete drives "
        f"({detection.catch_candidates_raw.size} raw catches -> {detection.catches_filtered.size} filtered catches)"
    )
    print("Outputs:")
    print(f"  {events_csv}")
    print(f"  {frame_csv}")
    if write_overlay_video:
        print(f"  {overlay_video_path}")
    if rp3_summary is not None:
        print(f"  {rp3_manifest_csv}")
        print(f"  {rp3_segments_csv}")
        print(f"  {rp3_segment_status_csv}")
        print(f"  {rp3_summary_json}")
        td = rp3_summary["outputs"].get("training_dataset_dir")
        if td:
            print(f"  {td}/ (training dataset)")
    print(f"  {summary_json}")

    return PipelineResult(
        exit_code=0,
        run_dir=run_dir,
        summary=summary,
        rp3_summary=rp3_summary,
        detection=detection,
        calibration=calibration,
        events_csv=events_csv,
        frame_csv=frame_csv,
        summary_json=summary_json,
        overlay_video=overlay_video_path if write_overlay_video else None,
        rp3_manifest_csv=rp3_manifest_csv if rp3_summary is not None else None,
        rp3_segments_csv=rp3_segments_csv if rp3_summary is not None else None,
        rp3_segment_status_csv=rp3_segment_status_csv if rp3_summary is not None else None,
        rp3_summary_json=rp3_summary_json if rp3_summary is not None else None,
        training_dataset_dir=training_dataset_dir,
    )
