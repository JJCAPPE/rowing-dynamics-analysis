#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sports2d_app"))
from drive_detection import (
    DriveEvent,
    DetectionResult,
    CalibrationResult,
    detect_drive_events,
    calibrate_velocity_fracs,
    _infer_time_s,
    _interp_nans,
    FINISH_METHOD_VELOCITY_THRESHOLD,
    FINISH_METHOD_POSITION_MAX,
    FINISH_METHOD_VELOCITY_CALIBRATED,
    VALID_FINISH_METHODS,
    DEFAULT_CATCH_VELOCITY_FRAC,
    DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED,
)
sys.path.pop(0)

from match_rp3_cli import (
    MatchConfig as Rp3MatchConfig,
    _build_match_manifest as _build_rp3_match_manifest,
    _load_rp3 as _load_rp3_clean_csv,
    _resolve_anchor_rp3_idx as _resolve_rp3_anchor_idx,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
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
FORCE_COL_RE = re.compile(r"^force_at_([0-9]+(?:\.[0-9]+)?)cm$")
PDF_AREA_EPS = 1e-9
PDF_AREA_TOL = 1e-6
_RP3_EXPAND_MODULE: ModuleType | None = None

SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 3
CHAIN_RULE_EPS = 1e-6
DS_DT_STALL_FRAC = 0.05
MAX_ANGULAR_VEL_DEG_S = 600.0
MAX_NAN_FRAC_ANGLES = 0.3



# DriveEvent, DetectionResult, detect_drive_events imported from drive_detection


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    if not runs_root.is_dir():
        raise NotADirectoryError(f"Runs path is not a directory: {runs_root}")

    runs: list[Path] = []
    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue
        stroke_csv = candidate / "stroke" / "stroke_signal.csv"
        if stroke_csv.exists():
            runs.append(candidate.resolve())

    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


def _pick_run_with_curses(options: Sequence[Path]) -> Path:
    def _inner(stdscr: Any) -> Path:
        idx = 0
        top = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, "Select Sports2D run (Enter to confirm)", width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "UP/DOWN move, ENTER select, q quit", width, curses.A_DIM)

            visible = max(1, height - 3)
            if idx < top:
                top = idx
            elif idx >= top + visible:
                top = idx - visible + 1

            for row in range(visible):
                i = top + row
                if i >= len(options):
                    break
                label = options[i].name
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(row + 2, 0, label, width, attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                idx = min(len(options) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return options[idx]

    return curses.wrapper(_inner)


def _pick_run_with_prompt(options: Sequence[Path]) -> Path:
    if not options:
        raise ValueError("No run options available.")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return options[0]

    print("\nAvailable Sports2D runs:")
    for i, option in enumerate(options, start=1):
        print(f"  {i:2d}. {option.name}")

    while True:
        raw = input("Select run number [1]: ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            pick = int(raw) - 1
            if 0 <= pick < len(options):
                return options[pick]
        print("Invalid selection. Enter a listed number.")


def _pick_file_with_curses(options: Sequence[Path], title: str) -> Path:
    def _inner(stdscr: Any) -> Path:
        idx = 0
        top = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, title, width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "UP/DOWN move, ENTER select, q quit", width, curses.A_DIM)

            visible = max(1, height - 3)
            if idx < top:
                top = idx
            elif idx >= top + visible:
                top = idx - visible + 1

            for row in range(visible):
                i = top + row
                if i >= len(options):
                    break
                label = options[i].name
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(row + 2, 0, label, width, attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                idx = min(len(options) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return options[idx]

    return curses.wrapper(_inner)


def _pick_file_with_prompt(options: Sequence[Path], title: str) -> Path:
    if not options:
        raise ValueError("No options available.")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return options[0]

    print(f"\n{title}")
    for i, option in enumerate(options, start=1):
        print(f"  {i:2d}. {option.name}")

    while True:
        raw = input("Select number [1]: ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            pick = int(raw) - 1
            if 0 <= pick < len(options):
                return options[pick]
        print("Invalid selection.")


def _pick_yes_no_with_curses(prompt: str, default_no: bool = True) -> bool:
    labels = ["No", "Yes"]
    idx = 0 if default_no else 1

    def _inner(stdscr: Any) -> bool:
        nonlocal idx
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, prompt, width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "LEFT/RIGHT or UP/DOWN, ENTER select, q cancel", width, curses.A_DIM)

            for i, label in enumerate(labels):
                x = 2 + i * 12
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(3, x, f"[ {label} ]", max(1, width - x), attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_LEFT, curses.KEY_UP, ord("h"), ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_RIGHT, curses.KEY_DOWN, ord("l"), ord("j")):
                idx = min(len(labels) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return idx == 1

    return curses.wrapper(_inner)


def _pick_yes_no_with_prompt(prompt: str, default_no: bool = True) -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return not default_no

    suffix = "[y/N]" if default_no else "[Y/n]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if raw == "":
            return not default_no
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _select_yes_no(prompt: str, *, default_no: bool = True) -> bool:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _pick_yes_no_with_curses(prompt, default_no=default_no)
        except Exception:
            pass
    return _pick_yes_no_with_prompt(prompt, default_no=default_no)


def _prompt_int(prompt: str, default: int) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return int(default)
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return int(default)
        if raw.lstrip("-").isdigit():
            return int(raw)
        print("Please enter an integer.")


def _prompt_choice(prompt: str, options: Sequence[str], default: str) -> str:
    options_norm = [str(x).strip().lower() for x in options]
    default_norm = str(default).strip().lower()
    if default_norm not in options_norm:
        raise ValueError("default must be one of options")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default_norm

    opt_txt = "/".join(options_norm)
    while True:
        raw = input(f"{prompt} [{default_norm}] ({opt_txt}): ").strip().lower()
        if raw == "":
            return default_norm
        if raw in options_norm:
            return raw
        print(f"Please enter one of: {opt_txt}")


def _select_run(runs_root: Path) -> Path:
    options = _discover_run_dirs(runs_root)
    if not options:
        raise FileNotFoundError(f"No runs with stroke/stroke_signal.csv found in {runs_root}")

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _pick_run_with_curses(options)
        except Exception:
            pass
    return _pick_run_with_prompt(options)


def _resolve_run_dir(run_dir: Path | None, runs_root: Path) -> Path:
    if run_dir is None:
        return _select_run(runs_root)

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    if not stroke_csv.exists():
        raise FileNotFoundError(f"Missing stroke signal CSV at: {stroke_csv}")
    return run_dir


def _ensure_path_in_dir(path: Path, parent_dir: Path, *, label: str) -> None:
    parent_dir = parent_dir.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        path.relative_to(parent_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent_dir}: {path}") from exc


def _discover_run_rp3_dirty_csvs(run_dir: Path) -> list[Path]:
    rp3_dir = (run_dir / "rp3").resolve()
    if not rp3_dir.exists() or not rp3_dir.is_dir():
        return []
    options = [
        p.resolve()
        for p in sorted(rp3_dir.glob("*.csv"))
        if p.is_file() and not p.name.startswith(".") and not p.name.lower().endswith("-clean.csv")
    ]
    return options


def _resolve_rp3_dirty_csv(
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
        _ensure_path_in_dir(csv_path, rp3_dir, label="--rp3-dirty-csv")
        if csv_path.name.lower().endswith("-clean.csv"):
            raise ValueError(f"Expected dirty RP3 CSV, got clean CSV: {csv_path.name}")
        return csv_path

    options = _discover_run_rp3_dirty_csvs(run_dir)
    if not options:
        raise FileNotFoundError(
            f"No RP3 dirty CSV files found in {rp3_dir}. Add one or run with --no-match-rp3."
        )
    if len(options) == 1:
        return options[0]

    if interactive:
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                return _pick_file_with_curses(options, "Select RP3 dirty CSV")
            except Exception:
                pass
        return _pick_file_with_prompt(options, "Select RP3 dirty CSV")

    raise ValueError(
        f"Multiple RP3 dirty CSV files found in {rp3_dir}. "
        "Specify one with --rp3-dirty-csv."
    )


def _load_rp3_expand_module() -> ModuleType:
    global _RP3_EXPAND_MODULE
    if _RP3_EXPAND_MODULE is not None:
        return _RP3_EXPAND_MODULE

    module_path = REPO_ROOT / "rp3-extraction" / "expand_rp3_curve_data.py"
    if not module_path.exists():
        raise FileNotFoundError(f"RP3 cleaning script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("rp3_expand_routine", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RP3_EXPAND_MODULE = module
    return module


def _clean_rp3_dirty_csv(dirty_csv: Path) -> Path:
    if dirty_csv.name.lower().endswith("-clean.csv"):
        raise ValueError(f"Expected dirty RP3 CSV, got clean CSV: {dirty_csv}")

    clean_csv = dirty_csv.with_name(f"{dirty_csv.stem}-clean.csv")
    module = _load_rp3_expand_module()
    process_file = getattr(module, "process_file", None)
    if process_file is None:
        raise AttributeError("expand_rp3_curve_data.py is missing process_file().")

    process_file(
        input_csv=dirty_csv,
        output_csv=clean_csv,
        max_stroke_length_cm=RP3_CLEAN_MAX_STROKE_LENGTH_CM,
        step_cm=RP3_CLEAN_STEP_CM,
        drop_curve_data=False,
        truncate=False,
    )
    return clean_csv.resolve()



# _moving_average, _interp_nans, _infer_time_s, _fill_zero_signs,
# _find_catch_candidates, _filter_catches_by_cycle, detect_drive_events
# imported from sports2d_app.drive_detection


def _build_frame_level_recomputed_columns(
    df: pd.DataFrame,
    detection: DetectionResult,
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    is_catch = np.zeros((n,), dtype=np.uint8)
    is_finish = np.zeros((n,), dtype=np.uint8)
    is_drive = np.zeros((n,), dtype=np.uint8)
    stroke_idx = np.full((n,), -1, dtype=np.int32)
    stroke_phase = np.full((n,), np.nan, dtype=np.float32)

    catches_idx = detection.catches_filtered
    catches_idx = catches_idx[(catches_idx >= 0) & (catches_idx < n)]
    is_catch[catches_idx] = 1

    time_s = _infer_time_s(out)
    frame_to_row: dict[int, int] = {}
    if "frame_idx" in out.columns:
        frame_vals = pd.to_numeric(out["frame_idx"], errors="coerce").to_numpy(dtype=np.float64)
        for row_idx, frame_val in enumerate(frame_vals):
            if not np.isfinite(frame_val):
                continue
            frame_to_row.setdefault(int(round(float(frame_val))), row_idx)

    for event in detection.events:
        if frame_to_row:
            c0 = int(frame_to_row.get(int(event.catch_frame_idx), int(event.catch_frame_idx)))
            f0 = int(frame_to_row.get(int(event.finish_frame_idx), int(event.finish_frame_idx)))
            c1 = int(frame_to_row.get(int(event.next_catch_frame_idx), int(event.next_catch_frame_idx)))
        else:
            c0 = int(event.catch_frame_idx)
            f0 = int(event.finish_frame_idx)
            c1 = int(event.next_catch_frame_idx)
        c0 = int(np.clip(c0, 0, n - 1))
        f0 = int(np.clip(f0, 0, n - 1))
        c1 = int(np.clip(c1, 0, n - 1))
        if not (c0 < f0 < c1):
            continue

        is_finish[f0] = 1
        stroke_idx[c0 : c1 + 1] = int(event.stroke_idx)
        is_drive[c0 : f0 + 1] = 1

        drive_len = max(1, f0 - c0)
        for t in range(c0, f0 + 1):
            stroke_phase[t] = np.float32(0.5 * (t - c0) / drive_len)

        rec_len = max(1, c1 - f0)
        for t in range(f0, c1 + 1):
            stroke_phase[t] = np.float32(0.5 + 0.5 * (t - f0) / rec_len)

    out["relative_axis_px_smooth"] = detection.signal_smooth_px
    out["velocity_axis_recomputed_px_s"] = detection.slope_px_s
    out["stroke_idx_recomputed"] = stroke_idx
    out["stroke_phase_recomputed"] = stroke_phase
    out["is_drive_recomputed"] = is_drive
    out["is_catch_recomputed"] = is_catch
    out["is_finish_recomputed"] = is_finish
    out["time_s_recomputed"] = time_s.astype(np.float32)
    return out


def _events_to_dataframe(events: Iterable[DriveEvent]) -> pd.DataFrame:
    rows = [asdict(event) for event in events]
    if not rows:
        return pd.DataFrame(
            columns=[
                "stroke_idx",
                "catch_frame_idx",
                "catch_time_s",
                "finish_frame_idx",
                "finish_time_s",
                "next_catch_frame_idx",
                "next_catch_time_s",
                "drive_duration_s",
                "recover_duration_s",
                "cycle_duration_s",
                "catch_distance_px",
                "finish_distance_px",
                "drive_displacement_px",
            ]
        )
    return pd.DataFrame(rows)


def _find_input_video(run_dir: Path) -> Path:
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


def _write_drive_overlay_video(
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


def _find_force_columns(rp3_df: pd.DataFrame) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for col in rp3_df.columns:
        m = FORCE_COL_RE.match(str(col))
        if m is None:
            continue
        found.append((str(col), float(m.group(1))))
    found.sort(key=lambda x: x[1])
    return found


def _interp_feature_on_progress(
    s_video: np.ndarray,
    feature_values: np.ndarray,
    s_targets: np.ndarray,
) -> np.ndarray:
    s_video = np.asarray(s_video, dtype=np.float64)
    y = np.asarray(feature_values, dtype=np.float64)
    s_targets = np.asarray(s_targets, dtype=np.float64)

    mask = np.isfinite(s_video) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(s_targets, np.nan, dtype=np.float64)

    s = s_video[mask]
    y = y[mask]
    order = np.argsort(s)
    s = s[order]
    y = y[order]

    s = np.clip(s, 0.0, 1.0)
    s = np.maximum.accumulate(s)

    unique_s, unique_idx = np.unique(s, return_index=True)
    y_unique = y[unique_idx]
    if unique_s.size == 0:
        return np.full_like(s_targets, np.nan, dtype=np.float64)
    if unique_s.size == 1:
        return np.full_like(s_targets, float(y_unique[0]), dtype=np.float64)

    clipped_targets = np.clip(s_targets, unique_s[0], unique_s[-1])
    return np.interp(clipped_targets, unique_s, y_unique)


def _savgol_smooth(arr: np.ndarray, window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
    finite = np.isfinite(arr)
    if finite.sum() < max(window, polyorder + 1):
        return arr.copy()
    filled = _interp_nans(arr.copy())
    eff_win = min(window, len(filled))
    if eff_win % 2 == 0:
        eff_win -= 1
    eff_win = max(eff_win, 1)
    eff_poly = min(polyorder, eff_win - 1)
    smoothed = savgol_filter(filled, eff_win, eff_poly)
    out = arr.copy()
    out[finite] = smoothed[finite]
    return out


def _build_force_pose_segments(
    *,
    run_dir: Path,
    frame_df: pd.DataFrame,
    events_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    rp3_df: pd.DataFrame,
    active_side: str,
    rp3_clean_csv: Path,
    use_rp3_finish: bool = False,
    include_second_derivatives: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    angles_csv = run_dir / "motionbert" / "angles_h36m.csv"
    if not angles_csv.exists():
        raise FileNotFoundError(f"angles_h36m.csv not found at: {angles_csv}")
    angles_df = pd.read_csv(angles_csv)
    if "frame_idx" not in angles_df.columns:
        raise ValueError("angles_h36m.csv missing frame_idx column.")

    merged = frame_df.merge(angles_df, on="frame_idx", how="left", suffixes=("", "_angle"))
    force_cols = _find_force_columns(rp3_df)
    if not force_cols:
        raise ValueError("No force_at_*cm columns found in RP3 CSV.")

    if active_side not in {"left", "right"}:
        raise ValueError("active_side must be 'left' or 'right'.")
    side_map = {
        "knee_active_deg": f"{active_side}_knee_deg",
        "hip_active_deg": f"{active_side}_hip_deg",
        "elbow_active_deg": f"{active_side}_elbow_deg",
        "trunk_vs_horizontal_deg": "trunk_vs_horizontal_deg",
        "spine_flexion_deg": "spine_flexion_deg",
    }
    missing = [src for src in side_map.values() if src not in merged.columns]
    if missing:
        raise ValueError(f"Missing angle columns for active side mapping: {missing}")

    events_by_stroke: dict[int, pd.Series] = {}
    for _, row in events_df.iterrows():
        events_by_stroke[int(row["stroke_idx"])] = row

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for match_seq_idx, (_, mrow) in enumerate(manifest_df.iterrows()):
        v_stroke = int(mrow["video_stroke_idx"])
        rp3_idx = int(mrow["rp3_row_idx"])
        rp3_stroke_number = int(mrow["rp3_stroke_number"])
        status_row: dict[str, Any] = {
            "match_seq_idx": int(match_seq_idx),
            "video_stroke_idx": int(v_stroke),
            "rp3_row_idx": int(rp3_idx),
            "rp3_stroke_number": int(rp3_stroke_number),
            "segment_exported": False,
            "segment_rows_written": 0,
            "drop_reason": "",
            "raw_area_trapz": float("nan"),
            "normalized_area_trapz": float("nan"),
        }
        if v_stroke not in events_by_stroke:
            status_row["drop_reason"] = "missing_events"
            status_rows.append(status_row)
            continue
        if not (0 <= rp3_idx < len(rp3_df)):
            status_row["drop_reason"] = "rp3_row_out_of_range"
            status_rows.append(status_row)
            continue

        ev = events_by_stroke[v_stroke]
        c_frame = int(ev["catch_frame_idx"])
        f_frame = int(ev["finish_frame_idx"])

        if use_rp3_finish:
            rp3_drive_s = float(mrow["rp3_drive_s"])
            catch_time = float(ev["catch_time_s"])
            rp3_finish_time = catch_time + rp3_drive_s
            time_col = "time_s_recomputed" if "time_s_recomputed" in merged.columns else "time_s"
            merged_times = pd.to_numeric(merged[time_col], errors="coerce")
            candidates = merged[
                (merged["frame_idx"] >= c_frame)
                & (merged_times <= rp3_finish_time + 1e-6)
            ]
            if not candidates.empty:
                f_frame = int(candidates["frame_idx"].iloc[-1])

        drive = merged[(merged["frame_idx"] >= c_frame) & (merged["frame_idx"] <= f_frame)].copy()
        if len(drive) < 2:
            status_row["drop_reason"] = "invalid_drive_window"
            status_rows.append(status_row)
            continue

        dist_col = "relative_axis_px_smooth" if "relative_axis_px_smooth" in drive.columns else "relative_axis_px"
        dist = pd.to_numeric(drive[dist_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(dist).sum() < 2:
            status_row["drop_reason"] = "invalid_distance_signal"
            status_rows.append(status_row)
            continue
        dist = _interp_nans(dist)

        x0 = float(ev["catch_distance_px"]) if "catch_distance_px" in ev.index else float(np.nanmin(dist))
        x1 = float(np.nanmax(dist))
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
            x0 = float(np.nanmin(dist))
            x1 = float(np.nanmax(dist))
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
            status_row["drop_reason"] = "invalid_drive_range"
            status_rows.append(status_row)
            continue

        s_video = (dist - x0) / (x1 - x0)
        s_video = np.clip(s_video, 0.0, 1.0)
        s_video = np.maximum.accumulate(s_video)
        if s_video[-1] > 1e-9:
            s_video = s_video / s_video[-1]

        rp3_row = rp3_df.iloc[rp3_idx]
        stroke_len = float(rp3_row["stroke_length"]) if np.isfinite(rp3_row["stroke_length"]) else float("nan")
        if not np.isfinite(stroke_len) or stroke_len <= 0:
            status_row["drop_reason"] = "invalid_stroke_length"
            status_rows.append(status_row)
            continue

        distances: list[float] = []
        force_raw: list[float] = []
        force_cols_used: list[str] = []
        for col, d_cm in force_cols:
            raw = rp3_row.get(col)
            f = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if not np.isfinite(f):
                continue
            if d_cm > stroke_len + 1e-6:
                continue
            distances.append(float(d_cm))
            force_raw.append(float(f))
            force_cols_used.append(col)

        if not distances:
            status_row["drop_reason"] = "no_valid_force_bins"
            status_rows.append(status_row)
            continue
        s_targets = np.asarray(distances, dtype=np.float64) / stroke_len
        s_targets = np.clip(s_targets, 0.0, 1.0)
        force_raw_arr = np.asarray(force_raw, dtype=np.float64)

        raw_area = float(np.trapz(force_raw_arr, s_targets))
        status_row["raw_area_trapz"] = raw_area
        if not np.isfinite(raw_area) or raw_area <= PDF_AREA_EPS:
            status_row["drop_reason"] = "zero_or_invalid_pdf_area"
            status_rows.append(status_row)
            continue
        force_pdf_arr = force_raw_arr / raw_area
        normalized_area = float(np.trapz(force_pdf_arr, s_targets))
        status_row["normalized_area_trapz"] = normalized_area
        if not np.isfinite(normalized_area):
            status_row["drop_reason"] = "invalid_normalized_pdf_area"
            status_rows.append(status_row)
            continue

        # -- Time array for time-domain derivatives --
        time_col = "time_s_recomputed" if "time_s_recomputed" in drive.columns else "time_s"
        time_arr = pd.to_numeric(drive[time_col], errors="coerce").to_numpy(dtype=np.float64)
        time_arr = _interp_nans(time_arr)

        # -- QC: NaN fraction across angle columns --
        qc_flags: list[str] = []
        angle_src_cols = list(side_map.values())
        nan_counts = sum(
            pd.to_numeric(drive[c], errors="coerce").isna().sum() for c in angle_src_cols
        )
        total_angle_vals = len(drive) * len(angle_src_cols)
        nan_frac = nan_counts / max(total_angle_vals, 1)
        status_row["nan_frac_angles"] = float(nan_frac)
        if nan_frac > MAX_NAN_FRAC_ANGLES:
            qc_flags.append("qc_tracking_sparse")

        # -- Savitzky-Golay smoothing + time-domain derivatives --
        smoothed_angles: dict[str, np.ndarray] = {}
        dtheta_dt_time: dict[str, np.ndarray] = {}
        max_angular_vel = 0.0
        for out_col, src_col in side_map.items():
            raw = pd.to_numeric(drive[src_col], errors="coerce").to_numpy(dtype=np.float64)
            smooth = _savgol_smooth(raw)
            smoothed_angles[out_col] = smooth
            if np.isfinite(smooth).sum() >= 2 and np.isfinite(time_arr).sum() >= 2:
                dt_arr = np.gradient(_interp_nans(smooth), time_arr)
                dtheta_dt_time[out_col] = dt_arr
                finite_dt = np.abs(dt_arr[np.isfinite(dt_arr)])
                if finite_dt.size > 0:
                    max_angular_vel = max(max_angular_vel, float(finite_dt.max()))
            else:
                dtheta_dt_time[out_col] = np.full_like(smooth, np.nan)

        status_row["max_deriv_deg_s"] = float(max_angular_vel)
        if max_angular_vel > MAX_ANGULAR_VEL_DEG_S:
            qc_flags.append("qc_nonphysio_deriv")

        # -- ds/dt in time domain --
        ds_dt_time = np.gradient(s_video, time_arr) if np.isfinite(time_arr).sum() >= 2 else np.full_like(s_video, np.nan)
        finite_ds_dt = ds_dt_time[np.isfinite(ds_dt_time)]
        ds_dt_min = float(finite_ds_dt.min()) if finite_ds_dt.size > 0 else float("nan")
        ds_dt_median = float(np.median(finite_ds_dt)) if finite_ds_dt.size > 0 else float("nan")
        status_row["ds_dt_min"] = ds_dt_min
        if np.isfinite(ds_dt_median) and ds_dt_median > 0 and np.isfinite(ds_dt_min):
            if ds_dt_min < DS_DT_STALL_FRAC * ds_dt_median:
                qc_flags.append("qc_ds_dt_stall")

        status_row["qc_flags"] = ",".join(qc_flags) if qc_flags else ""

        # -- Interpolate smoothed angles to s_targets --
        interp_features: dict[str, np.ndarray] = {}
        for out_col in side_map:
            interp_features[out_col] = _interp_feature_on_progress(s_video, smoothed_angles[out_col], s_targets)

        # -- Interpolate dtheta/dt and ds/dt to s_targets, then chain-rule --
        ds_dt_on_s = _interp_feature_on_progress(s_video, ds_dt_time, s_targets)

        deriv_features: dict[str, np.ndarray] = {}
        all_angle_keys = list(side_map.keys())
        for out_col in all_angle_keys:
            dtheta_dt_on_s = _interp_feature_on_progress(s_video, dtheta_dt_time[out_col], s_targets)
            dtheta_ds = dtheta_dt_on_s / (ds_dt_on_s + CHAIN_RULE_EPS)
            deriv_features[f"{out_col.replace('_deg', '')}_ddeg_ds"] = dtheta_ds

        # -- Optional second derivatives --
        second_deriv_features: dict[str, np.ndarray] = {}
        if include_second_derivatives:
            for out_col in all_angle_keys:
                d1_key = f"{out_col.replace('_deg', '')}_ddeg_ds"
                d1 = deriv_features[d1_key]
                if d1.size >= 2 and np.isfinite(d1).sum() >= 2:
                    second_deriv_features[f"{out_col.replace('_deg', '')}_d2deg_ds2"] = np.gradient(d1, s_targets)
                else:
                    second_deriv_features[f"{out_col.replace('_deg', '')}_d2deg_ds2"] = np.full_like(d1, np.nan)

        # -- Handle velocity/acceleration support features --
        vel_col = "velocity_axis_recomputed_px_s" if "velocity_axis_recomputed_px_s" in drive.columns else "velocity_axis_px_s"
        handle_vel_raw = pd.to_numeric(drive.get(vel_col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64) if vel_col in drive.columns else np.full(len(drive), np.nan)
        handle_vel_on_s = _interp_feature_on_progress(s_video, handle_vel_raw, s_targets)

        handle_accel_time = np.gradient(_interp_nans(handle_vel_raw), time_arr) if (np.isfinite(handle_vel_raw).sum() >= 2 and np.isfinite(time_arr).sum() >= 2) else np.full_like(handle_vel_raw, np.nan)
        handle_accel_on_s = _interp_feature_on_progress(s_video, handle_accel_time, s_targets)

        # -- Build output rows --
        for i in range(len(distances)):
            row = {
                "run_name": run_dir.name,
                "rp3_clean_csv": rp3_clean_csv.name,
                "active_side": active_side,
                "match_seq_idx": int(match_seq_idx),
                "video_stroke_idx": v_stroke,
                "rp3_row_idx": rp3_idx,
                "rp3_stroke_number": int(rp3_row["stroke_number"]),
                "video_catch_time_s": float(mrow["video_catch_time_s"]),
                "video_finish_time_s": float(mrow["video_finish_time_s"]),
                "video_drive_s": float(mrow["video_drive_s"]),
                "video_recover_s": float(mrow["video_recover_s"]),
                "video_cycle_s": float(mrow["video_cycle_s"]),
                "rp3_drive_s": float(mrow["rp3_drive_s"]),
                "rp3_recover_s": float(mrow["rp3_recover_s"]),
                "rp3_cycle_s": float(mrow["rp3_cycle_s"]),
                "cum_catch_err_s": float(mrow["cum_catch_err_s"]),
                "interval_err_s": float(mrow["interval_err_s"]),
                "rp3_rows_skipped_since_prev": int(mrow["rp3_rows_skipped_since_prev"]),
                "drive_bin_idx": i,
                "force_col": force_cols_used[i],
                "distance_cm": float(distances[i]),
                "stroke_length_cm": float(stroke_len),
                "s_force": float(s_targets[i]),
                "force_raw": float(force_raw_arr[i]),
                "force_n": float(force_pdf_arr[i]),
                "qc_flags": status_row["qc_flags"],
                "handle_velocity_px_s": float(handle_vel_on_s[i]) if np.isfinite(handle_vel_on_s[i]) else float("nan"),
                "handle_accel_px_s2": float(handle_accel_on_s[i]) if np.isfinite(handle_accel_on_s[i]) else float("nan"),
            }
            for key, arr in interp_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            for key, arr in deriv_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            for key, arr in second_deriv_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            rows.append(row)

        status_row["segment_exported"] = True
        status_row["segment_rows_written"] = int(len(distances))
        status_rows.append(status_row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["match_seq_idx", "distance_cm"]).reset_index(drop=True)
    status = pd.DataFrame(status_rows)
    if not status.empty:
        status = status.sort_values(["match_seq_idx"]).reset_index(drop=True)
    return out, status


def _validate_force_segment_exports(
    *,
    manifest_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> None:
    if len(status_df) != len(manifest_df):
        raise RuntimeError(
            f"Segment export status row mismatch: status_rows={len(status_df)} manifest_rows={len(manifest_df)}"
        )

    if "match_seq_idx" not in status_df.columns:
        raise RuntimeError("Segment export status is missing match_seq_idx.")
    seq = status_df["match_seq_idx"].to_numpy(dtype=np.int64)
    expected = np.arange(len(manifest_df), dtype=np.int64)
    if not np.array_equal(np.sort(seq), expected):
        raise RuntimeError("match_seq_idx must be unique and contiguous from 0..N-1 in export status.")

    exported_status = status_df[status_df["segment_exported"].astype(bool)]
    if not exported_status.empty:
        normalized = pd.to_numeric(exported_status["normalized_area_trapz"], errors="coerce")
        bad = (~np.isfinite(normalized.to_numpy(dtype=np.float64))) | (
            np.abs(normalized.to_numpy(dtype=np.float64) - 1.0) > PDF_AREA_TOL
        )
        if bool(np.any(bad)):
            raise RuntimeError("Exported stroke PDF normalization failed area-to-one invariant.")

    if segments_df.empty:
        return
    if "match_seq_idx" not in segments_df.columns:
        raise RuntimeError("Segments CSV missing match_seq_idx.")

    seg_keys = set(pd.to_numeric(segments_df["match_seq_idx"], errors="coerce").dropna().astype(int).tolist())
    exported_keys = set(pd.to_numeric(exported_status["match_seq_idx"], errors="coerce").dropna().astype(int).tolist())
    if not seg_keys.issubset(exported_keys):
        raise RuntimeError("Segments contain stroke keys that are not marked exported in status table.")

    seg_counts = segments_df.groupby("match_seq_idx").size().to_dict()
    for _, row in exported_status.iterrows():
        k = int(row["match_seq_idx"])
        expected_count = int(row["segment_rows_written"])
        actual_count = int(seg_counts.get(k, 0))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Segment row-count mismatch for match_seq_idx={k}: status={expected_count} actual={actual_count}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute drive event timings from stroke_signal.csv using only handle distance "
            "(catch=minima, finish=maxima) for downstream RP3 force-curve pairing."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Sports2D runs root (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory to process directly (skip interactive selection).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/inference).",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=None,
        help=(
            "Smoothing window (seconds) for relative distance signal. "
            "Default: 0.04 for velocity_calibrated, 0.08 otherwise."
        ),
    )
    parser.add_argument(
        "--min-cycle-s",
        type=float,
        default=0.8,
        help="Minimum time between consecutive catches (default: 0.8).",
    )
    parser.add_argument(
        "--min-drive-s",
        type=float,
        default=0.2,
        help="Minimum drive duration from catch to finish (default: 0.2).",
    )
    parser.add_argument(
        "--min-recover-s",
        type=float,
        default=0.2,
        help="Minimum recovery duration from finish to next catch (default: 0.2).",
    )
    parser.add_argument(
        "--min-drive-disp-frac",
        type=float,
        default=0.05,
        help="Minimum drive displacement as fraction of signal span (default: 0.05).",
    )
    parser.add_argument(
        "--slope-tol-frac",
        type=float,
        default=0.05,
        help="Flat-slope tolerance as fraction of slope std (default: 0.05).",
    )
    parser.add_argument(
        "--overlay-opacity",
        type=float,
        default=0.10,
        help="Opacity for full-frame drive overlay video (default: 0.10).",
    )
    parser.add_argument(
        "--overlay-video",
        action="store_true",
        help="Force writing the drive-phase overlay video.",
    )
    parser.add_argument(
        "--no-overlay-video",
        action="store_true",
        help="Skip writing the drive-phase overlay video.",
    )
    parser.add_argument(
        "--match-rp3",
        action="store_true",
        help="Build RP3 stroke matches and export per-2.2cm force/pose segment CSV.",
    )
    parser.add_argument(
        "--no-match-rp3",
        action="store_true",
        help="Skip RP3 matching and segment CSV export.",
    )
    parser.add_argument(
        "--rp3-dirty-csv",
        type=Path,
        default=None,
        help="Optional RP3 dirty CSV in <run>/rp3 to clean and use for matching.",
    )
    parser.add_argument(
        "--anchor-video-stroke-idx",
        type=int,
        default=1,
        help="Video stroke index to anchor matching from (default: 1).",
    )
    parser.add_argument(
        "--anchor-rp3-row-idx",
        type=int,
        default=None,
        help="RP3 row index anchor for the anchor video stroke.",
    )
    parser.add_argument(
        "--anchor-rp3-stroke-number",
        type=int,
        default=None,
        help="RP3 stroke_number anchor for the anchor video stroke (recommended).",
    )
    parser.add_argument(
        "--active-side",
        type=str,
        default=None,
        choices=["left", "right"],
        help="Active side to export canonical one-side features from.",
    )
    parser.add_argument(
        "--finish-velocity-frac",
        type=float,
        default=None,
        help=(
            "Finish velocity threshold as fraction of peak drive velocity. "
            "Default: 0.75 for velocity_calibrated, 0.85 for velocity_threshold."
        ),
    )
    parser.add_argument(
        "--catch-velocity-frac",
        type=float,
        default=None,
        help=(
            "Catch velocity threshold as fraction of peak drive velocity. "
            "Only used with velocity_calibrated. Default: 0.43 (or auto-calibrated from RP3)."
        ),
    )
    parser.add_argument(
        "--finish-method",
        type=str,
        default=FINISH_METHOD_VELOCITY_CALIBRATED,
        choices=sorted(VALID_FINISH_METHODS),
        help=f"Finish detection method (default: {FINISH_METHOD_VELOCITY_CALIBRATED}).",
    )
    parser.add_argument(
        "--use-rp3-finish",
        action="store_true",
        default=True,
        help="Override video finish with catch + rp3_drive_s for segment export (default: enabled).",
    )
    parser.add_argument(
        "--no-use-rp3-finish",
        action="store_true",
        help="Disable RP3 finish override for segment export.",
    )
    parser.add_argument("--max-jump-rows", type=int, default=10, help="Max RP3 row jump between matched strokes.")
    parser.add_argument("--max-interval-error-s", type=float, default=2.0)
    parser.add_argument("--max-cumulative-error-base-s", type=float, default=1.5)
    parser.add_argument("--max-cumulative-error-per-s", type=float, default=0.15)
    parser.add_argument("--max-abs-cum-error-s", type=float, default=4.0,
                        help="Hard cap on absolute cumulative timing error (default: 4.0s).")
    parser.add_argument("--w-drive", type=float, default=0.4)
    parser.add_argument("--w-recover", type=float, default=0.4)
    parser.add_argument("--w-interval", type=float, default=1.0)
    parser.add_argument("--w-cumulative", type=float, default=1.0)
    parser.add_argument("--w-skip", type=float, default=0.08)
    parser.add_argument(
        "--include-second-derivatives",
        action="store_true",
        default=False,
        help="Include d2theta/ds2 columns in segment export (ablation-gated, default off).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.overlay_video and args.no_overlay_video:
        print("Input error: use only one of --overlay-video or --no-overlay-video.")
        return 1
    if args.match_rp3 and args.no_match_rp3:
        print("Input error: use only one of --match-rp3 or --no-match-rp3.")
        return 1
    if args.anchor_rp3_row_idx is not None and args.anchor_rp3_stroke_number is not None:
        print("Input error: use only one of --anchor-rp3-row-idx or --anchor-rp3-stroke-number.")
        return 1

    use_rp3_finish = args.use_rp3_finish and not args.no_use_rp3_finish

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    selected_run_via_selector = args.run_dir is None
    try:
        run_dir = _resolve_run_dir(args.run_dir, args.runs_root)
    except Exception as exc:
        print(f"Run selection failed: {exc}")
        return 1
    has_dirty_rp3 = bool(_discover_run_rp3_dirty_csvs(run_dir))

    if args.overlay_video:
        write_overlay_video = True
    elif args.no_overlay_video:
        write_overlay_video = False
    elif selected_run_via_selector:
        write_overlay_video = _select_yes_no(
            "Write drive-phase overlay video?",
            default_no=True,
        )
    else:
        write_overlay_video = False

    if args.match_rp3:
        run_rp3_matching = True
    elif args.no_match_rp3:
        run_rp3_matching = False
    elif args.rp3_dirty_csv is not None or args.anchor_rp3_row_idx is not None or args.anchor_rp3_stroke_number is not None:
        run_rp3_matching = True
    else:
        run_rp3_matching = has_dirty_rp3

    if args.active_side is not None:
        active_side = str(args.active_side)
    elif run_rp3_matching and selected_run_via_selector:
        active_side = _prompt_choice(
            "Active side for unilateral features",
            options=["right", "left"],
            default="right",
        )
    else:
        active_side = "right"

    # ------------------------------------------------------------------
    # Resolve effective detection parameters
    # ------------------------------------------------------------------
    finish_method = str(args.finish_method)
    is_calibrated = finish_method == FINISH_METHOD_VELOCITY_CALIBRATED

    smooth_window_s: float = (
        float(args.smooth_window_s) if args.smooth_window_s is not None
        else (0.04 if is_calibrated else 0.08)
    )
    finish_velocity_frac: float = (
        float(args.finish_velocity_frac) if args.finish_velocity_frac is not None
        else (DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED if is_calibrated else 0.85)
    )
    catch_velocity_frac: float = (
        float(args.catch_velocity_frac) if args.catch_velocity_frac is not None
        else DEFAULT_CATCH_VELOCITY_FRAC
    )

    # ------------------------------------------------------------------
    # Shared detection kwargs
    # ------------------------------------------------------------------
    detect_kwargs: dict[str, Any] = dict(
        smooth_window_s=smooth_window_s,
        min_cycle_s=float(args.min_cycle_s),
        min_drive_s=float(args.min_drive_s),
        min_recover_s=float(args.min_recover_s),
        min_drive_disp_frac=float(args.min_drive_disp_frac),
        slope_tol_frac=float(args.slope_tol_frac),
        finish_velocity_frac=finish_velocity_frac,
        catch_velocity_frac=catch_velocity_frac,
    )

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    try:
        df = pd.read_csv(stroke_csv)
    except Exception as exc:
        print(f"Failed to read {stroke_csv}: {exc}")
        return 2

    # ------------------------------------------------------------------
    # Two-pass calibration flow when velocity_calibrated + RP3
    # ------------------------------------------------------------------
    calibration: CalibrationResult | None = None

    if is_calibrated and run_rp3_matching:
        try:
            rp3_dirty_csv_path_cal = _resolve_rp3_dirty_csv(
                run_dir=run_dir,
                rp3_dirty_csv=args.rp3_dirty_csv,
                interactive=interactive,
            )
            rp3_clean_csv_path_cal = _clean_rp3_dirty_csv(rp3_dirty_csv_path_cal)
            rp3_df_cal = _load_rp3_clean_csv(rp3_clean_csv_path_cal)
            anchor_rp3_idx_cal = _resolve_rp3_anchor_idx(
                rp3_df_cal,
                anchor_rp3_row_idx=args.anchor_rp3_row_idx,
                anchor_rp3_stroke_number=args.anchor_rp3_stroke_number,
                interactive=interactive,
            )

            coarse_detect = detect_drive_events(
                df,
                **{**detect_kwargs, "finish_method": FINISH_METHOD_VELOCITY_THRESHOLD},
            )
            if not coarse_detect.events:
                print("Pass-1 coarse detection found 0 drives; cannot calibrate.")
                return 2

            coarse_events_df = _events_to_dataframe(coarse_detect.events)
            match_cfg_cal = Rp3MatchConfig(
                max_jump_rows=int(args.max_jump_rows),
                max_interval_error_s=float(args.max_interval_error_s),
                max_cumulative_error_base_s=float(args.max_cumulative_error_base_s),
                max_cumulative_error_per_s=float(args.max_cumulative_error_per_s),
                max_abs_cum_error_s=float(args.max_abs_cum_error_s),
                w_drive=float(args.w_drive),
                w_recover=float(args.w_recover),
                w_interval=float(args.w_interval),
                w_cumulative=float(args.w_cumulative),
                w_skip=float(args.w_skip),
            )
            coarse_match = _build_rp3_match_manifest(
                video_df=coarse_events_df,
                rp3_df=rp3_df_cal,
                anchor_video_idx=int(args.anchor_video_stroke_idx),
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
                min_cycle_s=float(args.min_cycle_s),
                slope_tol_frac=float(args.slope_tol_frac),
            )

            if args.catch_velocity_frac is None:
                catch_velocity_frac = calibration.catch_velocity_frac
            if args.finish_velocity_frac is None:
                finish_velocity_frac = calibration.finish_velocity_frac
            detect_kwargs["catch_velocity_frac"] = catch_velocity_frac
            detect_kwargs["finish_velocity_frac"] = finish_velocity_frac

            print(
                f"Calibration: catch_frac={catch_velocity_frac:.3f} "
                f"finish_frac={finish_velocity_frac:.3f} "
                f"(MAE={calibration.mae_s * 1000:.1f}ms, "
                f"ME={calibration.me_s * 1000:.1f}ms, "
                f"n={calibration.n_strokes})"
            )
        except Exception as exc:
            print(f"Calibration failed ({exc}); using default fracs.")

    # ------------------------------------------------------------------
    # Final detection (Pass 2 if calibrated, or single-pass otherwise)
    # ------------------------------------------------------------------
    try:
        detection = detect_drive_events(
            df,
            **{**detect_kwargs, "finish_method": finish_method},
        )
    except Exception as exc:
        print(f"Failed to process {stroke_csv}: {exc}")
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "inference").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df = _events_to_dataframe(detection.events)
    frame_df = _build_frame_level_recomputed_columns(df, detection)

    events_csv = output_dir / "drive_events.csv"
    frame_csv = output_dir / "stroke_signal_with_drive_events.csv"
    summary_json = output_dir / "drive_events_summary.json"
    overlay_video = output_dir / "drive_phase_overlay.mp4"
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
        input_video = _find_input_video(run_dir)
        input_video_path = str(input_video)
        is_drive_flags = frame_df["is_drive_recomputed"].to_numpy(dtype=np.uint8)
        overlay_frames_written, overlay_drive_frames = _write_drive_overlay_video(
            input_video=input_video,
            is_drive_flags=is_drive_flags,
            out_video=overlay_video,
            alpha=float(args.overlay_opacity),
        )

    # ------------------------------------------------------------------
    # RP3 matching (final pass)
    # ------------------------------------------------------------------
    rp3_summary: dict[str, Any] | None = None
    rp3_dirty_csv_path: Path | None = None
    rp3_clean_csv_path: Path | None = None
    if run_rp3_matching:
        if events_df.empty:
            print("RP3 match failed: no detected drive events available for matching.")
            return 3
        try:
            rp3_dirty_csv_path = _resolve_rp3_dirty_csv(
                run_dir=run_dir,
                rp3_dirty_csv=args.rp3_dirty_csv,
                interactive=interactive,
            )
            rp3_clean_csv_path = _clean_rp3_dirty_csv(rp3_dirty_csv_path)
            rp3_df = _load_rp3_clean_csv(rp3_clean_csv_path)
            anchor_rp3_idx = _resolve_rp3_anchor_idx(
                rp3_df,
                anchor_rp3_row_idx=args.anchor_rp3_row_idx,
                anchor_rp3_stroke_number=args.anchor_rp3_stroke_number,
                interactive=interactive,
            )

            cum_err_base = float(args.max_cumulative_error_base_s)
            if calibration is not None:
                cum_err_base = max(cum_err_base, 2.0)
            match_cfg = Rp3MatchConfig(
                max_jump_rows=int(args.max_jump_rows),
                max_interval_error_s=float(args.max_interval_error_s),
                max_cumulative_error_base_s=cum_err_base,
                max_cumulative_error_per_s=float(args.max_cumulative_error_per_s),
                max_abs_cum_error_s=float(args.max_abs_cum_error_s),
                w_drive=float(args.w_drive),
                w_recover=float(args.w_recover),
                w_interval=float(args.w_interval),
                w_cumulative=float(args.w_cumulative),
                w_skip=float(args.w_skip),
            )

            match_result = _build_rp3_match_manifest(
                video_df=events_df,
                rp3_df=rp3_df,
                anchor_video_idx=int(args.anchor_video_stroke_idx),
                anchor_rp3_idx=int(anchor_rp3_idx),
                cfg=match_cfg,
            )
            manifest_df = match_result.manifest
            manifest_df.to_csv(rp3_manifest_csv, index=False)

            segments_df, segment_status_df = _build_force_pose_segments(
                run_dir=run_dir,
                frame_df=frame_df,
                events_df=events_df,
                manifest_df=manifest_df,
                rp3_df=rp3_df,
                active_side=active_side,
                rp3_clean_csv=rp3_clean_csv_path,
                use_rp3_finish=use_rp3_finish,
                include_second_derivatives=args.include_second_derivatives,
            )
            _validate_force_segment_exports(
                manifest_df=manifest_df,
                segments_df=segments_df,
                status_df=segment_status_df,
            )
            segments_df.to_csv(rp3_segments_csv, index=False)
            segment_status_df.to_csv(rp3_segment_status_csv, index=False)

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

            rp3_summary = {
                "run_dir": str(run_dir),
                "rp3_dirty_csv": str(rp3_dirty_csv_path),
                "rp3_clean_csv": str(rp3_clean_csv_path),
                "active_side": active_side,
                "anchor_video_stroke_idx": int(args.anchor_video_stroke_idx),
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
                },
            }
            with rp3_summary_json.open("w", encoding="utf-8") as f:
                json.dump(rp3_summary, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            print(f"RP3 match failed: {exc}")
            return 3

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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
            "min_cycle_s": float(args.min_cycle_s),
            "min_drive_s": float(args.min_drive_s),
            "min_recover_s": float(args.min_recover_s),
            "min_drive_disp_frac": float(args.min_drive_disp_frac),
            "slope_tol_frac": float(args.slope_tol_frac),
            "catch_velocity_frac": catch_velocity_frac,
            "finish_velocity_frac": finish_velocity_frac,
            "finish_method": finish_method,
            "use_rp3_finish": use_rp3_finish if run_rp3_matching else None,
            "calibration": cal_info,
        },
        "outputs": {
            "drive_events_csv": str(events_csv),
            "stroke_signal_with_drive_events_csv": str(frame_csv),
            "drive_phase_overlay_video": str(overlay_video) if write_overlay_video else None,
            "rp3_match_manifest_csv": str(rp3_manifest_csv) if rp3_summary is not None else None,
            "rp3_pose_force_segments_csv": str(rp3_segments_csv) if rp3_summary is not None else None,
            "rp3_pose_force_export_status_csv": str(rp3_segment_status_csv) if rp3_summary is not None else None,
            "rp3_match_summary_json": str(rp3_summary_json) if rp3_summary is not None else None,
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
    print(f"Outputs:")
    print(f"  {events_csv}")
    print(f"  {frame_csv}")
    if write_overlay_video:
        print(f"  {overlay_video}")
    if rp3_summary is not None:
        print(f"  {rp3_manifest_csv}")
        print(f"  {rp3_segments_csv}")
        print(f"  {rp3_segment_status_csv}")
        print(f"  {rp3_summary_json}")
    print(f"  {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
