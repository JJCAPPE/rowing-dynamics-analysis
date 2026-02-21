#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from match_rp3_cli import (
    MatchConfig as Rp3MatchConfig,
    _build_match_manifest as _build_rp3_match_manifest,
    _load_rp3 as _load_rp3_clean_csv,
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
FORCE_COL_RE = re.compile(r"^force_at_([0-9]+(?:\.[0-9]+)?)cm$")
PDF_AREA_EPS = 1e-9
PDF_AREA_TOL = 1e-6
_RP3_EXPAND_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class DriveEvent:
    stroke_idx: int
    catch_frame_idx: int
    catch_time_s: float
    finish_frame_idx: int
    finish_time_s: float
    next_catch_frame_idx: int
    next_catch_time_s: float
    drive_duration_s: float
    recover_duration_s: float
    cycle_duration_s: float
    catch_distance_px: float
    finish_distance_px: float
    drive_displacement_px: float


@dataclass(frozen=True)
class DetectionResult:
    events: list[DriveEvent]
    catch_candidates_raw: np.ndarray
    catches_filtered: np.ndarray
    signal_smooth_px: np.ndarray
    slope_px_s: np.ndarray
    fps_estimate: float
    min_drive_disp_px: float


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


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if signal.size < window:
        return signal.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _interp_nans(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    if values.size == 0:
        return values
    x = np.arange(values.size, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.all():
        return values
    if not mask.any():
        raise ValueError("Signal is all-NaN; cannot detect drive events.")
    values[~mask] = np.interp(x[~mask], x[mask], values[mask])
    return values


def _infer_time_s(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df.columns:
        time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(time_s).sum() >= 2:
            return _interp_nans(time_s)

    if "frame_idx" in df.columns:
        frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=np.float64)
        frame_idx = _interp_nans(frame_idx)
        start = float(frame_idx[0])
        return (frame_idx - start).astype(np.float64)

    return np.arange(len(df), dtype=np.float64)


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    out = signs.astype(np.int8, copy=True)
    if out.size == 0:
        return out

    for i in range(1, out.size):
        if out[i] == 0:
            out[i] = out[i - 1]
    for i in range(out.size - 2, -1, -1):
        if out[i] == 0:
            out[i] = out[i + 1]
    return out


def _find_catch_candidates(signs: np.ndarray) -> np.ndarray:
    catches: list[int] = []
    for i in range(1, signs.size):
        if signs[i - 1] < 0 and signs[i] > 0:
            catches.append(i)
    return np.asarray(catches, dtype=np.int32)


def _filter_catches_by_cycle(
    catches: np.ndarray,
    signal_smooth_px: np.ndarray,
    time_s: np.ndarray,
    min_cycle_s: float,
) -> np.ndarray:
    if catches.size <= 1:
        return catches

    kept: list[int] = [int(catches[0])]
    for c in catches[1:]:
        c = int(c)
        prev = kept[-1]
        if float(time_s[c] - time_s[prev]) >= min_cycle_s:
            kept.append(c)
            continue
        if float(signal_smooth_px[c]) < float(signal_smooth_px[prev]):
            kept[-1] = c
    return np.asarray(kept, dtype=np.int32)


def detect_drive_events(
    df: pd.DataFrame,
    *,
    smooth_window_s: float,
    min_cycle_s: float,
    min_drive_s: float,
    min_recover_s: float,
    min_drive_disp_frac: float,
    slope_tol_frac: float,
) -> DetectionResult:
    if "relative_axis_px" not in df.columns:
        raise ValueError("stroke_signal.csv must contain 'relative_axis_px'.")

    time_s = _infer_time_s(df)
    if time_s.size < 3:
        raise ValueError("Need at least 3 samples to detect catch/finish events.")

    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    fps_estimate = float(1.0 / np.median(valid_dt)) if valid_dt.size else 30.0

    signal_raw = pd.to_numeric(df["relative_axis_px"], errors="coerce").to_numpy(dtype=np.float64)
    signal_raw = _interp_nans(signal_raw)

    window_frames = max(3, int(round(max(0.01, smooth_window_s) * fps_estimate)))
    signal_smooth_px = _moving_average(signal_raw, window=window_frames)
    slope_px_s = np.gradient(signal_smooth_px, time_s)

    slope_std = float(np.nanstd(slope_px_s))
    slope_tol = max(1e-6, slope_std * max(0.0, slope_tol_frac))
    signs = np.zeros_like(slope_px_s, dtype=np.int8)
    signs[slope_px_s > slope_tol] = 1
    signs[slope_px_s < -slope_tol] = -1
    signs = _fill_zero_signs(signs)

    catches_raw = _find_catch_candidates(signs)
    catches_filtered = _filter_catches_by_cycle(
        catches=catches_raw,
        signal_smooth_px=signal_smooth_px,
        time_s=time_s,
        min_cycle_s=max(0.0, min_cycle_s),
    )

    p95 = float(np.nanpercentile(signal_smooth_px, 95.0))
    p05 = float(np.nanpercentile(signal_smooth_px, 5.0))
    signal_span_px = max(0.0, p95 - p05)
    min_drive_disp_px = max(1.0, signal_span_px * max(0.0, min_drive_disp_frac))

    events: list[DriveEvent] = []
    if catches_filtered.size >= 2:
        for i in range(catches_filtered.size - 1):
            c0 = int(catches_filtered[i])
            c1 = int(catches_filtered[i + 1])
            if c1 <= c0 + 1:
                continue

            segment = signal_smooth_px[c0 : c1 + 1]
            finish_rel = int(np.argmax(segment))
            f0 = int(c0 + finish_rel)
            if not (c0 < f0 < c1):
                continue

            drive_duration_s = float(time_s[f0] - time_s[c0])
            recover_duration_s = float(time_s[c1] - time_s[f0])
            cycle_duration_s = float(time_s[c1] - time_s[c0])
            drive_disp_px = float(signal_smooth_px[f0] - signal_smooth_px[c0])

            if drive_duration_s < min_drive_s:
                continue
            if recover_duration_s < min_recover_s:
                continue
            if drive_disp_px < min_drive_disp_px:
                continue

            events.append(
                DriveEvent(
                    stroke_idx=len(events),
                    catch_frame_idx=int(df.iloc[c0]["frame_idx"]) if "frame_idx" in df.columns else c0,
                    catch_time_s=float(time_s[c0]),
                    finish_frame_idx=int(df.iloc[f0]["frame_idx"]) if "frame_idx" in df.columns else f0,
                    finish_time_s=float(time_s[f0]),
                    next_catch_frame_idx=int(df.iloc[c1]["frame_idx"]) if "frame_idx" in df.columns else c1,
                    next_catch_time_s=float(time_s[c1]),
                    drive_duration_s=drive_duration_s,
                    recover_duration_s=recover_duration_s,
                    cycle_duration_s=cycle_duration_s,
                    catch_distance_px=float(signal_smooth_px[c0]),
                    finish_distance_px=float(signal_smooth_px[f0]),
                    drive_displacement_px=drive_disp_px,
                )
            )

    return DetectionResult(
        events=events,
        catch_candidates_raw=catches_raw,
        catches_filtered=catches_filtered,
        signal_smooth_px=signal_smooth_px.astype(np.float32),
        slope_px_s=slope_px_s.astype(np.float32),
        fps_estimate=fps_estimate,
        min_drive_disp_px=min_drive_disp_px,
    )


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
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory not found: {input_dir}")

    candidates = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(f"No input video found in: {input_dir}")
    return candidates[0]


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


def _build_force_pose_segments(
    *,
    run_dir: Path,
    frame_df: pd.DataFrame,
    events_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    rp3_df: pd.DataFrame,
    active_side: str,
    rp3_clean_csv: Path,
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
        x1 = float(ev["finish_distance_px"]) if "finish_distance_px" in ev.index else float(np.nanmax(dist))
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

        interp_features: dict[str, np.ndarray] = {}
        for out_col, src_col in side_map.items():
            values = pd.to_numeric(drive[src_col], errors="coerce").to_numpy(dtype=np.float64)
            interp_features[out_col] = _interp_feature_on_progress(s_video, values, s_targets)

        deriv_features: dict[str, np.ndarray] = {}
        for out_col in ["knee_active_deg", "hip_active_deg", "elbow_active_deg", "trunk_vs_horizontal_deg"]:
            vals = interp_features[out_col]
            if vals.size < 2:
                deriv_features[f"{out_col.replace('_deg', '')}_ddeg_ds"] = np.full_like(vals, np.nan)
                continue
            deriv_features[f"{out_col.replace('_deg', '')}_ddeg_ds"] = np.gradient(vals, s_targets)

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
            }
            for key, arr in interp_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            for key, arr in deriv_features.items():
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
        default=0.08,
        help="Smoothing window (seconds) for relative distance signal (default: 0.08).",
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
        default=0,
        help="Video stroke index to anchor matching from (default: 0).",
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
    parser.add_argument("--max-jump-rows", type=int, default=6, help="Max RP3 row jump between matched strokes.")
    parser.add_argument("--max-interval-error-s", type=float, default=1.2)
    parser.add_argument("--max-cumulative-error-base-s", type=float, default=0.8)
    parser.add_argument("--max-cumulative-error-per-s", type=float, default=0.08)
    parser.add_argument("--w-drive", type=float, default=0.4)
    parser.add_argument("--w-recover", type=float, default=0.4)
    parser.add_argument("--w-interval", type=float, default=1.0)
    parser.add_argument("--w-cumulative", type=float, default=1.0)
    parser.add_argument("--w-skip", type=float, default=0.08)
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

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    try:
        df = pd.read_csv(stroke_csv)
        detection = detect_drive_events(
            df,
            smooth_window_s=float(args.smooth_window_s),
            min_cycle_s=float(args.min_cycle_s),
            min_drive_s=float(args.min_drive_s),
            min_recover_s=float(args.min_recover_s),
            min_drive_disp_frac=float(args.min_drive_disp_frac),
            slope_tol_frac=float(args.slope_tol_frac),
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

            if args.anchor_rp3_row_idx is not None:
                anchor_rp3_idx = int(args.anchor_rp3_row_idx)
                if not (0 <= anchor_rp3_idx < len(rp3_df)):
                    raise ValueError(f"anchor_rp3_row_idx out of range: {anchor_rp3_idx}")
            elif args.anchor_rp3_stroke_number is not None:
                stroke_no = int(args.anchor_rp3_stroke_number)
                hit = rp3_df.index[rp3_df["stroke_number"].astype(int) == stroke_no].to_numpy()
                if hit.size == 0:
                    raise ValueError(f"anchor_rp3_stroke_number {stroke_no} not found in RP3 CSV.")
                anchor_rp3_idx = int(hit[0])
            elif interactive:
                min_stroke = int(rp3_df["stroke_number"].min())
                stroke_no = _prompt_int(
                    "Anchor RP3 stroke_number for first matched video stroke",
                    default=min_stroke,
                )
                hit = rp3_df.index[rp3_df["stroke_number"].astype(int) == stroke_no].to_numpy()
                if hit.size == 0:
                    raise ValueError(f"anchor_rp3_stroke_number {stroke_no} not found in RP3 CSV.")
                anchor_rp3_idx = int(hit[0])
            else:
                raise ValueError(
                    "Missing anchor. Provide --anchor-rp3-stroke-number (recommended) or --anchor-rp3-row-idx."
                )

            match_cfg = Rp3MatchConfig(
                max_jump_rows=int(args.max_jump_rows),
                max_interval_error_s=float(args.max_interval_error_s),
                max_cumulative_error_base_s=float(args.max_cumulative_error_base_s),
                max_cumulative_error_per_s=float(args.max_cumulative_error_per_s),
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

    inferred_time = _infer_time_s(df)
    if inferred_time.size:
        t0 = float(inferred_time[0])
        t1 = float(inferred_time[-1])
    else:
        t0 = 0.0
        t1 = 0.0

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
            "smooth_window_s": float(args.smooth_window_s),
            "min_cycle_s": float(args.min_cycle_s),
            "min_drive_s": float(args.min_drive_s),
            "min_recover_s": float(args.min_recover_s),
            "min_drive_disp_frac": float(args.min_drive_disp_frac),
            "slope_tol_frac": float(args.slope_tol_frac),
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
