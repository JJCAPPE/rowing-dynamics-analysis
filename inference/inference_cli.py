#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "sports2d_app" / "runs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference" / "outputs"


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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        run_dir = _resolve_run_dir(args.run_dir, args.runs_root)
    except Exception as exc:
        print(f"Run selection failed: {exc}")
        return 1

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

    output_dir = args.output_root.expanduser().resolve() / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df = _events_to_dataframe(detection.events)
    frame_df = _build_frame_level_recomputed_columns(df, detection)

    events_csv = output_dir / "drive_events.csv"
    frame_csv = output_dir / "stroke_signal_with_drive_events.csv"
    summary_json = output_dir / "drive_events_summary.json"

    events_df.to_csv(events_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)

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
        },
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
    print(f"  {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
