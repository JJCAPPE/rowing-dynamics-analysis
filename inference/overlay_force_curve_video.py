#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
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


def _pick_with_curses(options: Sequence[Path], title: str) -> Path:
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


def _pick_with_prompt(options: Sequence[Path], title: str) -> Path:
    if not options:
        raise ValueError("No options available.")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return options[0]

    print(f"\n{title}")
    for i, p in enumerate(options, start=1):
        print(f"  {i:2d}. {p.name}")
    while True:
        raw = input("Select number [1]: ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


def _pick_path(options: Sequence[Path], title: str) -> Path:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _pick_with_curses(options, title)
        except Exception:
            pass
    return _pick_with_prompt(options, title)


def _discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    runs: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        inf = p / "inference"
        if (inf / "stroke_signal_with_drive_events.csv").exists() and (
            inf / "rp3_pose_force_matched_segments.csv"
        ).exists():
            runs.append(p.resolve())
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _resolve_run_dir(run_dir: Path | None, runs_root: Path) -> Path:
    if run_dir is not None:
        p = run_dir.expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Run directory not found: {p}")
        return p
    runs = _discover_runs(runs_root)
    if not runs:
        raise FileNotFoundError(
            f"No runs with inference/stroke_signal_with_drive_events.csv and "
            f"inference/rp3_pose_force_matched_segments.csv under {runs_root}"
        )
    return _pick_path(runs, "Select run for force overlay video")


def _find_input_video(run_dir: Path) -> Path:
    input_dir = run_dir / "input"
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    vids = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES])
    if not vids:
        raise FileNotFoundError(f"No input video found in: {input_dir}")
    return vids[0]


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    text = str(value).strip()
    if text == "":
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _fmt_seconds(seconds: float | None, decimals: int = 2) -> str:
    if seconds is None or math.isnan(seconds):
        return "-"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if decimals == 0:
        return f"{minutes:02d}:{int(round(rem)):02d}"
    width = 3 + decimals
    return f"{minutes:02d}:{rem:0{width}.{decimals}f}"


def _fmt_number(value: float | None, suffix: str = "", precision: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{precision}f}{suffix}"


def _ensure_path_in_dir(path: Path, parent_dir: Path, *, label: str) -> None:
    parent_dir = parent_dir.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        path.relative_to(parent_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent_dir}: {path}") from exc


def _curve_panel_rect(width: int, height: int) -> tuple[int, int, int, int]:
    w = int(width * 0.42)
    h = int(height * 0.34)
    x0 = max(0, width - w - 18)
    y0 = 18
    return x0, y0, w, h


def _metrics_panel_rect(width: int, height: int, curve_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, w, h_curve = curve_rect
    gap = 10
    y1 = y0 + h_curve + gap
    h = max(120, min(int(height * 0.55), height - y1 - 18))
    return x0, y1, w, h


def _resolve_rp3_clean_csv(
    *,
    run_dir: Path,
    seg_df: pd.DataFrame,
    rp3_clean_csv: Path | None,
) -> Path:
    rp3_dir = (run_dir / "rp3").resolve()
    if rp3_clean_csv is not None:
        p = rp3_clean_csv.expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"RP3 clean CSV not found: {p}")
        _ensure_path_in_dir(p, rp3_dir, label="--rp3-clean-csv")
        return p

    if "rp3_clean_csv" not in seg_df.columns:
        raise ValueError(
            "rp3_pose_force_matched_segments.csv has no rp3_clean_csv column. "
            "Provide --rp3-clean-csv."
        )

    vals = [str(v).strip() for v in seg_df["rp3_clean_csv"].dropna().unique().tolist() if str(v).strip()]
    if not vals:
        raise ValueError("No rp3_clean_csv value found in segments CSV. Provide --rp3-clean-csv.")

    candidate = vals[0]
    p = Path(candidate)
    if p.is_absolute() and p.exists() and p.is_file():
        return p.resolve()
    p2 = (rp3_dir / p).resolve()
    if p2.exists() and p2.is_file():
        _ensure_path_in_dir(p2, rp3_dir, label="segments rp3_clean_csv")
        return p2
    p3 = (rp3_dir / p.name).resolve()
    if p3.exists() and p3.is_file():
        _ensure_path_in_dir(p3, rp3_dir, label="segments rp3_clean_csv")
        return p3
    raise FileNotFoundError(
        f"Could not resolve RP3 clean CSV from segments value '{candidate}'. "
        f"Expected it under {rp3_dir} or as an absolute existing path. "
        "Use --rp3-clean-csv."
    )


def _build_stroke_metrics(
    *,
    rp3_row: pd.Series | None,
    avg_force_pos_cm: float | None,
    stroke_label: str,
) -> dict[str, str]:
    if rp3_row is None:
        return {
            "stroke_no": stroke_label,
            "time": "-",
            "distance": "-",
            "stroke_rate": "-",
            "split": "-",
            "power": "-",
            "stroke_length": "-",
            "energy_per_stroke": "-",
            "peak_force": "-",
            "peak_force_pos": "-",
            "avg_force_pos": _fmt_number(avg_force_pos_cm, " cm", precision=1),
            "avg_force_pos_rel": "-",
            "rel_peak_pos": "-",
            "drive_time": "-",
            "recover_time": "-",
            "avg_calc_power": "-",
        }

    distance = _to_float(rp3_row.get("distance"))
    stroke_rate = _to_float(rp3_row.get("stroke_rate"))
    split_s = _to_float(rp3_row.get("estimated_500m_time"))
    power = _to_float(rp3_row.get("power"))
    stroke_length_cm = _to_float(rp3_row.get("stroke_length"))
    energy = _to_float(rp3_row.get("energy_per_stroke"))
    peak_force = _to_float(rp3_row.get("peak_force"))
    peak_pos = _to_float(rp3_row.get("peak_force_pos"))
    rel_peak = _to_float(rp3_row.get("rel_peak_force_pos"))
    drive = _to_float(rp3_row.get("drive_time"))
    recover = _to_float(rp3_row.get("recover_time"))
    avg_calc_power = _to_float(rp3_row.get("avg_calculated_power"))

    avg_force_pos_rel = None
    if avg_force_pos_cm is not None and stroke_length_cm is not None and stroke_length_cm > 0:
        avg_force_pos_rel = (avg_force_pos_cm / stroke_length_cm) * 100.0

    return {
        "stroke_no": stroke_label,
        "time": _fmt_seconds(_to_float(rp3_row.get("time")), decimals=2),
        "distance": _fmt_number(distance, " m", precision=1),
        "stroke_rate": _fmt_number(stroke_rate, " s/m", precision=1),
        "split": f"{_fmt_seconds(split_s, decimals=2)}/500m" if split_s is not None else "-",
        "power": _fmt_number(power, " W", precision=0),
        "stroke_length": _fmt_number((stroke_length_cm / 100.0) if stroke_length_cm is not None else None, " m", precision=2),
        "energy_per_stroke": _fmt_number(energy, " J", precision=1),
        "peak_force": _fmt_number(peak_force, " N", precision=0),
        "peak_force_pos": _fmt_number(peak_pos, " cm", precision=1),
        "avg_force_pos": _fmt_number(avg_force_pos_cm, " cm", precision=1),
        "avg_force_pos_rel": _fmt_number(avg_force_pos_rel, " %", precision=1),
        "rel_peak_pos": _fmt_number((rel_peak * 100.0) if rel_peak is not None else None, " %", precision=1),
        "drive_time": _fmt_number(drive, " s", precision=2),
        "recover_time": _fmt_number(recover, " s", precision=2),
        "avg_calc_power": _fmt_number(avg_calc_power, " W", precision=0),
    }


def _draw_metrics_panel(
    frame: np.ndarray,
    metrics: dict[str, str],
    rect: tuple[int, int, int, int],
    *,
    panel_alpha: float,
) -> None:
    x0, y0, w, h = rect

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (8, 8, 8), thickness=-1)
    blend_alpha = float(np.clip(panel_alpha, 0.0, 1.0))
    frame[:] = cv2.addWeighted(overlay, blend_alpha, frame, 1.0 - blend_alpha, 0.0)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (160, 160, 160), 1)

    cv2.putText(frame, "RP3 Metrics", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

    left_specs = [
        ("stroke_no", "Stroke"),
        ("time", "Time"),
        ("distance", "Distance"),
        ("stroke_rate", "Stroke rate"),
        ("split", "Split"),
        ("power", "Power"),
        ("stroke_length", "Stroke length"),
    ]
    right_specs = [
        ("energy_per_stroke", "Energy/stroke"),
        ("peak_force", "Peak force"),
        ("peak_force_pos", "Peak force pos"),
        ("avg_force_pos", "Avg force pos"),
        ("avg_force_pos_rel", "Avg force pos rel"),
        ("rel_peak_pos", "Rel. peak pos"),
        ("drive_time", "Drive time"),
        ("recover_time", "Recover time"),
        ("avg_calc_power", "Avg calc power"),
    ]

    rows = max(len(left_specs), len(right_specs))
    y_start = y0 + 42
    line_h = max(15, min(24, int((h - 52) / max(1, rows))))
    left_x = x0 + 10
    right_x = x0 + (w // 2) + 6
    text_color = (220, 220, 220)

    for i in range(rows):
        y = y_start + i * line_h
        if y > y0 + h - 6:
            break
        if i < len(left_specs):
            key, label = left_specs[i]
            text = f"{label}: {metrics.get(key, '-')}"
            cv2.putText(frame, text, (left_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
        if i < len(right_specs):
            key, label = right_specs[i]
            text = f"{label}: {metrics.get(key, '-')}"
            cv2.putText(frame, text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)


def _curve_points(
    s_vals: np.ndarray,
    f_vals: np.ndarray,
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    y_max: float,
) -> np.ndarray:
    if s_vals.size == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)
    sx = np.clip(s_vals, 0.0, 1.0)
    fy = np.clip(f_vals, 0.0, y_max)
    px = x0 + 8 + (sx * (w - 16))
    py = y0 + h - 8 - ((fy / max(y_max, 1e-9)) * (h - 16))
    pts = np.stack([px, py], axis=1).astype(np.int32).reshape((-1, 1, 2))
    return pts


def build_overlay_video(
    *,
    run_dir: Path,
    input_video: Path,
    stroke_csv: Path,
    segments_csv: Path,
    rp3_clean_csv: Path | None,
    output_video: Path,
    panel_alpha: float,
) -> tuple[int, int]:
    frame_df = pd.read_csv(stroke_csv)
    seg_df = pd.read_csv(segments_csv)
    if frame_df.empty:
        raise ValueError(f"Empty stroke signal file: {stroke_csv}")
    if seg_df.empty:
        raise ValueError(f"Empty segments file: {segments_csv}")

    required_frame_cols = {"stroke_idx_recomputed", "stroke_phase_recomputed", "is_drive_recomputed"}
    missing_frame_cols = [c for c in required_frame_cols if c not in frame_df.columns]
    if missing_frame_cols:
        raise ValueError(f"stroke_signal_with_drive_events.csv missing columns: {missing_frame_cols}")

    required_seg_cols = {"video_stroke_idx", "s_force", "rp3_row_idx"}
    missing_seg_cols = [c for c in required_seg_cols if c not in seg_df.columns]
    if missing_seg_cols:
        raise ValueError(f"rp3_pose_force_matched_segments.csv missing columns: {missing_seg_cols}")
    if "force_raw" in seg_df.columns:
        force_col = "force_raw"
    elif "force_n" in seg_df.columns:
        force_col = "force_n"
    else:
        raise ValueError("rp3_pose_force_matched_segments.csv missing both force_raw and force_n columns.")

    rp3_csv_path = _resolve_rp3_clean_csv(
        run_dir=run_dir,
        seg_df=seg_df,
        rp3_clean_csv=rp3_clean_csv,
    )
    rp3_df = pd.read_csv(rp3_csv_path)
    if rp3_df.empty:
        raise ValueError(f"Empty RP3 CSV: {rp3_csv_path}")
    rp3_df = rp3_df.reset_index(drop=True)
    rp3_df["rp3_row_idx"] = np.arange(len(rp3_df), dtype=np.int32)

    seg_df["video_stroke_idx"] = pd.to_numeric(seg_df["video_stroke_idx"], errors="coerce").astype("Int64")
    seg_df["rp3_row_idx"] = pd.to_numeric(seg_df["rp3_row_idx"], errors="coerce").astype("Int64")
    seg_df["s_force"] = _safe_numeric(seg_df["s_force"])
    seg_df[force_col] = _safe_numeric(seg_df[force_col])
    seg_df = seg_df.dropna(subset=["video_stroke_idx", "rp3_row_idx", "s_force", force_col]).copy()
    seg_df["video_stroke_idx"] = seg_df["video_stroke_idx"].astype(int)
    seg_df["rp3_row_idx"] = seg_df["rp3_row_idx"].astype(int)
    if "distance_cm" in seg_df.columns:
        seg_df["distance_cm"] = _safe_numeric(seg_df["distance_cm"])

    stroke_ids = sorted(int(v) for v in seg_df["video_stroke_idx"].dropna().unique().tolist())
    total_strokes = len(stroke_ids)
    curves: dict[int, dict[str, Any]] = {}
    for ord_idx, stroke_idx in enumerate(stroke_ids):
        grp = seg_df[seg_df["video_stroke_idx"] == stroke_idx].copy()
        g = grp.sort_values("s_force")
        s = g["s_force"].to_numpy(dtype=np.float64)
        f = g[force_col].to_numpy(dtype=np.float64)
        if len(s) < 2:
            continue
        if "distance_cm" in g.columns:
            d = g["distance_cm"].to_numpy(dtype=np.float64)
        else:
            d = s * 100.0

        rp3_row_idx = int(g["rp3_row_idx"].iloc[0])
        rp3_row = rp3_df.iloc[rp3_row_idx] if 0 <= rp3_row_idx < len(rp3_df) else None
        total_force = float(np.sum(f))
        avg_force_pos = float(np.sum(d * f) / total_force) if total_force > 1e-9 else None

        stroke_no = _to_float(rp3_row.get("stroke_number")) if rp3_row is not None else None
        if stroke_no is None:
            stroke_label = f"{stroke_idx} ({ord_idx + 1}/{total_strokes})"
        else:
            stroke_label = f"{int(round(stroke_no))} ({ord_idx + 1}/{total_strokes})"

        metrics = _build_stroke_metrics(
            rp3_row=rp3_row,
            avg_force_pos_cm=avg_force_pos,
            stroke_label=stroke_label,
        )
        peak_pos_cm = _to_float(rp3_row.get("peak_force_pos")) if rp3_row is not None else None
        stroke_length_cm = _to_float(rp3_row.get("stroke_length")) if rp3_row is not None else None
        if stroke_length_cm is None or stroke_length_cm <= 0:
            stroke_length_cm = float(np.nanmax(d)) if np.isfinite(d).any() else None

        curves[int(stroke_idx)] = {
            "s": s,
            "f": f,
            "d": d,
            "metrics": metrics,
            "peak_pos_cm": peak_pos_cm,
            "avg_force_pos_cm": avg_force_pos,
            "stroke_length_cm": stroke_length_cm,
        }
    if not curves:
        raise ValueError("No valid per-stroke force curves found in segment CSV.")

    global_max_force = max(float(np.nanmax(data["f"])) for data in curves.values())
    global_max_force = max(10.0, global_max_force * 1.05)

    stroke_idx_arr = _safe_numeric(frame_df["stroke_idx_recomputed"])
    phase_arr = _safe_numeric(frame_df["stroke_phase_recomputed"])
    is_drive_arr = _safe_numeric(frame_df["is_drive_recomputed"])

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create writer for: {output_video}")

    curve_rect = _curve_panel_rect(width, height)
    metrics_rect = _metrics_panel_rect(width, height, curve_rect)
    x0, y0, w, h = curve_rect
    panel_alpha = float(np.clip(panel_alpha, 0.0, 1.0))

    processed = 0
    overlay_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx = processed
            if idx >= len(frame_df):
                writer.write(frame)
                processed += 1
                continue

            stroke_idx_val = stroke_idx_arr[idx]
            phase_val = phase_arr[idx]
            drive_flag = bool(is_drive_arr[idx] >= 0.5) if np.isfinite(is_drive_arr[idx]) else False

            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (8, 8, 8), thickness=-1)
            frame = cv2.addWeighted(overlay, panel_alpha, frame, 1.0 - panel_alpha, 0.0)
            cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (160, 160, 160), 1)

            cv2.line(frame, (x0 + 8, y0 + h - 8), (x0 + w - 8, y0 + h - 8), (130, 130, 130), 1)
            cv2.line(frame, (x0 + 8, y0 + h - 8), (x0 + 8, y0 + 8), (130, 130, 130), 1)

            title = "Force Curve (drive-synced)" if force_col == "force_raw" else "Force PDF (drive-synced)"
            cv2.putText(frame, title, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

            if np.isfinite(stroke_idx_val):
                stroke_idx = int(round(float(stroke_idx_val)))
            else:
                stroke_idx = -1

            if stroke_idx in curves:
                stroke_data = curves[stroke_idx]
                s_vals = stroke_data["s"]
                f_vals = stroke_data["f"]
                pts = _curve_points(s_vals, f_vals, x0=x0, y0=y0, w=w, h=h, y_max=global_max_force)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], False, (40, 205, 245), 2, cv2.LINE_AA)

                stroke_len = stroke_data["stroke_length_cm"]
                peak_pos_cm = stroke_data["peak_pos_cm"]
                avg_pos_cm = stroke_data["avg_force_pos_cm"]
                if stroke_len is not None and stroke_len > 1e-9:
                    if peak_pos_cm is not None and np.isfinite(peak_pos_cm):
                        s_peak = float(np.clip(float(peak_pos_cm) / float(stroke_len), 0.0, 1.0))
                        peak_pts = _curve_points(
                            np.asarray([s_peak, s_peak], dtype=np.float64),
                            np.asarray([0.0, global_max_force], dtype=np.float64),
                            x0=x0,
                            y0=y0,
                            w=w,
                            h=h,
                            y_max=global_max_force,
                        )
                        if len(peak_pts) == 2:
                            cv2.line(
                                frame,
                                (int(peak_pts[0, 0, 0]), int(peak_pts[0, 0, 1])),
                                (int(peak_pts[1, 0, 0]), int(peak_pts[1, 0, 1])),
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    if avg_pos_cm is not None and np.isfinite(avg_pos_cm):
                        s_avg = float(np.clip(float(avg_pos_cm) / float(stroke_len), 0.0, 1.0))
                        avg_pts = _curve_points(
                            np.asarray([s_avg, s_avg], dtype=np.float64),
                            np.asarray([0.0, global_max_force], dtype=np.float64),
                            x0=x0,
                            y0=y0,
                            w=w,
                            h=h,
                            y_max=global_max_force,
                        )
                        if len(avg_pts) == 2:
                            cv2.line(
                                frame,
                                (int(avg_pts[0, 0, 0]), int(avg_pts[0, 0, 1])),
                                (int(avg_pts[1, 0, 0]), int(avg_pts[1, 0, 1])),
                                (95, 95, 95),
                                1,
                                cv2.LINE_AA,
                            )

                if drive_flag and np.isfinite(phase_val):
                    s_prog = float(np.clip(float(phase_val) * 2.0, 0.0, 1.0))
                    y_prog = float(np.interp(s_prog, s_vals, f_vals))

                    done_mask = s_vals <= s_prog
                    if int(done_mask.sum()) >= 2:
                        done_pts = _curve_points(
                            s_vals[done_mask],
                            f_vals[done_mask],
                            x0=x0,
                            y0=y0,
                            w=w,
                            h=h,
                            y_max=global_max_force,
                        )
                        cv2.polylines(frame, [done_pts], False, (60, 255, 120), 3, cv2.LINE_AA)

                    cur_pt = _curve_points(
                        np.asarray([s_prog], dtype=np.float64),
                        np.asarray([y_prog], dtype=np.float64),
                        x0=x0,
                        y0=y0,
                        w=w,
                        h=h,
                        y_max=global_max_force,
                    )
                    if len(cur_pt) == 1:
                        cx, cy = int(cur_pt[0, 0, 0]), int(cur_pt[0, 0, 1])
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                        cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)

                    if force_col == "force_raw":
                        txt = f"stroke {stroke_idx}  drive {s_prog*100:.1f}%  F={y_prog:.0f}N"
                    else:
                        txt = f"stroke {stroke_idx}  drive {s_prog*100:.1f}%  density={y_prog:.3f}"
                    cv2.putText(frame, txt, (x0 + 10, y0 + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                    overlay_frames += 1
                else:
                    txt = f"stroke {stroke_idx}  recovery"
                    cv2.putText(frame, txt, (x0 + 10, y0 + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

                _draw_metrics_panel(
                    frame,
                    stroke_data["metrics"],
                    metrics_rect,
                    panel_alpha=panel_alpha,
                )
            else:
                cv2.putText(
                    frame,
                    "No matched force curve for this stroke" if force_col == "force_raw" else "No matched force PDF for this stroke",
                    (x0 + 10, y0 + h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                _draw_metrics_panel(
                    frame,
                    _build_stroke_metrics(rp3_row=None, avg_force_pos_cm=None, stroke_label=f"{stroke_idx}"),
                    metrics_rect,
                    panel_alpha=panel_alpha,
                )

            writer.write(frame)
            processed += 1
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raise RuntimeError("No frames processed from input video.")
    return processed, overlay_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a run video with per-stroke RP3 force curve overlay synchronized to drive progression."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--stroke-csv", type=Path, default=None, help="Default: <run>/inference/stroke_signal_with_drive_events.csv")
    parser.add_argument("--segments-csv", type=Path, default=None, help="Default: <run>/inference/rp3_pose_force_matched_segments.csv")
    parser.add_argument("--rp3-clean-csv", type=Path, default=None, help="Optional RP3 clean CSV (auto-resolved from segments by default).")
    parser.add_argument("--output-video", type=Path, default=None, help="Default: <run>/inference/force_curve_overlay.mp4")
    parser.add_argument("--panel-alpha", type=float, default=0.35, help="Overlay panel opacity (default: 0.35).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = _resolve_run_dir(args.run_dir, args.runs_root)
        input_video = args.input_video.expanduser().resolve() if args.input_video is not None else _find_input_video(run_dir)
        stroke_csv = (
            args.stroke_csv.expanduser().resolve()
            if args.stroke_csv is not None
            else (run_dir / "inference" / "stroke_signal_with_drive_events.csv").resolve()
        )
        segments_csv = (
            args.segments_csv.expanduser().resolve()
            if args.segments_csv is not None
            else (run_dir / "inference" / "rp3_pose_force_matched_segments.csv").resolve()
        )
        output_video = (
            args.output_video.expanduser().resolve()
            if args.output_video is not None
            else (run_dir / "inference" / "force_curve_overlay.mp4").resolve()
        )

        if not stroke_csv.exists():
            raise FileNotFoundError(f"Missing stroke CSV: {stroke_csv}")
        if not segments_csv.exists():
            raise FileNotFoundError(
                f"Missing matched segments CSV: {segments_csv}. "
                "Run inference_cli.py with --match-rp3 first."
            )

        processed, overlay_frames = build_overlay_video(
            run_dir=run_dir,
            input_video=input_video,
            stroke_csv=stroke_csv,
            segments_csv=segments_csv,
            rp3_clean_csv=args.rp3_clean_csv,
            output_video=output_video,
            panel_alpha=float(args.panel_alpha),
        )
    except Exception as exc:
        print(f"Failed: {exc}")
        return 1

    print(f"Run: {run_dir.name}")
    print(f"Input video: {input_video}")
    print(f"Stroke CSV: {stroke_csv}")
    print(f"Segments CSV: {segments_csv}")
    print(f"Output video: {output_video}")
    print(f"Frames processed: {processed}")
    print(f"Drive-overlay frames: {overlay_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
