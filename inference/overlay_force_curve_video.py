#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "sports2d_app" / "runs"
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


def _panel_rect(width: int, height: int) -> tuple[int, int, int, int]:
    w = int(width * 0.42)
    h = int(height * 0.38)
    x0 = max(0, width - w - 18)
    y0 = 18
    return x0, y0, w, h


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
    input_video: Path,
    stroke_csv: Path,
    segments_csv: Path,
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

    required_seg_cols = {"video_stroke_idx", "s_force", "force_n"}
    missing_seg_cols = [c for c in required_seg_cols if c not in seg_df.columns]
    if missing_seg_cols:
        raise ValueError(f"rp3_pose_force_matched_segments.csv missing columns: {missing_seg_cols}")

    seg_df["video_stroke_idx"] = pd.to_numeric(seg_df["video_stroke_idx"], errors="coerce").astype("Int64")
    seg_df["s_force"] = _safe_numeric(seg_df["s_force"])
    seg_df["force_n"] = _safe_numeric(seg_df["force_n"])
    seg_df = seg_df.dropna(subset=["video_stroke_idx", "s_force", "force_n"]).copy()
    seg_df["video_stroke_idx"] = seg_df["video_stroke_idx"].astype(int)

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for stroke_idx, grp in seg_df.groupby("video_stroke_idx"):
        g = grp.sort_values("s_force")
        s = g["s_force"].to_numpy(dtype=np.float64)
        f = g["force_n"].to_numpy(dtype=np.float64)
        if len(s) < 2:
            continue
        curves[int(stroke_idx)] = (s, f)
    if not curves:
        raise ValueError("No valid per-stroke force curves found in segment CSV.")

    global_max_force = max(float(np.nanmax(f)) for _, f in curves.values())
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

    x0, y0, w, h = _panel_rect(width, height)
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

            title = "Force Curve (drive-synced)"
            cv2.putText(frame, title, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

            if np.isfinite(stroke_idx_val):
                stroke_idx = int(round(float(stroke_idx_val)))
            else:
                stroke_idx = -1

            if stroke_idx in curves:
                s_vals, f_vals = curves[stroke_idx]
                pts = _curve_points(s_vals, f_vals, x0=x0, y0=y0, w=w, h=h, y_max=global_max_force)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], False, (40, 205, 245), 2, cv2.LINE_AA)

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

                    txt = f"stroke {stroke_idx}  drive {s_prog*100:.1f}%  F={y_prog:.0f}N"
                    cv2.putText(frame, txt, (x0 + 10, y0 + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                    overlay_frames += 1
                else:
                    txt = f"stroke {stroke_idx}  recovery"
                    cv2.putText(frame, txt, (x0 + 10, y0 + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
            else:
                cv2.putText(
                    frame,
                    "No matched force curve for this stroke",
                    (x0 + 10, y0 + h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
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
            input_video=input_video,
            stroke_csv=stroke_csv,
            segments_csv=segments_csv,
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
