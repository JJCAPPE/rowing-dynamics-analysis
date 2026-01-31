#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def parse_args() -> argparse.Namespace:
    default_csv = Path(__file__).resolve().parent / "out_single1" / "angles.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Plot all angle columns from angles.csv over time. "
            "Use the Matplotlib toolbar to pan/zoom."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to angles.csv (default: {default_csv})",
    )
    parser.add_argument(
        "--time-column",
        default="time_s",
        help="Column to use for the x-axis (default: time_s)",
    )
    parser.add_argument(
        "--min-distance-s",
        type=float,
        default=0.8,
        help="Minimum time (s) between detected peaks/troughs (default: 0.8)",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help="Peak/trough prominence in degrees (default: auto from signal range)",
    )
    parser.add_argument(
        "--prominence-frac",
        type=float,
        default=0.1,
        help="Fraction of signal range to use for auto prominence (default: 0.1)",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.2,
        help="Savitzky-Golay smoothing window in seconds (0 to disable)",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order (default: 2)",
    )
    parser.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="Disable crop thumbnail overlays",
    )
    parser.add_argument(
        "--crop-boxes",
        type=Path,
        default=None,
        help="Path to crop_boxes.npy (default: <csv_dir>/crop_boxes.npy)",
    )
    parser.add_argument(
        "--stabilized-video",
        type=Path,
        default=None,
        help="Path to stabilized video (default: <csv_dir>/debug/stabilized.mp4)",
    )
    parser.add_argument(
        "--thumb-max-px",
        type=int,
        default=60,
        help="Maximum thumbnail width/height in pixels (default: 60)",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class StrokeTimes:
    catch_time_s: float
    finish_time_s: float
    catch_idx: int
    finish_idx: int


@dataclass(frozen=True)
class StrokeEvent:
    kind: str
    time_s: float
    frame_idx: int


def _median_dt(time_s: np.ndarray) -> float:
    diffs = np.diff(time_s.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Unable to infer time step from time column.")
    return float(np.median(diffs))


def _auto_prominence(signal: np.ndarray, frac: float) -> float:
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return 0.0
    p95 = float(np.percentile(finite, 95))
    p05 = float(np.percentile(finite, 5))
    return max((p95 - p05) * frac, 1e-6)


def _smooth_signal(
    signal: np.ndarray, *, time_s: np.ndarray, window_s: float, polyorder: int
) -> np.ndarray:
    if window_s <= 0:
        return signal
    if polyorder < 1:
        return signal
    if signal.size < 3:
        return signal

    series = pd.Series(signal).interpolate(limit_direction="both").bfill().ffill()
    filled = series.to_numpy(dtype=float)
    dt = _median_dt(time_s)
    window = int(round(window_s / dt))
    if window < 3:
        return filled
    if window % 2 == 0:
        window += 1
    if window > filled.size:
        window = filled.size if filled.size % 2 == 1 else filled.size - 1
    if window <= polyorder:
        return filled
    return savgol_filter(filled, window_length=window, polyorder=polyorder, mode="interp")


def _detect_strokes(
    time_s: np.ndarray,
    knee_signal: np.ndarray,
    hip_signal: np.ndarray,
    elbow_signal: np.ndarray,
    *,
    min_distance_s: float,
    prominence: Optional[float],
    prominence_frac: float,
    smooth_window_s: float,
    smooth_polyorder: int,
) -> list[StrokeTimes]:
    dt = _median_dt(time_s)
    min_distance = max(1, int(round(min_distance_s / dt)))

    knee_s = _smooth_signal(
        knee_signal, time_s=time_s, window_s=smooth_window_s, polyorder=smooth_polyorder
    )
    hip_s = _smooth_signal(
        hip_signal, time_s=time_s, window_s=smooth_window_s, polyorder=smooth_polyorder
    )
    elbow_s = _smooth_signal(
        elbow_signal, time_s=time_s, window_s=smooth_window_s, polyorder=smooth_polyorder
    )

    prom_knee = prominence if prominence is not None else _auto_prominence(knee_s, prominence_frac)
    prom_hip = prominence if prominence is not None else _auto_prominence(hip_s, prominence_frac)
    prom_elbow = (
        prominence if prominence is not None else _auto_prominence(elbow_s, prominence_frac)
    )

    knee_troughs, _ = find_peaks(-knee_s, distance=min_distance, prominence=prom_knee)
    hip_troughs, _ = find_peaks(-hip_s, distance=min_distance, prominence=prom_hip)
    elbow_troughs, _ = find_peaks(-elbow_s, distance=min_distance, prominence=prom_elbow)

    stroke_times: list[StrokeTimes] = []
    prev_finish_idx = -1
    for finish_idx in elbow_troughs:
        window_start = prev_finish_idx + 1
        window_end = finish_idx
        knee_candidates = knee_troughs[(knee_troughs >= window_start) & (knee_troughs < window_end)]
        hip_candidates = hip_troughs[(hip_troughs >= window_start) & (hip_troughs < window_end)]

        candidate_idxs: list[int] = []
        if knee_candidates.size:
            candidate_idxs.append(int(knee_candidates[0]))
        if hip_candidates.size:
            candidate_idxs.append(int(hip_candidates[0]))
        if not candidate_idxs:
            prev_finish_idx = int(finish_idx)
            continue

        catch_idx = int(min(candidate_idxs))
        if catch_idx >= finish_idx:
            prev_finish_idx = int(finish_idx)
            continue

        stroke_times.append(
            StrokeTimes(
                catch_time_s=float(time_s[catch_idx]),
                finish_time_s=float(time_s[int(finish_idx)]),
                catch_idx=catch_idx,
                finish_idx=int(finish_idx),
            )
        )
        prev_finish_idx = int(finish_idx)

    return stroke_times


def _event_frame_idx(frame_idx_series: Optional[np.ndarray], event_idx: int) -> int:
    if frame_idx_series is None:
        return event_idx
    val = frame_idx_series[event_idx]
    if not np.isfinite(val):
        return event_idx
    return int(round(float(val)))


def _load_crop_boxes(crop_boxes_path: Path) -> np.ndarray:
    boxes = np.load(crop_boxes_path)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected crop_boxes shape (T,4). Got {boxes.shape}")
    return boxes.astype(np.float32, copy=False)


def _extract_thumbnails(
    video_path: Path,
    boxes_xywh: np.ndarray,
    frame_indices: list[int],
    *,
    thumb_max_px: int,
) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    thumbnails: dict[int, np.ndarray] = {}
    for idx in sorted(set(frame_indices)):
        if idx < 0 or idx >= boxes_xywh.shape[0]:
            continue

        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        box = boxes_xywh[idx].astype(float, copy=False)
        if not np.all(np.isfinite(box)):
            continue
        x, y, w, h = box
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(frame.shape[1], int(round(x + w)))
        y1 = min(frame.shape[0], int(round(y + h)))
        if x1 <= x0 or y1 <= y0:
            continue

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        h_px, w_px = crop_rgb.shape[:2]
        scale = float(thumb_max_px) / float(max(h_px, w_px))
        if scale < 1.0:
            new_w = max(1, int(round(w_px * scale)))
            new_h = max(1, int(round(h_px * scale)))
            crop_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        thumbnails[idx] = crop_rgb

    cap.release()
    return thumbnails


def _overlay_thumbnails(
    ax: plt.Axes,
    events: list[StrokeEvent],
    thumbnails: dict[int, np.ndarray],
) -> None:
    if not events:
        return

    for event in events:
        img = thumbnails.get(event.frame_idx)
        if img is None:
            continue
        y_axes = 1.02 if event.kind == "catch" else 1.12
        edge_color = "tab:green" if event.kind == "catch" else "tab:red"
        imagebox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(
            imagebox,
            (event.time_s, y_axes),
            xycoords=ax.get_xaxis_transform(),
            box_alignment=(0.5, 0.0),
            frameon=True,
            pad=0.2,
            bboxprops=dict(edgecolor=edge_color, linewidth=1.0, facecolor="white", alpha=0.9),
        )
        ab.set_clip_on(False)
        ax.add_artist(ab)


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.time_column not in df.columns:
        raise ValueError(f"Missing time column: {args.time_column}")

    if "elbow_avg_deg" not in df.columns and {"left_elbow_deg", "right_elbow_deg"}.issubset(
        df.columns
    ):
        df["elbow_avg_deg"] = df[["left_elbow_deg", "right_elbow_deg"]].mean(
            axis=1, skipna=True
        )

    numeric_df = df.select_dtypes(include="number")
    if args.time_column not in numeric_df.columns:
        raise ValueError(f"Time column is not numeric: {args.time_column}")

    required_cols = {
        "left_knee_deg",
        "right_knee_deg",
        "left_hip_deg",
        "right_hip_deg",
        "left_elbow_deg",
        "right_elbow_deg",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns for stroke detection: {missing_list}")

    time_s = df[args.time_column].to_numpy(dtype=float)
    knee_signal = df[["left_knee_deg", "right_knee_deg"]].mean(axis=1, skipna=True).to_numpy()
    hip_signal = df[["left_hip_deg", "right_hip_deg"]].mean(axis=1, skipna=True).to_numpy()

    elbow_signal = df["elbow_avg_deg"].to_numpy() if "elbow_avg_deg" in df.columns else None
    if elbow_signal is None:
        raise ValueError("Missing elbow angle columns for finish detection.")

    stroke_times = _detect_strokes(
        time_s,
        knee_signal,
        hip_signal,
        elbow_signal,
        min_distance_s=args.min_distance_s,
        prominence=args.prominence,
        prominence_frac=args.prominence_frac,
        smooth_window_s=args.smooth_window_s,
        smooth_polyorder=args.smooth_polyorder,
    )
    stroke_windows = [(stroke.catch_time_s, stroke.finish_time_s) for stroke in stroke_times]

    frame_idx_series = None
    if "frame_idx" in df.columns:
        frame_idx_series = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=float)

    stroke_events: list[StrokeEvent] = []
    for stroke in stroke_times:
        catch_frame = _event_frame_idx(frame_idx_series, stroke.catch_idx)
        finish_frame = _event_frame_idx(frame_idx_series, stroke.finish_idx)
        stroke_events.append(
            StrokeEvent(kind="catch", time_s=stroke.catch_time_s, frame_idx=catch_frame)
        )
        stroke_events.append(
            StrokeEvent(kind="finish", time_s=stroke.finish_time_s, frame_idx=finish_frame)
        )
    stroke_events.sort(key=lambda e: e.time_s)

    excluded = {args.time_column, "frame_idx"}
    plot_cols = [col for col in numeric_df.columns if col not in excluded]
    plot_cols = [col for col in plot_cols if not numeric_df[col].isna().all()]

    if not plot_cols:
        raise ValueError("No numeric columns available to plot.")

    y_label = "Angle (deg)" if all("deg" in col for col in plot_cols) else "Value"

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in plot_cols:
        ax.plot(df[args.time_column], df[col], label=col, linewidth=1.5, alpha=0.9)

    for i, stroke in enumerate(stroke_times):
        catch_label = "catch" if i == 0 else None
        finish_label = "finish" if i == 0 else None
        ax.axvline(
            stroke.catch_time_s,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=catch_label,
        )
        ax.axvline(
            stroke.finish_time_s,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=finish_label,
        )

    ax.set_title("Angles over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    ax.legend(
        title="Signals and events",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    if not args.no_thumbnails and stroke_events:
        out_dir = csv_path.parent
        crop_boxes_path = args.crop_boxes or (out_dir / "crop_boxes.npy")
        stabilized_video_path = args.stabilized_video or (out_dir / "debug" / "stabilized.mp4")
        if crop_boxes_path.exists() and stabilized_video_path.exists():
            boxes = _load_crop_boxes(crop_boxes_path)
            thumbnails = _extract_thumbnails(
                stabilized_video_path,
                boxes,
                [event.frame_idx for event in stroke_events],
                thumb_max_px=args.thumb_max_px,
            )
            _overlay_thumbnails(ax, stroke_events, thumbnails)
            fig.tight_layout()
            fig.subplots_adjust(top=0.82)
        else:
            fig.tight_layout()
            print(
                "Thumbnail overlay skipped (missing crop_boxes.npy or stabilized.mp4). "
                "Use --crop-boxes/--stabilized-video to provide paths."
            )
    else:
        fig.tight_layout()

    if stroke_windows:
        print("Detected strokes (catch -> finish):")
        for i, (catch_t, finish_t) in enumerate(stroke_windows, start=1):
            print(f"  {i:02d}: catch={catch_t:.3f}s " f"finish={finish_t:.3f}s")
    else:
        print("No strokes detected with current parameters.")
    plt.show()


if __name__ == "__main__":
    main()
