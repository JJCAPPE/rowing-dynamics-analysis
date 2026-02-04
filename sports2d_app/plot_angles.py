#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
import cv2


@dataclass(frozen=True)
class StrokeTimes:
    catch_time_s: float
    finish_time_s: float
    catch_idx: int
    finish_idx: int


@dataclass(frozen=True)
class PlotResult:
    plot_path: Path
    time_column: str
    plotted_columns: list[str]
    stroke_times: list[StrokeTimes]


@dataclass(frozen=True)
class StrokeEvent:
    kind: str
    time_s: float
    frame_idx: int


def _normalize_name(name: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "", name.lower())
    for suffix in ("degrees", "deg"):
        if norm.endswith(suffix):
            norm = norm[: -len(suffix)]
            break
    return norm


def _find_column(columns: Iterable[str], target: str) -> Optional[str]:
    target_norm = _normalize_name(target)
    for col in columns:
        if _normalize_name(col) == target_norm:
            return col
    return None


def _auto_time_column(columns: Iterable[str]) -> Optional[str]:
    for candidate in ("time", "time_s", "timestamp", "t"):
        col = _find_column(columns, candidate)
        if col is not None:
            return col
    for col in columns:
        if "time" in col.lower():
            return col
    return None


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


def _fill_signal(signal: np.ndarray) -> np.ndarray:
    series = pd.Series(signal).interpolate(limit_direction="both").bfill().ffill()
    return series.to_numpy(dtype=float)


def _smooth_signal(signal: np.ndarray, *, time_s: np.ndarray, window_s: float) -> np.ndarray:
    if window_s <= 0 or signal.size < 3:
        return _fill_signal(signal)
    filled = _fill_signal(signal)
    dt = _median_dt(time_s)
    window = int(round(window_s / dt))
    if window < 3:
        return filled
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(filled)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _local_minima_indices(signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    left = signal[:-2]
    mid = signal[1:-1]
    right = signal[2:]
    mask = (mid <= left) & (mid <= right)
    return np.where(mask)[0] + 1


def _prominent_troughs(
    signal: np.ndarray, candidates: np.ndarray, *, min_distance: int, prominence: float
) -> np.ndarray:
    if prominence <= 0 or candidates.size == 0:
        return candidates
    keep: list[int] = []
    n = signal.size
    for idx in candidates:
        left = max(0, idx - min_distance)
        right = min(n - 1, idx + min_distance)
        left_max = np.nanmax(signal[left : idx + 1])
        right_max = np.nanmax(signal[idx : right + 1])
        prom = min(left_max, right_max) - signal[idx]
        if np.isfinite(prom) and prom >= prominence:
            keep.append(int(idx))
    return np.array(keep, dtype=int)


def _enforce_min_distance(
    signal: np.ndarray, candidates: np.ndarray, *, min_distance: int
) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    ordered = np.sort(candidates)
    selected: list[int] = []
    for idx in ordered:
        if not selected:
            selected.append(int(idx))
            continue
        if idx - selected[-1] >= min_distance:
            selected.append(int(idx))
            continue
        if signal[idx] < signal[selected[-1]]:
            selected[-1] = int(idx)
    return np.array(selected, dtype=int)


def _detect_troughs(
    time_s: np.ndarray,
    signal: np.ndarray,
    *,
    min_distance_s: float,
    prominence: Optional[float],
    prominence_frac: float,
    smooth_window_s: float,
) -> np.ndarray:
    dt = _median_dt(time_s)
    min_distance = max(1, int(round(min_distance_s / dt)))
    smooth = _smooth_signal(signal, time_s=time_s, window_s=smooth_window_s)
    prom = prominence if prominence is not None else _auto_prominence(smooth, prominence_frac)
    candidates = _local_minima_indices(smooth)
    candidates = _prominent_troughs(smooth, candidates, min_distance=min_distance, prominence=prom)
    return _enforce_min_distance(smooth, candidates, min_distance=min_distance)


def _angle_signal(df: pd.DataFrame, left_name: str, right_name: str) -> Optional[np.ndarray]:
    left_col = _find_column(df.columns, left_name)
    right_col = _find_column(df.columns, right_name)
    if left_col and right_col:
        return df[[left_col, right_col]].mean(axis=1, skipna=True).to_numpy(dtype=float)
    if left_col:
        return pd.to_numeric(df[left_col], errors="coerce").to_numpy(dtype=float)
    if right_col:
        return pd.to_numeric(df[right_col], errors="coerce").to_numpy(dtype=float)
    return None


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
) -> list[StrokeTimes]:
    knee_troughs = _detect_troughs(
        time_s,
        knee_signal,
        min_distance_s=min_distance_s,
        prominence=prominence,
        prominence_frac=prominence_frac,
        smooth_window_s=smooth_window_s,
    )
    hip_troughs = _detect_troughs(
        time_s,
        hip_signal,
        min_distance_s=min_distance_s,
        prominence=prominence,
        prominence_frac=prominence_frac,
        smooth_window_s=smooth_window_s,
    )
    elbow_troughs = _detect_troughs(
        time_s,
        elbow_signal,
        min_distance_s=min_distance_s,
        prominence=prominence,
        prominence_frac=prominence_frac,
        smooth_window_s=smooth_window_s,
    )

    stroke_times: list[StrokeTimes] = []
    prev_finish_idx = -1
    for finish_idx in elbow_troughs:
        window_start = prev_finish_idx + 1
        window_end = int(finish_idx)
        knee_candidates = knee_troughs[
            (knee_troughs >= window_start) & (knee_troughs < window_end)
        ]
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


def _event_frame_idx(
    frame_idx_series: Optional[np.ndarray],
    event_idx: int,
    *,
    time_s: np.ndarray,
    event_time_s: float,
    fps: Optional[float],
) -> int:
    if frame_idx_series is not None:
        val = frame_idx_series[event_idx]
        if np.isfinite(val):
            return int(round(float(val)))
    if fps is not None and fps > 0:
        return int(round(float(event_time_s) * float(fps)))
    if time_s.size > 1:
        return int(event_idx)
    return int(event_idx)


def _extract_frames(
    video_path: Path,
    frame_indices: list[int],
    *,
    thumb_max_px: Optional[int],
) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    thumbnails: dict[int, np.ndarray] = {}
    for idx in sorted(set(frame_indices)):
        if idx < 0:
            continue
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if thumb_max_px and thumb_max_px > 0:
            h_px, w_px = frame_rgb.shape[:2]
            scale = float(thumb_max_px) / float(max(h_px, w_px))
            if scale < 1.0:
                new_w = max(1, int(round(w_px * scale)))
                new_h = max(1, int(round(h_px * scale)))
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        thumbnails[idx] = frame_rgb

    cap.release()
    return thumbnails


def _overlay_thumbnails(
    ax: plt.Axes,
    events: list[StrokeEvent],
    thumbnails: dict[int, np.ndarray],
    *,
    zoom: float,
) -> None:
    if not events:
        return

    for event in events:
        img = thumbnails.get(event.frame_idx)
        if img is None:
            continue
        y_axes = 1.02 if event.kind == "catch" else 1.12
        edge_color = "tab:green" if event.kind == "catch" else "tab:red"
        imagebox = OffsetImage(img, zoom=zoom)
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


def generate_angles_plot(
    csv_path: Path,
    output_path: Path,
    *,
    time_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    include_strokes: bool = True,
    min_distance_s: float = 0.8,
    prominence: Optional[float] = None,
    prominence_frac: float = 0.1,
    smooth_window_s: float = 0.2,
    title: Optional[str] = None,
    video_path: Optional[Path] = None,
    include_thumbnails: bool = True,
    thumb_max_px: Optional[int] = None,
    thumb_zoom: float = 0.05,
    fig_dpi: int = 300,
) -> PlotResult:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    time_col = time_column or _auto_time_column(df.columns)
    if time_col is None:
        raise ValueError("Unable to determine time column in angles CSV.")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    for col in df.columns:
        if col == time_col:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_cols: list[str] = []
    columns_set: Optional[set[str]] = None
    if columns:
        columns_set = set()
        normalized = {_normalize_name(col): col for col in df.columns}
        for requested in columns:
            match = normalized.get(_normalize_name(requested))
            if match is not None:
                columns_set.add(match)
    for col in df.columns:
        if col == time_col:
            continue
        if columns_set is not None and col not in columns_set:
            continue
        if df[col].notna().any():
            plot_cols.append(col)

    if not plot_cols:
        raise ValueError("No numeric columns available to plot.")

    time_s = df[time_col].to_numpy(dtype=float)

    stroke_times: list[StrokeTimes] = []
    if include_strokes:
        knee_signal = _angle_signal(df, "left knee", "right knee")
        hip_signal = _angle_signal(df, "left hip", "right hip")
        elbow_signal = _angle_signal(df, "left elbow", "right elbow")
        if knee_signal is not None and hip_signal is not None and elbow_signal is not None:
            stroke_times = _detect_strokes(
                time_s,
                knee_signal,
                hip_signal,
                elbow_signal,
                min_distance_s=min_distance_s,
                prominence=prominence,
                prominence_frac=prominence_frac,
                smooth_window_s=smooth_window_s,
            )

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in plot_cols:
        ax.plot(time_s, df[col], label=col, linewidth=1.3, alpha=0.85)

    for i, stroke in enumerate(stroke_times):
        catch_label = "catch" if i == 0 else None
        finish_label = "finish" if i == 0 else None
        ax.axvline(
            stroke.catch_time_s,
            color="tab:green",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=catch_label,
        )
        ax.axvline(
            stroke.finish_time_s,
            color="tab:red",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=finish_label,
        )

    ax.set_title(title or "Sports2D angles over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    ax.legend(
        title="Signals",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=8,
    )

    if include_thumbnails and video_path and stroke_times:
        frame_idx_series = None
        if "frame_idx" in df.columns:
            frame_idx_series = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(
                dtype=float
            )
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        stroke_events: list[StrokeEvent] = []
        for stroke in stroke_times:
            catch_frame = _event_frame_idx(
                frame_idx_series,
                stroke.catch_idx,
                time_s=time_s,
                event_time_s=stroke.catch_time_s,
                fps=fps,
            )
            finish_frame = _event_frame_idx(
                frame_idx_series,
                stroke.finish_idx,
                time_s=time_s,
                event_time_s=stroke.finish_time_s,
                fps=fps,
            )
            stroke_events.append(
                StrokeEvent(kind="catch", time_s=stroke.catch_time_s, frame_idx=catch_frame)
            )
            stroke_events.append(
                StrokeEvent(kind="finish", time_s=stroke.finish_time_s, frame_idx=finish_frame)
            )
        stroke_events.sort(key=lambda e: e.time_s)

        thumbnails = _extract_frames(
            video_path,
            [event.frame_idx for event in stroke_events],
            thumb_max_px=thumb_max_px,
        )
        _overlay_thumbnails(ax, stroke_events, thumbnails, zoom=thumb_zoom)
        fig.subplots_adjust(top=0.78)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)

    return PlotResult(
        plot_path=output_path,
        time_column=time_col,
        plotted_columns=plot_cols,
        stroke_times=stroke_times,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static angles plot from a Sports2D angles CSV."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to angles CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <csv>_plot.png)",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="Time column to use (default: auto-detect)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Subset of columns to plot (default: all numeric)",
    )
    parser.add_argument(
        "--no-strokes",
        action="store_true",
        help="Disable catch/finish detection",
    )
    parser.add_argument(
        "--min-distance-s",
        type=float,
        default=0.8,
        help="Minimum time (s) between detected troughs (default: 0.8)",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help="Prominence threshold in degrees (default: auto)",
    )
    parser.add_argument(
        "--prominence-frac",
        type=float,
        default=0.1,
        help="Fraction of signal range for auto prominence (default: 0.1)",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.2,
        help="Smoothing window in seconds (default: 0.2)",
    )
    parser.add_argument("--title", default=None, help="Plot title override")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video path for overlay thumbnails",
    )
    parser.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="Disable thumbnail overlays",
    )
    parser.add_argument(
        "--thumb-max-px",
        type=int,
        default=None,
        help="Max width/height for thumbnails in pixels (default: keep full size)",
    )
    parser.add_argument(
        "--thumb-zoom",
        type=float,
        default=0.2,
        help="Zoom factor for thumbnail overlays (default: 0.2)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for the saved plot (default: 300)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    output_path = args.output or csv_path.with_name(f"{csv_path.stem}_plot.png")
    result = generate_angles_plot(
        csv_path,
        output_path,
        time_column=args.time_column,
        columns=args.columns,
        include_strokes=not args.no_strokes,
        min_distance_s=args.min_distance_s,
        prominence=args.prominence,
        prominence_frac=args.prominence_frac,
        smooth_window_s=args.smooth_window_s,
        title=args.title,
        video_path=args.video,
        include_thumbnails=not args.no_thumbnails,
        thumb_max_px=args.thumb_max_px,
        thumb_zoom=args.thumb_zoom,
        fig_dpi=args.dpi,
    )
    print(f"Saved plot: {result.plot_path}")
    if result.stroke_times:
        print("Detected strokes (catch -> finish):")
        for i, stroke in enumerate(result.stroke_times, start=1):
            print(
                f"  {i:02d}: catch={stroke.catch_time_s:.3f}s "
                f"finish={stroke.finish_time_s:.3f}s"
            )
    else:
        print("No strokes detected with current parameters.")


if __name__ == "__main__":
    main()
