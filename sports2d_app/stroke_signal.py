#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from plot_angles import generate_angles_plot


BBoxXYWH = Tuple[float, float, float, float]


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class TrackSeries:
    boxes_xywh: np.ndarray  # (T,4)
    centers_xy: np.ndarray  # (T,2)
    status: np.ndarray  # (T,) uint8


@dataclass(frozen=True)
class StrokeEvents:
    catch_idx: np.ndarray  # (N,)
    finish_idx: np.ndarray  # (N,)


@dataclass(frozen=True)
class StrokeTrackingOutputs:
    stroke_csv: Path
    stroke_npz: Path
    merged_angles_csv: Optional[Path]
    merged_angles_plot: Optional[Path]
    debug_video: Optional[Path]
    fps: float
    catch_idx: np.ndarray
    finish_idx: np.ndarray


def _video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frames)


def _read_frame(video_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        return frame
    finally:
        cap.release()


def _draw_help(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = img.copy()
    y = 30
    for line in lines:
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            out,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return out


def _select_bbox(frame_bgr: np.ndarray, title: str, prompt: Sequence[str]) -> BBoxXYWH:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    overlay = _draw_help(frame_bgr, list(prompt))
    x, y, w, h = cv2.selectROI(title, overlay, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    if w <= 1 or h <= 1:
        raise RuntimeError(f"Invalid bbox selection for '{title}'.")
    return float(x), float(y), float(w), float(h)


def annotate_handle_and_machine(video_path: Path, reference_frame_idx: int) -> Tuple[BBoxXYWH, BBoxXYWH]:
    frame = _read_frame(video_path, reference_frame_idx)
    try:
        machine_bbox = _select_bbox(
            frame,
            "Stroke Tracking: machine reference",
            [
                "Draw bbox around a rigid machine part (moves with the machine body).",
                "Press ENTER/SPACE to confirm. Press C to cancel.",
            ],
        )
        handle_bbox = _select_bbox(
            frame,
            "Stroke Tracking: handle",
            [
                "Draw bbox around the handle grip.",
                "Press ENTER/SPACE to confirm. Press C to cancel.",
            ],
        )
    finally:
        cv2.destroyAllWindows()
    return machine_bbox, handle_bbox


def _iter_frames(video_path: Path, start: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        idx = start
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _track_bbox_lk(
    video_path: Path,
    bbox0_px: BBoxXYWH,
    reference_frame_idx: int,
    *,
    ema_alpha: float = 0.4,
    min_points: int = 10,
) -> TrackSeries:
    meta = _video_meta(video_path)
    T = int(meta.frame_count)
    W = int(meta.width)
    H = int(meta.height)

    x0, y0, w0, h0 = [float(v) for v in bbox0_px]
    if w0 <= 2 or h0 <= 2:
        raise ValueError("bbox0_px is invalid (too small).")

    w = min(w0, float(W))
    h = min(h0, float(H))

    def clamp_center(cx: float, cy: float) -> Tuple[float, float]:
        cx = float(np.clip(cx, w / 2.0, W - w / 2.0))
        cy = float(np.clip(cy, h / 2.0, H - h / 2.0))
        return cx, cy

    def center_to_box(cx: float, cy: float) -> Tuple[float, float, float, float]:
        cx, cy = clamp_center(cx, cy)
        return float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)

    def box_mask(shape_hw: Tuple[int, int], box_xywh: Tuple[float, float, float, float]) -> np.ndarray:
        hh, ww = shape_hw
        x, y, bw, bh = box_xywh
        x0i = int(max(0, round(x)))
        y0i = int(max(0, round(y)))
        x1i = int(min(ww, round(x + bw)))
        y1i = int(min(hh, round(y + bh)))
        mask = np.zeros((hh, ww), dtype=np.uint8)
        mask[y0i:y1i, x0i:x1i] = 255
        return mask

    def init_features(gray: np.ndarray, box_xywh: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        mask = box_mask(gray.shape[:2], box_xywh)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=mask,
        )
        return pts

    def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    ref_idx = int(np.clip(reference_frame_idx, 0, max(0, T - 1)))
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    status = np.zeros((T,), dtype=np.uint8)

    cx0 = x0 + w0 / 2.0
    cy0 = y0 + h0 / 2.0
    centers[ref_idx] = (cx0, cy0)
    status[ref_idx] = 1

    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    prev_gray = None
    prev_pts = None
    prev_box = center_to_box(cx0, cy0)
    for idx, frame in _iter_frames(video_path, start=ref_idx):
        g = to_gray(frame)
        if idx == ref_idx:
            prev_gray = g
            prev_pts = init_features(prev_gray, prev_box)
            continue

        assert prev_gray is not None
        if prev_pts is None or prev_pts.shape[0] < min_points:
            prev_pts = init_features(prev_gray, prev_box)

        cx_prev, cy_prev = float(centers[idx - 1, 0]), float(centers[idx - 1, 1])
        cx_cur, cy_cur = cx_prev, cy_prev
        tracked = False

        if prev_pts is not None and prev_pts.shape[0] >= 1:
            nxt, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                g,
                prev_pts,
                None,
                winSize=lk_win_size,
                maxLevel=lk_max_level,
                criteria=lk_criteria,
            )
            if st is not None and nxt is not None:
                good = st.reshape(-1) == 1
                if int(good.sum()) >= max(3, min_points // 2):
                    disp = (nxt[good] - prev_pts[good]).reshape(-1, 2)
                    dx, dy = np.median(disp, axis=0)
                    cx_cur = float(cx_prev + dx)
                    cy_cur = float(cy_prev + dy)
                    prev_pts = nxt[good].reshape(-1, 1, 2)
                    tracked = True
                else:
                    prev_pts = None

        centers[idx] = clamp_center(cx_cur, cy_cur)
        status[idx] = 1 if tracked else 0
        prev_box = center_to_box(float(centers[idx, 0]), float(centers[idx, 1]))
        prev_gray = g

    if ref_idx > 0:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(ref_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {ref_idx}")

            prev_gray = to_gray(frame)
            prev_box = center_to_box(cx0, cy0)
            prev_pts = init_features(prev_gray, prev_box)

            for idx in range(ref_idx - 1, -1, -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                g = to_gray(frame)

                if prev_pts is None or prev_pts.shape[0] < min_points:
                    prev_pts = init_features(prev_gray, prev_box)

                cx_prev, cy_prev = float(centers[idx + 1, 0]), float(centers[idx + 1, 1])
                cx_cur, cy_cur = cx_prev, cy_prev
                tracked = False

                if prev_pts is not None and prev_pts.shape[0] >= 1:
                    nxt, st, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        g,
                        prev_pts,
                        None,
                        winSize=lk_win_size,
                        maxLevel=lk_max_level,
                        criteria=lk_criteria,
                    )
                    if st is not None and nxt is not None:
                        good = st.reshape(-1) == 1
                        if int(good.sum()) >= max(3, min_points // 2):
                            disp = (nxt[good] - prev_pts[good]).reshape(-1, 2)
                            dx, dy = np.median(disp, axis=0)
                            cx_cur = float(cx_prev + dx)
                            cy_cur = float(cy_prev + dy)
                            prev_pts = nxt[good].reshape(-1, 1, 2)
                            tracked = True
                        else:
                            prev_pts = None

                centers[idx] = clamp_center(cx_cur, cy_cur)
                status[idx] = 1 if tracked else 0
                prev_box = center_to_box(float(centers[idx, 0]), float(centers[idx, 1]))
                prev_gray = g
        finally:
            cap.release()

    def interp_nan(v: np.ndarray) -> np.ndarray:
        out = v.astype(np.float32, copy=True)
        n = out.shape[0]
        x = np.arange(n, dtype=np.float32)
        mask = np.isfinite(out)
        if mask.all():
            return out
        if not mask.any():
            out[:] = out[ref_idx]
            return out
        out[~mask] = np.interp(x[~mask], x[mask], out[mask]).astype(np.float32)
        return out

    centers[:, 0] = interp_nan(centers[:, 0])
    centers[:, 1] = interp_nan(centers[:, 1])

    alpha = float(np.clip(ema_alpha, 0.0, 0.999))
    smooth = centers.copy()
    for i in range(1, T):
        smooth[i, 0] = alpha * smooth[i - 1, 0] + (1.0 - alpha) * smooth[i, 0]
        smooth[i, 1] = alpha * smooth[i - 1, 1] + (1.0 - alpha) * smooth[i, 1]

    boxes = np.zeros((T, 4), dtype=np.float32)
    for i in range(T):
        boxes[i] = np.asarray(center_to_box(float(smooth[i, 0]), float(smooth[i, 1])), dtype=np.float32)

    return TrackSeries(boxes_xywh=boxes, centers_xy=smooth, status=status)


def _fill_signal(signal: np.ndarray) -> np.ndarray:
    series = pd.Series(signal).interpolate(limit_direction="both").bfill().ffill()
    return series.to_numpy(dtype=float)


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return _fill_signal(signal)
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(_fill_signal(signal))
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
    return np.where((mid <= left) & (mid <= right))[0] + 1


def _local_maxima_indices(signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    left = signal[:-2]
    mid = signal[1:-1]
    right = signal[2:]
    return np.where((mid >= left) & (mid >= right))[0] + 1


def _auto_prominence(signal: np.ndarray, frac: float) -> float:
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return 0.0
    p95 = float(np.percentile(finite, 95))
    p05 = float(np.percentile(finite, 5))
    return max((p95 - p05) * float(frac), 1e-6)


def _filter_prominence(
    signal: np.ndarray,
    candidates: np.ndarray,
    *,
    min_distance: int,
    prominence: float,
    mode: str,
) -> np.ndarray:
    if prominence <= 0 or candidates.size == 0:
        return candidates
    keep: list[int] = []
    n = signal.size
    for idx in candidates:
        left = max(0, idx - min_distance)
        right = min(n - 1, idx + min_distance)
        left_vals = signal[left : idx + 1]
        right_vals = signal[idx : right + 1]
        if mode == "min":
            prom = min(float(np.nanmax(left_vals)), float(np.nanmax(right_vals))) - float(signal[idx])
        else:
            prom = float(signal[idx]) - max(float(np.nanmin(left_vals)), float(np.nanmin(right_vals)))
        if np.isfinite(prom) and prom >= prominence:
            keep.append(int(idx))
    return np.asarray(keep, dtype=int)


def _enforce_min_distance(signal: np.ndarray, candidates: np.ndarray, *, min_distance: int, mode: str) -> np.ndarray:
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
        if mode == "min":
            replace = signal[idx] < signal[selected[-1]]
        else:
            replace = signal[idx] > signal[selected[-1]]
        if replace:
            selected[-1] = int(idx)
    return np.asarray(selected, dtype=int)


def _detect_stroke_events(
    signal_px: np.ndarray,
    *,
    fps: float,
    min_stroke_distance_s: float,
    prominence: Optional[float],
    prominence_frac: float,
    smooth_window_s: float,
) -> StrokeEvents:
    if signal_px.size < 5:
        return StrokeEvents(catch_idx=np.array([], dtype=int), finish_idx=np.array([], dtype=int))

    dt = 1.0 / float(max(fps, 1e-6))
    min_distance = max(2, int(round(float(min_stroke_distance_s) / dt)))
    smooth_window = max(1, int(round(float(smooth_window_s) / dt)))
    smooth = _smooth_signal(signal_px, window=smooth_window)
    prom = float(prominence) if prominence is not None else _auto_prominence(smooth, prominence_frac)

    catches = _local_minima_indices(smooth)
    catches = _filter_prominence(
        smooth,
        catches,
        min_distance=min_distance,
        prominence=prom,
        mode="min",
    )
    catches = _enforce_min_distance(smooth, catches, min_distance=min_distance, mode="min")
    if catches.size < 2:
        return StrokeEvents(catch_idx=np.array([], dtype=int), finish_idx=np.array([], dtype=int))

    finish_candidates = _local_maxima_indices(smooth)
    finish_candidates = _filter_prominence(
        smooth,
        finish_candidates,
        min_distance=max(2, min_distance // 2),
        prominence=max(prom * 0.5, 1e-6),
        mode="max",
    )
    finish_candidates = _enforce_min_distance(
        smooth,
        finish_candidates,
        min_distance=max(2, min_distance // 2),
        mode="max",
    )

    finish_idx: list[int] = []
    kept_catches: list[int] = []
    for i in range(len(catches) - 1):
        c0 = int(catches[i])
        c1 = int(catches[i + 1])
        local = finish_candidates[(finish_candidates > c0) & (finish_candidates < c1)]
        if local.size == 0:
            continue
        peak = int(local[np.argmax(smooth[local])])
        kept_catches.append(c0)
        finish_idx.append(peak)

    if finish_idx:
        # Ensure the final catch closes the last stroke.
        final_catch = int(catches[-1])
        kept_catches.append(final_catch)

    return StrokeEvents(
        catch_idx=np.asarray(kept_catches, dtype=int),
        finish_idx=np.asarray(finish_idx, dtype=int),
    )


def _principal_axis(vectors_xy: np.ndarray) -> np.ndarray:
    valid = np.isfinite(vectors_xy).all(axis=1)
    pts = vectors_xy[valid]
    if pts.shape[0] < 2:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    axis = axis / norm
    ref = pts[0]
    if float(np.dot(axis, ref)) < 0:
        axis = -axis
    return axis


def _stroke_phase_series(
    catch_idx: np.ndarray,
    finish_idx: np.ndarray,
    *,
    length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.full((length,), np.nan, dtype=np.float32)
    stroke_idx = np.full((length,), -1, dtype=np.int32)
    drive = np.zeros((length,), dtype=np.uint8)

    if catch_idx.size < 2 or finish_idx.size == 0:
        return phase, stroke_idx, drive

    n_strokes = min(finish_idx.size, catch_idx.size - 1)
    for i in range(n_strokes):
        c0 = int(catch_idx[i])
        f0 = int(finish_idx[i])
        c1 = int(catch_idx[i + 1])
        if not (0 <= c0 < f0 < c1 <= length - 1):
            continue

        stroke_idx[c0 : c1 + 1] = i
        drive[c0 : f0 + 1] = 1

        drive_len = max(1, f0 - c0)
        for t in range(c0, f0 + 1):
            phase[t] = np.float32(0.5 * (t - c0) / drive_len)

        rec_len = max(1, c1 - f0)
        for t in range(f0, c1 + 1):
            phase[t] = np.float32(0.5 + 0.5 * (t - f0) / rec_len)

    return phase, stroke_idx, drive


def _draw_bbox(frame: np.ndarray, box: Sequence[float], color: Tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _write_debug_video(
    video_path: Path,
    out_path: Path,
    *,
    handle_track: TrackSeries,
    machine_track: TrackSeries,
    rel_axis_px: np.ndarray,
    phase: np.ndarray,
    catch_idx: np.ndarray,
    finish_idx: np.ndarray,
) -> None:
    meta = _video_meta(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {out_path}")

    catch_set = set(int(v) for v in catch_idx.tolist())
    finish_set = set(int(v) for v in finish_idx.tolist())

    try:
        for idx, frame in _iter_frames(video_path, start=0):
            if idx >= handle_track.centers_xy.shape[0]:
                break
            _draw_bbox(frame, machine_track.boxes_xywh[idx], (0, 128, 255), "machine")
            _draw_bbox(frame, handle_track.boxes_xywh[idx], (0, 255, 0), "handle")

            pm = machine_track.centers_xy[idx]
            ph = handle_track.centers_xy[idx]
            if np.isfinite(pm).all() and np.isfinite(ph).all():
                p0 = (int(round(float(pm[0]))), int(round(float(pm[1]))))
                p1 = (int(round(float(ph[0]))), int(round(float(ph[1]))))
                cv2.line(frame, p0, p1, (255, 220, 0), 2, cv2.LINE_AA)

            text = f"rel_px={float(rel_axis_px[idx]):.2f}"
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if np.isfinite(phase[idx]):
                ptxt = f"phase={float(phase[idx]):.3f}"
                cv2.putText(
                    frame,
                    ptxt,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    ptxt,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if idx in catch_set:
                cv2.putText(
                    frame,
                    "CATCH",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (40, 180, 40),
                    3,
                    cv2.LINE_AA,
                )
            elif idx in finish_set:
                cv2.putText(
                    frame,
                    "FINISH",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (30, 30, 220),
                    3,
                    cv2.LINE_AA,
                )

            writer.write(frame)
    finally:
        writer.release()


def _parse_bbox(text: str) -> BBoxXYWH:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox must have 4 comma-separated values: x,y,w,h")
    x, y, w, h = [float(p) for p in parts]
    if w <= 1 or h <= 1:
        raise ValueError("BBox width/height must be > 1")
    return float(x), float(y), float(w), float(h)


def _build_stroke_dataframe(
    *,
    fps: float,
    handle_track: TrackSeries,
    machine_track: TrackSeries,
    m_per_px: Optional[float],
    min_stroke_distance_s: float,
    prominence: Optional[float],
    prominence_frac: float,
    smooth_window_s: float,
) -> Tuple[pd.DataFrame, StrokeEvents, np.ndarray]:
    T = int(handle_track.centers_xy.shape[0])
    if machine_track.centers_xy.shape[0] != T:
        raise ValueError("handle and machine tracks must have same frame count.")

    rel_vec = handle_track.centers_xy - machine_track.centers_xy
    axis = _principal_axis(rel_vec)
    axis_perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    rel_axis_px = np.sum(rel_vec * axis[None, :], axis=1).astype(np.float32)
    rel_perp_px = np.sum(rel_vec * axis_perp[None, :], axis=1).astype(np.float32)

    dt = 1.0 / float(max(fps, 1e-6))
    smooth_window = max(1, int(round(float(smooth_window_s) / dt)))
    rel_axis_smooth = _smooth_signal(rel_axis_px, window=smooth_window).astype(np.float32)
    vel_axis_px_s = np.gradient(rel_axis_smooth.astype(float), dt).astype(np.float32)

    events = _detect_stroke_events(
        rel_axis_smooth.astype(float),
        fps=fps,
        min_stroke_distance_s=min_stroke_distance_s,
        prominence=prominence,
        prominence_frac=prominence_frac,
        smooth_window_s=smooth_window_s,
    )
    phase, stroke_idx, drive = _stroke_phase_series(
        events.catch_idx,
        events.finish_idx,
        length=T,
    )

    frame_idx = np.arange(T, dtype=np.int32)
    time_s = frame_idx.astype(np.float32) / float(max(fps, 1e-6))

    catch_flag = np.zeros((T,), dtype=np.uint8)
    finish_flag = np.zeros((T,), dtype=np.uint8)
    catch_flag[events.catch_idx] = 1
    finish_flag[events.finish_idx] = 1

    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "time_s": time_s,
            "handle_cx_px": handle_track.centers_xy[:, 0].astype(np.float32),
            "handle_cy_px": handle_track.centers_xy[:, 1].astype(np.float32),
            "machine_cx_px": machine_track.centers_xy[:, 0].astype(np.float32),
            "machine_cy_px": machine_track.centers_xy[:, 1].astype(np.float32),
            "handle_status": handle_track.status.astype(np.uint8),
            "machine_status": machine_track.status.astype(np.uint8),
            "relative_axis_px": rel_axis_smooth.astype(np.float32),
            "relative_perp_px": rel_perp_px.astype(np.float32),
            "velocity_axis_px_s": vel_axis_px_s.astype(np.float32),
            "stroke_idx": stroke_idx.astype(np.int32),
            "stroke_phase": phase.astype(np.float32),
            "is_drive": drive.astype(np.uint8),
            "is_catch": catch_flag.astype(np.uint8),
            "is_finish": finish_flag.astype(np.uint8),
            "axis_x": np.full((T,), float(axis[0]), dtype=np.float32),
            "axis_y": np.full((T,), float(axis[1]), dtype=np.float32),
        }
    )

    if m_per_px is not None and m_per_px > 0:
        scale = float(m_per_px)
        df["relative_axis_m"] = df["relative_axis_px"].astype(float) * scale
        df["velocity_axis_m_s"] = df["velocity_axis_px_s"].astype(float) * scale

    return df, events, rel_axis_smooth.astype(np.float32)


def _merge_angles_with_stroke(angles_csv: Path, stroke_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    angles = pd.read_csv(angles_csv)
    if "frame_idx" in angles.columns and "frame_idx" in stroke_df.columns:
        merged = angles.merge(stroke_df, on="frame_idx", how="left", suffixes=("", "_stroke"))
    elif "time_s" in angles.columns and "time_s" in stroke_df.columns:
        merged = angles.merge(stroke_df, on="time_s", how="left", suffixes=("", "_stroke"))
    else:
        raise ValueError("Cannot merge: missing both frame_idx and time_s in one of the CSV files.")

    dup_cols = [c for c in merged.columns if c.endswith("_stroke")]
    for col in dup_cols:
        base = col[: -len("_stroke")]
        if base in merged.columns:
            merged.drop(columns=[col], inplace=True)
        else:
            merged.rename(columns={col: base}, inplace=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def run_stroke_signal_tracking(
    *,
    video_path: Path,
    out_dir: Path,
    angles_csv: Optional[Path] = None,
    reference_frame_idx: int = 0,
    machine_bbox: Optional[BBoxXYWH] = None,
    handle_bbox: Optional[BBoxXYWH] = None,
    annotate: bool = False,
    m_per_px: Optional[float] = None,
    ema_alpha: float = 0.4,
    min_points: int = 10,
    min_stroke_distance_s: float = 0.8,
    prominence: Optional[float] = None,
    prominence_frac: float = 0.1,
    smooth_window_s: float = 0.2,
    create_plot: bool = True,
    plot_video_path: Optional[Path] = None,
    debug_video: bool = False,
) -> StrokeTrackingOutputs:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if annotate or machine_bbox is None or handle_bbox is None:
        machine_bbox, handle_bbox = annotate_handle_and_machine(
            video_path=video_path,
            reference_frame_idx=int(reference_frame_idx),
        )
    assert machine_bbox is not None and handle_bbox is not None

    meta = _video_meta(video_path)
    machine_track = _track_bbox_lk(
        video_path,
        machine_bbox,
        int(reference_frame_idx),
        ema_alpha=float(ema_alpha),
        min_points=int(min_points),
    )
    handle_track = _track_bbox_lk(
        video_path,
        handle_bbox,
        int(reference_frame_idx),
        ema_alpha=float(ema_alpha),
        min_points=int(min_points),
    )

    stroke_df, events, rel_axis_px = _build_stroke_dataframe(
        fps=float(meta.fps),
        handle_track=handle_track,
        machine_track=machine_track,
        m_per_px=m_per_px,
        min_stroke_distance_s=float(min_stroke_distance_s),
        prominence=prominence,
        prominence_frac=float(prominence_frac),
        smooth_window_s=float(smooth_window_s),
    )

    stroke_csv = out_dir / "stroke_signal.csv"
    stroke_npz = out_dir / "stroke_signal.npz"
    stroke_df.to_csv(stroke_csv, index=False)
    np.savez_compressed(
        stroke_npz,
        handle_boxes_xywh=handle_track.boxes_xywh,
        machine_boxes_xywh=machine_track.boxes_xywh,
        handle_centers_xy=handle_track.centers_xy,
        machine_centers_xy=machine_track.centers_xy,
        handle_status=handle_track.status,
        machine_status=machine_track.status,
        stroke_table=stroke_df.to_records(index=False),
        catch_idx=events.catch_idx,
        finish_idx=events.finish_idx,
        fps=np.array(float(meta.fps), dtype=np.float32),
    )

    debug_video_path: Optional[Path] = None
    if debug_video:
        debug_video_path = out_dir / "stroke_tracking_debug.mp4"
        _write_debug_video(
            video_path,
            debug_video_path,
            handle_track=handle_track,
            machine_track=machine_track,
            rel_axis_px=rel_axis_px,
            phase=stroke_df["stroke_phase"].to_numpy(dtype=np.float32),
            catch_idx=events.catch_idx,
            finish_idx=events.finish_idx,
        )

    merged_csv: Optional[Path] = None
    merged_plot: Optional[Path] = None
    if angles_csv is not None:
        angles_csv = Path(angles_csv)
        if not angles_csv.exists():
            raise FileNotFoundError(f"angles_csv not found: {angles_csv}")

        merged_csv = out_dir / f"{angles_csv.stem}_with_stroke.csv"
        merged_df = _merge_angles_with_stroke(angles_csv, stroke_df, merged_csv)

        if create_plot:
            merged_plot = out_dir / f"{angles_csv.stem}_with_stroke_plot.png"
            aux_cols: list[str] = []
            if "relative_axis_m" in merged_df.columns:
                aux_cols.append("relative_axis_m")
            elif "relative_axis_px" in merged_df.columns:
                aux_cols.append("relative_axis_px")
            if "stroke_phase" in merged_df.columns:
                aux_cols.append("stroke_phase")
            generate_angles_plot(
                merged_csv,
                merged_plot,
                title="Rowing angles with handle-machine stroke signal",
                video_path=plot_video_path if plot_video_path is not None else video_path,
                include_thumbnails=(plot_video_path if plot_video_path is not None else video_path)
                is not None,
                thumb_max_px=None,
                thumb_zoom=0.045,
                fig_dpi=300,
                aux_columns=aux_cols if aux_cols else None,
                aux_ylabel="Handle signal",
            )

    return StrokeTrackingOutputs(
        stroke_csv=stroke_csv,
        stroke_npz=stroke_npz,
        merged_angles_csv=merged_csv,
        merged_angles_plot=merged_plot,
        debug_video=debug_video_path,
        fps=float(meta.fps),
        catch_idx=events.catch_idx,
        finish_idx=events.finish_idx,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track rowing handle and machine reference across frames, compute relative stroke "
            "distance/velocity/phase, and optionally merge with an angles CSV."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for stroke tracking artifacts",
    )
    parser.add_argument(
        "--reference-frame-idx",
        type=int,
        default=0,
        help="Frame index used for annotation defaults (default: 0)",
    )
    parser.add_argument(
        "--machine-bbox",
        type=str,
        default=None,
        help="Machine bbox as x,y,w,h in pixels",
    )
    parser.add_argument(
        "--handle-bbox",
        type=str,
        default=None,
        help="Handle bbox as x,y,w,h in pixels",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Open OpenCV ROI selector for machine/handle bbox annotation",
    )
    parser.add_argument(
        "--m-per-px",
        type=float,
        default=None,
        help="Optional scale factor for meter outputs",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.4,
        help="EMA alpha for tracked centers (default: 0.4)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="Minimum LK points per frame before re-initialization (default: 10)",
    )
    parser.add_argument(
        "--min-stroke-distance-s",
        type=float,
        default=0.8,
        help="Minimum separation between catches in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help="Optional prominence threshold in pixels for catch detection",
    )
    parser.add_argument(
        "--prominence-frac",
        type=float,
        default=0.1,
        help="Auto prominence fraction of signal range (default: 0.1)",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.2,
        help="Smoothing window in seconds for stroke signal (default: 0.2)",
    )
    parser.add_argument(
        "--angles-csv",
        type=Path,
        default=None,
        help="Optional angles CSV to merge with stroke signals",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, skip generating merged angles plot.",
    )
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="Write debug video with tracked boxes and event labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    machine_bbox = _parse_bbox(args.machine_bbox) if args.machine_bbox else None
    handle_bbox = _parse_bbox(args.handle_bbox) if args.handle_bbox else None

    outputs = run_stroke_signal_tracking(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        angles_csv=Path(args.angles_csv) if args.angles_csv is not None else None,
        reference_frame_idx=int(args.reference_frame_idx),
        machine_bbox=machine_bbox,
        handle_bbox=handle_bbox,
        annotate=bool(args.annotate),
        m_per_px=args.m_per_px,
        ema_alpha=float(args.ema_alpha),
        min_points=int(args.min_points),
        min_stroke_distance_s=float(args.min_stroke_distance_s),
        prominence=args.prominence,
        prominence_frac=float(args.prominence_frac),
        smooth_window_s=float(args.smooth_window_s),
        create_plot=not args.no_plot,
        plot_video_path=Path(args.video),
        debug_video=bool(args.debug_video),
    )

    print(f"Stroke CSV: {outputs.stroke_csv}")
    print(f"Stroke NPZ: {outputs.stroke_npz}")
    if outputs.merged_angles_csv is not None:
        print(f"Merged angles CSV: {outputs.merged_angles_csv}")
    if outputs.merged_angles_plot is not None:
        print(f"Merged angles plot: {outputs.merged_angles_plot}")
    if outputs.catch_idx.size and outputs.finish_idx.size:
        print("Detected strokes (catch -> finish):")
        n = min(outputs.finish_idx.size, max(0, outputs.catch_idx.size - 1))
        for i in range(n):
            c = int(outputs.catch_idx[i])
            f = int(outputs.finish_idx[i])
            print(f"  {i + 1:02d}: catch={c / outputs.fps:.3f}s finish={f / outputs.fps:.3f}s")
    else:
        print("No strokes detected. Try adjusting min-stroke-distance/prominence/smoothing.")


if __name__ == "__main__":
    main()
