from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .io_video import VideoWriter, get_video_metadata, iter_frames, read_frame
from .viz import draw_bbox
from .progress import ProgressReporter, get_progress

BBoxXYWH = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RiggerTrackResult:
    boxes_xywh: np.ndarray  # (T, 4) float32
    centers_xy: np.ndarray  # (T, 2) float32
    status: np.ndarray  # (T,) uint8 (1=tracked, 0=interpolated)


def track_rigger_bbox(
    video_path: Path,
    bbox0_px: BBoxXYWH,
    reference_frame_idx: int,
    out_npz: Path,
    debug_video_path: Optional[Path] = None,
    ema_alpha: float = 0.5,
    min_points: int = 10,
    progress: Optional[ProgressReporter] = None,
) -> RiggerTrackResult:
    """Track a rigid rigger bbox across frames using LK feature tracking.

    This is similar to crop tracking but operates on original frames and aims for
    highly stable hardware tracking (rigger/oarlock).
    """

    meta = get_video_metadata(video_path)
    T = int(meta.frame_count)
    width = int(meta.width)
    height = int(meta.height)
    fps = float(meta.fps) if meta.fps > 0 else 0.0
    prog = get_progress(progress)

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    x0, y0, w0, h0 = [float(v) for v in bbox0_px]
    if w0 <= 2 or h0 <= 2:
        raise ValueError("rigger bbox is invalid (too small).")

    w = min(float(w0), float(width))
    h = min(float(h0), float(height))

    def clamp_center(cx: float, cy: float) -> Tuple[float, float]:
        cx = float(np.clip(cx, w / 2.0, width - w / 2.0))
        cy = float(np.clip(cy, h / 2.0, height - h / 2.0))
        return cx, cy

    def center_to_box(cx: float, cy: float) -> Tuple[float, float, float, float]:
        cx, cy = clamp_center(cx, cy)
        x = float(cx - w / 2.0)
        y = float(cy - h / 2.0)
        return x, y, float(w), float(h)

    def box_mask(shape_hw: Tuple[int, int], box_xywh: Tuple[float, float, float, float]) -> np.ndarray:
        H, W = shape_hw
        x, y, bw, bh = box_xywh
        x0i = int(max(0, round(x)))
        y0i = int(max(0, round(y)))
        x1i = int(min(W, round(x + bw)))
        y1i = int(min(H, round(y + bh)))
        mask = np.zeros((H, W), dtype=np.uint8)
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

    centers = np.full((T, 2), np.nan, dtype=np.float32)
    status = np.zeros((T,), dtype=np.uint8)
    cx0 = x0 + w0 / 2.0
    cy0 = y0 + h0 / 2.0
    centers[reference_frame_idx] = (cx0, cy0)
    status[reference_frame_idx] = 1

    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Forward pass
    prev_gray = None
    prev_pts = None
    prev_box = center_to_box(cx0, cy0)
    track_stage = prog.start("Stage B: rigger tracking", total=T, unit="frame")
    for idx, frame in iter_frames(video_path, start=reference_frame_idx):
        g = to_gray(frame)
        if idx == reference_frame_idx:
            prev_gray = g
            prev_pts = init_features(prev_gray, prev_box)
            track_stage.update(1)
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
        track_stage.update(1)

    # Backward pass
    if reference_frame_idx > 0:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(reference_frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {reference_frame_idx}")
            prev_gray = to_gray(frame)
            prev_box = center_to_box(cx0, cy0)
            prev_pts = init_features(prev_gray, prev_box)

            for idx in range(reference_frame_idx - 1, -1, -1):
                ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
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
                track_stage.update(1)
        finally:
            cap.release()
    track_stage.close()

    def interp_nan(v: np.ndarray) -> np.ndarray:
        out = v.astype(np.float32, copy=True)
        n = out.shape[0]
        x = np.arange(n, dtype=np.float32)
        mask = np.isfinite(out)
        if mask.all():
            return out
        if not mask.any():
            out[:] = out[reference_frame_idx]
            return out
        out[~mask] = np.interp(x[~mask], x[mask], out[mask]).astype(np.float32)
        return out

    centers[:, 0] = interp_nan(centers[:, 0])
    centers[:, 1] = interp_nan(centers[:, 1])

    alpha = float(np.clip(float(ema_alpha), 0.0, 0.999))
    smooth = centers.copy()
    for i in range(1, T):
        smooth[i, 0] = alpha * smooth[i - 1, 0] + (1.0 - alpha) * smooth[i, 0]
        smooth[i, 1] = alpha * smooth[i - 1, 1] + (1.0 - alpha) * smooth[i, 1]

    boxes = np.zeros((T, 4), dtype=np.float32)
    for i in range(T):
        boxes[i] = np.array(center_to_box(float(smooth[i, 0]), float(smooth[i, 1])), dtype=np.float32)

    np.savez_compressed(
        out_npz,
        boxes_xywh=boxes,
        centers_xy=smooth,
        status=status,
        reference_frame_idx=int(reference_frame_idx),
        width=int(width),
        height=int(height),
        fps=float(fps),
    )

    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        with VideoWriter(debug_video_path, fps=fps, frame_size=(width, height)) as vw:
            debug_stage = prog.start("Stage B: rigger debug video", total=T, unit="frame")
            for idx, frame in iter_frames(video_path):
                out = draw_bbox(frame, boxes[idx], color=(0, 128, 255))
                vw.write(out)
                debug_stage.update(1)
            debug_stage.close()

    return RiggerTrackResult(boxes_xywh=boxes, centers_xy=smooth, status=status)
