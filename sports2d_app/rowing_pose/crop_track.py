from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .progress import ProgressReporter, get_progress

BBoxXYWH = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CropTrackResult:
    boxes_xywh: np.ndarray  # (T, 4) float32


def track_crop_boxes(
    video_path: Path,
    stabilization_npz: Path,
    bbox0_px: BBoxXYWH,
    out_npy: Path,
    rigger_track_npz: Optional[Path] = None,
    rigger_bbox0_px: Optional[BBoxXYWH] = None,
    debug_video_path: Optional[Path] = None,
    padding: float = 0.2,
    ema_alpha: float = 0.9,
    min_points: int = 10,
    progress: Optional[ProgressReporter] = None,
) -> CropTrackResult:
    """Track a smooth crop bbox over stabilized frames.

    Implements Stage D in `planning.MD`:
    - If rigger_track_npz is provided: anchor crop relative to rigger center.
    - Else: goodFeaturesToTrack + LK tracking on stabilized frames.
    - EMA smoothing of bbox center.
    """

    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)  # (T,2,3)
    T = int(A.shape[0])
    width = int(stab["width"])
    height = int(stab["height"])
    fps = float(stab["fps"]) if "fps" in stab.files else 0.0
    ref_idx = int(stab["reference_frame_idx"]) if "reference_frame_idx" in stab.files else 0
    prog = get_progress(progress)

    out_npy.parent.mkdir(parents=True, exist_ok=True)

    # Fixed crop size/aspect based on the initial bbox (expanded by padding).
    x0, y0, w0, h0 = [float(v) for v in bbox0_px]
    if w0 <= 2 or h0 <= 2:
        raise ValueError("bbox0_px is invalid (too small).")

    w = float(w0 * (1.0 + float(padding)))
    h = float(h0 * (1.0 + float(padding)))
    w = min(w, float(width))
    h = min(h, float(height))

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

    # Storage for centers (raw, then smoothed).
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    cx0 = x0 + w0 / 2.0
    cy0 = y0 + h0 / 2.0
    centers[ref_idx] = (cx0, cy0)

    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    from .io_video import VideoWriter, iter_frames
    from .stabilize import warp_frame

    def stabilized_gray(frame_bgr: np.ndarray, idx: int) -> np.ndarray:
        stab_bgr = warp_frame(frame_bgr, A[idx], (width, height))
        return cv2.cvtColor(stab_bgr, cv2.COLOR_BGR2GRAY)

    if rigger_track_npz is not None and rigger_bbox0_px is not None:
        track = np.load(rigger_track_npz)
        if "centers_xy" in track:
            rigger_centers = track["centers_xy"].astype(np.float32)
        else:
            boxes = track["boxes_xywh"].astype(np.float32)
            rigger_centers = np.stack(
                [boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0], axis=1
            )

        # Convert rigger centers to stabilized coords (translation-only A).
        rigger_centers_stab = rigger_centers.copy()
        rigger_centers_stab[:, 0] += A[:, 0, 2]
        rigger_centers_stab[:, 1] += A[:, 1, 2]

        rx0, ry0, rw0, rh0 = [float(v) for v in rigger_bbox0_px]
        rigger_center_ref = np.array([rx0 + rw0 / 2.0, ry0 + rh0 / 2.0], dtype=np.float32)
        athlete_center_ref = np.array([cx0, cy0], dtype=np.float32)
        offset = athlete_center_ref - rigger_center_ref

        track_stage = prog.start("Stage D: crop tracking (rigger)", total=T, unit="frame")
        for i in range(T):
            centers[i] = rigger_centers_stab[i] + offset
            track_stage.update(1)
        track_stage.close()
    else:
        # Forward pass.
        prev_gray = None
        prev_pts = None
        prev_box = center_to_box(cx0, cy0)
        track_stage = prog.start("Stage D: crop tracking", total=T, unit="frame")
        for idx, frame in iter_frames(video_path, start=ref_idx):
            g = stabilized_gray(frame, idx)
            if idx == ref_idx:
                prev_gray = g
                prev_pts = init_features(prev_gray, prev_box)
                track_stage.update(1)
                continue

            assert prev_gray is not None
            if prev_pts is None or prev_pts.shape[0] < min_points:
                prev_pts = init_features(prev_gray, prev_box)

            cx_prev, cy_prev = float(centers[idx - 1, 0]), float(centers[idx - 1, 1])
            cx_cur, cy_cur = cx_prev, cy_prev

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
                    else:
                        prev_pts = None

            centers[idx] = clamp_center(cx_cur, cy_cur)
            prev_box = center_to_box(float(centers[idx, 0]), float(centers[idx, 1]))
            prev_gray = g
            track_stage.update(1)

        # Backward pass (fills frames < ref_idx).
        if ref_idx > 0:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            try:
                # Initialize at ref.
                ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(ref_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read frame {ref_idx}")
                prev_gray = stabilized_gray(frame, ref_idx)
                prev_box = center_to_box(cx0, cy0)
                prev_pts = init_features(prev_gray, prev_box)

                for idx in range(ref_idx - 1, -1, -1):
                    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    g = stabilized_gray(frame, idx)

                    if prev_pts is None or prev_pts.shape[0] < min_points:
                        prev_pts = init_features(prev_gray, prev_box)

                    cx_prev, cy_prev = float(centers[idx + 1, 0]), float(centers[idx + 1, 1])
                    cx_cur, cy_cur = cx_prev, cy_prev

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
                            else:
                                prev_pts = None

                    centers[idx] = clamp_center(cx_cur, cy_cur)
                    prev_box = center_to_box(float(centers[idx, 0]), float(centers[idx, 1]))
                    prev_gray = g
                    track_stage.update(1)
            finally:
                cap.release()
        track_stage.close()

    # Interpolate any missing centers.
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

    # EMA smoothing (forward).
    alpha = float(ema_alpha)
    alpha = float(np.clip(alpha, 0.0, 0.999))
    smooth = centers.copy()
    for i in range(1, T):
        smooth[i, 0] = alpha * smooth[i - 1, 0] + (1.0 - alpha) * smooth[i, 0]
        smooth[i, 1] = alpha * smooth[i - 1, 1] + (1.0 - alpha) * smooth[i, 1]

    boxes = np.zeros((T, 4), dtype=np.float32)
    for i in range(T):
        boxes[i] = np.array(center_to_box(float(smooth[i, 0]), float(smooth[i, 1])), dtype=np.float32)

    np.save(out_npy, boxes)

    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        from .viz import draw_bbox

        with VideoWriter(debug_video_path, fps=fps, frame_size=(width, height)) as vw:
            debug_stage = prog.start("Stage D: crop debug video", total=T, unit="frame")
            for idx, frame in iter_frames(video_path):
                stab_bgr = warp_frame(frame, A[idx], (width, height))
                out = draw_bbox(stab_bgr, boxes[idx])
                vw.write(out)
                debug_stage.update(1)
            debug_stage.close()

    return CropTrackResult(boxes_xywh=boxes)

