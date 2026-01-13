from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .io_video import VideoWriter, get_video_metadata, iter_frames, read_frame

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class StabilizationResult:
    """Per-frame translation-only stabilization."""

    A: np.ndarray  # (T, 2, 3) float32
    anchor_xy: np.ndarray  # (T, 2) float32
    status: np.ndarray  # (T,) uint8  (1=tracked, 0=bad)


def compute_stabilization(
    video_path: Path,
    anchor0_px: Point2D,
    reference_frame_idx: int,
    out_npz: Path,
    debug_video_path: Optional[Path] = None,
) -> StabilizationResult:
    """Track anchor point and create per-frame affine translations.

    Notes:
    - This is translation-only stabilization (Stage B in `planning.MD`).
    - If LK fails, we attempt template matching around the last known good anchor.
    - Any missing anchors are linearly interpolated before computing transforms.
    """

    meta = get_video_metadata(video_path)
    if reference_frame_idx < 0 or reference_frame_idx >= max(1, meta.frame_count):
        raise ValueError("reference_frame_idx out of range.")

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # Params (kept here for now; `run.json` also stores defaults)
    lk_win_size = (21, 21)
    lk_max_level = 3
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    template_half_size = 16
    template_search_radius = 50
    template_min_score = 0.6

    T = int(meta.frame_count)
    anchor_xy = np.full((T, 2), np.nan, dtype=np.float32)
    status = np.zeros((T,), dtype=np.uint8)

    def extract_patch(gray: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
        cx, cy = center
        x = int(round(cx))
        y = int(round(cy))
        hs = int(template_half_size)
        x0 = max(0, x - hs)
        y0 = max(0, y - hs)
        x1 = min(gray.shape[1], x + hs + 1)
        y1 = min(gray.shape[0], y + hs + 1)
        patch = gray[y0:y1, x0:x1]
        return patch

    def template_match(
        gray: np.ndarray, last_good_gray: np.ndarray, last_good_pt: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        templ = extract_patch(last_good_gray, last_good_pt)
        if templ.size == 0 or templ.shape[0] < 5 or templ.shape[1] < 5:
            return None

        cx, cy = last_good_pt
        hs = int(template_half_size)
        sr = int(template_search_radius)
        x0 = int(round(cx)) - sr - hs
        y0 = int(round(cy)) - sr - hs
        x1 = int(round(cx)) + sr + hs + 1
        y1 = int(round(cy)) + sr + hs + 1
        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(gray.shape[1], x1)
        y1c = min(gray.shape[0], y1)
        search = gray[y0c:y1c, x0c:x1c]
        if search.shape[0] < templ.shape[0] or search.shape[1] < templ.shape[1]:
            return None

        res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < template_min_score:
            return None

        # max_loc is top-left of the best match in search ROI
        x_tl = max_loc[0] + x0c
        y_tl = max_loc[1] + y0c
        x_center = float(x_tl + templ.shape[1] / 2.0)
        y_center = float(y_tl + templ.shape[0] / 2.0)
        return x_center, y_center

    def track_forward() -> None:
        frame_ref = read_frame(video_path, reference_frame_idx)
        prev_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        prev_pt = np.array(anchor0_px, dtype=np.float32).reshape(1, 1, 2)
        prev_has_anchor = True
        last_good_gray = prev_gray
        last_good_pt = (float(anchor0_px[0]), float(anchor0_px[1]))

        for idx, frame in iter_frames(video_path, start=reference_frame_idx + 1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pt_xy: Optional[Tuple[float, float]] = None
            if prev_has_anchor:
                nxt, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    prev_pt,
                    None,
                    winSize=lk_win_size,
                    maxLevel=lk_max_level,
                    criteria=lk_criteria,
                )
                if st is not None and int(st.reshape(-1)[0]) == 1 and nxt is not None:
                    x, y = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
                    if 0 <= x < meta.width and 0 <= y < meta.height:
                        pt_xy = (x, y)

            if pt_xy is None:
                pt_xy = template_match(gray, last_good_gray, last_good_pt)

            if pt_xy is None:
                anchor_xy[idx, :] = np.nan
                status[idx] = 0
                prev_has_anchor = False
                prev_gray = gray
                continue

            anchor_xy[idx, :] = (pt_xy[0], pt_xy[1])
            status[idx] = 1
            prev_pt = np.array([[pt_xy]], dtype=np.float32)
            prev_gray = gray
            prev_has_anchor = True
            last_good_gray = gray
            last_good_pt = pt_xy

    def _open_capture() -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        return cap

    def _read_gray_with_cap(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        if not ok:
            raise RuntimeError(f"Failed to seek to frame {frame_idx}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def track_backward() -> None:
        cap = _open_capture()
        try:
            prev_gray = _read_gray_with_cap(cap, reference_frame_idx)
            prev_pt = np.array(anchor0_px, dtype=np.float32).reshape(1, 1, 2)
            prev_has_anchor = True
            last_good_gray = prev_gray
            last_good_pt = (float(anchor0_px[0]), float(anchor0_px[1]))

            for idx in range(reference_frame_idx - 1, -1, -1):
                gray = _read_gray_with_cap(cap, idx)

                pt_xy: Optional[Tuple[float, float]] = None
                if prev_has_anchor:
                    nxt, st, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray,
                        prev_pt,
                        None,
                        winSize=lk_win_size,
                        maxLevel=lk_max_level,
                        criteria=lk_criteria,
                    )
                    if st is not None and int(st.reshape(-1)[0]) == 1 and nxt is not None:
                        x, y = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
                        if 0 <= x < meta.width and 0 <= y < meta.height:
                            pt_xy = (x, y)

                if pt_xy is None:
                    pt_xy = template_match(gray, last_good_gray, last_good_pt)

                if pt_xy is None:
                    anchor_xy[idx, :] = np.nan
                    status[idx] = 0
                    prev_has_anchor = False
                    prev_gray = gray
                    continue

                anchor_xy[idx, :] = (pt_xy[0], pt_xy[1])
                status[idx] = 1
                prev_pt = np.array([[pt_xy]], dtype=np.float32)
                prev_gray = gray
                prev_has_anchor = True
                last_good_gray = gray
                last_good_pt = pt_xy
        finally:
            cap.release()

    # Seed reference.
    anchor_xy[reference_frame_idx, :] = (float(anchor0_px[0]), float(anchor0_px[1]))
    status[reference_frame_idx] = 1

    # Track forward and backward from reference.
    if reference_frame_idx < T - 1:
        track_forward()
    if reference_frame_idx > 0:
        track_backward()

    # Interpolate any missing anchors (per coordinate).
    def interp_nan(v: np.ndarray) -> np.ndarray:
        out = v.astype(np.float32, copy=True)
        n = out.shape[0]
        x = np.arange(n, dtype=np.float32)
        mask = np.isfinite(out)
        if mask.all():
            return out
        if not mask.any():
            # catastrophic failure; fall back to constant anchor0
            out[:] = out[reference_frame_idx]
            return out
        out[~mask] = np.interp(x[~mask], x[mask], out[mask]).astype(np.float32)
        return out

    anchor_xy[:, 0] = interp_nan(anchor_xy[:, 0])
    anchor_xy[:, 1] = interp_nan(anchor_xy[:, 1])

    # Build dx/dy translation series and smooth gently (while preserving ref=0 shift).
    dx = float(anchor0_px[0]) - anchor_xy[:, 0]
    dy = float(anchor0_px[1]) - anchor_xy[:, 1]
    try:
        # window length must be odd and <= T
        win = min(31, T if T % 2 == 1 else T - 1)
        if win >= 5:
            from scipy.signal import savgol_filter  # type: ignore

            dx_s = savgol_filter(dx, window_length=win, polyorder=2, mode="interp")
            dy_s = savgol_filter(dy, window_length=win, polyorder=2, mode="interp")
            dx = dx_s.astype(np.float32)
            dy = dy_s.astype(np.float32)
    except Exception:
        # If SciPy isn't available or filter fails, keep raw dx/dy.
        dx = dx.astype(np.float32)
        dy = dy.astype(np.float32)

    dx = dx - float(dx[reference_frame_idx])
    dy = dy - float(dy[reference_frame_idx])

    A = np.zeros((T, 2, 3), dtype=np.float32)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 0, 2] = dx
    A[:, 1, 2] = dy

    np.savez_compressed(
        out_npz,
        A=A,
        anchor_xy=anchor_xy,
        status=status,
        reference_frame_idx=int(reference_frame_idx),
        anchor0_px=np.array(anchor0_px, dtype=np.float32),
        width=int(meta.width),
        height=int(meta.height),
        fps=float(meta.fps),
    )

    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_size = (int(meta.width), int(meta.height))
        with VideoWriter(debug_video_path, fps=meta.fps, frame_size=out_size) as vw:
            for idx, frame in iter_frames(video_path):
                stab = warp_frame(frame, A[idx], out_size)
                x0, y0 = int(round(anchor0_px[0])), int(round(anchor0_px[1]))
                cv2.circle(stab, (x0, y0), 6, (0, 0, 255), -1)
                vw.write(stab)

    return StabilizationResult(A=A, anchor_xy=anchor_xy, status=status)


def warp_frame(frame_bgr: np.ndarray, A_2x3: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Apply a 2x3 affine matrix to a BGR frame. (Implemented in Task 3.)"""
    W, H = int(out_size[0]), int(out_size[1])
    return cv2.warpAffine(
        frame_bgr,
        A_2x3.astype(np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

