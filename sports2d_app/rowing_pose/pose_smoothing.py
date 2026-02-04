from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import POSE_SMOOTHING_DEFAULTS


@dataclass(frozen=True)
class PoseSmoothingParams:
    enabled: bool
    conf_threshold: float
    max_gap: int
    median_window: int


def pose_smoothing_params_from_dict(params: Optional[dict]) -> PoseSmoothingParams:
    merged = dict(POSE_SMOOTHING_DEFAULTS)
    if isinstance(params, dict):
        merged.update(params)

    enabled = bool(merged.get("enabled", True))
    conf_threshold = float(merged.get("conf_threshold", 0.3))
    conf_threshold = min(max(conf_threshold, 0.0), 1.0)
    max_gap = int(merged.get("max_gap", 5))
    max_gap = max(0, max_gap)
    median_window = int(merged.get("median_window", 5))
    if median_window < 1:
        median_window = 1
    if median_window % 2 == 0:
        median_window += 1

    return PoseSmoothingParams(
        enabled=enabled,
        conf_threshold=conf_threshold,
        max_gap=max_gap,
        median_window=median_window,
    )


def smooth_keypoints_2d(J2d_px: np.ndarray, params: PoseSmoothingParams) -> np.ndarray:
    """Smooth 2D keypoints (T,J,3) using conf-aware gap fill + median filter."""
    k = np.asarray(J2d_px, dtype=np.float32)
    if k.ndim != 3 or k.shape[2] < 3:
        raise ValueError(f"Expected J2d_px shape (T,J,3). Got {k.shape}")
    xy = k[:, :, 0:2]
    conf = k[:, :, 2]
    xy_s = _smooth_coords(
        xy,
        conf=conf,
        conf_threshold=params.conf_threshold,
        max_gap=params.max_gap,
        median_window=params.median_window,
    )
    out = k.copy()
    out[:, :, 0:2] = xy_s
    return out


def smooth_joints_3d(J3d: np.ndarray, params: PoseSmoothingParams) -> np.ndarray:
    """Smooth 3D joints (T,J,3) using median filter (no confidence)."""
    k = np.asarray(J3d, dtype=np.float32)
    if k.ndim != 3 or k.shape[2] != 3:
        raise ValueError(f"Expected J3d shape (T,J,3). Got {k.shape}")
    return _smooth_coords(
        k,
        conf=None,
        conf_threshold=params.conf_threshold,
        max_gap=params.max_gap,
        median_window=params.median_window,
    )


def _smooth_coords(
    coords: np.ndarray,
    *,
    conf: Optional[np.ndarray],
    conf_threshold: float,
    max_gap: int,
    median_window: int,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3:
        raise ValueError(f"Expected coords shape (T,J,D). Got {coords.shape}")

    T, J, D = coords.shape
    out = coords.copy()

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.ndim == 3 and conf.shape[2] == 1:
            conf = conf[:, :, 0]
        if conf.shape != (T, J):
            raise ValueError(f"Expected conf shape (T,J). Got {conf.shape}")
        valid_joint = conf >= float(conf_threshold)
    else:
        valid_joint = np.ones((T, J), dtype=bool)

    win = _normalize_window(median_window, T)

    for j in range(J):
        valid_j = valid_joint[:, j]
        for d in range(D):
            v = coords[:, j, d].astype(np.float32, copy=True)
            valid = valid_j & np.isfinite(v)
            v[~valid] = np.nan
            v_filled, filled = _fill_short_gaps(v, valid, max_gap=max_gap)
            v_med = _median_filter_1d(v_filled, window=win)
            mask_smooth = valid | filled
            v_med[~mask_smooth] = np.nan
            out[:, j, d] = v_med

    return out.astype(np.float32, copy=False)


def _normalize_window(window: int, length: int) -> int:
    if window <= 1 or length <= 1:
        return 1
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > length:
        w = length if length % 2 == 1 else max(1, length - 1)
    return max(1, w)


def _fill_short_gaps(
    v: np.ndarray, valid: np.ndarray, *, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    out = v.astype(np.float32, copy=True)
    filled = np.zeros_like(valid, dtype=bool)
    n = out.shape[0]
    if max_gap <= 0 or n == 0:
        return out, filled

    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        start = i
        while i < n and not valid[i]:
            i += 1
        end = i
        gap_len = end - start
        if gap_len <= max_gap and start > 0 and end < n:
            v0 = out[start - 1]
            v1 = out[end]
            if np.isfinite(v0) and np.isfinite(v1):
                for k in range(gap_len):
                    alpha = float(k + 1) / float(gap_len + 1)
                    out[start + k] = v0 + (v1 - v0) * alpha
                    filled[start + k] = True
    return out, filled


def _median_filter_1d(v: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1:
        return v.astype(np.float32, copy=False)
    n = v.shape[0]
    half = window // 2
    out = v.astype(np.float32, copy=True)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = v[lo:hi]
        if np.isfinite(window_vals).any():
            out[i] = np.nanmedian(window_vals)
        else:
            out[i] = np.nan
    return out
