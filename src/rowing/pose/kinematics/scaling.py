from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _root_center(arr: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """Root-center along joint axis (T,J,*) by subtracting the root joint."""

    return arr - arr[:, root_idx : root_idx + 1]


def compute_alpha_scale_xy(
    J3d_raw: np.ndarray,  # (T,17,3)
    J2d_m_xy: np.ndarray,  # (T,17,2)
    conf: Optional[np.ndarray] = None,  # (T,17)
    root_idx: int = 0,
    min_conf: float = 0.2,
) -> Optional[float]:
    """Solve a single scalar alpha for best-fitting XY scale (least squares).

    Minimizes: || alpha * P - Q || where P = root-centered J3d_raw_xy and
    Q = root-centered J2d_m_xy. If `conf` is provided, it is used as weights.
    """

    J3d_raw = np.asarray(J3d_raw, dtype=np.float32)
    J2d_m_xy = np.asarray(J2d_m_xy, dtype=np.float32)
    if J3d_raw.ndim != 3 or J3d_raw.shape[1] != 17 or J3d_raw.shape[2] < 2:
        raise ValueError(f"Expected J3d_raw shape (T,17,3). Got {J3d_raw.shape}")
    if J2d_m_xy.shape[:2] != (J3d_raw.shape[0], 17) or J2d_m_xy.shape[2] != 2:
        raise ValueError(f"Expected J2d_m_xy shape (T,17,2). Got {J2d_m_xy.shape}")

    P = _root_center(J3d_raw[:, :, :2], root_idx=root_idx)
    Q = _root_center(J2d_m_xy, root_idx=root_idx)

    P_flat = P.reshape(-1)
    Q_flat = Q.reshape(-1)
    mask = np.isfinite(P_flat) & np.isfinite(Q_flat)

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.shape != (J3d_raw.shape[0], 17):
            raise ValueError(f"conf must be (T,17). Got {conf.shape}")
        w = np.repeat(conf[:, :, None], 2, axis=2).reshape(-1)
        mask = mask & (w >= float(min_conf))
        w = w[mask]
        Pm = P_flat[mask]
        Qm = Q_flat[mask]
        den = float(np.sum(w * Pm * Pm))
        if den <= 1e-12:
            return None
        num = float(np.sum(w * Pm * Qm))
        return float(num / den)

    Pm = P_flat[mask]
    Qm = Q_flat[mask]
    den = float(np.sum(Pm * Pm))
    if den <= 1e-12:
        return None
    num = float(np.sum(Pm * Qm))
    return float(num / den)

