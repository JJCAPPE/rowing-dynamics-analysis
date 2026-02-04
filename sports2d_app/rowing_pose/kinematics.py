from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def angle_abc(A: np.ndarray, B: np.ndarray, C: np.ndarray, eps: float = 1e-9) -> float:
    """Angle at B (in radians) formed by A-B-C for 2D or 3D points."""

    u = A - B
    v = C - B
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return float("nan")
    cos = float(np.dot(u, v) / (nu * nv))
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.arccos(cos))


@dataclass(frozen=True)
class AnglesPerFrame:
    names: Tuple[str, ...]
    values_rad: np.ndarray  # (T, K)


def compute_basic_angles_h36m17(J3d: np.ndarray, joint_names: Sequence[str]) -> AnglesPerFrame:
    """Compute a small set of rowing-relevant joint angles.

    Expected joints: H36M-17 naming (pelvis, hips, knees, ankles, thorax, shoulders, elbows, wrists).
    Angles are returned in radians.
    """
    J3d = np.asarray(J3d, dtype=np.float32)
    if J3d.ndim != 3 or J3d.shape[2] < 2:
        raise ValueError(f"Expected J3d shape (T,J,3) or (T,J,2+). Got {J3d.shape}")

    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}

    def idx(name: str) -> int:
        if name not in name_to_idx:
            raise KeyError(f"Missing joint '{name}' in joint_names")
        return int(name_to_idx[name])

    pelvis = idx("pelvis")
    thorax = idx("thorax")
    lhip, lknee, lank = idx("left_hip"), idx("left_knee"), idx("left_ankle")
    rhip, rknee, rank = idx("right_hip"), idx("right_knee"), idx("right_ankle")
    lsho, lelb, lwri = idx("left_shoulder"), idx("left_elbow"), idx("left_wrist")
    rsho, relb, rwri = idx("right_shoulder"), idx("right_elbow"), idx("right_wrist")

    T = int(J3d.shape[0])
    vals = np.full((T, 7), np.nan, dtype=np.float32)

    # Knee angles.
    for t in range(T):
        vals[t, 0] = angle_abc(J3d[t, lhip, :3], J3d[t, lknee, :3], J3d[t, lank, :3])
        vals[t, 1] = angle_abc(J3d[t, rhip, :3], J3d[t, rknee, :3], J3d[t, rank, :3])

        # Hip angles: thorax–hip–knee
        vals[t, 2] = angle_abc(J3d[t, thorax, :3], J3d[t, lhip, :3], J3d[t, lknee, :3])
        vals[t, 3] = angle_abc(J3d[t, thorax, :3], J3d[t, rhip, :3], J3d[t, rknee, :3])

        # Elbow angles.
        vals[t, 4] = angle_abc(J3d[t, lsho, :3], J3d[t, lelb, :3], J3d[t, lwri, :3])
        vals[t, 5] = angle_abc(J3d[t, rsho, :3], J3d[t, relb, :3], J3d[t, rwri, :3])

        # Trunk angle vs horizontal (image X axis) in XY plane: angle(trunk_xy, +x).
        v = (J3d[t, thorax, :2] - J3d[t, pelvis, :2]).astype(np.float32)
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            cos = float(np.clip(v[0] / nv, -1.0, 1.0))
            vals[t, 6] = float(np.arccos(cos))

    names = (
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
        "left_elbow",
        "right_elbow",
        "trunk_vs_horizontal",
    )
    return AnglesPerFrame(names=names, values_rad=vals)

