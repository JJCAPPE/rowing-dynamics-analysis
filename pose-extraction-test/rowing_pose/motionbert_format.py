from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from .skeletons import H36M17_JOINT_NAMES, coco17_to_h36m17


MotionBERTNormMode = Literal["pixel", "bbox"]


def normalize_by_image_size(X_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize (x,y) pixel coordinates by image center and min(w,h)/2.

    Matches MotionBERT's `read_input(..., vid_size=(w,h), scale_range=None)` path.
    """

    X = np.asarray(X_px, dtype=np.float32).copy()
    scale = float(min(width, height)) / 2.0
    if scale <= 1e-9:
        raise ValueError("Invalid image size for normalization.")
    X[:, :, 0] = (X[:, :, 0] - float(width) / 2.0) / scale
    X[:, :, 1] = (X[:, :, 1] - float(height) / 2.0) / scale
    return X


def crop_scale_deterministic(motion: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """Deterministic version of MotionBERT's `crop_scale` (no randomness).

    Motion: (T, 17, 3) in arbitrary 2D coordinate system. Conf==0 marks invalid joints.
    Output XY clipped to [-1, 1].
    """

    motion = np.asarray(motion, dtype=np.float32)
    out = motion.copy()
    valid = motion[motion[..., 2] != 0][:, :2]
    if valid.shape[0] < 4:
        return np.zeros_like(out)
    xmin, xmax = float(valid[:, 0].min()), float(valid[:, 0].max())
    ymin, ymax = float(valid[:, 1].min()), float(valid[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin) * float(ratio)
    if scale <= 1e-9:
        return np.zeros_like(out)
    xs = (xmin + xmax - scale) / 2.0
    ys = (ymin + ymax - scale) / 2.0
    out[:, :, :2] = (motion[:, :, :2] - np.array([xs, ys], dtype=np.float32)) / float(scale)
    out[:, :, :2] = (out[:, :, :2] - 0.5) * 2.0
    out[:, :, :2] = np.clip(out[:, :, :2], -1.0, 1.0)
    return out


@dataclass(frozen=True)
class MotionBERTInput:
    X_h36m17: np.ndarray  # (T,17,3) in MotionBERT normalized coords
    joint_names: Tuple[str, ...]


def prepare_motionbert_input_from_coco(
    J2d_coco_px: np.ndarray,
    width: int,
    height: int,
    mode: MotionBERTNormMode = "pixel",
) -> MotionBERTInput:
    """Stage F: COCO-17 (px) → H36M-17 → MotionBERT input coords."""

    J2d_coco_px = np.asarray(J2d_coco_px, dtype=np.float32)
    if J2d_coco_px.ndim != 3 or J2d_coco_px.shape[1:] != (17, 3):
        raise ValueError(f"Expected COCO input shape (T,17,3). Got {J2d_coco_px.shape}")

    X_h36m_px = coco17_to_h36m17(J2d_coco_px)
    X = normalize_by_image_size(X_h36m_px, width=width, height=height)
    if mode == "bbox":
        X = crop_scale_deterministic(X, ratio=1.0)
    elif mode == "pixel":
        pass
    else:
        raise ValueError(f"Unknown MotionBERT norm mode: {mode}")

    return MotionBERTInput(X_h36m17=X.astype(np.float32, copy=False), joint_names=H36M17_JOINT_NAMES)

