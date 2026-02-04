from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def draw_bbox(frame_bgr: np.ndarray, box_xywh: Sequence[float], color=(0, 255, 0)) -> np.ndarray:
    vals = [float(v) for v in box_xywh]
    if not np.isfinite(vals).all():
        return frame_bgr.copy()
    x, y, w, h = [int(round(v)) for v in vals]
    out = frame_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out


def draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints_xyc: np.ndarray,  # (J,3) x,y,conf
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    min_conf: float = 0.2,
) -> np.ndarray:
    out = frame_bgr.copy()
    pts = keypoints_xyc
    for j in range(pts.shape[0]):
        x, y, c = float(pts[j, 0]), float(pts[j, 1]), float(pts[j, 2])
        if not np.isfinite([x, y, c]).all() or c < min_conf:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 255, 255), -1)
    if edges is not None:
        for a, b in edges:
            xa, ya, ca = float(pts[a, 0]), float(pts[a, 1]), float(pts[a, 2])
            xb, yb, cb = float(pts[b, 0]), float(pts[b, 1]), float(pts[b, 2])
            if not np.isfinite([xa, ya, ca, xb, yb, cb]).all() or ca < min_conf or cb < min_conf:
                continue
            cv2.line(
                out,
                (int(round(xa)), int(round(ya))),
                (int(round(xb)), int(round(yb))),
                (255, 0, 0),
                2,
            )
    return out


def apply_affine_to_xy(
    xy: np.ndarray,
    A_2x3: np.ndarray,
) -> np.ndarray:
    """Apply a 2x3 affine transform to (N,2) points."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy shape (N,2). Got {xy.shape}")
    A = np.asarray(A_2x3, dtype=np.float32)
    if A.shape != (2, 3):
        raise ValueError(f"Expected A shape (2,3). Got {A.shape}")

    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy1 = np.concatenate([xy, ones], axis=1)  # (N,3)
    out = (xy1 @ A.T).astype(np.float32)  # (N,2)
    return out


def apply_affine_to_keypoints_xyc(
    keypoints_xyc: np.ndarray,
    A_2x3: np.ndarray,
) -> np.ndarray:
    """Apply affine to keypoints (J,3) x,y,conf (conf unchanged)."""
    k = np.asarray(keypoints_xyc, dtype=np.float32)
    if k.ndim != 2 or k.shape[1] < 3:
        raise ValueError(f"Expected keypoints shape (J,3+). Got {k.shape}")
    out = k.copy()
    out[:, 0:2] = apply_affine_to_xy(out[:, 0:2], A_2x3)
    return out


def draw_text_panel(
    frame_bgr: np.ndarray,
    lines: Sequence[str],
    origin_xy: Tuple[int, int] = (10, 20),
    line_height: int = 18,
    font_scale: float = 0.5,
    color=(255, 255, 255),
    thickness: int = 1,
    bg_color=(0, 0, 0),
    bg_alpha: float = 0.5,
) -> np.ndarray:
    """Draw a semi-transparent text panel (top-left by default)."""
    out = frame_bgr.copy()
    if not lines:
        return out

    x0, y0 = int(origin_xy[0]), int(origin_xy[1])
    pad = 6

    # Estimate panel size.
    widths = []
    for s in lines:
        (w, _), _ = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        widths.append(int(w))
    panel_w = max(widths) + 2 * pad
    panel_h = len(lines) * line_height + 2 * pad

    x1 = min(out.shape[1], x0 + panel_w)
    y1 = min(out.shape[0], y0 + panel_h)
    x0c = max(0, x0)
    y0c = max(0, y0 - pad)  # allow a bit above first baseline

    overlay = out.copy()
    cv2.rectangle(overlay, (x0c, y0c), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, float(bg_alpha), out, 1.0 - float(bg_alpha), 0.0, out)

    y = y0
    for s in lines:
        cv2.putText(
            out,
            str(s),
            (x0 + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += int(line_height)
    return out


def draw_values_at_named_joints(
    frame_bgr: np.ndarray,
    keypoints_xyc: np.ndarray,  # (J,3) x,y,conf
    joint_names: Sequence[str],
    values: dict[str, float],
    *,
    min_conf: float = 0.2,
    fmt: str = "{name}: {value:.1f} deg",
    color=(0, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
    offset_px: Tuple[int, int] = (6, -6),
) -> np.ndarray:
    """Draw per-joint labels for a subset of joints by name."""
    out = frame_bgr.copy()
    name_to_idx = {str(n): int(i) for i, n in enumerate(joint_names)}
    for name, val in values.items():
        if name not in name_to_idx:
            continue
        j = name_to_idx[name]
        x, y, c = float(keypoints_xyc[j, 0]), float(keypoints_xyc[j, 1]), float(keypoints_xyc[j, 2])
        if c < min_conf or not np.isfinite(x) or not np.isfinite(y):
            continue
        text = fmt.format(name=name, value=float(val))
        org = (int(round(x + offset_px[0])), int(round(y + offset_px[1])))
        cv2.putText(
            out,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )
    return out
