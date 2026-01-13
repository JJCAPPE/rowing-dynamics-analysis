from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def draw_bbox(frame_bgr: np.ndarray, box_xywh: Sequence[float], color=(0, 255, 0)) -> np.ndarray:
    x, y, w, h = [int(round(float(v))) for v in box_xywh]
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
        if c < min_conf:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 255, 255), -1)
    if edges is not None:
        for a, b in edges:
            xa, ya, ca = float(pts[a, 0]), float(pts[a, 1]), float(pts[a, 2])
            xb, yb, cb = float(pts[b, 0]), float(pts[b, 1]), float(pts[b, 2])
            if ca < min_conf or cb < min_conf:
                continue
            cv2.line(out, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (255, 0, 0), 2)
    return out

