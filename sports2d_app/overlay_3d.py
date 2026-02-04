from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


H36M17_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


def get_video_metadata(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def iter_frames(video_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _render_3d_inset(
    J3d: np.ndarray,
    size: Tuple[int, int] = (520, 520),
    mirror_x: bool = True,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    W, H = int(size[0]), int(size[1])
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    J = J3d.copy().astype(np.float32)
    if flip_z:
        J[:, 2] *= -1.0

    # Center around pelvis (root)
    root = J[0].copy()
    Jr = J - root[None, :]

    xs = Jr[:, 0]
    ys = Jr[:, 1]
    zs = Jr[:, 2]

    if mirror_x:
        xs = -xs
    if flip_y:
        ys = -ys

    x_min = np.nanmin(xs)
    x_max = np.nanmax(xs)
    y_min = np.nanmin(ys)
    y_max = np.nanmax(ys)
    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        return canvas

    pad = 20
    rx = max(1e-6, x_max - x_min)
    ry = max(1e-6, y_max - y_min)
    sx = float(W - 2 * pad) / rx
    sy = float(H - 2 * pad) / ry
    s = float(min(sx, sy))

    x0 = float((x_min + x_max) / 2.0)
    y0 = float((y_min + y_max) / 2.0)

    x_img = (xs - x0) * s + float(W) / 2.0
    y_img = (ys - y0) * s + float(H) / 2.0

    def depth_color(zv: float) -> Tuple[int, int, int]:
        if not np.isfinite(zv):
            return (180, 180, 180)
        t = float(np.clip((zv + 1.0) / 2.0, 0.0, 1.0))
        b = int(round(255 * (1.0 - t)))
        g = int(round(255 * t))
        return (b, g, 60)

    for a, b in H36M17_EDGES:
        xa, ya = float(x_img[a]), float(y_img[a])
        xb, yb = float(x_img[b]), float(y_img[b])
        if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            continue
        c = depth_color(float((zs[a] + zs[b]) / 2.0))
        cv2.line(canvas, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), c, 2)

    for j in range(17):
        xj, yj = float(x_img[j]), float(y_img[j])
        if not (np.isfinite(xj) and np.isfinite(yj)):
            continue
        cv2.circle(canvas, (int(round(xj)), int(round(yj))), 3, (255, 255, 255), -1)

    cv2.putText(canvas, "3D", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def generate_pose3d_overlay_video(
    video_path: Path,
    pose3d_npz: Path,
    out_video_path: Path,
    inset_size: Tuple[int, int] = (520, 520),
    mirror_3d: bool = True,
    flip_3d: bool = False,
    flip_3d_depth: bool = False,
) -> None:
    video_path = Path(video_path)
    out_video_path = Path(out_video_path)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    meta = get_video_metadata(video_path)

    d3 = dict(np.load(pose3d_npz, allow_pickle=False))
    if d3.get("J3d_m") is not None and np.asarray(d3["J3d_m"]).size:
        J3d = np.asarray(d3["J3d_m"], dtype=np.float32).reshape(-1, 17, 3)
    else:
        J3d = np.asarray(d3["J3d_raw"], dtype=np.float32).reshape(-1, 17, 3)

    inset_w, inset_h = int(inset_size[0]), int(inset_size[1])
    margin = 12
    x0 = max(0, meta.width - inset_w - margin)
    y0 = margin

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(
        str(out_video_path), fourcc, meta.fps, (meta.width, meta.height)
    )
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, meta.fps, (meta.width, meta.height)
        )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_video_path}")

    try:
        for idx, frame in iter_frames(video_path):
            if idx >= len(J3d):
                break
            inset = _render_3d_inset(
                J3d[idx],
                size=(inset_w, inset_h),
                mirror_x=mirror_3d,
                flip_y=flip_3d,
                flip_z=flip_3d_depth,
            )
            # Draw border and composite
            cv2.rectangle(
                frame,
                (x0 - 2, y0 - 2),
                (x0 + inset_w + 2, y0 + inset_h + 2),
                (0, 0, 0),
                -1,
            )
            frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
            writer.write(frame)
    finally:
        writer.release()
