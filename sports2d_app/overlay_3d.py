from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import cv2
import numpy as np

from rowing_pose.kinematics import compute_basic_angles_h36m17
from rowing_pose.skeletons import H36M17_JOINT_NAMES

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

ANGLE_LABEL_JOINT = {
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "trunk_vs_horizontal": "thorax",
    "spine_flexion": "spine",
    "head_vs_trunk": "neck",
}


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


def _draw_bbox_xywh(
    frame_bgr: np.ndarray, box_xywh: Sequence[float], *, color: Tuple[int, int, int], label: str
) -> None:
    x, y, w, h = [int(round(float(v))) for v in box_xywh]
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame_bgr,
        label,
        (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _render_3d_inset(
    J3d: np.ndarray,
    size: Tuple[int, int] = (520, 520),
    mirror_x: bool = True,
    flip_y: bool = False,
    flip_z: bool = False,
    angle_labels: Tuple[Tuple[int, str], ...] | None = None,
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

    if angle_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        for joint_idx, label in angle_labels:
            if not (0 <= joint_idx < 17):
                continue
            xj, yj = float(x_img[joint_idx]), float(y_img[joint_idx])
            if not (np.isfinite(xj) and np.isfinite(yj)):
                continue
            text = str(label)
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int(round(xj + 6))
            y = int(round(yj - 6))
            x = max(2, min(x, W - tw - 2))
            y = max(th + 2, min(y, H - 2))
            cv2.putText(
                canvas,
                text,
                (x + 1, y + 1),
                font,
                scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (x, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    cv2.putText(canvas, "3D", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def generate_pose3d_overlay_video(
    video_path: Path,
    pose3d_npz: Path,
    out_video_path: Path,
    stroke_signal_npz: Path | None = None,
    inset_size: Tuple[int, int] = (520, 520),
    mirror_3d: bool = True,
    flip_3d: bool = False,
    flip_3d_depth: bool = False,
    inset_bg_alpha: float = 0.5,
    show_joint_angles: bool = True,
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

    angle_specs: Tuple[Tuple[int, int], ...] = ()
    angles_deg: np.ndarray | None = None
    if show_joint_angles:
        try:
            if d3.get("joint_names") is not None:
                joint_names = tuple(str(x) for x in np.asarray(d3["joint_names"]).tolist())
            else:
                joint_names = H36M17_JOINT_NAMES
            ang = compute_basic_angles_h36m17(J3d, joint_names)
            angles_deg = np.degrees(ang.values_rad)
            name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
            specs = []
            for a_idx, a_name in enumerate(ang.names):
                joint_name = ANGLE_LABEL_JOINT.get(a_name)
                if joint_name is None:
                    continue
                joint_idx = name_to_idx.get(joint_name)
                if joint_idx is None:
                    continue
                specs.append((a_idx, int(joint_idx)))
            angle_specs = tuple(specs)
        except Exception:
            angles_deg = None
            angle_specs = ()

    handle_boxes_xywh: np.ndarray | None = None
    machine_boxes_xywh: np.ndarray | None = None
    stroke_phase: np.ndarray | None = None
    rel_axis_px: np.ndarray | None = None
    catch_set: set[int] = set()
    finish_set: set[int] = set()
    if stroke_signal_npz is not None:
        try:
            ds = np.load(stroke_signal_npz, allow_pickle=False)
            if "handle_boxes_xywh" in ds.files:
                handle_boxes_xywh = np.asarray(ds["handle_boxes_xywh"], dtype=np.float32)
            if "machine_boxes_xywh" in ds.files:
                machine_boxes_xywh = np.asarray(ds["machine_boxes_xywh"], dtype=np.float32)
            if "catch_idx" in ds.files:
                catch_set = set(int(v) for v in np.asarray(ds["catch_idx"]).astype(int).tolist())
            if "finish_idx" in ds.files:
                finish_set = set(int(v) for v in np.asarray(ds["finish_idx"]).astype(int).tolist())
            if "stroke_table" in ds.files:
                rec = ds["stroke_table"]
                names = tuple(rec.dtype.names or ())
                if "stroke_phase" in names:
                    stroke_phase = np.asarray(rec["stroke_phase"], dtype=np.float32)
                if "relative_axis_px" in names:
                    rel_axis_px = np.asarray(rec["relative_axis_px"], dtype=np.float32)
        except Exception:
            handle_boxes_xywh = None
            machine_boxes_xywh = None
            stroke_phase = None
            rel_axis_px = None
            catch_set = set()
            finish_set = set()

    inset_w, inset_h = int(inset_size[0]), int(inset_size[1])
    margin = 12
    x0 = margin
    y0 = meta.height - inset_h - margin
    if x0 + inset_w > meta.width:
        x0 = max(0, meta.width - inset_w)
    if y0 < 0:
        y0 = 0

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
                angle_labels=(
                    tuple(
                        (
                            joint_idx,
                            f"{int(round(float(angles_deg[idx, a_idx])))}deg",
                        )
                        for a_idx, joint_idx in angle_specs
                        if np.isfinite(angles_deg[idx, a_idx])
                    )
                    if angles_deg is not None and angle_specs
                    else None
                ),
            )
            # Draw border and composite
            roi = frame[y0 : y0 + inset_h, x0 : x0 + inset_w]
            bg_alpha = float(np.clip(inset_bg_alpha, 0.0, 1.0))
            if bg_alpha > 0:
                dimmed = (roi.astype(np.float32) * (1.0 - bg_alpha)).astype(np.uint8)
            else:
                dimmed = roi.copy()
            composite = dimmed
            mask = np.any(inset != 0, axis=2)
            composite[mask] = inset[mask]
            frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = composite

            bx0 = max(0, x0 - 2)
            by0 = max(0, y0 - 2)
            bx1 = min(meta.width - 1, x0 + inset_w + 1)
            by1 = min(meta.height - 1, y0 + inset_h + 1)
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), (0, 0, 0), 1)

            if (
                handle_boxes_xywh is not None
                and machine_boxes_xywh is not None
                and idx < len(handle_boxes_xywh)
                and idx < len(machine_boxes_xywh)
            ):
                _draw_bbox_xywh(
                    frame, machine_boxes_xywh[idx], color=(0, 128, 255), label="machine"
                )
                _draw_bbox_xywh(
                    frame, handle_boxes_xywh[idx], color=(0, 255, 0), label="handle"
                )
                hm = handle_boxes_xywh[idx]
                mm = machine_boxes_xywh[idx]
                hcx = int(round(float(hm[0] + hm[2] * 0.5)))
                hcy = int(round(float(hm[1] + hm[3] * 0.5)))
                mcx = int(round(float(mm[0] + mm[2] * 0.5)))
                mcy = int(round(float(mm[1] + mm[3] * 0.5)))
                cv2.line(frame, (mcx, mcy), (hcx, hcy), (255, 220, 0), 2, cv2.LINE_AA)

            if rel_axis_px is not None and idx < len(rel_axis_px):
                txt = f"handle_rel_px={float(rel_axis_px[idx]):.2f}"
                cv2.putText(
                    frame,
                    txt,
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    txt,
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if stroke_phase is not None and idx < len(stroke_phase) and np.isfinite(stroke_phase[idx]):
                ptxt = f"phase={float(stroke_phase[idx]):.3f}"
                cv2.putText(
                    frame,
                    ptxt,
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    ptxt,
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if idx in catch_set:
                cv2.putText(
                    frame,
                    "CATCH",
                    (16, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (40, 180, 40),
                    2,
                    cv2.LINE_AA,
                )
            elif idx in finish_set:
                cv2.putText(
                    frame,
                    "FINISH",
                    (16, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (30, 30, 220),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()
