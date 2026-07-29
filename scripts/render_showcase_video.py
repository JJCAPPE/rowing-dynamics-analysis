#!/usr/bin/env python3
"""Render a website-ready rowing biomechanics showcase video.

The compositor combines the existing whole-body Sports2D debug render,
MotionBERT 3D joints, frame-level stroke/angle data, and matched RP3 curves.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


CANVAS_W = 1920
CANVAS_H = 1080
MARGIN = 42

VIDEO_RECT = (42, 122, 1280, 720)
ANGLES_RECT = (42, 864, 1280, 174)
FORCE_RECT = (1348, 122, 530, 336)
POSE_RECT = (1348, 476, 530, 300)
METRICS_RECT = (1348, 794, 530, 244)

# Website palette from https://cv-nu-sage.vercel.app/.
BG = (10, 11, 11)  # #0b0b0a
FG = (232, 239, 241)  # #f1efe8
MUTED = (159, 167, 170)  # #aaa79f
SUBTLE = (117, 124, 127)  # #7f7c75
LINE = (52, 54, 55)
PANEL = (20, 22, 22)  # #161614
SURFACE = (26, 29, 29)  # #1d1d1a
ACCENT = (42, 209, 230)  # #e6d12a
ACCENT_INK = (15, 17, 17)

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

ANGLE_SPECS = (
    ("KNEE", "right_knee_deg"),
    ("HIP", "right_hip_deg"),
    ("ELBOW", "right_elbow_deg"),
    ("TRUNK", "trunk_vs_horizontal_deg"),
    ("SPINE", "spine_flexion_deg"),
)


@dataclass(frozen=True)
class StrokeCurve:
    video_stroke_idx: int
    ordinal: int
    total: int
    rp3_stroke_number: int
    s: np.ndarray
    force: np.ndarray
    peak_force: float
    drive_time_s: float
    cycle_time_s: float

    @property
    def stroke_rate(self) -> float:
        if self.cycle_time_s <= 0:
            return float("nan")
        return 60.0 / self.cycle_time_s


@dataclass(frozen=True)
class ShowcaseData:
    stroke: pd.DataFrame
    angles: dict[str, np.ndarray]
    angle_ranges: dict[str, tuple[float, float]]
    pose3d: np.ndarray
    pose_bounds: tuple[float, float, float, float]
    curves: dict[int, StrokeCurve]
    force_max: float
    catch_indices: np.ndarray
    finish_indices: np.ndarray


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _prepare_force_curves(segments: pd.DataFrame) -> dict[int, StrokeCurve]:
    required = {
        "video_stroke_idx",
        "rp3_stroke_number",
        "s_force",
        "force_raw",
        "rp3_drive_s",
        "rp3_cycle_s",
    }
    missing = sorted(required - set(segments.columns))
    if missing:
        raise ValueError(f"Matched segments CSV missing columns: {missing}")

    clean = segments.copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["video_stroke_idx", "s_force", "force_raw"])

    stroke_ids = sorted(int(value) for value in clean["video_stroke_idx"].unique())
    curves: dict[int, StrokeCurve] = {}
    for ordinal, stroke_idx in enumerate(stroke_ids, start=1):
        group = clean[clean["video_stroke_idx"] == stroke_idx].sort_values("s_force")
        if len(group) < 2:
            continue
        s = group["s_force"].to_numpy(dtype=np.float64)
        force = group["force_raw"].to_numpy(dtype=np.float64)
        valid = np.isfinite(s) & np.isfinite(force)
        s = s[valid]
        force = force[valid]
        if len(s) < 2:
            continue

        curves[stroke_idx] = StrokeCurve(
            video_stroke_idx=stroke_idx,
            ordinal=ordinal,
            total=len(stroke_ids),
            rp3_stroke_number=int(round(_safe_float(group["rp3_stroke_number"].iloc[0], stroke_idx))),
            s=s,
            force=force,
            peak_force=float(np.nanmax(force)),
            drive_time_s=_safe_float(group["rp3_drive_s"].iloc[0]),
            cycle_time_s=_safe_float(group["rp3_cycle_s"].iloc[0]),
        )
    if not curves:
        raise ValueError("No usable force curves found in matched segments CSV.")
    return curves


def _project_pose_values(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root = pose[..., :1, :]
    centered = pose - root
    x = -centered[..., 0] + 0.28 * centered[..., 2]
    y = centered[..., 1] - 0.08 * centered[..., 2]
    return x, y


def _load_data(run_dir: Path) -> ShowcaseData:
    stroke_path = run_dir / "inference" / "stroke_signal_with_drive_events.csv"
    angles_path = run_dir / "stroke" / "angles_h36m_with_stroke.csv"
    segments_path = run_dir / "inference" / "rp3_pose_force_matched_segments.csv"
    pose_path = run_dir / "motionbert" / "pose3d.npz"
    for path in (stroke_path, angles_path, segments_path, pose_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing showcase input: {path}")

    stroke = pd.read_csv(stroke_path)
    angle_df = pd.read_csv(angles_path)
    segments = pd.read_csv(segments_path)
    pose_npz = np.load(pose_path, allow_pickle=False)
    if "J3d_m" in pose_npz.files and np.asarray(pose_npz["J3d_m"]).size:
        pose3d = np.asarray(pose_npz["J3d_m"], dtype=np.float32).reshape(-1, 17, 3)
    else:
        pose3d = np.asarray(pose_npz["J3d_raw"], dtype=np.float32).reshape(-1, 17, 3)

    angles: dict[str, np.ndarray] = {}
    angle_ranges: dict[str, tuple[float, float]] = {}
    for _, column in ANGLE_SPECS:
        values = _numeric(angle_df, column)
        angles[column] = values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            angle_ranges[column] = (0.0, 180.0)
            continue
        lo, hi = np.quantile(finite, [0.01, 0.99])
        padding = max(3.0, float(hi - lo) * 0.08)
        angle_ranges[column] = (float(lo - padding), float(hi + padding))

    curves = _prepare_force_curves(segments)
    raw_force_max = max(curve.peak_force for curve in curves.values())
    force_max = max(100.0, math.ceil(raw_force_max * 1.08 / 100.0) * 100.0)

    px, py = _project_pose_values(pose3d)
    finite_x = px[np.isfinite(px)]
    finite_y = py[np.isfinite(py)]
    pose_bounds = (
        float(np.quantile(finite_x, 0.002)),
        float(np.quantile(finite_x, 0.998)),
        float(np.quantile(finite_y, 0.002)),
        float(np.quantile(finite_y, 0.998)),
    )

    catch_indices = np.flatnonzero(_numeric(stroke, "is_catch_recomputed") >= 0.5)
    finish_indices = np.flatnonzero(_numeric(stroke, "is_finish_recomputed") >= 0.5)
    return ShowcaseData(
        stroke=stroke,
        angles=angles,
        angle_ranges=angle_ranges,
        pose3d=pose3d,
        pose_bounds=pose_bounds,
        curves=curves,
        force_max=force_max,
        catch_indices=catch_indices,
        finish_indices=finish_indices,
    )


def _put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = FG,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(image, text, xy, font, scale, color, thickness, cv2.LINE_AA)


def _put_condensed_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
    width_scale: float = 0.72,
) -> int:
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    mask = np.zeros((text_h + baseline + 8, text_w + 8), dtype=np.uint8)
    cv2.putText(mask, text, (4, text_h + 2), font, scale, 255, thickness, cv2.LINE_AA)
    target_w = max(1, int(round(mask.shape[1] * width_scale)))
    mask = cv2.resize(mask, (target_w, mask.shape[0]), interpolation=cv2.INTER_AREA)

    x, baseline_y = xy
    y = baseline_y - text_h - 2
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.shape[1], x + mask.shape[1])
    y1 = min(image.shape[0], y + mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    mask_crop = mask[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32) / 255.0
    roi = image[y0:y1, x0:x1].astype(np.float32)
    color_array = np.asarray(color, dtype=np.float32)
    image[y0:y1, x0:x1] = (
        roi * (1.0 - mask_crop[..., None]) + color_array * mask_crop[..., None]
    ).astype(np.uint8)
    return target_w


def _blend_rect(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x, y, w, h = rect
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return
    fill = np.empty_like(roi)
    fill[:] = color
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _panel(frame: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int] = PANEL) -> None:
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), LINE, 1)


def _build_static_canvas() -> np.ndarray:
    canvas = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = BG

    cv2.line(canvas, (0, 102), (CANVAS_W, 102), LINE, 1)
    _put_condensed_text(
        canvas,
        "MOTION BECOMES SIGNAL.",
        (MARGIN, 68),
        scale=1.75,
        color=FG,
        thickness=2,
        width_scale=0.68,
    )
    _put_text(
        canvas,
        "ROWING BIOMECHANICS / SYNCHRONIZED VIDEO + RP3 TELEMETRY",
        (MARGIN, 91),
        scale=0.36,
        color=MUTED,
    )
    _put_text(canvas, "GIACOMO CAPPELLETTO", (1625, 49), scale=0.42, color=FG)
    _put_text(canvas, "RESEARCH / 2026", (1717, 76), scale=0.34, color=ACCENT)

    x, y, w, h = VIDEO_RECT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), LINE, 1)
    for rect in (ANGLES_RECT, FORCE_RECT, POSE_RECT, METRICS_RECT):
        _panel(canvas, rect)

    fx, fy, _, _ = FORCE_RECT
    _put_text(canvas, "RP3 / FORCE CURVE", (fx + 22, fy + 30), scale=0.42, color=FG)
    _put_text(canvas, "DRIVE-SYNCHRONIZED", (fx + 22, fy + 49), scale=0.31, color=MUTED)

    px, py, _, _ = POSE_RECT
    _put_text(canvas, "MOTIONBERT / 3D LIFT", (px + 22, py + 30), scale=0.42, color=FG)
    _put_text(canvas, "H36M-17 BODY MODEL", (px + 22, py + 49), scale=0.31, color=MUTED)

    mx, my, _, _ = METRICS_RECT
    _put_text(canvas, "STROKE / LIVE STATE", (mx + 22, my + 30), scale=0.42, color=FG)

    ax, ay, aw, ah = ANGLES_RECT
    _put_text(canvas, "JOINT ANGLES / ROLLING 3.0 S", (ax + 18, ay + 28), scale=0.38, color=FG)
    _put_text(canvas, "ACTIVE SIDE: RIGHT", (ax + aw - 160, ay + 28), scale=0.31, color=MUTED)
    card_top = ay + 42
    card_w = aw / len(ANGLE_SPECS)
    for index, (label, _) in enumerate(ANGLE_SPECS):
        card_x = int(round(ax + index * card_w))
        if index:
            cv2.line(canvas, (card_x, card_top), (card_x, ay + ah), LINE, 1)
        _put_text(canvas, label, (card_x + 16, card_top + 24), scale=0.34, color=MUTED)
    return canvas


def _phase_state(stroke_idx: int, phase: float, is_drive: bool) -> tuple[str, float]:
    if stroke_idx < 0 or not np.isfinite(phase):
        return "READY", 0.0
    if is_drive:
        return "DRIVE", float(np.clip(phase * 2.0, 0.0, 1.0))
    return "RECOVERY", float(np.clip((phase - 0.5) * 2.0, 0.0, 1.0))


def _latest_event(frame_idx: int, indices: np.ndarray, pulse_frames: int) -> float:
    if indices.size == 0:
        return 0.0
    pos = int(np.searchsorted(indices, frame_idx, side="right") - 1)
    if pos < 0:
        return 0.0
    delta = frame_idx - int(indices[pos])
    if delta < 0 or delta > pulse_frames:
        return 0.0
    return 1.0 - delta / max(1, pulse_frames)


def _recolor_tracking(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((35, 110, 105)), np.array((98, 255, 255)))
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    result = frame.astype(np.float32)
    accent = np.asarray(ACCENT, dtype=np.float32)
    result = result * (1.0 - 0.72 * mask[..., None]) + accent * (0.72 * mask[..., None])
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_video(
    canvas: np.ndarray,
    video_frame: np.ndarray,
    *,
    frame_idx: int,
    source_fps: float,
    stroke_idx: int,
    phase_name: str,
    phase_progress: float,
    curve: StrokeCurve | None,
    catch_pulse: float,
    finish_pulse: float,
) -> None:
    x, y, w, h = VIDEO_RECT
    resized = cv2.resize(video_frame, (w, h), interpolation=cv2.INTER_AREA)
    resized = _recolor_tracking(resized)
    resized = cv2.convertScaleAbs(resized, alpha=0.97, beta=-3)
    canvas[y : y + h, x : x + w] = resized

    # Covers the old debug value list while preserving the whole-body render.
    card = (x, y, 332, 214)
    _blend_rect(canvas, card, BG, 0.96)
    cv2.rectangle(canvas, (card[0], card[1]), (card[0] + card[2], card[1] + card[3]), LINE, 1)
    cv2.rectangle(canvas, (card[0], card[1]), (card[0] + 5, card[1] + card[3]), ACCENT, -1)
    _put_text(canvas, "SPORTS2D / WHOLE-BODY", (card[0] + 20, card[1] + 27), scale=0.35, color=MUTED)
    _put_condensed_text(
        canvas,
        phase_name,
        (card[0] + 20, card[1] + 82),
        scale=1.45,
        color=ACCENT if phase_name == "DRIVE" else FG,
        thickness=2,
        width_scale=0.69,
    )
    if curve is not None:
        stroke_label = f"STROKE {curve.ordinal:02d} / {curve.total:02d}  |  RP3 #{curve.rp3_stroke_number:02d}"
    elif stroke_idx >= 0:
        stroke_label = f"STROKE {stroke_idx:02d}  |  ALIGNING"
    else:
        stroke_label = "POSE LOCK / READY"
    _put_text(canvas, stroke_label, (card[0] + 20, card[1] + 112), scale=0.34, color=FG)
    cv2.line(
        canvas,
        (card[0] + 20, card[1] + 138),
        (card[0] + card[2] - 20, card[1] + 138),
        LINE,
        1,
    )
    _put_text(canvas, "FACE / HANDS / LIMBS / HANDLE", (card[0] + 20, card[1] + 165), scale=0.31, color=MUTED)
    _put_text(canvas, "TRACKING CONFIDENCE / LOCKED", (card[0] + 20, card[1] + 190), scale=0.29, color=ACCENT)

    _blend_rect(canvas, (x + 16, y + h - 54, w - 32, 38), BG, 0.82)
    bar_x0 = x + 132
    bar_x1 = x + w - 136
    bar_y = y + h - 35
    cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), LINE, 3, cv2.LINE_AA)
    progress_x = int(round(bar_x0 + phase_progress * (bar_x1 - bar_x0)))
    cv2.line(
        canvas,
        (bar_x0, bar_y),
        (progress_x, bar_y),
        ACCENT if phase_name == "DRIVE" else FG,
        4,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, (progress_x, bar_y), 6, ACCENT, -1, cv2.LINE_AA)
    left_label, right_label = ("CATCH", "FINISH") if phase_name == "DRIVE" else ("FINISH", "CATCH")
    _put_text(canvas, left_label, (x + 34, bar_y + 5), scale=0.32, color=MUTED)
    _put_text(canvas, right_label, (x + w - 102, bar_y + 5), scale=0.32, color=MUTED)
    _put_text(
        canvas,
        f"{phase_progress * 100:05.1f}%",
        (bar_x1 - 66, bar_y - 10),
        scale=0.33,
        color=ACCENT if phase_name == "DRIVE" else FG,
    )

    _blend_rect(canvas, (x + w - 232, y + 16, 216, 34), BG, 0.78)
    _put_text(
        canvas,
        f"SYNC {source_fps:.0f} HZ  /  T+{frame_idx / source_fps:05.1f} S",
        (x + w - 216, y + 38),
        scale=0.31,
        color=FG,
    )

    pulse = max(catch_pulse, finish_pulse)
    if pulse > 0:
        pulse_color = tuple(int(round(LINE[i] * (1.0 - pulse) + ACCENT[i] * pulse)) for i in range(3))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), pulse_color, 3)
        event = "CATCH" if catch_pulse >= finish_pulse else "FINISH"
        _blend_rect(canvas, (x + w - 158, y + 62, 142, 44), ACCENT, 0.88 * pulse)
        _put_text(canvas, event, (x + w - 137, y + 91), scale=0.58, color=ACCENT_INK, thickness=2)


def _curve_points(
    s: np.ndarray,
    force: np.ndarray,
    rect: tuple[int, int, int, int],
    force_max: float,
) -> np.ndarray:
    x, y, w, h = rect
    px = x + np.clip(s, 0.0, 1.0) * w
    py = y + h - np.clip(force, 0.0, force_max) / force_max * h
    return np.stack([px, py], axis=1).round().astype(np.int32).reshape(-1, 1, 2)


def _draw_force_panel(
    canvas: np.ndarray,
    curve: StrokeCurve | None,
    *,
    phase_name: str,
    phase_progress: float,
    force_max: float,
) -> float:
    x, y, w, h = FORCE_RECT
    graph = (x + 34, y + 82, w - 62, h - 124)
    gx, gy, gw, gh = graph
    for frac in (0.0, 0.5, 1.0):
        yy = int(round(gy + gh * (1.0 - frac)))
        cv2.line(canvas, (gx, yy), (gx + gw, yy), LINE, 1)
        _put_text(canvas, f"{force_max * frac:.0f}", (gx - 29, yy + 4), scale=0.25, color=SUBTLE)
    for frac in (0.0, 0.5, 1.0):
        xx = int(round(gx + gw * frac))
        cv2.line(canvas, (xx, gy), (xx, gy + gh), LINE, 1)
        _put_text(canvas, f"{frac * 100:.0f}", (xx - 7, gy + gh + 18), scale=0.25, color=SUBTLE)
    _put_text(canvas, "N", (gx - 27, gy - 8), scale=0.28, color=MUTED)
    _put_text(canvas, "DRIVE %", (gx + gw - 49, gy + gh + 18), scale=0.25, color=MUTED)

    if curve is None:
        _put_text(canvas, "AWAITING MATCHED RP3 STROKE", (gx + 64, gy + gh // 2), scale=0.36, color=MUTED)
        _put_text(canvas, "--- N", (x + w - 95, y + 32), scale=0.46, color=FG, thickness=1)
        return float("nan")

    progress = phase_progress if phase_name == "DRIVE" else 1.0
    progress = float(np.clip(progress, 0.0, 1.0))
    current_force = float(np.interp(progress, curve.s, curve.force))
    all_points = _curve_points(curve.s, curve.force, graph, force_max)
    if len(all_points) >= 2:
        cv2.polylines(canvas, [all_points], False, MUTED, 2, cv2.LINE_AA)

    completed = curve.s <= progress
    if int(completed.sum()) < 2:
        completed[: min(2, len(completed))] = True
    done_s = curve.s[completed]
    done_f = curve.force[completed]
    if done_s.size and done_s[-1] < progress:
        done_s = np.append(done_s, progress)
        done_f = np.append(done_f, current_force)
    done_points = _curve_points(done_s, done_f, graph, force_max)
    if len(done_points) >= 2:
        fill_points = np.concatenate(
            [
                done_points.reshape(-1, 2),
                np.asarray([[done_points[-1, 0, 0], gy + gh], [done_points[0, 0, 0], gy + gh]]),
            ]
        ).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [fill_points], ACCENT)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0.0, dst=canvas)
        cv2.polylines(canvas, [done_points], False, ACCENT, 4, cv2.LINE_AA)

    dot = _curve_points(
        np.asarray([progress]),
        np.asarray([current_force]),
        graph,
        force_max,
    )[0, 0]
    cv2.circle(canvas, tuple(int(v) for v in dot), 7, ACCENT_INK, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(int(v) for v in dot), 5, ACCENT, -1, cv2.LINE_AA)
    _put_text(canvas, f"{current_force:03.0f} N", (x + w - 103, y + 32), scale=0.46, color=ACCENT, thickness=1)
    _put_text(
        canvas,
        f"PEAK {curve.peak_force:.0f} N  /  RP3 STROKE {curve.rp3_stroke_number}",
        (gx, y + h - 16),
        scale=0.29,
        color=MUTED,
    )
    return current_force


def _project_pose(
    pose: np.ndarray,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x_values, y_values = _project_pose_values(pose[None, ...])
    x_values = x_values[0]
    y_values = y_values[0]
    xmin, xmax, ymin, ymax = bounds
    x, y, w, h = rect
    scale = min(w / max(1e-6, xmax - xmin), h / max(1e-6, ymax - ymin))
    used_w = (xmax - xmin) * scale
    used_h = (ymax - ymin) * scale
    x_offset = x + (w - used_w) / 2.0
    y_offset = y + (h - used_h) / 2.0
    px = x_offset + (x_values - xmin) * scale
    py = y_offset + (y_values - ymin) * scale
    return np.stack([px, py], axis=1)


def _draw_pose_panel(data: ShowcaseData, canvas: np.ndarray, frame_idx: int) -> None:
    x, y, w, h = POSE_RECT
    draw_rect = (x + 40, y + 62, w - 80, h - 82)
    dx, dy, dw, dh = draw_rect
    cv2.line(canvas, (dx, dy + dh), (dx + dw, dy + dh), LINE, 1)
    cv2.line(canvas, (dx + dw // 2, dy), (dx + dw // 2, dy + dh), LINE, 1)
    cv2.line(canvas, (dx, dy + dh // 2), (dx + dw, dy + dh // 2), LINE, 1)

    pose_idx = min(frame_idx, len(data.pose3d) - 1)
    pose = data.pose3d[pose_idx]
    points = _project_pose(pose, draw_rect, data.pose_bounds)

    start = max(0, pose_idx - 60)
    trail_frames = data.pose3d[start : pose_idx + 1 : 6]
    if len(trail_frames) >= 2:
        trail = np.asarray(
            [_project_pose(item, draw_rect, data.pose_bounds)[16] for item in trail_frames],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [trail], False, (32, 105, 115), 2, cv2.LINE_AA)

    depth = pose[:, 2] - pose[0, 2]
    depth_min = float(np.nanmin(depth))
    depth_span = max(1e-6, float(np.nanmax(depth) - depth_min))
    for a, b in H36M17_EDGES:
        if not np.isfinite(points[[a, b]]).all():
            continue
        depth_t = float(np.clip(((depth[a] + depth[b]) * 0.5 - depth_min) / depth_span, 0.0, 1.0))
        color = tuple(
            int(round(MUTED[channel] * (1.0 - depth_t) + FG[channel] * depth_t))
            for channel in range(3)
        )
        cv2.line(
            canvas,
            tuple(points[a].round().astype(int)),
            tuple(points[b].round().astype(int)),
            color,
            3,
            cv2.LINE_AA,
        )
    for point in points:
        if np.isfinite(point).all():
            cv2.circle(canvas, tuple(point.round().astype(int)), 5, ACCENT_INK, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(point.round().astype(int)), 3, ACCENT, -1, cv2.LINE_AA)
    _put_text(canvas, "3D TRAJECTORY / RIGHT WRIST", (x + 275, y + 48), scale=0.27, color=SUBTLE)


def _draw_angle_cards(
    data: ShowcaseData,
    canvas: np.ndarray,
    *,
    frame_idx: int,
    source_fps: float,
) -> None:
    x, y, w, h = ANGLES_RECT
    card_top = y + 42
    card_w = w / len(ANGLE_SPECS)
    history_frames = max(2, int(round(source_fps * 3.0)))
    history_start = max(0, frame_idx - history_frames)
    sample_step = max(1, int(round(source_fps / 24.0)))
    indices = np.arange(history_start, frame_idx + 1, sample_step, dtype=int)
    if indices.size == 0 or indices[-1] != frame_idx:
        indices = np.append(indices, frame_idx)

    for index, (_, column) in enumerate(ANGLE_SPECS):
        card_x = int(round(x + index * card_w))
        card_right = int(round(x + (index + 1) * card_w))
        graph_x0 = card_x + 16
        graph_x1 = card_right - 16
        graph_y0 = card_top + 42
        graph_y1 = y + h - 14
        values = data.angles[column]
        safe_indices = np.clip(indices, 0, len(values) - 1)
        trace = values[safe_indices]
        lo, hi = data.angle_ranges[column]
        valid = np.isfinite(trace)
        if valid.any():
            px = np.linspace(graph_x0, graph_x1, len(trace))
            py = graph_y1 - np.clip((trace - lo) / max(1e-6, hi - lo), 0.0, 1.0) * (graph_y1 - graph_y0)
            points = np.stack([px[valid], py[valid]], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, MUTED, 2, cv2.LINE_AA)
                accent_start = max(0, len(points) - 12)
                cv2.polylines(canvas, [points[accent_start:]], False, ACCENT, 3, cv2.LINE_AA)
            current = trace[-1]
            if np.isfinite(current):
                _put_text(
                    canvas,
                    f"{current:05.1f}",
                    (card_right - 67, card_top + 25),
                    scale=0.37,
                    color=FG,
                )
                cv2.circle(canvas, (graph_x1, int(round(py[-1]))), 4, ACCENT, -1, cv2.LINE_AA)
        cv2.line(canvas, (graph_x0, graph_y1), (graph_x1, graph_y1), LINE, 1)


def _metric(
    canvas: np.ndarray,
    label: str,
    value: str,
    xy: tuple[int, int],
) -> None:
    x, y = xy
    _put_text(canvas, label, (x, y), scale=0.28, color=MUTED)
    _put_text(canvas, value, (x, y + 25), scale=0.48, color=FG, thickness=1)


def _draw_metrics_panel(
    data: ShowcaseData,
    canvas: np.ndarray,
    *,
    frame_idx: int,
    stroke_idx: int,
    phase_name: str,
    phase_progress: float,
    curve: StrokeCurve | None,
) -> None:
    x, y, w, h = METRICS_RECT
    phase_color = ACCENT if phase_name == "DRIVE" else FG
    _put_condensed_text(
        canvas,
        phase_name,
        (x + 22, y + 88),
        scale=1.48,
        color=phase_color,
        thickness=2,
        width_scale=0.68,
    )
    _put_text(canvas, f"{phase_progress * 100:05.1f}%", (x + w - 103, y + 79), scale=0.50, color=phase_color)
    bar_x0 = x + 22
    bar_x1 = x + w - 22
    bar_y = y + 105
    cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), LINE, 4, cv2.LINE_AA)
    progress_x = int(round(bar_x0 + phase_progress * (bar_x1 - bar_x0)))
    cv2.line(canvas, (bar_x0, bar_y), (progress_x, bar_y), phase_color, 4, cv2.LINE_AA)

    velocity = _safe_float(data.stroke.iloc[min(frame_idx, len(data.stroke) - 1)].get("velocity_axis_recomputed_px_s"))
    if curve is None:
        stroke_value = f"{max(0, stroke_idx):02d} / --"
        peak_value = "--- N"
        rate_value = "--.- /M"
        drive_value = "-.-- S"
    else:
        stroke_value = f"{curve.ordinal:02d} / {curve.total:02d}"
        peak_value = f"{curve.peak_force:.0f} N"
        rate_value = f"{curve.stroke_rate:.1f} /M"
        drive_value = f"{curve.drive_time_s:.2f} S"

    metric_y = y + 138
    _metric(canvas, "STROKE", stroke_value, (x + 22, metric_y))
    _metric(canvas, "PEAK FORCE", peak_value, (x + 145, metric_y))
    _metric(canvas, "RATE", rate_value, (x + 288, metric_y))
    _metric(canvas, "HANDLE VELOCITY", f"{velocity:+.0f} PX/S" if np.isfinite(velocity) else "---", (x + 390, metric_y))
    _put_text(canvas, f"RP3 DRIVE {drive_value}", (x + 22, y + h - 16), scale=0.29, color=MUTED)


def _render_frame(
    static_canvas: np.ndarray,
    data: ShowcaseData,
    video_frame: np.ndarray,
    *,
    frame_idx: int,
    source_fps: float,
) -> np.ndarray:
    canvas = static_canvas.copy()
    row_idx = min(frame_idx, len(data.stroke) - 1)
    row = data.stroke.iloc[row_idx]
    stroke_value = _safe_float(row.get("stroke_idx_recomputed"), -1.0)
    stroke_idx = int(round(stroke_value)) if np.isfinite(stroke_value) else -1
    phase = _safe_float(row.get("stroke_phase_recomputed"))
    is_drive = _safe_float(row.get("is_drive_recomputed"), 0.0) >= 0.5
    phase_name, phase_progress = _phase_state(stroke_idx, phase, is_drive)
    curve = data.curves.get(stroke_idx)
    pulse_frames = max(1, int(round(source_fps * 0.22)))
    catch_pulse = _latest_event(frame_idx, data.catch_indices, pulse_frames)
    finish_pulse = _latest_event(frame_idx, data.finish_indices, pulse_frames)

    _draw_video(
        canvas,
        video_frame,
        frame_idx=frame_idx,
        source_fps=source_fps,
        stroke_idx=stroke_idx,
        phase_name=phase_name,
        phase_progress=phase_progress,
        curve=curve,
        catch_pulse=catch_pulse,
        finish_pulse=finish_pulse,
    )
    _draw_force_panel(
        canvas,
        curve,
        phase_name=phase_name,
        phase_progress=phase_progress,
        force_max=data.force_max,
    )
    _draw_pose_panel(data, canvas, frame_idx)
    _draw_angle_cards(data, canvas, frame_idx=frame_idx, source_fps=source_fps)
    _draw_metrics_panel(
        data,
        canvas,
        frame_idx=frame_idx,
        stroke_idx=stroke_idx,
        phase_name=phase_name,
        phase_progress=phase_progress,
        curve=curve,
    )

    time_s = frame_idx / source_fps
    if time_s < 0.65:
        alpha = float(np.clip(time_s / 0.65, 0.0, 1.0))
        alpha = 1.0 - (1.0 - alpha) ** 3
        canvas = (canvas.astype(np.float32) * alpha).astype(np.uint8)
    return canvas


def _open_writer(output: Path, fps: float) -> tuple[cv2.VideoWriter, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (CANVAS_W, CANVAS_H),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Unable to create video writer: {output}")


def render_showcase(
    *,
    run_dir: Path,
    output: Path,
    input_video: Path | None = None,
    output_fps: float = 60.0,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> tuple[int, float, str]:
    run_dir = run_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    input_video = (
        input_video.expanduser().resolve()
        if input_video is not None
        else (run_dir / "stroke" / "stroke_tracking_debug.mp4").resolve()
    )
    if not input_video.exists():
        raise FileNotFoundError(f"Missing whole-body tracking video: {input_video}")
    if input_video == output:
        raise ValueError("Input and output video paths must differ.")
    if output_fps <= 0:
        raise ValueError("output_fps must be positive.")
    if start_s < 0:
        raise ValueError("start_s must be non-negative.")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive when provided.")

    data = _load_data(run_dir)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_frames = min(source_frames, len(data.stroke), len(data.pose3d))
    stride = max(1, int(round(source_fps / output_fps)))
    actual_fps = source_fps / stride

    start_frame = min(usable_frames, int(round(start_s * source_fps)))
    end_frame = usable_frames
    if duration_s is not None:
        end_frame = min(end_frame, start_frame + int(round(duration_s * source_fps)))
    if end_frame <= start_frame:
        cap.release()
        raise ValueError("Selected render range contains no frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer, codec = _open_writer(output, actual_fps)
    static_canvas = _build_static_canvas()
    expected = math.ceil((end_frame - start_frame) / stride)
    rendered = 0
    source_idx = start_frame
    next_report = 0.1
    try:
        while source_idx < end_frame:
            ok, video_frame = cap.read()
            if not ok or video_frame is None:
                break
            if (source_idx - start_frame) % stride == 0:
                composed = _render_frame(
                    static_canvas,
                    data,
                    video_frame,
                    frame_idx=source_idx,
                    source_fps=source_fps,
                )
                writer.write(composed)
                rendered += 1
                progress = rendered / max(1, expected)
                if progress >= next_report:
                    print(f"Render {progress * 100:5.1f}% ({rendered}/{expected} frames)", flush=True)
                    next_report += 0.1
            source_idx += 1
    finally:
        cap.release()
        writer.release()

    if rendered == 0:
        raise RuntimeError("No frames were rendered.")
    return rendered, actual_fps, codec


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run containing pose, stroke, and RP3 artifacts.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Default: <run>/stroke/stroke_tracking_debug.mp4",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS; source frames are evenly decimated.")
    parser.add_argument("--start-s", type=float, default=0.0, help="Start time for preview renders.")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional preview duration.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        rendered, fps, codec = render_showcase(
            run_dir=args.run_dir,
            output=args.output,
            input_video=args.input_video,
            output_fps=args.fps,
            start_s=args.start_s,
            duration_s=args.duration_s,
        )
    except Exception as exc:
        print(f"Showcase render failed: {exc}")
        return 1
    print(f"Output: {args.output.expanduser().resolve()}")
    print(f"Frames: {rendered}")
    print(f"FPS: {fps:.3f}")
    print(f"Writer codec: {codec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
