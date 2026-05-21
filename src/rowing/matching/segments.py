"""Force/pose segment extraction and QC scoring.

The functions in this module turn a per-frame stroke signal + RP3 match
manifest into the canonical per-stroke segment table consumed by
``rowing.dataset.build``. They were extracted from the old
``inference/inference_cli.py`` orchestrator during the Phase 1 refactor and
keep their original semantics intact.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from rowing.dataset.feature_contract import (
    apply_mirror_normalization,
    build_side_map,
)
from rowing.matching.detect import (
    DetectionResult,
    DriveEvent,
    _infer_time_s,
    _interp_nans,
)


# ---------------------------------------------------------------------------
# Constants (kept identical to the legacy CLI for byte-equivalent outputs)
# ---------------------------------------------------------------------------

FORCE_COL_RE = re.compile(r"^force_at_([0-9]+(?:\.[0-9]+)?)cm$")
PDF_AREA_EPS = 1e-9
PDF_AREA_TOL = 1e-6

SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 3
CHAIN_RULE_EPS = 1e-6
DS_DT_STALL_FRAC = 0.05
MAX_ANGULAR_VEL_DEG_S = 600.0
MAX_NAN_FRAC_ANGLES = 0.3
PROGRESS_MONOTONICITY_VIOLATION_FRAC = 0.15
DRIVE_DURATION_MIN_S = 0.4
DRIVE_DURATION_MAX_S = 2.0
CYCLE_DURATION_MIN_S = 1.5
CYCLE_DURATION_MAX_S = 4.0
MIN_DETECTION_CONFIDENCE = 0.15


__all__ = [
    "FORCE_COL_RE",
    "PDF_AREA_EPS",
    "PDF_AREA_TOL",
    "SAVGOL_WINDOW",
    "SAVGOL_POLYORDER",
    "CHAIN_RULE_EPS",
    "DS_DT_STALL_FRAC",
    "MAX_ANGULAR_VEL_DEG_S",
    "MAX_NAN_FRAC_ANGLES",
    "PROGRESS_MONOTONICITY_VIOLATION_FRAC",
    "DRIVE_DURATION_MIN_S",
    "DRIVE_DURATION_MAX_S",
    "CYCLE_DURATION_MIN_S",
    "CYCLE_DURATION_MAX_S",
    "MIN_DETECTION_CONFIDENCE",
    "events_to_dataframe",
    "build_frame_level_recomputed_columns",
    "build_force_pose_segments",
    "validate_force_segment_exports",
    "compute_quality_score",
    "find_force_columns",
    "interp_feature_on_progress",
    "savgol_smooth",
    "detect_rower_facing",
]


# ---------------------------------------------------------------------------
# DataFrame adapters
# ---------------------------------------------------------------------------


def events_to_dataframe(events: Iterable[DriveEvent]) -> pd.DataFrame:
    """Render a list of :class:`DriveEvent` records as a stable-schema DataFrame."""
    rows = [asdict(event) for event in events]
    if not rows:
        return pd.DataFrame(
            columns=[
                "stroke_idx",
                "catch_frame_idx",
                "catch_time_s",
                "finish_frame_idx",
                "finish_time_s",
                "next_catch_frame_idx",
                "next_catch_time_s",
                "drive_duration_s",
                "recover_duration_s",
                "cycle_duration_s",
                "catch_distance_px",
                "finish_distance_px",
                "drive_displacement_px",
                "catch_velocity_contrast",
                "finish_velocity_contrast",
                "drive_prominence",
            ]
        )
    return pd.DataFrame(rows)


def build_frame_level_recomputed_columns(
    df: pd.DataFrame,
    detection: DetectionResult,
) -> pd.DataFrame:
    """Append recomputed catch/finish/drive columns aligned with *detection*.

    The new columns suffixed ``_recomputed`` are the canonical per-frame
    representation downstream stages consume.
    """
    out = df.copy()
    n = len(out)

    is_catch = np.zeros((n,), dtype=np.uint8)
    is_finish = np.zeros((n,), dtype=np.uint8)
    is_drive = np.zeros((n,), dtype=np.uint8)
    stroke_idx = np.full((n,), -1, dtype=np.int32)
    stroke_phase = np.full((n,), np.nan, dtype=np.float32)

    catches_idx = detection.catches_filtered
    catches_idx = catches_idx[(catches_idx >= 0) & (catches_idx < n)]
    is_catch[catches_idx] = 1

    time_s = _infer_time_s(out)
    frame_to_row: dict[int, int] = {}
    if "frame_idx" in out.columns:
        frame_vals = pd.to_numeric(out["frame_idx"], errors="coerce").to_numpy(dtype=np.float64)
        for row_idx, frame_val in enumerate(frame_vals):
            if not np.isfinite(frame_val):
                continue
            frame_to_row.setdefault(int(round(float(frame_val))), row_idx)

    for event in detection.events:
        if frame_to_row:
            c0 = int(frame_to_row.get(int(event.catch_frame_idx), int(event.catch_frame_idx)))
            f0 = int(frame_to_row.get(int(event.finish_frame_idx), int(event.finish_frame_idx)))
            c1 = int(frame_to_row.get(int(event.next_catch_frame_idx), int(event.next_catch_frame_idx)))
        else:
            c0 = int(event.catch_frame_idx)
            f0 = int(event.finish_frame_idx)
            c1 = int(event.next_catch_frame_idx)
        c0 = int(np.clip(c0, 0, n - 1))
        f0 = int(np.clip(f0, 0, n - 1))
        c1 = int(np.clip(c1, 0, n - 1))
        if not (c0 < f0 < c1):
            continue

        is_finish[f0] = 1
        stroke_idx[c0 : c1 + 1] = int(event.stroke_idx)
        is_drive[c0 : f0 + 1] = 1

        drive_len = max(1, f0 - c0)
        for t in range(c0, f0 + 1):
            stroke_phase[t] = np.float32(0.5 * (t - c0) / drive_len)

        rec_len = max(1, c1 - f0)
        for t in range(f0, c1 + 1):
            stroke_phase[t] = np.float32(0.5 + 0.5 * (t - f0) / rec_len)

    out["relative_axis_px_smooth"] = detection.signal_smooth_px
    out["velocity_axis_recomputed_px_s"] = detection.slope_px_s
    out["stroke_idx_recomputed"] = stroke_idx
    out["stroke_phase_recomputed"] = stroke_phase
    out["is_drive_recomputed"] = is_drive
    out["is_catch_recomputed"] = is_catch
    out["is_finish_recomputed"] = is_finish
    out["time_s_recomputed"] = time_s.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# QC scoring helpers
# ---------------------------------------------------------------------------


def _sigmoid_penalty(
    value: float, good: float, bad: float, *, k: float = 12.0,
) -> float:
    """Smooth penalty in [0, 1]: ~1.0 at *good*, ~0.0 at *bad*.

    Works for both "lower is better" (good < bad) and "higher is better"
    (good > bad) by flipping the sigmoid direction.
    """
    if not np.isfinite(value):
        return 0.5
    midpoint = (good + bad) / 2.0
    scale = abs(bad - good)
    if scale < 1e-12:
        return 1.0 if abs(value - good) < abs(value - bad) else 0.0
    x = (value - midpoint) / scale
    if good < bad:
        x = -x
    return float(1.0 / (1.0 + np.exp(-k * x)))


def _range_penalty(
    value: float,
    good_lo: float, good_hi: float,
    bad_lo: float, bad_hi: float,
    *, k: float = 12.0,
) -> float:
    """Penalty ~1.0 inside [good_lo, good_hi], dropping toward 0 outside [bad_lo, bad_hi]."""
    if not np.isfinite(value):
        return 0.5
    if good_lo <= value <= good_hi:
        return 1.0
    if value < good_lo:
        return _sigmoid_penalty(value, good=good_lo, bad=bad_lo, k=k)
    return _sigmoid_penalty(value, good=good_hi, bad=bad_hi, k=k)


def compute_quality_score(
    nan_frac: float,
    max_deriv_deg_s: float,
    mono_violation_frac: float,
    detection_confidence: float,
    drive_duration_s: float,
) -> float:
    """Aggregate stroke quality in [0, 1] (product of per-dimension penalties)."""
    penalties = [
        _sigmoid_penalty(nan_frac, good=0.05, bad=0.3),
        _sigmoid_penalty(max_deriv_deg_s, good=300.0, bad=600.0),
        _sigmoid_penalty(mono_violation_frac, good=0.05, bad=0.15),
        _sigmoid_penalty(detection_confidence, good=0.30, bad=0.15),
        _range_penalty(drive_duration_s, good_lo=0.6, good_hi=1.5, bad_lo=0.4, bad_hi=2.0),
    ]
    score = 1.0
    for p in penalties:
        score *= p
    return score


# ---------------------------------------------------------------------------
# Force/progress feature builders
# ---------------------------------------------------------------------------


def find_force_columns(rp3_df: pd.DataFrame) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for col in rp3_df.columns:
        m = FORCE_COL_RE.match(str(col))
        if m is None:
            continue
        found.append((str(col), float(m.group(1))))
    found.sort(key=lambda x: x[1])
    return found


def interp_feature_on_progress(
    s_video: np.ndarray,
    feature_values: np.ndarray,
    s_targets: np.ndarray,
) -> np.ndarray:
    s_video = np.asarray(s_video, dtype=np.float64)
    y = np.asarray(feature_values, dtype=np.float64)
    s_targets = np.asarray(s_targets, dtype=np.float64)

    mask = np.isfinite(s_video) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(s_targets, np.nan, dtype=np.float64)

    s = s_video[mask]
    y = y[mask]
    order = np.argsort(s)
    s = s[order]
    y = y[order]

    s = np.clip(s, 0.0, 1.0)
    s = np.maximum.accumulate(s)

    unique_s, unique_idx = np.unique(s, return_index=True)
    y_unique = y[unique_idx]
    if unique_s.size == 0:
        return np.full_like(s_targets, np.nan, dtype=np.float64)
    if unique_s.size == 1:
        return np.full_like(s_targets, float(y_unique[0]), dtype=np.float64)

    clipped_targets = np.clip(s_targets, unique_s[0], unique_s[-1])
    return np.interp(clipped_targets, unique_s, y_unique)


def savgol_smooth(
    arr: np.ndarray, window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    finite = np.isfinite(arr)
    if finite.sum() < max(window, polyorder + 1):
        return arr.copy()
    filled = _interp_nans(arr.copy())
    eff_win = min(window, len(filled))
    if eff_win % 2 == 0:
        eff_win -= 1
    eff_win = max(eff_win, 1)
    eff_poly = min(polyorder, eff_win - 1)
    smoothed = savgol_filter(filled, eff_win, eff_poly)
    out = arr.copy()
    out[finite] = smoothed[finite]
    return out


def detect_rower_facing(
    trunk_deg: np.ndarray,
    catch_frac: float = 0.15,
) -> str:
    """Return ``'right'`` or ``'left'`` based on early-drive trunk posture."""
    n_catch = max(1, int(len(trunk_deg) * catch_frac))
    early = trunk_deg[:n_catch]
    finite = early[np.isfinite(early)]
    if finite.size == 0:
        return "right"
    return "left" if float(np.median(finite)) > 90.0 else "right"


# ---------------------------------------------------------------------------
# Main segment builder + post-build invariant check
# ---------------------------------------------------------------------------


def build_force_pose_segments(
    *,
    run_dir: Path,
    frame_df: pd.DataFrame,
    events_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    rp3_df: pd.DataFrame,
    active_side: str,
    rp3_clean_csv: Path,
    use_rp3_finish: bool = False,
    include_second_derivatives: bool = False,
    rower_facing: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    angles_csv = run_dir / "motionbert" / "angles_h36m.csv"
    if not angles_csv.exists():
        raise FileNotFoundError(f"angles_h36m.csv not found at: {angles_csv}")
    angles_df = pd.read_csv(angles_csv)
    if "frame_idx" not in angles_df.columns:
        raise ValueError("angles_h36m.csv missing frame_idx column.")

    merged = frame_df.merge(angles_df, on="frame_idx", how="left", suffixes=("", "_angle"))
    force_cols = find_force_columns(rp3_df)
    if not force_cols:
        raise ValueError("No force_at_*cm columns found in RP3 CSV.")

    if active_side not in {"left", "right"}:
        raise ValueError("active_side must be 'left' or 'right'.")
    include_head = "head_vs_trunk_deg" in merged.columns
    side_map = build_side_map(active_side, include_head=include_head)
    required_sources = [
        src for canonical, src in side_map.items()
        if canonical != "head_vs_trunk_deg"
    ]
    missing = [src for src in required_sources if src not in merged.columns]
    if missing:
        raise ValueError(f"Missing angle columns for active side mapping: {missing}")

    events_by_stroke: dict[int, pd.Series] = {}
    for _, row in events_df.iterrows():
        events_by_stroke[int(row["stroke_idx"])] = row

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for match_seq_idx, (_, mrow) in enumerate(manifest_df.iterrows()):
        v_stroke = int(mrow["video_stroke_idx"])
        rp3_idx = int(mrow["rp3_row_idx"])
        rp3_stroke_number = int(mrow["rp3_stroke_number"])
        status_row: dict[str, Any] = {
            "match_seq_idx": int(match_seq_idx),
            "video_stroke_idx": int(v_stroke),
            "rp3_row_idx": int(rp3_idx),
            "rp3_stroke_number": int(rp3_stroke_number),
            "segment_exported": False,
            "segment_rows_written": 0,
            "drop_reason": "",
            "raw_area_trapz": float("nan"),
            "normalized_area_trapz": float("nan"),
        }
        if v_stroke not in events_by_stroke:
            status_row["drop_reason"] = "missing_events"
            status_rows.append(status_row)
            continue
        if not (0 <= rp3_idx < len(rp3_df)):
            status_row["drop_reason"] = "rp3_row_out_of_range"
            status_rows.append(status_row)
            continue

        ev = events_by_stroke[v_stroke]
        c_frame = int(ev["catch_frame_idx"])
        f_frame = int(ev["finish_frame_idx"])

        if use_rp3_finish:
            rp3_drive_s = float(mrow["rp3_drive_s"])
            catch_time = float(ev["catch_time_s"])
            rp3_finish_time = catch_time + rp3_drive_s
            time_col = "time_s_recomputed" if "time_s_recomputed" in merged.columns else "time_s"
            merged_times = pd.to_numeric(merged[time_col], errors="coerce")
            candidates = merged[
                (merged["frame_idx"] >= c_frame)
                & (merged_times <= rp3_finish_time + 1e-6)
            ]
            if not candidates.empty:
                f_frame = int(candidates["frame_idx"].iloc[-1])

        drive = merged[(merged["frame_idx"] >= c_frame) & (merged["frame_idx"] <= f_frame)].copy()
        if len(drive) < 2:
            status_row["drop_reason"] = "invalid_drive_window"
            status_rows.append(status_row)
            continue

        dist_col = "relative_axis_px_smooth" if "relative_axis_px_smooth" in drive.columns else "relative_axis_px"
        dist = pd.to_numeric(drive[dist_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(dist).sum() < 2:
            status_row["drop_reason"] = "invalid_distance_signal"
            status_rows.append(status_row)
            continue
        dist = _interp_nans(dist)

        x0 = float(ev["catch_distance_px"]) if "catch_distance_px" in ev.index else float(np.nanmin(dist))
        x1 = float(np.nanmax(dist))
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
            x0 = float(np.nanmin(dist))
            x1 = float(np.nanmax(dist))
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
            status_row["drop_reason"] = "invalid_drive_range"
            status_rows.append(status_row)
            continue

        s_video_raw = (dist - x0) / (x1 - x0)
        s_video_raw = np.clip(s_video_raw, 0.0, 1.0)
        s_video = np.maximum.accumulate(s_video_raw)
        mono_violations = int(np.sum(s_video_raw < s_video - 1e-9))
        mono_violation_frac = mono_violations / max(len(s_video_raw), 1)
        status_row["progress_mono_violation_frac"] = float(mono_violation_frac)
        if s_video[-1] > 1e-9:
            s_video = s_video / s_video[-1]

        rp3_row = rp3_df.iloc[rp3_idx]
        stroke_len = float(rp3_row["stroke_length"]) if np.isfinite(rp3_row["stroke_length"]) else float("nan")
        if not np.isfinite(stroke_len) or stroke_len <= 0:
            status_row["drop_reason"] = "invalid_stroke_length"
            status_rows.append(status_row)
            continue

        distances: list[float] = []
        force_raw: list[float] = []
        force_cols_used: list[str] = []
        for col, d_cm in force_cols:
            raw = rp3_row.get(col)
            f = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if not np.isfinite(f):
                continue
            if d_cm > stroke_len + 1e-6:
                continue
            distances.append(float(d_cm))
            force_raw.append(float(f))
            force_cols_used.append(col)

        if not distances:
            status_row["drop_reason"] = "no_valid_force_bins"
            status_rows.append(status_row)
            continue
        s_targets = np.asarray(distances, dtype=np.float64) / stroke_len
        s_targets = np.clip(s_targets, 0.0, 1.0)
        force_raw_arr = np.asarray(force_raw, dtype=np.float64)

        raw_area = float(np.trapz(force_raw_arr, s_targets))
        status_row["raw_area_trapz"] = raw_area
        if not np.isfinite(raw_area) or raw_area <= PDF_AREA_EPS:
            status_row["drop_reason"] = "zero_or_invalid_pdf_area"
            status_rows.append(status_row)
            continue
        force_pdf_arr = force_raw_arr / raw_area
        normalized_area = float(np.trapz(force_pdf_arr, s_targets))
        status_row["normalized_area_trapz"] = normalized_area
        if not np.isfinite(normalized_area):
            status_row["drop_reason"] = "invalid_normalized_pdf_area"
            status_rows.append(status_row)
            continue

        time_col = "time_s_recomputed" if "time_s_recomputed" in drive.columns else "time_s"
        time_arr = pd.to_numeric(drive[time_col], errors="coerce").to_numpy(dtype=np.float64)
        time_arr = _interp_nans(time_arr)

        qc_flags: list[str] = []
        angle_src_cols = list(side_map.values())
        nan_counts = sum(
            pd.to_numeric(drive[c], errors="coerce").isna().sum() for c in angle_src_cols
        )
        total_angle_vals = len(drive) * len(angle_src_cols)
        nan_frac = nan_counts / max(total_angle_vals, 1)
        status_row["nan_frac_angles"] = float(nan_frac)
        if nan_frac > MAX_NAN_FRAC_ANGLES:
            qc_flags.append("qc_tracking_sparse")

        trunk_src = side_map["trunk_vs_horizontal_deg"]
        trunk_raw_for_detect = pd.to_numeric(
            drive[trunk_src], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if rower_facing == "auto":
            detected_facing = detect_rower_facing(trunk_raw_for_detect)
        else:
            detected_facing = rower_facing

        mirrored_angles = apply_mirror_normalization(
            drive, side_map, facing=detected_facing,
        )

        smoothed_angles: dict[str, np.ndarray] = {}
        dtheta_dt_time: dict[str, np.ndarray] = {}
        max_angular_vel = 0.0
        for out_col in side_map.keys():
            raw = mirrored_angles[out_col]
            smooth = savgol_smooth(raw)
            smoothed_angles[out_col] = smooth
            if np.isfinite(smooth).sum() >= 2 and np.isfinite(time_arr).sum() >= 2:
                dt_arr = np.gradient(_interp_nans(smooth), time_arr)
                dtheta_dt_time[out_col] = dt_arr
                finite_dt = np.abs(dt_arr[np.isfinite(dt_arr)])
                if finite_dt.size > 0:
                    max_angular_vel = max(max_angular_vel, float(finite_dt.max()))
            else:
                dtheta_dt_time[out_col] = np.full_like(smooth, np.nan)

        status_row["max_deriv_deg_s"] = float(max_angular_vel)
        if max_angular_vel > MAX_ANGULAR_VEL_DEG_S:
            qc_flags.append("qc_nonphysio_deriv")

        ds_dt_time = (
            np.gradient(s_video, time_arr)
            if np.isfinite(time_arr).sum() >= 2
            else np.full_like(s_video, np.nan)
        )
        finite_ds_dt = ds_dt_time[np.isfinite(ds_dt_time)]
        ds_dt_min = float(finite_ds_dt.min()) if finite_ds_dt.size > 0 else float("nan")
        ds_dt_median = float(np.median(finite_ds_dt)) if finite_ds_dt.size > 0 else float("nan")
        status_row["ds_dt_min"] = ds_dt_min
        if np.isfinite(ds_dt_median) and ds_dt_median > 0 and np.isfinite(ds_dt_min):
            if ds_dt_min < DS_DT_STALL_FRAC * ds_dt_median:
                qc_flags.append("qc_ds_dt_stall")

        if mono_violation_frac > PROGRESS_MONOTONICITY_VIOLATION_FRAC:
            qc_flags.append("qc_progress_nonmonotonic")

        catch_vc = float(ev["catch_velocity_contrast"]) if "catch_velocity_contrast" in ev.index else float("nan")
        finish_vc = float(ev["finish_velocity_contrast"]) if "finish_velocity_contrast" in ev.index else float("nan")
        if np.isfinite(catch_vc) and np.isfinite(finish_vc):
            detection_confidence = min(catch_vc, finish_vc)
        elif np.isfinite(catch_vc):
            detection_confidence = catch_vc
        elif np.isfinite(finish_vc):
            detection_confidence = finish_vc
        else:
            detection_confidence = float("nan")
        status_row["detection_confidence"] = float(detection_confidence)
        if np.isfinite(detection_confidence) and detection_confidence < MIN_DETECTION_CONFIDENCE:
            qc_flags.append("qc_weak_detection")

        video_drive_s = float(mrow["video_drive_s"])
        video_cycle_s = float(mrow["video_cycle_s"])
        if not (DRIVE_DURATION_MIN_S <= video_drive_s <= DRIVE_DURATION_MAX_S):
            qc_flags.append("qc_duration_implausible")
        elif not (CYCLE_DURATION_MIN_S <= video_cycle_s <= CYCLE_DURATION_MAX_S):
            qc_flags.append("qc_duration_implausible")

        status_row["qc_flags"] = ",".join(qc_flags) if qc_flags else ""

        stroke_quality_score = compute_quality_score(
            nan_frac=nan_frac,
            max_deriv_deg_s=max_angular_vel,
            mono_violation_frac=mono_violation_frac,
            detection_confidence=detection_confidence,
            drive_duration_s=video_drive_s,
        )
        status_row["stroke_quality_score"] = float(stroke_quality_score)

        interp_features: dict[str, np.ndarray] = {}
        for out_col in side_map:
            interp_features[out_col] = interp_feature_on_progress(
                s_video, smoothed_angles[out_col], s_targets,
            )

        ds_dt_on_s = interp_feature_on_progress(s_video, ds_dt_time, s_targets)

        deriv_features: dict[str, np.ndarray] = {}
        all_angle_keys = list(side_map.keys())
        for out_col in all_angle_keys:
            dtheta_dt_on_s = interp_feature_on_progress(
                s_video, dtheta_dt_time[out_col], s_targets,
            )
            dtheta_ds = dtheta_dt_on_s / (ds_dt_on_s + CHAIN_RULE_EPS)
            deriv_features[f"{out_col.replace('_deg', '')}_ddeg_ds"] = dtheta_ds

        second_deriv_features: dict[str, np.ndarray] = {}
        if include_second_derivatives:
            for out_col in all_angle_keys:
                d1_key = f"{out_col.replace('_deg', '')}_ddeg_ds"
                d1 = deriv_features[d1_key]
                if d1.size >= 2 and np.isfinite(d1).sum() >= 2:
                    second_deriv_features[f"{out_col.replace('_deg', '')}_d2deg_ds2"] = np.gradient(d1, s_targets)
                else:
                    second_deriv_features[f"{out_col.replace('_deg', '')}_d2deg_ds2"] = np.full_like(d1, np.nan)

        vel_col = "velocity_axis_recomputed_px_s" if "velocity_axis_recomputed_px_s" in drive.columns else "velocity_axis_px_s"
        handle_vel_raw = (
            pd.to_numeric(drive.get(vel_col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
            if vel_col in drive.columns
            else np.full(len(drive), np.nan)
        )
        handle_vel_on_s = interp_feature_on_progress(s_video, handle_vel_raw, s_targets)

        handle_accel_time = (
            np.gradient(_interp_nans(handle_vel_raw), time_arr)
            if (np.isfinite(handle_vel_raw).sum() >= 2 and np.isfinite(time_arr).sum() >= 2)
            else np.full_like(handle_vel_raw, np.nan)
        )
        handle_accel_on_s = interp_feature_on_progress(s_video, handle_accel_time, s_targets)

        for i in range(len(distances)):
            row = {
                "run_name": run_dir.name,
                "rp3_clean_csv": rp3_clean_csv.name,
                "active_side": active_side,
                "rower_facing": detected_facing,
                "match_seq_idx": int(match_seq_idx),
                "video_stroke_idx": v_stroke,
                "rp3_row_idx": rp3_idx,
                "rp3_stroke_number": int(rp3_row["stroke_number"]),
                "video_catch_time_s": float(mrow["video_catch_time_s"]),
                "video_finish_time_s": float(mrow["video_finish_time_s"]),
                "video_drive_s": float(mrow["video_drive_s"]),
                "video_recover_s": float(mrow["video_recover_s"]),
                "video_cycle_s": float(mrow["video_cycle_s"]),
                "rp3_drive_s": float(mrow["rp3_drive_s"]),
                "rp3_recover_s": float(mrow["rp3_recover_s"]),
                "rp3_cycle_s": float(mrow["rp3_cycle_s"]),
                "cum_catch_err_s": float(mrow["cum_catch_err_s"]),
                "interval_err_s": float(mrow["interval_err_s"]),
                "rp3_rows_skipped_since_prev": int(mrow["rp3_rows_skipped_since_prev"]),
                "drive_bin_idx": i,
                "force_col": force_cols_used[i],
                "distance_cm": float(distances[i]),
                "stroke_length_cm": float(stroke_len),
                "s_force": float(s_targets[i]),
                "force_raw": float(force_raw_arr[i]),
                "force_n": float(force_pdf_arr[i]),
                "qc_flags": status_row["qc_flags"],
                "stroke_quality_score": float(stroke_quality_score),
                "handle_velocity_px_s": float(handle_vel_on_s[i]) if np.isfinite(handle_vel_on_s[i]) else float("nan"),
                "handle_accel_px_s2": float(handle_accel_on_s[i]) if np.isfinite(handle_accel_on_s[i]) else float("nan"),
            }
            for key, arr in interp_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            for key, arr in deriv_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            for key, arr in second_deriv_features.items():
                row[key] = float(arr[i]) if np.isfinite(arr[i]) else float("nan")
            rows.append(row)

        status_row["segment_exported"] = True
        status_row["segment_rows_written"] = int(len(distances))
        status_rows.append(status_row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["match_seq_idx", "distance_cm"]).reset_index(drop=True)
    status = pd.DataFrame(status_rows)
    if not status.empty:
        status = status.sort_values(["match_seq_idx"]).reset_index(drop=True)
    return out, status


def validate_force_segment_exports(
    *,
    manifest_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> None:
    """Cross-check segment export against the match manifest.

    Raises if any post-conditions fail (row count parity, contiguous
    ``match_seq_idx``, PDF-area-to-one invariants, etc.).
    """
    if len(status_df) != len(manifest_df):
        raise RuntimeError(
            f"Segment export status row mismatch: status_rows={len(status_df)} manifest_rows={len(manifest_df)}"
        )

    if "match_seq_idx" not in status_df.columns:
        raise RuntimeError("Segment export status is missing match_seq_idx.")
    seq = status_df["match_seq_idx"].to_numpy(dtype=np.int64)
    expected = np.arange(len(manifest_df), dtype=np.int64)
    if not np.array_equal(np.sort(seq), expected):
        raise RuntimeError("match_seq_idx must be unique and contiguous from 0..N-1 in export status.")

    exported_status = status_df[status_df["segment_exported"].astype(bool)]
    if not exported_status.empty:
        normalized = pd.to_numeric(exported_status["normalized_area_trapz"], errors="coerce")
        bad = (~np.isfinite(normalized.to_numpy(dtype=np.float64))) | (
            np.abs(normalized.to_numpy(dtype=np.float64) - 1.0) > PDF_AREA_TOL
        )
        if bool(np.any(bad)):
            raise RuntimeError("Exported stroke PDF normalization failed area-to-one invariant.")

    if segments_df.empty:
        return
    if "match_seq_idx" not in segments_df.columns:
        raise RuntimeError("Segments CSV missing match_seq_idx.")

    seg_keys = set(pd.to_numeric(segments_df["match_seq_idx"], errors="coerce").dropna().astype(int).tolist())
    exported_keys = set(pd.to_numeric(exported_status["match_seq_idx"], errors="coerce").dropna().astype(int).tolist())
    if not seg_keys.issubset(exported_keys):
        raise RuntimeError("Segments contain stroke keys that are not marked exported in status table.")

    seg_counts = segments_df.groupby("match_seq_idx").size().to_dict()
    for _, row in exported_status.iterrows():
        k = int(row["match_seq_idx"])
        expected_count = int(row["segment_rows_written"])
        actual_count = int(seg_counts.get(k, 0))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Segment row-count mismatch for match_seq_idx={k}: status={expected_count} actual={actual_count}"
            )
