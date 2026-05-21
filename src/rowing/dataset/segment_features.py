"""Per-stroke pose feature builders used by both training and inference.

Two public entry points:

* :func:`build_stroke_feature_sequence` — from a single drive window's
  merged pose/stroke-signal frame, produce a ``(G, K)`` feature tensor on
  a given ``s_grid`` plus a QC dict.  Used internally by both the RP3
  matched exporter (training path) and the video-only inference path.

* :func:`build_pose_drive_segments` — no-RP3 entry point used by
  ``predict_force_cli.py``.  Given a ``run_dir`` and a drive-events
  dataframe, returns a dataframe of per-stroke, per-grid-bin pose
  features plus a status dataframe mirroring the training-side status.

The goal is that inference uses the same smoothing, mirror-normalization,
and chain-rule derivative pipeline as training so that a model trained
on RP3-matched segments generalizes when RP3 is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from rowing.dataset.feature_contract import (
    apply_mirror_normalization,
    build_side_map,
    canonical_columns,
)


SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 3
CHAIN_RULE_EPS = 1e-6
MAX_ANGULAR_VEL_DEG_S = 600.0
MAX_NAN_FRAC_ANGLES = 0.3
PROGRESS_MONOTONICITY_VIOLATION_FRAC = 0.15
DS_DT_STALL_FRAC = 0.05
DRIVE_DURATION_MIN_S = 0.4
DRIVE_DURATION_MAX_S = 2.0
CYCLE_DURATION_MIN_S = 1.5
CYCLE_DURATION_MAX_S = 4.0
MIN_DETECTION_CONFIDENCE = 0.15


def _interp_nans(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if finite.all() or finite.sum() == 0:
        return x
    idx = np.arange(len(x))
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def _savgol_smooth(arr: np.ndarray, window: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
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


def _detect_rower_facing(trunk_deg: np.ndarray, catch_frac: float = 0.15) -> str:
    n_catch = max(1, int(len(trunk_deg) * catch_frac))
    early = trunk_deg[:n_catch]
    finite = early[np.isfinite(early)]
    if finite.size == 0:
        return "right"
    return "left" if float(np.median(finite)) > 90.0 else "right"


def _interp_on_progress(s_video: np.ndarray, values: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
    s_video = np.asarray(s_video, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
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


# ---------------------------------------------------------------------------
# Feature column generation
# ---------------------------------------------------------------------------


def feature_column_names(
    *,
    include_head: bool,
    include_second_derivatives: bool,
) -> list[str]:
    """Return the canonical ordered feature-column list for per-bin output."""
    angle_cols = canonical_columns(include_head=include_head)
    deriv_cols = [col.replace("_deg", "_ddeg_ds") for col in angle_cols]
    support_cols = ["handle_velocity_px_s", "handle_accel_px_s2"]
    cols = list(angle_cols) + list(deriv_cols) + support_cols
    if include_second_derivatives:
        cols.extend(col.replace("_deg", "_d2deg_ds2") for col in angle_cols)
    return cols


# ---------------------------------------------------------------------------
# Core per-stroke feature builder
# ---------------------------------------------------------------------------


@dataclass
class StrokeFeatureResult:
    """Per-stroke outputs on a fixed ``s_grid``.

    ``features`` maps canonical feature names (angles, first derivatives,
    optional second derivatives, handle velocity/accel) to arrays of
    length ``len(s_grid)``.
    """

    s_grid: np.ndarray
    features: dict[str, np.ndarray]
    rower_facing: str
    stroke_quality_score: float
    qc_flags: list[str]
    stats: dict[str, float]
    nan_frac_angles: float


def _resolve_time_column(df: pd.DataFrame) -> str:
    if "time_s_recomputed" in df.columns:
        return "time_s_recomputed"
    return "time_s"


def _resolve_velocity_column(df: pd.DataFrame) -> str | None:
    for col in ("velocity_axis_recomputed_px_s", "velocity_axis_px_s"):
        if col in df.columns:
            return col
    return None


def _resolve_distance_column(df: pd.DataFrame) -> str:
    if "relative_axis_px_smooth" in df.columns:
        return "relative_axis_px_smooth"
    return "relative_axis_px"


def build_stroke_feature_sequence(
    *,
    drive_df: pd.DataFrame,
    s_grid: np.ndarray,
    side_map: dict[str, str],
    rower_facing: str = "auto",
    include_second_derivatives: bool = False,
    catch_distance_px: float | None = None,
    drive_duration_s: float | None = None,
    cycle_duration_s: float | None = None,
    detection_confidence: float | None = None,
) -> StrokeFeatureResult | None:
    """Compute ``(G, K)`` per-stroke features on ``s_grid``.

    ``drive_df`` is a slice of the stroke signal covering a single drive
    (catch .. finish).  Returns ``None`` when the window is too short or
    the progress signal cannot be normalized; the caller is expected to
    drop such strokes.
    """
    if len(drive_df) < 2:
        return None

    dist_col = _resolve_distance_column(drive_df)
    dist = pd.to_numeric(drive_df[dist_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(dist).sum() < 2:
        return None
    dist = _interp_nans(dist)

    x0 = float(catch_distance_px) if catch_distance_px is not None else float(np.nanmin(dist))
    x1 = float(np.nanmax(dist))
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
        x0 = float(np.nanmin(dist))
        x1 = float(np.nanmax(dist))
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0 + 1e-6:
        return None

    s_video_raw = np.clip((dist - x0) / (x1 - x0), 0.0, 1.0)
    s_video = np.maximum.accumulate(s_video_raw)
    mono_violations = int(np.sum(s_video_raw < s_video - 1e-9))
    mono_violation_frac = mono_violations / max(len(s_video_raw), 1)
    if s_video[-1] > 1e-9:
        s_video = s_video / s_video[-1]

    time_col = _resolve_time_column(drive_df)
    time_arr = pd.to_numeric(drive_df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    time_arr = _interp_nans(time_arr)

    qc_flags: list[str] = []
    angle_src_cols = [src for src in side_map.values() if src in drive_df.columns]
    if angle_src_cols:
        nan_counts = sum(
            pd.to_numeric(drive_df[c], errors="coerce").isna().sum() for c in angle_src_cols
        )
        total = len(drive_df) * len(angle_src_cols)
        nan_frac = nan_counts / max(total, 1)
    else:
        nan_frac = 1.0
    if nan_frac > MAX_NAN_FRAC_ANGLES:
        qc_flags.append("qc_tracking_sparse")

    # Detect rower facing before mirror normalization.
    trunk_src = side_map.get("trunk_vs_horizontal_deg")
    if trunk_src is not None and trunk_src in drive_df.columns:
        trunk_raw = pd.to_numeric(drive_df[trunk_src], errors="coerce").to_numpy(dtype=np.float64)
    else:
        trunk_raw = np.zeros(len(drive_df), dtype=np.float64)
    detected_facing = _detect_rower_facing(trunk_raw) if rower_facing == "auto" else rower_facing

    mirrored = apply_mirror_normalization(drive_df, side_map, facing=detected_facing)

    smoothed: dict[str, np.ndarray] = {}
    dtheta_dt: dict[str, np.ndarray] = {}
    max_angular_vel = 0.0
    for name, raw in mirrored.items():
        smooth = _savgol_smooth(raw)
        smoothed[name] = smooth
        if np.isfinite(smooth).sum() >= 2 and np.isfinite(time_arr).sum() >= 2:
            dt_arr = np.gradient(_interp_nans(smooth), time_arr)
            dtheta_dt[name] = dt_arr
            finite_dt = np.abs(dt_arr[np.isfinite(dt_arr)])
            if finite_dt.size > 0:
                max_angular_vel = max(max_angular_vel, float(finite_dt.max()))
        else:
            dtheta_dt[name] = np.full_like(smooth, np.nan)

    if max_angular_vel > MAX_ANGULAR_VEL_DEG_S:
        qc_flags.append("qc_nonphysio_deriv")

    if np.isfinite(time_arr).sum() >= 2:
        ds_dt_time = np.gradient(s_video, time_arr)
    else:
        ds_dt_time = np.full_like(s_video, np.nan)
    finite_ds_dt = ds_dt_time[np.isfinite(ds_dt_time)]
    ds_dt_min = float(finite_ds_dt.min()) if finite_ds_dt.size > 0 else float("nan")
    ds_dt_median = float(np.median(finite_ds_dt)) if finite_ds_dt.size > 0 else float("nan")
    if np.isfinite(ds_dt_median) and ds_dt_median > 0 and np.isfinite(ds_dt_min):
        if ds_dt_min < DS_DT_STALL_FRAC * ds_dt_median:
            qc_flags.append("qc_ds_dt_stall")
    if mono_violation_frac > PROGRESS_MONOTONICITY_VIOLATION_FRAC:
        qc_flags.append("qc_progress_nonmonotonic")
    if detection_confidence is not None and np.isfinite(detection_confidence) and detection_confidence < MIN_DETECTION_CONFIDENCE:
        qc_flags.append("qc_weak_detection")
    if drive_duration_s is not None and not (DRIVE_DURATION_MIN_S <= drive_duration_s <= DRIVE_DURATION_MAX_S):
        qc_flags.append("qc_duration_implausible")
    elif cycle_duration_s is not None and not (CYCLE_DURATION_MIN_S <= cycle_duration_s <= CYCLE_DURATION_MAX_S):
        qc_flags.append("qc_duration_implausible")

    # Interpolate everything onto s_grid.
    features: dict[str, np.ndarray] = {}
    for name in side_map.keys():
        features[name] = _interp_on_progress(s_video, smoothed[name], s_grid)

    ds_dt_on_s = _interp_on_progress(s_video, ds_dt_time, s_grid)

    for name in side_map.keys():
        dtheta_dt_on_s = _interp_on_progress(s_video, dtheta_dt[name], s_grid)
        dtheta_ds = dtheta_dt_on_s / (ds_dt_on_s + CHAIN_RULE_EPS)
        features[name.replace("_deg", "_ddeg_ds")] = dtheta_ds

    if include_second_derivatives:
        for name in side_map.keys():
            d1_key = name.replace("_deg", "_ddeg_ds")
            d1 = features[d1_key]
            if d1.size >= 2 and np.isfinite(d1).sum() >= 2:
                features[name.replace("_deg", "_d2deg_ds2")] = np.gradient(d1, s_grid)
            else:
                features[name.replace("_deg", "_d2deg_ds2")] = np.full_like(d1, np.nan)

    vel_col = _resolve_velocity_column(drive_df)
    if vel_col is not None:
        handle_vel_raw = pd.to_numeric(drive_df[vel_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        handle_vel_raw = np.full(len(drive_df), np.nan)
    features["handle_velocity_px_s"] = _interp_on_progress(s_video, handle_vel_raw, s_grid)

    if np.isfinite(handle_vel_raw).sum() >= 2 and np.isfinite(time_arr).sum() >= 2:
        handle_accel_time = np.gradient(_interp_nans(handle_vel_raw), time_arr)
    else:
        handle_accel_time = np.full_like(handle_vel_raw, np.nan)
    features["handle_accel_px_s2"] = _interp_on_progress(s_video, handle_accel_time, s_grid)

    stroke_quality = _stroke_quality_score(
        nan_frac=nan_frac,
        max_deriv_deg_s=max_angular_vel,
        mono_violation_frac=mono_violation_frac,
        detection_confidence=detection_confidence if detection_confidence is not None else float("nan"),
        drive_duration_s=drive_duration_s if drive_duration_s is not None else float("nan"),
    )

    stats = {
        "max_deriv_deg_s": float(max_angular_vel),
        "ds_dt_min": float(ds_dt_min),
        "progress_mono_violation_frac": float(mono_violation_frac),
    }
    return StrokeFeatureResult(
        s_grid=np.asarray(s_grid, dtype=np.float64),
        features=features,
        rower_facing=detected_facing,
        stroke_quality_score=float(stroke_quality),
        qc_flags=qc_flags,
        stats=stats,
        nan_frac_angles=float(nan_frac),
    )


# ---------------------------------------------------------------------------
# Quality score (lightweight duplicate of inference_cli._compute_quality_score)
# ---------------------------------------------------------------------------


def _sigmoid_penalty(value: float, good: float, bad: float, k: float = 12.0) -> float:
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


def _range_penalty(value: float, good_lo: float, good_hi: float, bad_lo: float, bad_hi: float, k: float = 12.0) -> float:
    if not np.isfinite(value):
        return 0.5
    if good_lo <= value <= good_hi:
        return 1.0
    if value < good_lo:
        return _sigmoid_penalty(value, good=good_lo, bad=bad_lo, k=k)
    return _sigmoid_penalty(value, good=good_hi, bad=bad_hi, k=k)


def _stroke_quality_score(
    *,
    nan_frac: float,
    max_deriv_deg_s: float,
    mono_violation_frac: float,
    detection_confidence: float,
    drive_duration_s: float,
) -> float:
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
    return float(score)


# ---------------------------------------------------------------------------
# No-RP3 entry point for video-only inference
# ---------------------------------------------------------------------------


def build_pose_drive_segments(
    *,
    run_dir: Path,
    events_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    s_grid: np.ndarray,
    active_side: str,
    include_head: bool = True,
    include_second_derivatives: bool = False,
    rower_facing: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Video-only pose feature builder.

    Returns
    -------
    segments_df
        One row per (stroke, grid-bin) with canonical feature columns
        matching :func:`feature_column_names`.
    status_df
        One row per stroke with QC + summary metadata.
    """
    run_dir = Path(run_dir)
    angles_csv = run_dir / "motionbert" / "angles_h36m.csv"
    if not angles_csv.exists():
        raise FileNotFoundError(f"angles_h36m.csv not found at: {angles_csv}")
    angles_df = pd.read_csv(angles_csv)
    if "frame_idx" not in angles_df.columns:
        raise ValueError("angles_h36m.csv missing frame_idx column.")

    merged = frame_df.merge(angles_df, on="frame_idx", how="left", suffixes=("", "_angle"))

    run_include_head = include_head and ("head_vs_trunk_deg" in merged.columns)
    side_map = build_side_map(active_side, include_head=run_include_head)

    feature_cols = feature_column_names(
        include_head=run_include_head,
        include_second_derivatives=include_second_derivatives,
    )

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for seq_idx, (_, ev) in enumerate(events_df.iterrows()):
        stroke_idx = int(ev.get("stroke_idx", seq_idx))
        c_frame = int(ev["catch_frame_idx"])
        f_frame = int(ev["finish_frame_idx"])
        status_row: dict[str, Any] = {
            "seq_idx": int(seq_idx),
            "stroke_idx": int(stroke_idx),
            "catch_frame_idx": int(c_frame),
            "finish_frame_idx": int(f_frame),
            "segment_exported": False,
            "segment_rows_written": 0,
            "drop_reason": "",
        }

        drive = merged[(merged["frame_idx"] >= c_frame) & (merged["frame_idx"] <= f_frame)].copy()
        if len(drive) < 2:
            status_row["drop_reason"] = "invalid_drive_window"
            status_rows.append(status_row)
            continue

        catch_distance_px = float(ev["catch_distance_px"]) if "catch_distance_px" in ev.index else None
        drive_duration_s = None
        cycle_duration_s = None
        if "catch_time_s" in ev.index and "finish_time_s" in ev.index:
            drive_duration_s = float(ev["finish_time_s"]) - float(ev["catch_time_s"])
        detection_confidence = None
        if "catch_velocity_contrast" in ev.index or "finish_velocity_contrast" in ev.index:
            catch_vc = float(ev.get("catch_velocity_contrast", float("nan")))
            finish_vc = float(ev.get("finish_velocity_contrast", float("nan")))
            vals = [v for v in (catch_vc, finish_vc) if np.isfinite(v)]
            detection_confidence = min(vals) if vals else None

        result = build_stroke_feature_sequence(
            drive_df=drive,
            s_grid=s_grid,
            side_map=side_map,
            rower_facing=rower_facing,
            include_second_derivatives=include_second_derivatives,
            catch_distance_px=catch_distance_px,
            drive_duration_s=drive_duration_s,
            cycle_duration_s=cycle_duration_s,
            detection_confidence=detection_confidence,
        )
        if result is None:
            status_row["drop_reason"] = "invalid_drive_range"
            status_rows.append(status_row)
            continue

        status_row["rower_facing"] = result.rower_facing
        status_row["stroke_quality_score"] = float(result.stroke_quality_score)
        status_row["qc_flags"] = ",".join(result.qc_flags)
        status_row["nan_frac_angles"] = float(result.nan_frac_angles)
        status_row["max_deriv_deg_s"] = float(result.stats.get("max_deriv_deg_s", float("nan")))
        status_row["progress_mono_violation_frac"] = float(
            result.stats.get("progress_mono_violation_frac", float("nan"))
        )
        status_row["segment_exported"] = True
        status_row["segment_rows_written"] = int(len(s_grid))
        status_rows.append(status_row)

        for bin_idx, s_val in enumerate(s_grid):
            row: dict[str, Any] = {
                "run_name": run_dir.name,
                "seq_idx": int(seq_idx),
                "stroke_idx": int(stroke_idx),
                "active_side": active_side,
                "rower_facing": result.rower_facing,
                "drive_bin_idx": int(bin_idx),
                "s_force": float(s_val),
                "stroke_quality_score": float(result.stroke_quality_score),
                "qc_flags": status_row["qc_flags"],
            }
            for col in feature_cols:
                arr = result.features.get(col)
                if arr is None:
                    row[col] = float("nan")
                else:
                    v = arr[bin_idx]
                    row[col] = float(v) if np.isfinite(v) else float("nan")
            rows.append(row)

    segments_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    return segments_df, status_df


def stack_feature_tensor(
    segments_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_grid: int,
) -> tuple[np.ndarray, list[int]]:
    """Pivot a long-form segment dataframe into ``(N, G, K)`` tensor.

    Returns the tensor plus the ordered list of ``seq_idx`` so callers can
    map each row back to a stroke.
    """
    if segments_df.empty:
        return np.zeros((0, n_grid, len(feature_cols)), dtype=np.float64), []

    seq_order = (
        segments_df[["seq_idx", "drive_bin_idx"]]
        .drop_duplicates("seq_idx")
        .sort_values("seq_idx")["seq_idx"]
        .astype(int)
        .tolist()
    )
    K = len(feature_cols)
    N = len(seq_order)
    out = np.full((N, n_grid, K), np.nan, dtype=np.float64)
    for i, seq_idx in enumerate(seq_order):
        grp = segments_df[segments_df["seq_idx"] == seq_idx].sort_values("drive_bin_idx")
        if len(grp) != n_grid:
            continue
        for k, col in enumerate(feature_cols):
            if col in grp.columns:
                out[i, :, k] = pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=np.float64)
    return out, seq_order
