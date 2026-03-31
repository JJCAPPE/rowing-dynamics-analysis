"""Shared drive-event detection for rowing stroke signals.

Provides a single, canonical catch/finish detector used by both the
Sports2D stroke-tracking stage and the inference CLI.  The detector
operates on a **raw** (unsmoothed) ``relative_axis_px`` signal and
applies exactly one controlled smoothing pass internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriveEvent:
    stroke_idx: int
    catch_frame_idx: int
    catch_time_s: float
    finish_frame_idx: int
    finish_time_s: float
    next_catch_frame_idx: int
    next_catch_time_s: float
    drive_duration_s: float
    recover_duration_s: float
    cycle_duration_s: float
    catch_distance_px: float
    finish_distance_px: float
    drive_displacement_px: float
    catch_velocity_contrast: float
    finish_velocity_contrast: float
    drive_prominence: float


@dataclass(frozen=True)
class DetectionResult:
    events: list[DriveEvent]
    catch_candidates_raw: np.ndarray
    catches_filtered: np.ndarray
    signal_smooth_px: np.ndarray
    slope_px_s: np.ndarray
    fps_estimate: float
    min_drive_disp_px: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if signal.size < window:
        return signal.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _interp_nans(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    if values.size == 0:
        return values
    x = np.arange(values.size, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.all():
        return values
    if not mask.any():
        raise ValueError("Signal is all-NaN; cannot detect drive events.")
    values[~mask] = np.interp(x[~mask], x[mask], values[mask])
    return values


def _infer_time_s(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df.columns:
        time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(time_s).sum() >= 2:
            return _interp_nans(time_s)

    if "frame_idx" in df.columns:
        frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=np.float64)
        frame_idx = _interp_nans(frame_idx)
        start = float(frame_idx[0])
        return (frame_idx - start).astype(np.float64)

    return np.arange(len(df), dtype=np.float64)


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    out = signs.astype(np.int8, copy=True)
    if out.size == 0:
        return out

    for i in range(1, out.size):
        if out[i] == 0:
            out[i] = out[i - 1]
    for i in range(out.size - 2, -1, -1):
        if out[i] == 0:
            out[i] = out[i + 1]
    return out


def _find_catch_candidates(signs: np.ndarray) -> np.ndarray:
    catches: list[int] = []
    for i in range(1, signs.size):
        if signs[i - 1] < 0 and signs[i] > 0:
            catches.append(i)
    return np.asarray(catches, dtype=np.int32)


def _filter_catches_by_cycle(
    catches: np.ndarray,
    signal_smooth_px: np.ndarray,
    time_s: np.ndarray,
    min_cycle_s: float,
) -> np.ndarray:
    if catches.size <= 1:
        return catches

    kept: list[int] = [int(catches[0])]
    for c in catches[1:]:
        c = int(c)
        prev = kept[-1]
        if float(time_s[c] - time_s[prev]) >= min_cycle_s:
            kept.append(c)
            continue
        if float(signal_smooth_px[c]) < float(signal_smooth_px[prev]):
            kept[-1] = c
    return np.asarray(kept, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FINISH_METHOD_VELOCITY_THRESHOLD = "velocity_threshold"
FINISH_METHOD_POSITION_MAX = "position_max"
FINISH_METHOD_VELOCITY_CALIBRATED = "velocity_calibrated"
VALID_FINISH_METHODS = {
    FINISH_METHOD_VELOCITY_THRESHOLD,
    FINISH_METHOD_POSITION_MAX,
    FINISH_METHOD_VELOCITY_CALIBRATED,
}

DEFAULT_CATCH_VELOCITY_FRAC = 0.43
DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED = 0.75


def detect_drive_events(
    df: pd.DataFrame,
    *,
    smooth_window_s: float,
    min_cycle_s: float,
    min_drive_s: float,
    min_recover_s: float,
    min_drive_disp_frac: float,
    slope_tol_frac: float,
    finish_velocity_frac: float = 0.0,
    catch_velocity_frac: float = 0.0,
    finish_method: str = FINISH_METHOD_VELOCITY_THRESHOLD,
) -> DetectionResult:
    """Detect catch/finish events from a stroke signal DataFrame.

    The DataFrame must contain a ``relative_axis_px`` column (raw or
    smoothed -- the function applies its own single smoothing pass).

    ``finish_method`` controls how catch and finish are placed:

    * ``velocity_threshold`` -- catch at slope zero-crossing, finish
      where velocity drops to ``finish_velocity_frac`` of peak.
    * ``position_max`` -- catch at slope zero-crossing, finish at the
      position maximum (argmax).
    * ``velocity_calibrated`` -- both catch *and* finish placed by
      per-stroke velocity-fraction thresholds.  Catch fires when
      velocity rises above ``catch_velocity_frac * peak_vel``; finish
      fires when velocity drops below ``finish_velocity_frac * peak_vel``.
    """
    if finish_method not in VALID_FINISH_METHODS:
        raise ValueError(
            f"Unknown finish_method {finish_method!r}. "
            f"Valid: {sorted(VALID_FINISH_METHODS)}"
        )

    if "relative_axis_px" not in df.columns:
        raise ValueError("DataFrame must contain 'relative_axis_px'.")

    time_s = _infer_time_s(df)
    if time_s.size < 3:
        raise ValueError("Need at least 3 samples to detect catch/finish events.")

    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    fps_estimate = float(1.0 / np.median(valid_dt)) if valid_dt.size else 30.0

    signal_raw = pd.to_numeric(df["relative_axis_px"], errors="coerce").to_numpy(dtype=np.float64)
    signal_raw = _interp_nans(signal_raw)

    window_frames = max(3, int(round(max(0.01, smooth_window_s) * fps_estimate)))
    signal_smooth_px = _moving_average(signal_raw, window=window_frames)
    slope_px_s = np.gradient(signal_smooth_px, time_s)

    slope_std = float(np.nanstd(slope_px_s))
    slope_tol = max(1e-6, slope_std * max(0.0, slope_tol_frac))
    signs = np.zeros_like(slope_px_s, dtype=np.int8)
    signs[slope_px_s > slope_tol] = 1
    signs[slope_px_s < -slope_tol] = -1
    signs = _fill_zero_signs(signs)

    catches_raw = _find_catch_candidates(signs)
    catches_filtered = _filter_catches_by_cycle(
        catches=catches_raw,
        signal_smooth_px=signal_smooth_px,
        time_s=time_s,
        min_cycle_s=max(0.0, min_cycle_s),
    )

    p95 = float(np.nanpercentile(signal_smooth_px, 95.0))
    p05 = float(np.nanpercentile(signal_smooth_px, 5.0))
    signal_span_px = max(0.0, p95 - p05)
    min_drive_disp_px = max(1.0, signal_span_px * max(0.0, min_drive_disp_frac))

    use_velocity = (
        finish_method == FINISH_METHOD_VELOCITY_THRESHOLD
        and finish_velocity_frac > 0
    )
    use_calibrated = finish_method == FINISH_METHOD_VELOCITY_CALIBRATED

    has_frame_idx = "frame_idx" in df.columns
    events: list[DriveEvent] = []
    if catches_filtered.size >= 2:
        for i in range(catches_filtered.size - 1):
            c0 = int(catches_filtered[i])
            c1 = int(catches_filtered[i + 1])
            if c1 <= c0 + 1:
                continue

            segment = signal_smooth_px[c0 : c1 + 1]
            pos_max_rel = int(np.argmax(segment))
            f0 = int(c0 + pos_max_rel)

            if use_calibrated and pos_max_rel > 1:
                c0_orig = c0
                cycle_len = c1 - c0
                max_catch_shift = int(cycle_len * 0.25)
                seg_vel = slope_px_s[c0 : c0 + pos_max_rel + 1]
                peak_vel_rel = int(np.argmax(seg_vel))
                peak_vel = float(seg_vel[peak_vel_rel])
                if peak_vel > 0:
                    if catch_velocity_frac > 0:
                        catch_thresh = peak_vel * catch_velocity_frac
                        rising = np.where(seg_vel >= catch_thresh)[0]
                        if rising.size > 0:
                            shift = int(rising[0])
                            if shift <= max_catch_shift:
                                c0 = c0_orig + shift
                    finish_thresh = peak_vel * finish_velocity_frac
                    after_peak = seg_vel[peak_vel_rel:]
                    falling = np.where(after_peak <= finish_thresh)[0]
                    if falling.size > 0:
                        f0 = int(c0_orig + peak_vel_rel + int(falling[0]))
            elif use_velocity and pos_max_rel > 1:
                vel_seg = slope_px_s[c0 : c0 + pos_max_rel + 1]
                peak_vel_rel = int(np.argmax(vel_seg))
                peak_vel = float(vel_seg[peak_vel_rel])
                if peak_vel > 0:
                    threshold = peak_vel * finish_velocity_frac
                    after_peak = vel_seg[peak_vel_rel:]
                    below = np.where(after_peak <= threshold)[0]
                    if below.size > 0:
                        f0 = int(c0 + peak_vel_rel + int(below[0]))

            if not (c0 < f0 < c1):
                continue

            drive_duration_s = float(time_s[f0] - time_s[c0])
            recover_duration_s = float(time_s[c1] - time_s[f0])
            cycle_duration_s = float(time_s[c1] - time_s[c0])
            drive_disp_px = float(signal_smooth_px[f0] - signal_smooth_px[c0])

            if drive_duration_s < min_drive_s:
                continue
            if recover_duration_s < min_recover_s:
                continue
            if drive_disp_px < min_drive_disp_px:
                continue

            drive_vel = slope_px_s[c0 : f0 + 1]
            peak_drive_vel = float(np.nanmax(np.abs(drive_vel))) if drive_vel.size > 0 else 0.0
            vel_norm = max(peak_drive_vel, 1e-9)

            half_w = max(2, int(round(fps_estimate * 0.03)))
            c_before = slope_px_s[max(0, c0 - half_w) : c0]
            c_after = slope_px_s[c0 : min(len(slope_px_s), c0 + half_w)]
            catch_vel_contrast = float(
                (np.nanmean(np.abs(c_after)) if c_after.size else 0.0)
                - (np.nanmean(np.abs(c_before)) if c_before.size else 0.0)
            ) / vel_norm

            f_before = slope_px_s[max(0, f0 - half_w) : f0]
            f_after = slope_px_s[f0 : min(len(slope_px_s), f0 + half_w)]
            finish_vel_contrast = float(
                (np.nanmean(np.abs(f_before)) if f_before.size else 0.0)
                - (np.nanmean(np.abs(f_after)) if f_after.size else 0.0)
            ) / vel_norm

            prominence = drive_disp_px / max(min_drive_disp_px, 1e-9)

            events.append(
                DriveEvent(
                    stroke_idx=len(events),
                    catch_frame_idx=int(df.iloc[c0]["frame_idx"]) if has_frame_idx else c0,
                    catch_time_s=float(time_s[c0]),
                    finish_frame_idx=int(df.iloc[f0]["frame_idx"]) if has_frame_idx else f0,
                    finish_time_s=float(time_s[f0]),
                    next_catch_frame_idx=int(df.iloc[c1]["frame_idx"]) if has_frame_idx else c1,
                    next_catch_time_s=float(time_s[c1]),
                    drive_duration_s=drive_duration_s,
                    recover_duration_s=recover_duration_s,
                    cycle_duration_s=cycle_duration_s,
                    catch_distance_px=float(signal_smooth_px[c0]),
                    finish_distance_px=float(signal_smooth_px[f0]),
                    drive_displacement_px=drive_disp_px,
                    catch_velocity_contrast=catch_vel_contrast,
                    finish_velocity_contrast=finish_vel_contrast,
                    drive_prominence=prominence,
                )
            )

    return DetectionResult(
        events=events,
        catch_candidates_raw=catches_raw,
        catches_filtered=catches_filtered,
        signal_smooth_px=signal_smooth_px.astype(np.float32),
        slope_px_s=slope_px_s.astype(np.float32),
        fps_estimate=fps_estimate,
        min_drive_disp_px=min_drive_disp_px,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    catch_velocity_frac: float
    finish_velocity_frac: float
    mae_s: float
    me_s: float
    std_s: float
    n_strokes: int


def calibrate_velocity_fracs(
    df: pd.DataFrame,
    rp3_drive_durations: dict[int, float],
    *,
    smooth_window_s: float = 0.04,
    min_cycle_s: float = 0.8,
    slope_tol_frac: float = 0.05,
    catch_frac_range: tuple[float, float] = (0.30, 0.60),
    finish_frac_range: tuple[float, float] = (0.65, 0.85),
    step: float = 0.01,
    min_matched_strokes: int = 5,
) -> CalibrationResult:
    """Sweep catch/finish velocity fracs to minimise MAE vs RP3 drive durations.

    Uses coarse zero-crossing stroke boundaries, then for each
    ``(catch_frac, finish_frac)`` pair recomputes catch/finish within
    each cycle using per-stroke peak-velocity fractions.

    Parameters
    ----------
    df : DataFrame with ``relative_axis_px``, ``time_s``, ``frame_idx``.
    rp3_drive_durations : ``{video_stroke_idx: rp3_drive_s}``.
    """
    time_s = _infer_time_s(df)
    fps_est = 120.0
    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    if valid_dt.size:
        fps_est = float(1.0 / np.median(valid_dt))

    signal_raw = pd.to_numeric(
        df["relative_axis_px"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    signal_raw = _interp_nans(signal_raw)

    window_frames = max(3, int(round(max(0.01, smooth_window_s) * fps_est)))
    sig = _moving_average(signal_raw, window=window_frames)
    vel = np.gradient(sig, time_s)

    slope_std = float(np.nanstd(vel))
    slope_tol = max(1e-6, slope_std * max(0.0, slope_tol_frac))
    signs = np.zeros_like(vel, dtype=np.int8)
    signs[vel > slope_tol] = 1
    signs[vel < -slope_tol] = -1
    signs = _fill_zero_signs(signs)
    catches_raw = _find_catch_candidates(signs)
    catches = _filter_catches_by_cycle(
        catches_raw, sig, time_s, max(0.0, min_cycle_s),
    )

    if catches.size < 2:
        return CalibrationResult(
            DEFAULT_CATCH_VELOCITY_FRAC,
            DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED,
            mae_s=float("nan"), me_s=float("nan"),
            std_s=float("nan"), n_strokes=0,
        )

    has_frame_idx = "frame_idx" in df.columns
    frame_idx_arr = (
        pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(dtype=np.int64)
        if has_frame_idx else None
    )

    stroke_info: list[tuple[int, int, int, int, float]] = []
    event_counter = 0
    for ci in range(catches.size - 1):
        c0 = int(catches[ci])
        c1 = int(catches[ci + 1])
        if c1 <= c0 + 1:
            continue
        seg = sig[c0 : c1 + 1]
        pos_max_rel = int(np.argmax(seg))
        if pos_max_rel < 2:
            continue
        si = event_counter
        event_counter += 1
        if si not in rp3_drive_durations:
            continue
        stroke_info.append((si, c0, c1, pos_max_rel, rp3_drive_durations[si]))

    if len(stroke_info) < min_matched_strokes:
        return CalibrationResult(
            DEFAULT_CATCH_VELOCITY_FRAC,
            DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED,
            mae_s=float("nan"), me_s=float("nan"),
            std_s=float("nan"), n_strokes=len(stroke_info),
        )

    catch_fracs = np.arange(
        catch_frac_range[0], catch_frac_range[1] + step * 0.5, step,
    )
    finish_fracs = np.arange(
        finish_frac_range[0], finish_frac_range[1] + step * 0.5, step,
    )

    best_mae = float("inf")
    best_cf = DEFAULT_CATCH_VELOCITY_FRAC
    best_ff = DEFAULT_FINISH_VELOCITY_FRAC_CALIBRATED
    best_me = 0.0
    best_std = 0.0
    best_n = 0

    for cf in catch_fracs:
        for ff in finish_fracs:
            errors: list[float] = []
            for si, c0, c1, pos_max_rel, rp3_d in stroke_info:
                seg_vel = vel[c0 : c0 + pos_max_rel + 1]
                peak_vel_rel = int(np.argmax(seg_vel))
                peak_vel = float(seg_vel[peak_vel_rel])
                if peak_vel <= 0:
                    continue

                catch_cands = np.where(seg_vel >= peak_vel * cf)[0]
                if catch_cands.size == 0:
                    continue
                catch_rel = int(catch_cands[0])

                after_peak = seg_vel[peak_vel_rel:]
                finish_cands = np.where(after_peak <= peak_vel * ff)[0]
                if finish_cands.size == 0:
                    continue
                finish_rel = peak_vel_rel + int(finish_cands[0])
                if finish_rel <= catch_rel:
                    continue

                video_drive = float(
                    time_s[c0 + finish_rel] - time_s[c0 + catch_rel]
                )
                errors.append(video_drive - rp3_d)

            if len(errors) < min_matched_strokes:
                continue
            errs = np.array(errors)
            mae = float(np.abs(errs).mean())
            if mae < best_mae:
                best_mae = mae
                best_cf = float(cf)
                best_ff = float(ff)
                best_me = float(errs.mean())
                best_std = float(errs.std())
                best_n = len(errors)

    return CalibrationResult(
        catch_velocity_frac=round(best_cf, 4),
        finish_velocity_frac=round(best_ff, 4),
        mae_s=best_mae,
        me_s=best_me,
        std_s=best_std,
        n_strokes=best_n,
    )
