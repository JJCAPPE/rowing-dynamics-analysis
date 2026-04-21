"""Tests for :func:`inference.pair_session.coarse_sync_anchor`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pair_session import coarse_sync_anchor


def _video_df_from_intervals(intervals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "cycle_duration_s": intervals,
        "catch_time_s": np.cumsum(intervals),
    })


def _rp3_df_from_intervals(intervals: np.ndarray, start_stroke: int = 1) -> pd.DataFrame:
    drive = intervals * 0.4
    recover = intervals - drive
    stroke_number = np.arange(start_stroke, start_stroke + len(intervals))
    return pd.DataFrame({
        "stroke_number": stroke_number,
        "drive_time": drive,
        "recover_time": recover,
        "rp3_cycle_s": intervals,
    })


def test_coarse_sync_recovers_known_offset_zero_noise() -> None:
    rng = np.random.default_rng(42)
    video_intervals = rng.uniform(1.5, 2.3, size=40)
    offset = 5
    prefix = rng.uniform(1.5, 2.3, size=offset)
    rp3_intervals = np.concatenate([prefix, video_intervals])

    video_df = _video_df_from_intervals(video_intervals)
    rp3_df = _rp3_df_from_intervals(rp3_intervals, start_stroke=10)

    result = coarse_sync_anchor(video_df, rp3_df, anchor_video_idx=1)

    assert result.offset_strokes == offset
    assert result.anchor_rp3_idx == 1 + offset
    assert result.overlap_length >= 8
    assert result.best_cost < 1e-9
    assert result.mean_rate_diff_spm < 1e-6
    # Stroke numbers in rp3 start at 10, so idx=6 -> stroke 16.
    assert result.anchor_rp3_stroke_number == 10 + result.anchor_rp3_idx


def test_coarse_sync_robust_to_small_noise() -> None:
    rng = np.random.default_rng(0)
    video_intervals = rng.uniform(1.8, 2.2, size=30)
    offset = 3
    prefix = rng.uniform(1.8, 2.2, size=offset)
    rp3_intervals = np.concatenate([prefix, video_intervals + rng.normal(0, 0.02, size=30)])

    video_df = _video_df_from_intervals(video_intervals)
    rp3_df = _rp3_df_from_intervals(rp3_intervals)

    result = coarse_sync_anchor(video_df, rp3_df, anchor_video_idx=1, min_overlap=8)
    assert abs(result.offset_strokes - offset) <= 1
    assert result.normalized_cost < 0.5  # best offset well below median


def test_coarse_sync_raises_when_too_short() -> None:
    video_df = _video_df_from_intervals(np.array([2.0, 2.1]))
    rp3_df = _rp3_df_from_intervals(np.array([2.0, 2.1, 2.0]))
    with pytest.raises(ValueError):
        coarse_sync_anchor(video_df, rp3_df, anchor_video_idx=1, min_overlap=8)


def test_coarse_sync_raises_when_anchor_out_of_range() -> None:
    video_df = _video_df_from_intervals(np.array([2.0, 2.1, 2.0, 2.05]))
    rp3_df = _rp3_df_from_intervals(np.array([2.0, 2.1, 2.0, 2.05, 2.0]))
    with pytest.raises(ValueError):
        coarse_sync_anchor(video_df, rp3_df, anchor_video_idx=10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
