"""Tests for :mod:`rowing.matching.overrides` and DP override integration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowing.matching.match import MatchConfig, _build_match_manifest
from rowing.matching.overrides import (
    MatchOverrides,
    Pin,
    load_overrides,
    overrides_path,
    resolve_pin_to_row_idx,
    save_overrides,
    validate_overrides,
)


# ---------------------------------------------------------------------------
# Synthetic match fixtures
# ---------------------------------------------------------------------------


def _make_video_df(intervals: np.ndarray, drive_frac: float = 0.4) -> pd.DataFrame:
    """Build a fake `drive_events.csv` from per-stroke intervals."""
    catch_times = np.cumsum(intervals)
    drive = intervals * drive_frac
    finish_times = catch_times - intervals + drive  # catch + drive
    next_catch = catch_times + intervals
    return pd.DataFrame(
        {
            "stroke_idx": np.arange(len(intervals), dtype=int),
            "catch_time_s": catch_times,
            "finish_time_s": finish_times,
            "drive_duration_s": drive,
            "recover_duration_s": intervals - drive,
            "cycle_duration_s": intervals,
        }
    )


def _make_rp3_df(intervals: np.ndarray, *, start_stroke: int = 1) -> pd.DataFrame:
    """Build a fake RP3 clean CSV from per-stroke intervals."""
    drive = intervals * 0.4
    return pd.DataFrame(
        {
            "stroke_number": np.arange(start_stroke, start_stroke + len(intervals), dtype=int),
            "time": np.cumsum(intervals),
            "drive_time": drive,
            "recover_time": intervals - drive,
            "rp3_cycle_s": intervals,
        }
    )


def _default_cfg() -> MatchConfig:
    return MatchConfig(
        max_jump_rows=10,
        max_interval_error_s=2.0,
        max_cumulative_error_base_s=2.0,
        max_cumulative_error_per_s=0.20,
        max_abs_cum_error_s=4.0,
        w_drive=0.4,
        w_recover=0.4,
        w_interval=1.0,
        w_cumulative=1.0,
        w_skip=0.08,
    )


# ---------------------------------------------------------------------------
# Sidecar load/save/validate
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    overrides = MatchOverrides(
        anchor_video_stroke_idx=1,
        anchor_rp3_stroke_number=12,
        active_side="right",
        pinned=[Pin(video_stroke_idx=4, rp3_stroke_number=15)],
        excluded_video_stroke_idx=[7, 9],
    )
    saved_path = save_overrides(run_dir, overrides)
    assert saved_path == overrides_path(run_dir)
    assert saved_path.exists()

    loaded = load_overrides(run_dir)
    assert loaded.anchor_video_stroke_idx == 1
    assert loaded.anchor_rp3_stroke_number == 12
    assert loaded.active_side == "right"
    assert loaded.pinned == [Pin(video_stroke_idx=4, rp3_stroke_number=15)]
    assert loaded.excluded_video_stroke_idx == [7, 9]


def test_save_empty_deletes_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    save_overrides(run_dir, MatchOverrides(anchor_rp3_stroke_number=2))
    assert overrides_path(run_dir).exists()

    save_overrides(run_dir, MatchOverrides())
    assert not overrides_path(run_dir).exists()


def test_load_returns_empty_when_missing(tmp_path: Path) -> None:
    overrides = load_overrides(tmp_path / "nonexistent")
    assert overrides.is_empty


def test_load_raises_on_malformed_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    inference_dir = run_dir / "inference"
    inference_dir.mkdir(parents=True)
    (inference_dir / "match_overrides.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="Malformed match overrides"):
        load_overrides(run_dir)


def test_validate_rejects_overlap_between_pin_and_exclude() -> None:
    overrides = MatchOverrides(
        pinned=[Pin(video_stroke_idx=3, rp3_stroke_number=10)],
        excluded_video_stroke_idx=[3],
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_overrides(overrides)


def test_validate_unknown_video_stroke_raises() -> None:
    overrides = MatchOverrides(pinned=[Pin(video_stroke_idx=99, rp3_stroke_number=10)])
    with pytest.raises(ValueError, match="unknown video_stroke_idx"):
        validate_overrides(overrides, video_stroke_indices=range(5))


def test_validate_unknown_rp3_stroke_number_raises() -> None:
    overrides = MatchOverrides(pinned=[Pin(video_stroke_idx=2, rp3_stroke_number=999)])
    with pytest.raises(ValueError, match="rp3_stroke_number"):
        validate_overrides(overrides, rp3_stroke_numbers=range(20))


def test_resolve_pin_to_row_idx() -> None:
    rp3_df = _make_rp3_df(np.full(5, 2.0), start_stroke=10)
    assert resolve_pin_to_row_idx(rp3_df, 12) == 2
    with pytest.raises(KeyError):
        resolve_pin_to_row_idx(rp3_df, 99)


# ---------------------------------------------------------------------------
# Matcher DP — pin enforcement
# ---------------------------------------------------------------------------


def test_pin_forces_path_through_chosen_rp3_row() -> None:
    """Without a pin the natural alignment is identity; a pin must override it."""
    intervals = np.full(10, 2.0)
    video_df = _make_video_df(intervals)
    rp3_df = _make_rp3_df(np.full(15, 2.0))

    # Loosen tolerances so the pinned path is reachable
    cfg = MatchConfig(
        max_jump_rows=10,
        max_interval_error_s=5.0,
        max_cumulative_error_base_s=10.0,
        max_cumulative_error_per_s=1.0,
        max_abs_cum_error_s=20.0,
    )
    baseline = _build_match_manifest(
        video_df, rp3_df, anchor_video_idx=0, anchor_rp3_idx=0, cfg=cfg,
    )
    assert baseline.matched_rp3_indices[5] == 5

    pinned = _build_match_manifest(
        video_df,
        rp3_df,
        anchor_video_idx=0,
        anchor_rp3_idx=0,
        cfg=cfg,
        pinned_rp3_row_by_relative_idx={5: 7},
    )
    assert pinned.matched_rp3_indices[5] == 7
    # Pinned path scores worse than (or equal to) baseline since baseline is optimal
    assert pinned.total_score >= baseline.total_score


def test_pin_unreachable_with_tight_jump_raises() -> None:
    rng = np.random.default_rng(1)
    intervals = rng.uniform(1.9, 2.1, size=10)
    video_df = _make_video_df(intervals)
    rp3_df = _make_rp3_df(intervals)
    cfg = MatchConfig(
        max_jump_rows=1,  # very tight: only direct neighbour
        max_interval_error_s=2.0,
        max_cumulative_error_base_s=10.0,
        max_cumulative_error_per_s=1.0,
        max_abs_cum_error_s=20.0,
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        _build_match_manifest(
            video_df,
            rp3_df,
            anchor_video_idx=0,
            anchor_rp3_idx=0,
            cfg=cfg,
            pinned_rp3_row_by_relative_idx={3: 9},
        )


def test_pin_anchor_must_match_anchor_rp3_idx() -> None:
    intervals = np.full(8, 2.0)
    video_df = _make_video_df(intervals)
    rp3_df = _make_rp3_df(np.full(12, 2.0))
    with pytest.raises(ValueError, match="must agree"):
        _build_match_manifest(
            video_df,
            rp3_df,
            anchor_video_idx=0,
            anchor_rp3_idx=1,
            cfg=_default_cfg(),
            pinned_rp3_row_by_relative_idx={0: 2},
        )


# ---------------------------------------------------------------------------
# Matcher DP — exclusion handling
# ---------------------------------------------------------------------------


def test_excluded_video_stroke_is_skipped() -> None:
    intervals = np.full(8, 2.0)
    video_df = _make_video_df(intervals)
    # Drop stroke 3 from the video side (e.g. occluded handle)
    rp3_df = _make_rp3_df(intervals)

    cfg = _default_cfg()
    result = _build_match_manifest(
        video_df,
        rp3_df,
        anchor_video_idx=0,
        anchor_rp3_idx=0,
        cfg=cfg,
        excluded_relative_indices={3},
    )
    # Manifest has n - 1 rows (excluded stroke is gone)
    assert len(result.manifest) == len(video_df) - 1
    # Excluded video stroke is not in the manifest
    assert 3 not in result.manifest["video_stroke_idx"].astype(int).tolist()


def test_excluded_anchor_rejects() -> None:
    intervals = np.full(5, 2.0)
    video_df = _make_video_df(intervals)
    rp3_df = _make_rp3_df(intervals)
    with pytest.raises(ValueError, match="anchor"):
        _build_match_manifest(
            video_df,
            rp3_df,
            anchor_video_idx=0,
            anchor_rp3_idx=0,
            cfg=_default_cfg(),
            excluded_relative_indices={0},
        )


def test_pin_and_exclude_overlap_rejects() -> None:
    intervals = np.full(5, 2.0)
    video_df = _make_video_df(intervals)
    rp3_df = _make_rp3_df(intervals)
    with pytest.raises(ValueError, match="overlap"):
        _build_match_manifest(
            video_df,
            rp3_df,
            anchor_video_idx=0,
            anchor_rp3_idx=0,
            cfg=_default_cfg(),
            pinned_rp3_row_by_relative_idx={2: 2},
            excluded_relative_indices={2},
        )


# ---------------------------------------------------------------------------
# Anchor override precedence
# ---------------------------------------------------------------------------


def test_anchor_override_via_sidecar_takes_effect_when_cli_silent(tmp_path: Path) -> None:
    """Saved sidecar should provide anchor_rp3_stroke_number when CLI omits it."""
    run_dir = tmp_path / "run"
    save_overrides(
        run_dir,
        MatchOverrides(anchor_rp3_stroke_number=15, active_side="left"),
    )
    loaded = load_overrides(run_dir)
    assert loaded.anchor_rp3_stroke_number == 15
    assert loaded.active_side == "left"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
