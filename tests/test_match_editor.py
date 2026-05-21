"""Tests for :mod:`rowing.matching.editor` (Phase 4 visual editor).

These tests drive the editor's state machine without spinning up a matplotlib
window: ``MatchEditor._redraw`` is a no-op when no figure exists, so we only
exercise the recompute / edit logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd
import pytest

from rowing.matching.editor import (
    EditorRunData,
    MatchEditor,
    PAIR_ANCHOR,
    PAIR_AUTO,
    PAIR_EXCLUDED,
    PAIR_PINNED,
)
from rowing.matching.match import MatchConfig
from rowing.matching.overrides import MatchOverrides, Pin, save_overrides


# ---------------------------------------------------------------------------
# Synthetic fixtures (mirror tests/test_match_overrides.py)
# ---------------------------------------------------------------------------


def _make_video_df(intervals: np.ndarray) -> pd.DataFrame:
    catch = np.cumsum(intervals)
    drive = intervals * 0.4
    return pd.DataFrame(
        {
            "stroke_idx": np.arange(len(intervals), dtype=int),
            "catch_time_s": catch,
            "finish_time_s": catch - intervals + drive,
            "drive_duration_s": drive,
            "recover_duration_s": intervals - drive,
            "cycle_duration_s": intervals,
        }
    )


def _make_rp3_df(intervals: np.ndarray) -> pd.DataFrame:
    drive = intervals * 0.4
    return pd.DataFrame(
        {
            "stroke_number": np.arange(1, len(intervals) + 1, dtype=int),
            "time": np.cumsum(intervals),
            "drive_time": drive,
            "recover_time": intervals - drive,
            "rp3_cycle_s": intervals,
        }
    )


def _editor_data(
    tmp_path: Path,
    *,
    video_n: int = 10,
    rp3_n: int = 15,
    interval_s: float = 2.0,
) -> EditorRunData:
    run_dir = tmp_path / "run"
    (run_dir / "inference").mkdir(parents=True)

    intervals_v = np.full(video_n, interval_s)
    intervals_r = np.full(rp3_n, interval_s)
    events_df = _make_video_df(intervals_v)
    rp3_df = _make_rp3_df(intervals_r)

    cfg = MatchConfig(
        max_jump_rows=10,
        max_interval_error_s=5.0,
        max_cumulative_error_base_s=10.0,
        max_cumulative_error_per_s=1.0,
        max_abs_cum_error_s=20.0,
    )
    return EditorRunData(
        run_dir=run_dir,
        events_df=events_df,
        rp3_df=rp3_df,
        rp3_clean_csv=run_dir / "rp3" / "fake-clean.csv",
        rp3_dirty_csv=None,
        summary={"active_side": "left"},
        cfg=cfg,
        baseline_anchor_video_idx=0,
        baseline_anchor_rp3_idx=0,
    )


# ---------------------------------------------------------------------------
# Match recompute
# ---------------------------------------------------------------------------


def test_baseline_match_aligns_one_to_one(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)
    assert len(editor.manifest_df) == 10
    rp3_rows = editor.manifest_df["rp3_row_idx"].astype(int).tolist()
    assert rp3_rows == list(range(10))
    assert editor.last_match_error is None


def test_pair_status_marks_anchor_and_auto(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)
    assert editor._pair_status(0) == PAIR_ANCHOR
    assert editor._pair_status(3) == PAIR_AUTO


# ---------------------------------------------------------------------------
# Edit transitions
# ---------------------------------------------------------------------------


def test_exclude_drops_video_stroke_and_promotes_skip(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = 3
    editor._exclude_selected()

    assert editor.last_match_error is None
    matched = editor.manifest_df["video_stroke_idx"].astype(int).tolist()
    assert 3 not in matched
    assert len(matched) == 9
    assert editor._pair_status(3) == PAIR_EXCLUDED
    assert editor._dirty


def test_pin_forces_remap(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = 5
    editor._remap_selected_to_row(7)

    assert editor.last_match_error is None
    v5 = editor.manifest_df[editor.manifest_df["video_stroke_idx"].astype(int) == 5]
    assert int(v5.iloc[0]["rp3_row_idx"]) == 7
    assert editor._pair_status(5) == PAIR_PINNED


def test_unpin_restores_auto_match(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = 5
    editor._remap_selected_to_row(7)
    assert editor._pair_status(5) == PAIR_PINNED

    editor.selected_video_stroke_idx = 5
    editor._unpin_selected()
    assert editor._pair_status(5) == PAIR_AUTO
    v5 = editor.manifest_df[editor.manifest_df["video_stroke_idx"].astype(int) == 5]
    assert int(v5.iloc[0]["rp3_row_idx"]) == 5


def test_set_anchor_to_selected_resets_post_anchor_state(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = 3
    editor._set_anchor_to_selected()

    assert editor.anchor_video_idx == 3
    assert editor.anchor_rp3_idx == 3
    matched = editor.manifest_df["video_stroke_idx"].astype(int).tolist()
    assert matched[0] == 3  # anchor is the new first row
    assert editor._dirty


def test_reset_overrides_clears_state(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = 3
    editor._exclude_selected()
    editor.selected_video_stroke_idx = 5
    editor._remap_selected_to_row(7)
    assert not editor.overrides.is_empty

    editor._reset_overrides()
    assert editor.overrides.is_empty
    assert editor.anchor_video_idx == data.baseline_anchor_video_idx
    assert editor.anchor_rp3_idx == data.baseline_anchor_rp3_idx
    assert editor.manifest_df["rp3_row_idx"].astype(int).tolist() == list(range(10))


def test_anchor_excluded_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _editor_data(tmp_path)
    editor = MatchEditor(data)

    editor.selected_video_stroke_idx = data.baseline_anchor_video_idx
    editor._exclude_selected()

    captured = capsys.readouterr().out
    assert "Cannot exclude the anchor" in captured
    assert data.baseline_anchor_video_idx not in editor.overrides.excluded_set()


# ---------------------------------------------------------------------------
# Sidecar warmup
# ---------------------------------------------------------------------------


def test_loads_existing_overrides_on_init(tmp_path: Path) -> None:
    data = _editor_data(tmp_path)
    save_overrides(
        data.run_dir,
        MatchOverrides(
            pinned=[Pin(video_stroke_idx=4, rp3_stroke_number=6)],
            excluded_video_stroke_idx=[2],
        ),
    )
    editor = MatchEditor(data)
    assert editor.overrides.pinned == [Pin(video_stroke_idx=4, rp3_stroke_number=6)]
    assert editor.overrides.excluded_set() == {2}
    matched = editor.manifest_df["video_stroke_idx"].astype(int).tolist()
    assert 2 not in matched
    v4 = editor.manifest_df[editor.manifest_df["video_stroke_idx"].astype(int) == 4]
    assert int(v4.iloc[0]["rp3_stroke_number"]) == 6
