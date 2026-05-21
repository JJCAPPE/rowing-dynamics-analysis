"""Tests for :mod:`rowing.dataset.feature_contract` mirror normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rowing.dataset.feature_contract import (
    apply_mirror_normalization,
    build_side_map,
    canonical_columns,
    mirror_transform,
)


def test_build_side_map_left_right_symmetry() -> None:
    left = build_side_map("left", include_head=True)
    right = build_side_map("right", include_head=True)
    assert left["knee_active_deg"] == "left_knee_deg"
    assert right["knee_active_deg"] == "right_knee_deg"
    assert left["trunk_vs_horizontal_deg"] == "trunk_vs_horizontal_deg"
    assert "head_vs_trunk_deg" in left and "head_vs_trunk_deg" in right


def test_canonical_columns_includes_head() -> None:
    with_head = canonical_columns(include_head=True)
    without_head = canonical_columns(include_head=False)
    assert with_head[-1] == "head_vs_trunk_deg"
    assert "head_vs_trunk_deg" not in without_head
    assert len(with_head) == len(without_head) + 1


def test_mirror_transform_supplement_only_for_trunk() -> None:
    x = np.array([30.0, 90.0, 150.0])
    np.testing.assert_array_equal(mirror_transform("knee_active_deg", x, facing="left"), x)
    np.testing.assert_array_equal(mirror_transform("hip_active_deg", x, facing="left"), x)
    np.testing.assert_array_equal(mirror_transform("spine_flexion_deg", x, facing="left"), x)
    np.testing.assert_array_equal(mirror_transform("head_vs_trunk_deg", x, facing="left"), x)

    flipped = mirror_transform("trunk_vs_horizontal_deg", x, facing="left")
    np.testing.assert_array_equal(flipped, 180.0 - x)


def test_mirror_transform_identity_when_facing_right() -> None:
    x = np.array([30.0, 90.0, 150.0])
    for name in (
        "knee_active_deg",
        "hip_active_deg",
        "elbow_active_deg",
        "trunk_vs_horizontal_deg",
        "spine_flexion_deg",
        "head_vs_trunk_deg",
    ):
        np.testing.assert_array_equal(mirror_transform(name, x, facing="right"), x)


def test_apply_mirror_normalization_roundtrip_right() -> None:
    df = pd.DataFrame({
        "left_knee_deg": [20.0, 30.0, 40.0],
        "right_knee_deg": [22.0, 32.0, 42.0],
        "left_hip_deg": [50.0, 60.0, 70.0],
        "right_hip_deg": [52.0, 62.0, 72.0],
        "left_elbow_deg": [10.0, 20.0, 30.0],
        "right_elbow_deg": [12.0, 22.0, 32.0],
        "trunk_vs_horizontal_deg": [85.0, 95.0, 105.0],
        "spine_flexion_deg": [5.0, 10.0, 15.0],
        "head_vs_trunk_deg": [2.0, 4.0, 6.0],
    })
    side_map = build_side_map("right", include_head=True)

    facing_right = apply_mirror_normalization(df, side_map, facing="right")
    facing_left = apply_mirror_normalization(df, side_map, facing="left")

    # All canonical columns present on both facings.
    for k in side_map:
        assert k in facing_right and k in facing_left

    # Chain/spine/head are invariant to facing.
    for name in ("knee_active_deg", "hip_active_deg", "elbow_active_deg", "spine_flexion_deg", "head_vs_trunk_deg"):
        np.testing.assert_array_equal(facing_right[name], facing_left[name])

    # Trunk is flipped.
    np.testing.assert_allclose(
        facing_left["trunk_vs_horizontal_deg"],
        180.0 - facing_right["trunk_vs_horizontal_deg"],
    )


def test_apply_mirror_normalization_missing_column_is_nan() -> None:
    df = pd.DataFrame({
        "right_knee_deg": [20.0],
        "right_hip_deg": [50.0],
        "right_elbow_deg": [10.0],
        "spine_flexion_deg": [5.0],
        # deliberately missing trunk and head
    })
    side_map = build_side_map("right", include_head=True)
    out = apply_mirror_normalization(df, side_map, facing="right")
    assert np.isnan(out["trunk_vs_horizontal_deg"]).all()
    assert np.isnan(out["head_vs_trunk_deg"]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
