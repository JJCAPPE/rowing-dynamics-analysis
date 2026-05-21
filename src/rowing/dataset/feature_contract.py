"""Shared feature contract for segment building and model inference.

Defines:
- The canonical active-side feature names.
- The source-column mapping per ``active_side`` (left/right).
- The mirror-normalization policy applied when ``rower_facing == 'left'``.

Rationale (Section 6 of force-curve-inference-process.md)
--------------------------------------------------------
The upstream MotionBERT H36M angle pipeline produces angles with the
following conventions (see sports2d_app/rowing_pose/kinematics.py):

- ``{left|right}_knee_deg``, ``{left|right}_hip_deg``,
  ``{left|right}_elbow_deg``, ``spine_flexion_deg``, ``head_vs_trunk_deg``
  are UNSIGNED flexion magnitudes returned by ``angle_abc`` in
  ``[0, 180]`` degrees.  They are symmetric under a world-axis flip and
  therefore never require mirror normalization.
- ``trunk_vs_horizontal_deg`` is the trunk orientation relative to the
  +x image axis.  Rowers facing left produce angles greater than 90 deg
  near the catch; facing right produces angles less than 90 deg.  To
  give every session the same semantics (catch -> low angle, finish ->
  high angle) we apply ``180 - angle`` when the detected facing is
  ``left``.

The mirror policy is centralized here so training (segment export) and
video-only inference share the same transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_ACTIVE_CHAIN: tuple[str, ...] = (
    "knee_active_deg",
    "hip_active_deg",
    "elbow_active_deg",
)

CANONICAL_CENTRAL: tuple[str, ...] = (
    "trunk_vs_horizontal_deg",
    "spine_flexion_deg",
    "head_vs_trunk_deg",
)

# Full canonical feature ordering (chain first, then central signals).
CANONICAL_ANGLE_COLS: tuple[str, ...] = CANONICAL_ACTIVE_CHAIN + CANONICAL_CENTRAL


@dataclass(frozen=True)
class MirrorPolicy:
    """How to transform a source column when the rower faces ``left``."""

    name: str
    kind: str  # one of: "identity", "supplement_180"


# Mirror-flip policy per canonical feature.
#
# - ``identity``       : no change; flexion magnitudes are symmetric.
# - ``supplement_180`` : ``x -> 180 - x``; applies to orientation angles
#                        measured against the +x world axis.
_MIRROR_POLICIES: dict[str, MirrorPolicy] = {
    "knee_active_deg": MirrorPolicy("knee_active_deg", "identity"),
    "hip_active_deg": MirrorPolicy("hip_active_deg", "identity"),
    "elbow_active_deg": MirrorPolicy("elbow_active_deg", "identity"),
    "trunk_vs_horizontal_deg": MirrorPolicy("trunk_vs_horizontal_deg", "supplement_180"),
    "spine_flexion_deg": MirrorPolicy("spine_flexion_deg", "identity"),
    "head_vs_trunk_deg": MirrorPolicy("head_vs_trunk_deg", "identity"),
}


def build_side_map(
    active_side: str,
    *,
    include_head: bool = True,
) -> dict[str, str]:
    """Return ``{canonical_name: source_column}`` for the chosen active side.

    When ``include_head`` is False the map omits ``head_vs_trunk_deg`` so
    that legacy segment CSVs (written before head was part of the feature
    contract) can still be consumed.
    """
    if active_side not in {"left", "right"}:
        raise ValueError("active_side must be 'left' or 'right'.")
    mapping = {
        "knee_active_deg": f"{active_side}_knee_deg",
        "hip_active_deg": f"{active_side}_hip_deg",
        "elbow_active_deg": f"{active_side}_elbow_deg",
        "trunk_vs_horizontal_deg": "trunk_vs_horizontal_deg",
        "spine_flexion_deg": "spine_flexion_deg",
    }
    if include_head:
        mapping["head_vs_trunk_deg"] = "head_vs_trunk_deg"
    return mapping


def canonical_columns(include_head: bool = True) -> list[str]:
    """Return the ordered canonical feature column list."""
    cols = list(CANONICAL_ACTIVE_CHAIN) + ["trunk_vs_horizontal_deg", "spine_flexion_deg"]
    if include_head:
        cols.append("head_vs_trunk_deg")
    return cols


def mirror_transform(
    canonical_name: str,
    values: np.ndarray,
    *,
    facing: str,
) -> np.ndarray:
    """Apply the mirror policy for ``canonical_name`` when ``facing=='left'``.

    ``values`` is not modified in place; a new array is returned.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    if facing != "left":
        return arr
    policy = _MIRROR_POLICIES.get(canonical_name)
    if policy is None or policy.kind == "identity":
        return arr
    if policy.kind == "supplement_180":
        return 180.0 - arr
    return arr


def apply_mirror_normalization(
    drive_df: pd.DataFrame,
    side_map: dict[str, str],
    *,
    facing: str,
) -> dict[str, np.ndarray]:
    """Return a dict ``{canonical_name: np.ndarray}`` of mirror-normalized
    angle arrays read from ``drive_df``.

    Missing source columns yield NaN arrays the same length as ``drive_df``.
    """
    out: dict[str, np.ndarray] = {}
    n = len(drive_df)
    for canonical, src in side_map.items():
        if src not in drive_df.columns:
            out[canonical] = np.full(n, np.nan, dtype=np.float64)
            continue
        raw = pd.to_numeric(drive_df[src], errors="coerce").to_numpy(dtype=np.float64)
        out[canonical] = mirror_transform(canonical, raw, facing=facing)
    return out


def iter_canonical_columns(
    include_head: bool,
    feature_suffix: str = "",
) -> Iterable[str]:
    """Yield canonical angle columns (optionally with a shared suffix).

    Example::

        list(iter_canonical_columns(True, "_arr"))
        # -> ['knee_active_deg_arr', ..., 'head_vs_trunk_deg_arr']
    """
    for col in canonical_columns(include_head=include_head):
        yield f"{col}{feature_suffix}"
