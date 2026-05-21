"""Sidecar for visual match-editor edits.

Lives at ``<run>/inference/match_overrides.json`` and drives both the matcher
DP (pinned pairs / excluded strokes / anchor overrides) and the editor (which
displays the current edit state).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


__all__ = [
    "Pin",
    "MatchOverrides",
    "overrides_path",
    "load_overrides",
    "save_overrides",
    "validate_overrides",
    "resolve_pin_to_row_idx",
    "OVERRIDES_FILENAME",
]


OVERRIDES_FILENAME = "match_overrides.json"


@dataclass(frozen=True)
class Pin:
    """A user-pinned match between a video stroke and a specific RP3 row.

    ``video_stroke_idx`` is the absolute ``stroke_idx`` value from
    ``drive_events.csv`` (i.e. the same index the user sees in the editor).
    ``rp3_stroke_number`` is the ``stroke_number`` column of the RP3 clean
    CSV (NOT the row index — stroke numbers are stable across edits).
    """

    video_stroke_idx: int
    rp3_stroke_number: int


@dataclass
class MatchOverrides:
    """Editable overrides applied on top of the matcher's baseline behaviour."""

    anchor_video_stroke_idx: int | None = None
    anchor_rp3_stroke_number: int | None = None
    active_side: str | None = None
    rower_facing: str | None = None
    pinned: list[Pin] = field(default_factory=list)
    excluded_video_stroke_idx: list[int] = field(default_factory=list)
    notes: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.anchor_video_stroke_idx is None
            and self.anchor_rp3_stroke_number is None
            and self.active_side is None
            and self.rower_facing is None
            and not self.pinned
            and not self.excluded_video_stroke_idx
            and not self.notes
        )

    def pinned_map(self) -> dict[int, int]:
        """Return ``{video_stroke_idx: rp3_stroke_number}`` for fast lookup."""
        return {int(p.video_stroke_idx): int(p.rp3_stroke_number) for p in self.pinned}

    def excluded_set(self) -> set[int]:
        return {int(idx) for idx in self.excluded_video_stroke_idx}

    def to_json_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pinned"] = [asdict(p) for p in self.pinned]
        payload["excluded_video_stroke_idx"] = sorted(int(x) for x in self.excluded_video_stroke_idx)
        return {k: v for k, v in payload.items() if v is not None and v != []}


def overrides_path(run_dir: Path) -> Path:
    """Canonical location for a run's ``match_overrides.json``."""
    return Path(run_dir) / "inference" / OVERRIDES_FILENAME


def load_overrides(run_dir: Path) -> MatchOverrides:
    """Load the overrides sidecar; returns an empty :class:`MatchOverrides` if missing."""
    path = overrides_path(run_dir)
    if not path.exists():
        return MatchOverrides()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed match overrides JSON at {path}: {exc}") from exc
    return _from_payload(data)


def save_overrides(run_dir: Path, overrides: MatchOverrides) -> Path:
    """Persist the overrides sidecar atomically.

    A blank/empty ``MatchOverrides`` deletes the sidecar; this lets the editor
    "reset to defaults" cleanly without leaving a stale file.
    """
    path = overrides_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if overrides.is_empty:
        if path.exists():
            path.unlink()
        return path

    payload = overrides.to_json_payload()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)
    return path


def _from_payload(data: dict[str, Any]) -> MatchOverrides:
    pinned_raw = data.get("pinned") or []
    pinned: list[Pin] = []
    for entry in pinned_raw:
        try:
            pinned.append(
                Pin(
                    video_stroke_idx=int(entry["video_stroke_idx"]),
                    rp3_stroke_number=int(entry["rp3_stroke_number"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid pinned entry {entry!r}: requires video_stroke_idx + rp3_stroke_number"
            ) from exc
    excluded_raw = data.get("excluded_video_stroke_idx") or []
    excluded = sorted({int(x) for x in excluded_raw})

    def _opt_int(key: str) -> int | None:
        v = data.get(key)
        return int(v) if v is not None else None

    def _opt_str(key: str) -> str | None:
        v = data.get(key)
        return str(v) if v is not None else None

    return MatchOverrides(
        anchor_video_stroke_idx=_opt_int("anchor_video_stroke_idx"),
        anchor_rp3_stroke_number=_opt_int("anchor_rp3_stroke_number"),
        active_side=_opt_str("active_side"),
        rower_facing=_opt_str("rower_facing"),
        pinned=pinned,
        excluded_video_stroke_idx=excluded,
        notes=_opt_str("notes"),
    )


def validate_overrides(
    overrides: MatchOverrides,
    *,
    video_stroke_indices: Iterable[int] | None = None,
    rp3_stroke_numbers: Iterable[int] | None = None,
) -> None:
    """Raise ``ValueError`` if any reference is incoherent with the run state."""
    pinned_video = {p.video_stroke_idx for p in overrides.pinned}
    excluded = overrides.excluded_set()

    overlap = pinned_video & excluded
    if overlap:
        raise ValueError(
            f"Pinned and excluded sets overlap on video_stroke_idx={sorted(overlap)}"
        )

    if video_stroke_indices is not None:
        valid_v = set(video_stroke_indices)
        bad_v = (pinned_video | excluded) - valid_v
        if bad_v:
            raise ValueError(
                f"Override references unknown video_stroke_idx={sorted(bad_v)}; "
                f"valid={sorted(valid_v)[:8]}…"
            )
        if (
            overrides.anchor_video_stroke_idx is not None
            and overrides.anchor_video_stroke_idx not in valid_v
        ):
            raise ValueError(
                "anchor_video_stroke_idx="
                f"{overrides.anchor_video_stroke_idx} not in detected video strokes."
            )

    if rp3_stroke_numbers is not None:
        valid_r = set(rp3_stroke_numbers)
        bad_r = {p.rp3_stroke_number for p in overrides.pinned} - valid_r
        if bad_r:
            raise ValueError(
                f"Pinned rp3_stroke_number={sorted(bad_r)} not present in RP3 CSV."
            )
        if (
            overrides.anchor_rp3_stroke_number is not None
            and overrides.anchor_rp3_stroke_number not in valid_r
        ):
            raise ValueError(
                "anchor_rp3_stroke_number="
                f"{overrides.anchor_rp3_stroke_number} not in RP3 CSV."
            )


def resolve_pin_to_row_idx(rp3_df: pd.DataFrame, rp3_stroke_number: int) -> int:
    """Convert an RP3 ``stroke_number`` to the corresponding row index in *rp3_df*.

    Raises ``KeyError`` if the stroke number isn't present.
    """
    if "stroke_number" not in rp3_df.columns:
        raise KeyError("RP3 CSV missing stroke_number column.")
    matches = rp3_df.index[rp3_df["stroke_number"].astype("Int64") == int(rp3_stroke_number)]
    if len(matches) == 0:
        raise KeyError(f"rp3_stroke_number={rp3_stroke_number} not found in RP3 CSV.")
    return int(matches[0])
