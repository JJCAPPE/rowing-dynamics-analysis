"""Session pairing utilities.

Pair a video run directory to the correct RP3 workout export using
``session_registry.csv`` so that downstream CLIs (``match_rp3_cli``,
``predict_force_cli``) do not have to prompt the user.

Registry schema
---------------

Minimum columns (existing):

* ``video_run_id`` -- arbitrary unique id for the session.
* ``video_run_dir`` -- absolute or repo-relative path to the video file or
  run directory; only the stem is used for matching.
* ``rp3_clean_csv`` -- filename (inside the run's ``rp3/`` subdirectory)
  of the cleaned RP3 export.
* ``athlete_id`` -- athlete identifier, free text.
* ``session_id`` -- session identifier, free text.
* ``active_side`` -- ``Left`` or ``Right`` (case-insensitive).
* ``recording_distance`` -- distance to rower (optional, free-form).

Optional pairing columns (added for the auto-pair workflow):

* ``rower_facing`` -- ``left`` or ``right``. Overrides the automatic
  facing inference used by mirror normalization.
* ``anchor_rp3_stroke_number`` -- RP3 ``stroke_number`` value that
  corresponds to ``anchor_video_stroke_idx``.
* ``anchor_video_stroke_idx`` -- zero-based video stroke index used as
  the DP-match anchor (default: 1, matching ``match_rp3_cli``).
* ``date`` -- ISO date string (``YYYY-MM-DD``). Reserved for future
  context-based pairing.
* ``piece_id`` -- optional piece/workout identifier.

All optional fields default to ``None`` / empty.

Usage
-----

>>> from inference.pair_session import auto_pair_run
>>> ctx = auto_pair_run(run_dir=Path("runs/giacomo-10m"))
>>> ctx.rp3_clean_csv
PosixPath('runs/giacomo-10m/rp3/giacomo-10m-202602122020-rp3-row-clean.csv')
>>> ctx.anchor_rp3_stroke_number
5
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "session_registry.csv"


@dataclass(frozen=True)
class PairingContext:
    """Resolved pairing information for a single video run."""

    video_run_id: str
    run_dir: Path
    rp3_clean_csv: Path
    athlete_id: str
    session_id: str
    active_side: str
    rower_facing: str | None
    anchor_rp3_stroke_number: int | None
    anchor_video_stroke_idx: int
    date: str | None
    piece_id: str | None
    recording_distance: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        d["rp3_clean_csv"] = str(self.rp3_clean_csv)
        return d


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _coerce_optional_int(value: Any) -> int | None:
    s = _coerce_optional_str(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _coerce_active_side(value: Any) -> str:
    s = _coerce_optional_str(value) or "right"
    s = s.lower()
    if s not in {"left", "right"}:
        raise ValueError(
            f"active_side must be 'left' or 'right' (case-insensitive); got {value!r}"
        )
    return s


def _coerce_rower_facing(value: Any) -> str | None:
    s = _coerce_optional_str(value)
    if s is None:
        return None
    s = s.lower()
    if s not in {"left", "right"}:
        raise ValueError(
            f"rower_facing must be 'left' or 'right' (case-insensitive); got {value!r}"
        )
    return s


def load_registry(registry_path: Path | None = None) -> pd.DataFrame:
    """Load the session registry CSV with normalized optional columns."""
    path = Path(registry_path).expanduser().resolve() if registry_path else DEFAULT_REGISTRY
    if not path.exists():
        raise FileNotFoundError(f"Session registry not found: {path}")
    df = pd.read_csv(path)
    required = {"video_run_dir", "rp3_clean_csv", "athlete_id", "session_id", "active_side"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Session registry missing required columns: {missing} ({path})")

    for col in (
        "rower_facing",
        "anchor_rp3_stroke_number",
        "anchor_video_stroke_idx",
        "date",
        "piece_id",
        "recording_distance",
        "video_run_id",
    ):
        if col not in df.columns:
            df[col] = None
    return df


def _match_registry_row(df: pd.DataFrame, run_name: str) -> pd.Series | None:
    """Match registry row by longest video_run_dir stem prefix of run_name."""
    best_stem = ""
    best_row: pd.Series | None = None
    for _, row in df.iterrows():
        video_run_dir = _coerce_optional_str(row.get("video_run_dir"))
        if not video_run_dir:
            continue
        stem = Path(video_run_dir).stem
        if run_name == stem or run_name.startswith(stem + "-") or run_name.startswith(stem + "_") or run_name == stem:
            if len(stem) > len(best_stem):
                best_stem = stem
                best_row = row
    return best_row


def auto_pair_run(
    *,
    run_dir: Path,
    registry_path: Path | None = None,
) -> PairingContext:
    """Resolve a :class:`PairingContext` for *run_dir* from the session registry.

    Parameters
    ----------
    run_dir
        Path to the run directory (e.g. ``runs/giacomo-10m``).
    registry_path
        Optional explicit registry CSV path. Defaults to
        ``<repo_root>/session_registry.csv``.

    Raises
    ------
    FileNotFoundError
        If the registry or the RP3 clean CSV under ``run_dir/rp3/`` cannot
        be located.
    LookupError
        If no registry row matches ``run_dir``.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    df = load_registry(registry_path)
    row = _match_registry_row(df, run_dir.name)
    if row is None:
        raise LookupError(
            f"No session_registry row matches run '{run_dir.name}'. "
            "Add an entry with a matching video_run_dir stem."
        )

    rp3_clean_name = _coerce_optional_str(row.get("rp3_clean_csv"))
    if not rp3_clean_name:
        raise LookupError(
            f"Registry row for '{run_dir.name}' has an empty rp3_clean_csv field."
        )
    rp3_dir = run_dir / "rp3"
    candidate = rp3_dir / rp3_clean_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"RP3 clean CSV not found: {candidate}. "
            "Either run inference_cli.py first or correct the rp3_clean_csv "
            "field in the session registry."
        )

    return PairingContext(
        video_run_id=_coerce_optional_str(row.get("video_run_id")) or run_dir.name,
        run_dir=run_dir,
        rp3_clean_csv=candidate.resolve(),
        athlete_id=_coerce_optional_str(row.get("athlete_id")) or "unknown",
        session_id=_coerce_optional_str(row.get("session_id")) or run_dir.name,
        active_side=_coerce_active_side(row.get("active_side")),
        rower_facing=_coerce_rower_facing(row.get("rower_facing")),
        anchor_rp3_stroke_number=_coerce_optional_int(row.get("anchor_rp3_stroke_number")),
        anchor_video_stroke_idx=_coerce_optional_int(row.get("anchor_video_stroke_idx")) or 1,
        date=_coerce_optional_str(row.get("date")),
        piece_id=_coerce_optional_str(row.get("piece_id")),
        recording_distance=_coerce_optional_str(row.get("recording_distance")),
    )


@dataclass(frozen=True)
class CoarseSyncResult:
    """Outcome of cross-correlation based coarse synchronization."""

    anchor_video_idx: int
    anchor_rp3_idx: int
    anchor_rp3_stroke_number: int | None
    offset_strokes: int
    overlap_length: int
    best_cost: float
    normalized_cost: float
    mean_rate_diff_spm: float
    cost_curve: np.ndarray
    candidate_offsets: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_video_idx": int(self.anchor_video_idx),
            "anchor_rp3_idx": int(self.anchor_rp3_idx),
            "anchor_rp3_stroke_number": (
                int(self.anchor_rp3_stroke_number)
                if self.anchor_rp3_stroke_number is not None
                else None
            ),
            "offset_strokes": int(self.offset_strokes),
            "overlap_length": int(self.overlap_length),
            "best_cost": float(self.best_cost),
            "normalized_cost": float(self.normalized_cost),
            "mean_rate_diff_spm": float(self.mean_rate_diff_spm),
        }


def _interval_series_from_video(video_df: pd.DataFrame) -> np.ndarray:
    if "cycle_duration_s" in video_df.columns:
        arr = pd.to_numeric(video_df["cycle_duration_s"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        catches = pd.to_numeric(video_df["catch_time_s"], errors="coerce").to_numpy(dtype=np.float64)
        arr = np.diff(catches, prepend=catches[0])
    return arr


def _interval_series_from_rp3(rp3_df: pd.DataFrame) -> np.ndarray:
    if "rp3_cycle_s" in rp3_df.columns:
        arr = pd.to_numeric(rp3_df["rp3_cycle_s"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        drive = pd.to_numeric(rp3_df["drive_time"], errors="coerce").to_numpy(dtype=np.float64)
        recover = pd.to_numeric(rp3_df["recover_time"], errors="coerce").to_numpy(dtype=np.float64)
        arr = drive + recover
    return arr


def coarse_sync_anchor(
    video_df: pd.DataFrame,
    rp3_df: pd.DataFrame,
    *,
    anchor_video_idx: int = 1,
    max_offset: int | None = None,
    min_overlap: int = 8,
) -> CoarseSyncResult:
    """Estimate the RP3 anchor row from stroke-interval cross-correlation.

    Uses cycle-duration (``1/stroke_rate``) sequences. The ``offset``
    parameter represents how many RP3 strokes come before the RP3 row
    paired with video stroke ``0``.  For each feasible ``offset``, the
    cost is the mean absolute difference between video intervals
    starting at ``anchor_video_idx`` and the RP3 window
    ``rp3[offset + anchor_video_idx : offset + anchor_video_idx + len(v_tail)]``.
    The returned ``anchor_rp3_idx`` is the RP3 row that pairs with
    video stroke ``anchor_video_idx`` (i.e., ``offset + anchor_video_idx``).

    Returns a :class:`CoarseSyncResult` with diagnostics.  The caller is
    responsible for feeding ``anchor_rp3_idx`` / ``anchor_rp3_stroke_number``
    into the fine DP matcher.

    Raises
    ------
    ValueError
        If the sequences have too few strokes to estimate an offset.
    """
    v_int = _interval_series_from_video(video_df)
    r_int = _interval_series_from_rp3(rp3_df)

    if anchor_video_idx < 0 or anchor_video_idx >= len(v_int):
        raise ValueError(
            f"anchor_video_idx={anchor_video_idx} out of range (video strokes={len(v_int)})"
        )

    v_tail = v_int[anchor_video_idx:]
    n_v = int(np.sum(np.isfinite(v_tail)))
    if n_v < min_overlap:
        raise ValueError(
            f"Need at least {min_overlap} finite video intervals after the anchor "
            f"(have {n_v}). Record more strokes or adjust --anchor-video-stroke-idx."
        )

    # offset k means rp3[k + anchor_video_idx : k + anchor_video_idx + len(v_tail)]
    # aligns with video[anchor_video_idx:]. The maximum admissible offset is
    # bounded by the RP3 tail length.
    max_k = len(r_int) - anchor_video_idx - len(v_tail)
    if max_k < 0:
        raise ValueError(
            "RP3 export has fewer strokes than the video sequence starting at the anchor."
        )
    if max_offset is not None:
        max_k = min(max_k, int(max_offset))

    offsets = np.arange(0, max_k + 1, dtype=np.int64)
    costs = np.full(offsets.shape, np.inf, dtype=np.float64)
    overlaps = np.zeros(offsets.shape, dtype=np.int64)

    for i, k in enumerate(offsets):
        start = int(k) + int(anchor_video_idx)
        r_slice = r_int[start : start + len(v_tail)]
        diff = np.abs(v_tail - r_slice)
        mask = np.isfinite(diff)
        if mask.sum() < min_overlap:
            continue
        costs[i] = float(np.mean(diff[mask]))
        overlaps[i] = int(mask.sum())

    if not np.isfinite(costs).any():
        raise ValueError("Coarse sync failed: no offset yielded enough finite intervals.")

    best_i = int(np.argmin(costs))
    offset_strokes = int(offsets[best_i])
    anchor_rp3_idx = int(anchor_video_idx) + offset_strokes

    r_slice = r_int[anchor_rp3_idx : anchor_rp3_idx + len(v_tail)]
    finite = np.isfinite(v_tail) & np.isfinite(r_slice)
    if finite.any():
        rate_v = 60.0 / np.where(v_tail[finite] > 0, v_tail[finite], np.nan)
        rate_r = 60.0 / np.where(r_slice[finite] > 0, r_slice[finite], np.nan)
        mean_rate_diff_spm = float(np.nanmean(np.abs(rate_v - rate_r)))
    else:
        mean_rate_diff_spm = float("nan")

    anchor_rp3_stroke_number: int | None = None
    if "stroke_number" in rp3_df.columns and 0 <= anchor_rp3_idx < len(rp3_df):
        value = rp3_df.iloc[anchor_rp3_idx]["stroke_number"]
        try:
            anchor_rp3_stroke_number = int(round(float(value)))
        except (TypeError, ValueError):
            anchor_rp3_stroke_number = None

    finite_costs = costs[np.isfinite(costs)]
    normalized_cost = (
        float(costs[best_i] / float(np.median(finite_costs)))
        if finite_costs.size > 0 and float(np.median(finite_costs)) > 0
        else float("nan")
    )

    return CoarseSyncResult(
        anchor_video_idx=int(anchor_video_idx),
        anchor_rp3_idx=int(anchor_rp3_idx),
        anchor_rp3_stroke_number=anchor_rp3_stroke_number,
        offset_strokes=offset_strokes,
        overlap_length=int(overlaps[best_i]),
        best_cost=float(costs[best_i]),
        normalized_cost=normalized_cost,
        mean_rate_diff_spm=mean_rate_diff_spm,
        cost_curve=costs,
        candidate_offsets=offsets,
    )


__all__ = [
    "PairingContext",
    "CoarseSyncResult",
    "auto_pair_run",
    "coarse_sync_anchor",
    "load_registry",
]
