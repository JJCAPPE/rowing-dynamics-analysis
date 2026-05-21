"""Interactive matplotlib editor for RP3-to-video stroke matches.

Phase 4 of the unified-CLI plan. Builds on top of the read-only diagnostic
viewer (:mod:`rowing.matching.diagnostics`) but exposes editable state:

- **Pair column panel** — left column = video strokes, right column = RP3 rows,
  with ``Line2D`` connectors. Click a connector to select; click an empty RP3
  row to remap; key ``e`` excludes, ``u`` un-pins, ``a`` sets anchor pair.
- **Drift panel** — live ``cum_catch_err_s`` line; recomputed in-process after
  every edit by re-running :func:`rowing.matching.match._build_match_manifest`.
- **Save** (``ctrl+s``) writes ``<run>/inference/match_overrides.json`` and
  re-runs the inference pipeline so segments + dataset stay consistent.
- **Reset** (``r``) wipes all overrides and reverts to the baseline match.

The editor is intentionally non-modal: when a pair is selected, the next click
on an RP3-row marker in the right column re-pins the selected video stroke.
Pinning the anchor pair is rejected at edit time (matches the matcher's
constraint).

Design notes
~~~~~~~~~~~~

The matcher is deterministic and fast (DP over the post-anchor sequence), so
re-running it on every edit is cheap (< 50ms even on long sessions). The
heavy work (segment export, dataset rebuild) is deferred until ``ctrl+s``.

The editor only mutates :class:`rowing.matching.overrides.MatchOverrides`;
it never writes the matcher manifest directly. Persistence flows through the
sidecar JSON so the same edits are honoured by every CLI invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.selectors import (
    discover_run_dirs,
    pick_run_with_curses,
    pick_run_with_prompt,
)
from rowing.matching.match import (
    MatchConfig,
    _build_match_manifest as _build_rp3_match_manifest,
    _load_rp3 as _load_rp3_clean_csv,
)
from rowing.matching.overrides import (
    MatchOverrides,
    Pin,
    load_overrides,
    overrides_path,
    resolve_pin_to_row_idx,
    save_overrides,
    validate_overrides,
)


__all__ = ["MatchEditor", "open_editor", "main"]


# ---------------------------------------------------------------------------
# Status enum / state container
# ---------------------------------------------------------------------------


PAIR_AUTO = "auto"
PAIR_PINNED = "pinned"
PAIR_EXCLUDED = "excluded"
PAIR_ANCHOR = "anchor"


_STATUS_COLORS = {
    PAIR_AUTO: "#1f77b4",      # blue
    PAIR_PINNED: "#d62728",    # red
    PAIR_EXCLUDED: "#7f7f7f",  # gray
    PAIR_ANCHOR: "#2ca02c",    # green
}


@dataclass
class EditorRunData:
    """Static (read-only) inputs needed to drive the editor.

    These are loaded once when the editor opens. Mutable editing state lives
    on :class:`MatchEditor` directly so re-running the matcher and re-drawing
    panels can stay simple.
    """

    run_dir: Path
    events_df: pd.DataFrame
    rp3_df: pd.DataFrame
    rp3_clean_csv: Path
    rp3_dirty_csv: Path | None
    summary: dict[str, Any]
    cfg: MatchConfig
    baseline_anchor_video_idx: int
    baseline_anchor_rp3_idx: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_editor_inputs(run_dir: Path) -> EditorRunData:
    """Gather everything the editor needs from a finished inference run."""
    inf = run_dir / "inference"
    summary_path = inf / "rp3_match_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No rp3_match_summary.json under {inf}. Run inference with --match-rp3 first."
        )
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    events_path = inf / "drive_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}")
    events_df = pd.read_csv(events_path)

    rp3_clean_str = summary.get("rp3_clean_csv")
    if not rp3_clean_str:
        raise ValueError("rp3_match_summary.json is missing 'rp3_clean_csv'.")
    rp3_clean_csv = Path(rp3_clean_str)
    if not rp3_clean_csv.exists():
        raise FileNotFoundError(f"Cleaned RP3 CSV not found: {rp3_clean_csv}")
    rp3_df = _load_rp3_clean_csv(rp3_clean_csv)

    rp3_dirty_str = summary.get("rp3_dirty_csv")
    rp3_dirty_csv = Path(rp3_dirty_str) if rp3_dirty_str else None

    cfg = MatchConfig(
        max_jump_rows=10,
        max_interval_error_s=2.0,
        max_cumulative_error_base_s=2.0,
        max_cumulative_error_per_s=0.15,
        max_abs_cum_error_s=4.0,
        w_drive=0.4,
        w_recover=0.4,
        w_interval=1.0,
        w_cumulative=1.0,
        w_skip=0.08,
    )

    return EditorRunData(
        run_dir=run_dir,
        events_df=events_df,
        rp3_df=rp3_df,
        rp3_clean_csv=rp3_clean_csv,
        rp3_dirty_csv=rp3_dirty_csv,
        summary=summary,
        cfg=cfg,
        baseline_anchor_video_idx=int(summary.get("anchor_video_stroke_idx", 1)),
        baseline_anchor_rp3_idx=int(summary.get("anchor_rp3_row_idx", 0)),
    )


# ---------------------------------------------------------------------------
# Run selection (matches diagnostics module)
# ---------------------------------------------------------------------------


def _discover_diagnosed_runs(runs_root: Path) -> list[Path]:
    return [
        r for r in discover_run_dirs(runs_root)
        if (r / "inference" / "rp3_match_manifest.csv").exists()
        and (r / "inference" / "rp3_match_summary.json").exists()
    ]


def _select_run(runs_root: Path) -> Path:
    options = _discover_diagnosed_runs(runs_root)
    if not options:
        raise FileNotFoundError(
            f"No runs with inference/rp3_match_summary.json found under {runs_root}"
        )
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return pick_run_with_curses(options)
        except Exception:
            pass
    return pick_run_with_prompt(options)


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


@dataclass
class _PairArtists:
    """Cache of matplotlib artists per video stroke for fast updates."""

    video_marker: Line2D
    rp3_marker: Line2D
    connector: Line2D
    label_left: Any
    label_right: Any


class MatchEditor:
    """Interactive matplotlib editor over the RP3 match.

    Use :func:`open_editor` from external code; ``MatchEditor`` is exposed for
    advanced callers (e.g. tests that want to drive the state machine
    without a display).
    """

    # -- lifecycle ---------------------------------------------------------

    def __init__(self, data: EditorRunData) -> None:
        self.data = data

        self.overrides: MatchOverrides = load_overrides(data.run_dir)
        self._validate_or_warn(self.overrides)

        self.anchor_video_idx = self._resolve_anchor_video_idx()
        self.anchor_rp3_idx = self._resolve_anchor_rp3_idx()

        # Run baseline DP so the editor has something to draw immediately.
        self.manifest_df: pd.DataFrame = pd.DataFrame()
        self.matched_rp3_indices: list[int] = []
        self.last_match_error: str | None = None
        self.selected_video_stroke_idx: int | None = None

        self.fig: plt.Figure | None = None
        self.ax_pairs: plt.Axes | None = None
        self.ax_drift: plt.Axes | None = None
        self.ax_status: plt.Axes | None = None
        self._artists: dict[int, _PairArtists] = {}
        self._rp3_axis_artists: dict[int, Line2D] = {}
        self._drift_line: Line2D | None = None
        self._status_text: Any | None = None
        self._dirty: bool = False

        self._recompute_match()

    # -- override helpers --------------------------------------------------

    def _validate_or_warn(self, overrides: MatchOverrides) -> None:
        try:
            validate_overrides(
                overrides,
                video_stroke_indices=self.data.events_df["stroke_idx"].astype(int).tolist(),
                rp3_stroke_numbers=self.data.rp3_df["stroke_number"].astype(int).tolist(),
            )
        except ValueError as exc:
            print(f"Warning: existing overrides are inconsistent: {exc}")

    def _resolve_anchor_video_idx(self) -> int:
        if self.overrides.anchor_video_stroke_idx is not None:
            return int(self.overrides.anchor_video_stroke_idx)
        return int(self.data.baseline_anchor_video_idx)

    def _resolve_anchor_rp3_idx(self) -> int:
        if self.overrides.anchor_rp3_stroke_number is not None:
            try:
                return resolve_pin_to_row_idx(
                    self.data.rp3_df, self.overrides.anchor_rp3_stroke_number,
                )
            except KeyError:
                print(
                    f"Warning: override anchor_rp3_stroke_number="
                    f"{self.overrides.anchor_rp3_stroke_number} not present in RP3 CSV; "
                    "falling back to baseline."
                )
        return int(self.data.baseline_anchor_rp3_idx)

    # -- match recompute ---------------------------------------------------

    def _recompute_match(self) -> None:
        """Run the matcher with current overrides; update state in-place."""
        self.last_match_error = None

        excluded_relative_indices: set[int] = set()
        for idx in self.overrides.excluded_video_stroke_idx:
            rel = int(idx) - self.anchor_video_idx
            if rel > 0:
                excluded_relative_indices.add(rel)

        pinned_rp3_row_by_relative_idx: dict[int, int] = {}
        for pin in self.overrides.pinned:
            rel = int(pin.video_stroke_idx) - self.anchor_video_idx
            if rel < 0:
                # Skip pre-anchor pins — they're surfaced as warnings on save.
                continue
            try:
                rp3_row_idx = resolve_pin_to_row_idx(
                    self.data.rp3_df, pin.rp3_stroke_number,
                )
            except KeyError as exc:
                self.last_match_error = str(exc)
                continue
            pinned_rp3_row_by_relative_idx[rel] = rp3_row_idx

        try:
            result = _build_rp3_match_manifest(
                video_df=self.data.events_df,
                rp3_df=self.data.rp3_df,
                anchor_video_idx=int(self.anchor_video_idx),
                anchor_rp3_idx=int(self.anchor_rp3_idx),
                cfg=self.data.cfg,
                pinned_rp3_row_by_relative_idx=pinned_rp3_row_by_relative_idx or None,
                excluded_relative_indices=excluded_relative_indices or None,
            )
            self.manifest_df = result.manifest
            self.matched_rp3_indices = list(result.matched_rp3_indices)
        except (RuntimeError, ValueError) as exc:
            self.last_match_error = str(exc)
            # Keep last successful manifest so the user can recover.

    # -- pair status -------------------------------------------------------

    def _pair_status(self, video_stroke_idx: int) -> str:
        if int(video_stroke_idx) == int(self.anchor_video_idx):
            return PAIR_ANCHOR
        if int(video_stroke_idx) in self.overrides.excluded_set():
            return PAIR_EXCLUDED
        if int(video_stroke_idx) in self.overrides.pinned_map():
            return PAIR_PINNED
        return PAIR_AUTO

    # -- drawing -----------------------------------------------------------

    def _build_layout(self) -> None:
        self.fig = plt.figure("Match Editor", figsize=(15, 11))
        gs = self.fig.add_gridspec(
            nrows=2, ncols=2,
            left=0.06, right=0.96, top=0.95, bottom=0.06,
            height_ratios=[3, 1.2],
            width_ratios=[2.2, 1.0],
            hspace=0.18, wspace=0.18,
        )
        self.ax_pairs = self.fig.add_subplot(gs[0, 0])
        self.ax_status = self.fig.add_subplot(gs[0, 1])
        self.ax_drift = self.fig.add_subplot(gs[1, :])

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _draw_pair_column(self) -> None:
        """Render the editing surface: video strokes ↔ RP3 rows with connectors."""
        ax = self.ax_pairs
        assert ax is not None
        ax.clear()
        self._artists.clear()
        self._rp3_axis_artists.clear()

        events = self.data.events_df
        rp3_df = self.data.rp3_df

        post_anchor = events[events["stroke_idx"].astype(int) >= self.anchor_video_idx].copy()
        if post_anchor.empty:
            ax.text(0.5, 0.5, "No video strokes after anchor.",
                    transform=ax.transAxes, ha="center", va="center", color="red")
            return

        post_anchor["__rel_i"] = (
            post_anchor["stroke_idx"].astype(int) - int(self.anchor_video_idx)
        )

        # Y range: relative_idx for everything (anchor at 0 on top).
        manifest_lookup: dict[int, int] = {}
        if not self.manifest_df.empty:
            manifest_lookup = {
                int(r["video_stroke_idx"]): int(r["rp3_row_idx"])
                for _, r in self.manifest_df.iterrows()
            }

        max_rel = int(post_anchor["__rel_i"].max())
        max_rp3_y = max(
            max_rel,
            (max(manifest_lookup.values()) - self.anchor_rp3_idx) if manifest_lookup else max_rel,
        )

        # Visible RP3 window: anchor → max matched + buffer
        rp3_y_max = min(len(rp3_df) - self.anchor_rp3_idx - 1, max_rp3_y + 5)

        x_left = 0.0
        x_right = 1.0

        # Draw RP3 column (right) — every row in window is clickable
        for rp3_y in range(0, rp3_y_max + 1):
            rp3_row_idx = self.anchor_rp3_idx + rp3_y
            if rp3_row_idx >= len(rp3_df):
                break
            rp3_no = int(rp3_df.iloc[rp3_row_idx]["stroke_number"])
            is_matched = rp3_row_idx in set(manifest_lookup.values())
            color = "#888888" if not is_matched else "#000000"
            marker = ax.plot(
                x_right, -rp3_y,
                marker="o", linestyle="", markersize=8,
                markerfacecolor="white" if not is_matched else "#dddddd",
                markeredgecolor=color, markeredgewidth=1.2,
                picker=8, zorder=3,
            )[0]
            marker.set_gid(f"rp3:{rp3_row_idx}")
            self._rp3_axis_artists[rp3_row_idx] = marker
            ax.text(
                x_right + 0.06, -rp3_y, f"r{rp3_no}",
                fontsize=7, ha="left", va="center", color=color,
            )

        # Draw video column (left) + connectors
        for _, row in post_anchor.iterrows():
            v_idx = int(row["stroke_idx"])
            rel_i = int(row["__rel_i"])
            status = self._pair_status(v_idx)
            color = _STATUS_COLORS[status]

            edgecolor = color
            facecolor = color if status in (PAIR_PINNED, PAIR_ANCHOR) else "white"
            video_marker = ax.plot(
                x_left, -rel_i,
                marker="o", linestyle="", markersize=9,
                markerfacecolor=facecolor, markeredgecolor=edgecolor,
                markeredgewidth=1.4,
                picker=8, zorder=3,
            )[0]
            video_marker.set_gid(f"video:{v_idx}")
            label_left = ax.text(
                x_left - 0.06, -rel_i, f"v{v_idx}",
                fontsize=7, ha="right", va="center", color=color, fontweight="bold",
            )

            rp3_row_idx = manifest_lookup.get(v_idx)
            if status == PAIR_EXCLUDED or rp3_row_idx is None:
                connector = ax.plot(
                    [x_left, x_left + 0.18], [-rel_i, -rel_i],
                    color="#aaaaaa", linewidth=1.0, linestyle=":",
                    alpha=0.6, zorder=2,
                )[0]
                rp3_marker = video_marker  # placeholder; not used
                label_right = label_left
            else:
                rp3_y = rp3_row_idx - self.anchor_rp3_idx
                linestyle = "-" if status != PAIR_AUTO else "--"
                linewidth = 1.8 if status == PAIR_PINNED else 1.2
                connector = ax.plot(
                    [x_left, x_right], [-rel_i, -rp3_y],
                    color=color, linewidth=linewidth, linestyle=linestyle,
                    alpha=0.85, zorder=2,
                    picker=4,
                )[0]
                connector.set_gid(f"connector:{v_idx}")
                rp3_marker = self._rp3_axis_artists.get(rp3_row_idx, video_marker)
                label_right = ax.text(
                    x_right + 0.06, -rp3_y, "",
                    fontsize=1, ha="left", va="center",
                )  # spacer; rp3 label was drawn above

            self._artists[v_idx] = _PairArtists(
                video_marker=video_marker,
                rp3_marker=rp3_marker,
                connector=connector,
                label_left=label_left,
                label_right=label_right,
            )

        # Highlight selected pair
        if self.selected_video_stroke_idx is not None:
            self._highlight_selected()

        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-rp3_y_max - 1, 0.5)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["video", "RP3"])
        ax.set_yticks([])
        ax.set_title(
            f"Pair editor — {self.data.run_dir.name}    "
            f"(matches: {len(self.manifest_df)})",
            fontsize=11, fontweight="bold",
        )
        for spine in ("right", "top", "left"):
            ax.spines[spine].set_visible(False)

    def _highlight_selected(self) -> None:
        v_idx = self.selected_video_stroke_idx
        if v_idx is None:
            return
        artists = self._artists.get(v_idx)
        if artists is None:
            return
        artists.video_marker.set_markersize(14)
        artists.connector.set_linewidth(2.5)
        artists.connector.set_alpha(1.0)

    def _draw_drift(self) -> None:
        ax = self.ax_drift
        assert ax is not None
        ax.clear()

        if self.manifest_df.empty:
            ax.text(0.5, 0.5, "No matches available.",
                    transform=ax.transAxes, ha="center", va="center", color="red")
            return

        m = self.manifest_df
        x = m["video_stroke_idx"].astype(int).to_numpy()
        cum_err = m["cum_catch_err_s"].astype(float).to_numpy()

        ax.plot(
            x, cum_err,
            marker="o", linestyle="-", color="#1f77b4", linewidth=1.0,
        )
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax.fill_between(x, 0.0, cum_err, alpha=0.15, color="#1f77b4")

        for _, row in m.iterrows():
            v_idx = int(row["video_stroke_idx"])
            status = self._pair_status(v_idx)
            if status in (PAIR_PINNED, PAIR_ANCHOR):
                ax.plot(
                    v_idx, float(row["cum_catch_err_s"]),
                    marker="D", color=_STATUS_COLORS[status], markersize=7,
                )

        ax.set_xlabel("video_stroke_idx")
        ax.set_ylabel("cum catch err (s)")
        mean_abs = float(np.mean(np.abs(cum_err)))
        ax.set_title(
            f"Cumulative drift  —  mean |cum err| = {mean_abs:.3f}s",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)

    def _draw_status(self) -> None:
        ax = self.ax_status
        assert ax is not None
        ax.clear()
        ax.axis("off")

        lines: list[str] = []
        lines.append(f"Run: {self.data.run_dir.name}")
        lines.append(f"Anchor video: v{self.anchor_video_idx}")
        try:
            anchor_no = int(
                self.data.rp3_df.iloc[self.anchor_rp3_idx]["stroke_number"]
            )
            lines.append(f"Anchor RP3:   r{anchor_no} (row {self.anchor_rp3_idx})")
        except Exception:
            lines.append(f"Anchor RP3:   row {self.anchor_rp3_idx}")
        lines.append("")
        lines.append(f"Pinned: {len(self.overrides.pinned)}")
        lines.append(f"Excluded: {len(self.overrides.excluded_video_stroke_idx)}")
        if self.overrides.is_empty:
            lines.append("[dim]no overrides[/dim]")
        lines.append("")

        if self.selected_video_stroke_idx is not None:
            v_idx = self.selected_video_stroke_idx
            status = self._pair_status(v_idx)
            rp3_row_idx = None
            if not self.manifest_df.empty:
                row = self.manifest_df[
                    self.manifest_df["video_stroke_idx"].astype(int) == v_idx
                ]
                if not row.empty:
                    rp3_row_idx = int(row.iloc[0]["rp3_row_idx"])
            lines.append(f"Selected: v{v_idx} ({status})")
            if rp3_row_idx is not None:
                rp3_no = int(self.data.rp3_df.iloc[rp3_row_idx]["stroke_number"])
                lines.append(f"  → r{rp3_no} (row {rp3_row_idx})")
            else:
                lines.append("  → (no current match)")
        else:
            lines.append("Selected: none")
        lines.append("")

        if self.last_match_error:
            lines.append(f"⚠ {self.last_match_error[:80]}")
            lines.append("")

        if self._dirty:
            lines.append("● UNSAVED CHANGES (ctrl+s to save & re-run)")
        lines.append("")
        lines.append("Keys:")
        lines.append("  click connector → select")
        lines.append("  click empty RP3 → remap selected")
        lines.append("  e  exclude selected")
        lines.append("  u  un-pin selected")
        lines.append("  a  set anchor pair (selected → its current RP3)")
        lines.append("  r  reset all overrides")
        lines.append("  ctrl+s  save + re-run")
        lines.append("  q  close")

        ax.text(
            0.0, 1.0, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, family="monospace",
        )

    def _redraw(self) -> None:
        if self.fig is None or self.ax_pairs is None:
            # Headless edit path (used by tests / scripted callers): the
            # editor still recomputes the match in-place, but skip drawing.
            return
        self._draw_pair_column()
        self._draw_drift()
        self._draw_status()
        self.fig.canvas.draw_idle()

    # -- event handlers ----------------------------------------------------

    def _on_pick(self, event: Any) -> None:
        gid = getattr(event.artist, "get_gid", lambda: None)()
        if not gid:
            return
        kind, _, payload = gid.partition(":")
        if kind == "connector" or kind == "video":
            try:
                v_idx = int(payload)
            except ValueError:
                return
            self.selected_video_stroke_idx = v_idx
            self._redraw()
            return
        if kind == "rp3":
            try:
                rp3_row_idx = int(payload)
            except ValueError:
                return
            self._remap_selected_to_row(rp3_row_idx)

    def _on_button_press(self, event: Any) -> None:
        # Picking handles all selections / remaps. Empty-canvas clicks clear.
        if event.inaxes is self.ax_pairs and event.xdata is not None:
            # If they clicked far from any artist, drop selection.
            # (matplotlib's pick_event takes precedence; this is a fallback.)
            if event.xdata < -0.35 or event.xdata > 1.35:
                self.selected_video_stroke_idx = None
                self._redraw()

    def _on_key(self, event: Any) -> None:
        key = event.key
        if key in ("q", "escape"):
            plt.close(self.fig)
            return
        if key == "ctrl+s":
            self._save_and_rerun()
            return
        if key == "r":
            self._reset_overrides()
            return
        if self.selected_video_stroke_idx is None:
            return
        if key == "e":
            self._exclude_selected()
        elif key == "u":
            self._unpin_selected()
        elif key == "a":
            self._set_anchor_to_selected()

    # -- edits -------------------------------------------------------------

    def _remap_selected_to_row(self, rp3_row_idx: int) -> None:
        if self.selected_video_stroke_idx is None:
            return
        v_idx = int(self.selected_video_stroke_idx)
        if v_idx == int(self.anchor_video_idx):
            print("Cannot remap the anchor pair via row click; press 'a' to move the anchor.")
            return
        if rp3_row_idx == int(self.anchor_rp3_idx):
            print("Cannot pin a non-anchor video stroke onto the anchor RP3 row.")
            return
        try:
            rp3_no = int(self.data.rp3_df.iloc[rp3_row_idx]["stroke_number"])
        except IndexError:
            return

        # Remove any pre-existing pin for this video stroke, then re-add.
        new_pins = [p for p in self.overrides.pinned if p.video_stroke_idx != v_idx]
        new_pins.append(Pin(video_stroke_idx=v_idx, rp3_stroke_number=rp3_no))
        # Drop from excluded if present.
        new_excluded = [
            i for i in self.overrides.excluded_video_stroke_idx if int(i) != v_idx
        ]
        self.overrides.pinned = new_pins
        self.overrides.excluded_video_stroke_idx = new_excluded
        self._dirty = True
        self._recompute_match()
        self._redraw()

    def _exclude_selected(self) -> None:
        if self.selected_video_stroke_idx is None:
            return
        v_idx = int(self.selected_video_stroke_idx)
        if v_idx == int(self.anchor_video_idx):
            print("Cannot exclude the anchor video stroke. Move the anchor first.")
            return
        self.overrides.pinned = [
            p for p in self.overrides.pinned if p.video_stroke_idx != v_idx
        ]
        if v_idx not in self.overrides.excluded_video_stroke_idx:
            self.overrides.excluded_video_stroke_idx = sorted(
                set(self.overrides.excluded_video_stroke_idx) | {v_idx}
            )
        self._dirty = True
        self._recompute_match()
        self._redraw()

    def _unpin_selected(self) -> None:
        if self.selected_video_stroke_idx is None:
            return
        v_idx = int(self.selected_video_stroke_idx)
        before_pin = len(self.overrides.pinned)
        self.overrides.pinned = [
            p for p in self.overrides.pinned if p.video_stroke_idx != v_idx
        ]
        before_excl = len(self.overrides.excluded_video_stroke_idx)
        self.overrides.excluded_video_stroke_idx = [
            i for i in self.overrides.excluded_video_stroke_idx if int(i) != v_idx
        ]
        if (
            len(self.overrides.pinned) != before_pin
            or len(self.overrides.excluded_video_stroke_idx) != before_excl
        ):
            self._dirty = True
            self._recompute_match()
            self._redraw()

    def _set_anchor_to_selected(self) -> None:
        if self.selected_video_stroke_idx is None:
            return
        v_idx = int(self.selected_video_stroke_idx)
        # Find current matched RP3 for this video stroke (or fall back to a pin).
        rp3_row_idx: int | None = None
        if not self.manifest_df.empty:
            row = self.manifest_df[
                self.manifest_df["video_stroke_idx"].astype(int) == v_idx
            ]
            if not row.empty:
                rp3_row_idx = int(row.iloc[0]["rp3_row_idx"])
        if rp3_row_idx is None:
            print("Selected stroke has no current match; can't promote to anchor.")
            return

        rp3_no = int(self.data.rp3_df.iloc[rp3_row_idx]["stroke_number"])
        # New anchor wipes pins/excludes prior to its own video idx (they no
        # longer fit the post-anchor coordinate system).
        self.overrides.anchor_video_stroke_idx = v_idx
        self.overrides.anchor_rp3_stroke_number = rp3_no
        self.overrides.pinned = [
            p for p in self.overrides.pinned if p.video_stroke_idx >= v_idx
        ]
        self.overrides.excluded_video_stroke_idx = sorted(
            i for i in self.overrides.excluded_video_stroke_idx if int(i) > v_idx
        )
        self.anchor_video_idx = v_idx
        self.anchor_rp3_idx = rp3_row_idx
        self.selected_video_stroke_idx = None
        self._dirty = True
        self._recompute_match()
        self._redraw()

    def _reset_overrides(self) -> None:
        self.overrides = MatchOverrides()
        self.anchor_video_idx = int(self.data.baseline_anchor_video_idx)
        self.anchor_rp3_idx = int(self.data.baseline_anchor_rp3_idx)
        self.selected_video_stroke_idx = None
        self._dirty = True
        self._recompute_match()
        self._redraw()

    # -- persistence -------------------------------------------------------

    def _save_and_rerun(self) -> None:
        try:
            validate_overrides(
                self.overrides,
                video_stroke_indices=self.data.events_df["stroke_idx"].astype(int).tolist(),
                rp3_stroke_numbers=self.data.rp3_df["stroke_number"].astype(int).tolist(),
            )
        except ValueError as exc:
            print(f"Cannot save: {exc}")
            return

        path = save_overrides(self.data.run_dir, self.overrides)
        print(f"Saved overrides → {path}")

        # Re-run inference with the same toggles as the parent run, but skip
        # detection re-calibration if the run already has it. Keep behaviour
        # conservative — just trigger match + segment export from the menu's
        # default options. We import lazily so the editor is usable in
        # contexts where the full pipeline isn't importable (tests).
        from rowing.cli.pipeline import PipelineOptions, run_inference

        opts = PipelineOptions(
            runs_root=self.data.run_dir.parent,
            run_dir=self.data.run_dir,
            match_rp3=True,
            no_overlay_video=True,
            no_build_dataset=False,
            interactive=False,
            active_side=self.data.summary.get("active_side"),
        )
        result = run_inference(opts)
        if result.exit_code == 0:
            print("Pipeline re-ran successfully.")
            self._dirty = False
        else:
            print(f"Pipeline re-run failed (exit {result.exit_code}): {result.error}")

        # After the pipeline writes a fresh manifest, reload it to keep the
        # editor consistent with disk state.
        self._recompute_match()
        self._redraw()

    # -- entry point -------------------------------------------------------

    def show(self) -> None:
        self._build_layout()
        self._redraw()
        plt.show()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def open_editor(run_dir: Path) -> None:
    """Open the editor for *run_dir* (must already have an inference manifest)."""
    data = _load_editor_inputs(run_dir)
    editor = MatchEditor(data)
    editor.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive editor for RP3-to-video stroke matches.",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help=f"Sports2D runs root (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory to edit (skip interactive selection).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.run_dir is None:
        run_dir = _select_run(args.runs_root)
    else:
        run_dir = args.run_dir.expanduser().resolve()
        if not run_dir.is_dir():
            print(f"Run directory not found: {run_dir}")
            return 1

    try:
        open_editor(run_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
