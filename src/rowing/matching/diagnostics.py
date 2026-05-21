#!/usr/bin/env python3
"""Interactive diagnostic viewer for RP3-to-video stroke match alignment.

Opens an interactive matplotlib window with:
  Panel 1 – Handle tracking signal with catch/finish lines and drive/recovery shading
  Panel 2 – Video vs RP3 drive/recovery duration bars side-by-side
  Panel 3 – Per-stroke RP3 force curves (small multiples) with angle overlay

Navigate pages with slider, keyboard (LEFT/RIGHT, HOME/END), or scroll wheel.
Press 's' to save the current page as a PNG.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.widgets import Slider, TextBox
import numpy as np
import pandas as pd

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.selectors import (
    discover_run_dirs as _discover_run_dirs,
    pick_run_with_curses as _pick_run_with_curses,
    pick_run_with_prompt as _pick_run_with_prompt,
)


# ---------------------------------------------------------------------------
# Run discovery – only show runs that have completed RP3 matching
# ---------------------------------------------------------------------------

def _discover_diagnosed_runs(runs_root: Path) -> list[Path]:
    all_runs = _discover_run_dirs(runs_root)
    return [r for r in all_runs if (r / "inference" / "rp3_match_manifest.csv").exists()]


def _select_run(runs_root: Path) -> Path:
    options = _discover_diagnosed_runs(runs_root)
    if not options:
        raise FileNotFoundError(
            f"No runs with inference/rp3_match_manifest.csv found under {runs_root}"
        )
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _pick_run_with_curses(options)
        except Exception:
            pass
    return _pick_run_with_prompt(options)


def _resolve_run_dir(run_dir: Path | None, runs_root: Path) -> Path:
    if run_dir is None:
        return _select_run(runs_root)
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_active_side(summary: dict[str, Any], segments: pd.DataFrame) -> str | None:
    side = summary.get("active_side")
    if isinstance(side, str) and side.lower() in {"left", "right"}:
        return side.lower()
    if not segments.empty and "active_side" in segments.columns:
        non_null = segments["active_side"].dropna()
        if not non_null.empty:
            side = str(non_null.iloc[0]).strip().lower()
            if side in {"left", "right"}:
                return side
    return None


def _attach_active_side_angles(
    run_dir: Path,
    stroke_signal: pd.DataFrame,
    summary: dict[str, Any],
    segments: pd.DataFrame,
) -> pd.DataFrame:
    if "frame_idx" not in stroke_signal.columns:
        return stroke_signal
    if all(c in stroke_signal.columns for c in ("knee_active_deg", "hip_active_deg", "elbow_active_deg")):
        return stroke_signal

    side = _resolve_active_side(summary, segments)
    if side is None:
        return stroke_signal

    knee_col = f"{side}_knee_deg"
    hip_col = f"{side}_hip_deg"
    elbow_col = f"{side}_elbow_deg"
    angles_path = run_dir / "motionbert" / "angles_h36m.csv"
    if not angles_path.exists():
        return stroke_signal

    try:
        angles = pd.read_csv(angles_path, usecols=["frame_idx", knee_col, hip_col, elbow_col])
    except Exception:
        return stroke_signal

    out = stroke_signal.copy()
    lookup = angles.drop_duplicates(subset="frame_idx").set_index("frame_idx")
    if "knee_active_deg" not in out.columns:
        out["knee_active_deg"] = out["frame_idx"].map(lookup[knee_col])
    if "hip_active_deg" not in out.columns:
        out["hip_active_deg"] = out["frame_idx"].map(lookup[hip_col])
    if "elbow_active_deg" not in out.columns:
        out["elbow_active_deg"] = out["frame_idx"].map(lookup[elbow_col])
    return out


def _load_run_data(run_dir: Path) -> dict[str, Any]:
    inf = run_dir / "inference"

    manifest = pd.read_csv(inf / "rp3_match_manifest.csv")
    drive_events = pd.read_csv(inf / "drive_events.csv")
    stroke_signal = pd.read_csv(inf / "stroke_signal_with_drive_events.csv")

    segments_path = inf / "rp3_pose_force_matched_segments.csv"
    segments = pd.read_csv(segments_path) if segments_path.exists() else pd.DataFrame()

    summary: dict[str, Any] = {}
    summary_path = inf / "rp3_match_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    stroke_signal = _attach_active_side_angles(run_dir, stroke_signal, summary, segments)

    return {
        "manifest": manifest,
        "drive_events": drive_events,
        "stroke_signal": stroke_signal,
        "segments": segments,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _render_panel_handle(
    ax: plt.Axes,
    stroke_signal: pd.DataFrame,
    drive_events: pd.DataFrame,
    page_strokes: list[int],
    t_min: float,
    t_max: float,
) -> list[plt.Axes]:
    created_axes: list[plt.Axes] = []
    time_s = pd.to_numeric(stroke_signal["time_s"], errors="coerce").to_numpy(dtype=float)

    raw_handle: np.ndarray | None = None
    smooth_handle: np.ndarray | None = None
    if "relative_axis_px" in stroke_signal.columns:
        raw_handle = pd.to_numeric(
            stroke_signal["relative_axis_px"], errors="coerce"
        ).to_numpy(dtype=float)
    if "relative_axis_px_smooth" in stroke_signal.columns:
        smooth_handle = pd.to_numeric(
            stroke_signal["relative_axis_px_smooth"], errors="coerce"
        ).to_numpy(dtype=float)
    if raw_handle is None and smooth_handle is None:
        raise ValueError(
            "stroke_signal is missing both 'relative_axis_px' and 'relative_axis_px_smooth'"
        )

    mask = (time_s >= t_min) & (time_s <= t_max)
    if raw_handle is not None:
        ax.plot(
            time_s[mask], raw_handle[mask],
            color="dimgray", linewidth=0.8, alpha=0.45,
            label="Handle distance (raw)",
        )

    if smooth_handle is not None:
        ax.plot(
            time_s[mask], smooth_handle[mask],
            color="steelblue", linewidth=1.2, alpha=0.9,
            label="Handle distance (smoothed)",
        )

    handle_for_scale = smooth_handle if smooth_handle is not None else raw_handle
    sig_max = float(np.nanmax(handle_for_scale[mask])) if mask.any() else 1.0

    angle_series: list[tuple[str, str, np.ndarray]] = []
    if "knee_active_deg" in stroke_signal.columns:
        knee = pd.to_numeric(stroke_signal["knee_active_deg"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(knee[mask]).any():
            angle_series.append(("Knee angle (deg)", "tab:red", knee))
    if "hip_active_deg" in stroke_signal.columns:
        hip = pd.to_numeric(stroke_signal["hip_active_deg"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(hip[mask]).any():
            angle_series.append(("Hip angle (deg)", "tab:olive", hip))
    if "elbow_active_deg" in stroke_signal.columns:
        elbow = pd.to_numeric(stroke_signal["elbow_active_deg"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(elbow[mask]).any():
            angle_series.append(("Elbow angle (deg)", "tab:orange", elbow))

    angle_ax: plt.Axes | None = None
    if angle_series:
        angle_ax = ax.twinx()
        created_axes.append(angle_ax)
        for label, color, values in angle_series:
            angle_ax.plot(
                time_s[mask], values[mask], color=color, linewidth=1.0, alpha=0.75,
                label=label,
            )
        angle_ax.set_ylabel("Joint angle (deg)")
        angle_ax.tick_params(axis="y", labelsize=8, colors="dimgray")
        angle_ax.grid(False)

    page_events = drive_events[drive_events["stroke_idx"].isin(page_strokes)]
    for i, (_, ev) in enumerate(page_events.iterrows()):
        catch_t = float(ev["catch_time_s"])
        finish_t = float(ev["finish_time_s"])
        next_catch_t = float(ev["next_catch_time_s"])

        ax.axvspan(
            catch_t, finish_t, alpha=0.12, color="tab:green", zorder=0,
            label="Drive phase" if i == 0 else None,
        )
        ax.axvspan(
            finish_t, next_catch_t, alpha=0.08, color="tab:red", zorder=0,
            label="Recovery phase" if i == 0 else None,
        )

        ax.axvline(
            catch_t, color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8,
            label="Catch" if i == 0 else None,
        )
        ax.axvline(
            finish_t, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8,
            label="Finish" if i == 0 else None,
        )

        ax.text(
            catch_t + 0.03, sig_max * 1.01,
            f"v{int(ev['stroke_idx'])}", fontsize=8, ha="left", va="bottom",
            color="tab:green", fontweight="bold",
        )

    ax.set_xlim(t_min, t_max)
    ax.set_ylabel("Handle distance (px)")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    if angle_ax is not None:
        lines_2, labels_2 = angle_ax.get_legend_handles_labels()
    else:
        lines_2, labels_2 = ([], [])
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.2)
    ax.margins(x=0)
    ax.tick_params(axis="x", labelbottom=False)
    return created_axes


def _render_panel_durations(
    ax: plt.Axes,
    manifest: pd.DataFrame,
    drive_events: pd.DataFrame,
    page_strokes: list[int],
) -> None:
    bar_h = 0.32
    page_manifest = manifest[manifest["video_stroke_idx"].isin(page_strokes)]
    page_events = drive_events[drive_events["stroke_idx"].isin(page_strokes)]

    for i, stroke_idx in enumerate(page_strokes):
        mrow = page_manifest[page_manifest["video_stroke_idx"] == stroke_idx]
        ev_row = page_events[page_events["stroke_idx"] == stroke_idx]
        if mrow.empty or ev_row.empty:
            continue
        mrow = mrow.iloc[0]

        video_catch_t = float(mrow["video_catch_time_s"])
        video_drive = float(mrow["video_drive_s"])
        video_recover = float(mrow["video_recover_s"])
        rp3_drive = float(mrow["rp3_drive_s"])
        rp3_recover = float(mrow["rp3_recover_s"])
        rp3_num = int(mrow["rp3_stroke_number"])
        cum_err = float(mrow["cum_catch_err_s"])
        skipped = int(mrow["rp3_rows_skipped_since_prev"])

        y_video = i * 2.0
        y_rp3 = i * 2.0 + 0.50

        # Video bars (solid)
        ax.barh(y_video, video_drive, height=bar_h, left=video_catch_t,
                color="tab:blue", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.barh(y_video, video_recover, height=bar_h, left=video_catch_t + video_drive,
                color="tab:orange", alpha=0.5, edgecolor="white", linewidth=0.5)

        # RP3 bars (hatched)
        ax.barh(y_rp3, rp3_drive, height=bar_h, left=video_catch_t,
                color="tab:blue", alpha=0.4, edgecolor="white", linewidth=0.5, hatch="//")
        ax.barh(y_rp3, rp3_recover, height=bar_h, left=video_catch_t + rp3_drive,
                color="tab:orange", alpha=0.3, edgecolor="white", linewidth=0.5, hatch="//")

        # Duration text on drive bars
        if video_drive > 0.3:
            ax.text(video_catch_t + video_drive / 2.0, y_video,
                    f"{video_drive:.2f}s", fontsize=6, ha="center", va="center",
                    color="white", fontweight="bold")
        if rp3_drive > 0.3:
            ax.text(video_catch_t + rp3_drive / 2.0, y_rp3,
                    f"{rp3_drive:.2f}s", fontsize=6, ha="center", va="center",
                    color="white", fontweight="bold")

        # Duration text on recovery bars
        rec_x_video = video_catch_t + video_drive + video_recover / 2.0
        rec_x_rp3 = video_catch_t + rp3_drive + rp3_recover / 2.0
        if video_recover > 0.5:
            ax.text(rec_x_video, y_video,
                    f"{video_recover:.2f}s", fontsize=6, ha="center", va="center",
                    color="#664400")
        if rp3_recover > 0.5:
            ax.text(rec_x_rp3, y_rp3,
                    f"{rp3_recover:.2f}s", fontsize=6, ha="center", va="center",
                    color="#664400")

        # Stroke pairing + row labels
        mid_y = (y_video + y_rp3) / 2.0
        ax.text(
            video_catch_t - 0.08, mid_y,
            f"v{stroke_idx}\u2194r{rp3_num}",
            fontsize=7, ha="right", va="center", fontweight="bold",
        )
        ax.text(video_catch_t + 0.02, y_video - bar_h * 0.55, "Vid",
                fontsize=5, ha="left", va="top", color="dimgray", fontstyle="italic")
        ax.text(video_catch_t + 0.02, y_rp3 - bar_h * 0.55, "RP3",
                fontsize=5, ha="left", va="top", color="dimgray", fontstyle="italic")

        # Catch/finish vertical lines (match panel 1)
        video_finish_t = video_catch_t + video_drive
        ax.axvline(video_catch_t, color="tab:green", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.axvline(video_finish_t, color="tab:red", linestyle="--",
                   linewidth=0.8, alpha=0.6)

        # RP3 implied finish line (drive ends at a different point)
        rp3_finish_t = video_catch_t + rp3_drive
        if abs(rp3_finish_t - video_finish_t) > 0.02:
            ax.axvline(rp3_finish_t, color="tab:red", linestyle=":",
                       linewidth=0.7, alpha=0.4)

        # Error annotation
        right_edge = video_catch_t + max(
            video_drive + video_recover, rp3_drive + rp3_recover
        )
        err_colour = "red" if abs(cum_err) > 0.1 else "gray"
        ax.text(
            right_edge + 0.08, mid_y,
            f"cum \u0394={cum_err:+.3f}s",
            fontsize=6, ha="left", va="center", color=err_colour,
        )

        if skipped > 0:
            ax.text(
                video_catch_t - 0.04, y_rp3 + bar_h * 0.5 + 0.25,
                f"\u26A0 {skipped} skipped",
                fontsize=6, ha="right", va="bottom", color="darkorange",
            )

    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.7, label="Drive (video)"),
        Patch(facecolor="tab:orange", alpha=0.5, label="Recovery (video)"),
        Patch(facecolor="tab:blue", alpha=0.4, hatch="//", label="Drive (RP3)"),
        Patch(facecolor="tab:orange", alpha=0.3, hatch="//", label="Recovery (RP3)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2)

    ax.set_yticks([])
    ax.set_ylim(-0.5, len(page_strokes) * 2.0 + 0.3)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_ylabel("Duration comparison")
    ax.set_xlabel("Time (s)")


def _render_panel_force(
    parent_gs: gridspec.SubplotSpec,
    fig: plt.Figure,
    manifest: pd.DataFrame,
    segments: pd.DataFrame,
    page_strokes: list[int],
) -> list[plt.Axes]:
    """Returns every axes created so they can be removed later."""
    n = len(page_strokes)
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=parent_gs, wspace=0.35)

    page_manifest = manifest[manifest["video_stroke_idx"].isin(page_strokes)]
    created_axes: list[plt.Axes] = []

    for col_idx, stroke_idx in enumerate(page_strokes):
        ax = fig.add_subplot(inner_gs[0, col_idx])
        created_axes.append(ax)
        stroke_seg = (
            segments[segments["video_stroke_idx"] == stroke_idx]
            if not segments.empty
            else pd.DataFrame()
        )

        if stroke_seg.empty:
            ax.text(
                0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray",
            )
            ax.set_title(f"v{stroke_idx}", fontsize=8)
            ax.set_xlim(0, 1)
            continue

        s = stroke_seg["s_force"].to_numpy(dtype=float)
        force = stroke_seg["force_raw"].to_numpy(dtype=float)

        ax.plot(s, force, color="tab:blue", linewidth=1.2, label="Force (N)")
        ax.fill_between(s, 0, force, alpha=0.15, color="tab:blue")

        if "knee_active_deg" in stroke_seg.columns:
            ax_tw = ax.twinx()
            created_axes.append(ax_tw)
            knee = stroke_seg["knee_active_deg"].to_numpy(dtype=float)
            ax_tw.plot(s, knee, color="tab:red", linewidth=0.8, alpha=0.7,
                       label="Knee angle (\u00b0)")
            ax_tw.tick_params(axis="y", labelsize=5, colors="tab:red")
            if col_idx == n - 1:
                ax_tw.set_ylabel("Knee (\u00b0)", fontsize=6, color="tab:red")
            else:
                ax_tw.set_yticklabels([])

        mrow = page_manifest[page_manifest["video_stroke_idx"] == stroke_idx]
        rp3_num = int(mrow.iloc[0]["rp3_stroke_number"]) if not mrow.empty else "?"
        ax.set_title(f"v{stroke_idx} / r{rp3_num}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=5)
        if col_idx == 0:
            ax.set_ylabel("Force (N)", fontsize=7)
        ax.set_xlabel("Drive progress", fontsize=6)
        ax.grid(True, alpha=0.2)

        if col_idx == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ([], [])
            if "knee_active_deg" in stroke_seg.columns:
                lines_2, labels_2 = ax_tw.get_legend_handles_labels()
            ax.legend(
                lines_1 + lines_2, labels_1 + labels_2,
                loc="upper left", fontsize=5, framealpha=0.7,
            )

    return created_axes


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

class DiagnosticsViewer:
    def __init__(
        self,
        run_dir: Path,
        data: dict[str, Any],
        strokes_per_page: int = 10,
    ) -> None:
        self.run_dir = run_dir
        self.manifest: pd.DataFrame = data["manifest"]
        self.drive_events: pd.DataFrame = data["drive_events"]
        self.stroke_signal: pd.DataFrame = data["stroke_signal"]
        self.segments: pd.DataFrame = data["segments"]
        self.summary: dict[str, Any] = data["summary"]

        all_idxs = sorted(self.manifest["video_stroke_idx"].unique().tolist())
        self.pages: list[list[int]] = []
        for i in range(0, len(all_idxs), strokes_per_page):
            self.pages.append(all_idxs[i : i + strokes_per_page])

        self.total_pages = len(self.pages)
        self.page_index = 0

        self._ignore_slider = False
        self._ignore_text = False

        self.fig: plt.Figure | None = None
        self.main_gs: gridspec.GridSpec | None = None
        self.slider: Slider | None = None
        self.textbox: TextBox | None = None
        self._panel_axes: list[plt.Axes] = []
        self._connection_patches: list[ConnectionPatch] = []

    # -- drawing -----------------------------------------------------------

    def _clear_panels(self) -> None:
        for patch in self._connection_patches:
            patch.remove()
        self._connection_patches.clear()
        for ax in self._panel_axes:
            ax.remove()
        self._panel_axes.clear()

    def _draw_page(self) -> None:
        self._clear_panels()

        page_strokes = self.pages[self.page_index]
        page_events = self.drive_events[
            self.drive_events["stroke_idx"].isin(page_strokes)
        ]
        if page_events.empty:
            self.fig.canvas.draw_idle()
            return

        t_min = max(0.0, float(page_events["catch_time_s"].min()) - 0.5)
        t_max = float(page_events["next_catch_time_s"].max()) + 0.5

        # Nested gridspec: top block (panels 1+2 tight), bottom (panel 3)
        top_gs = self.main_gs[0].subgridspec(
            2, 1, height_ratios=[3, 1.5], hspace=0.06,
        )

        # Panel 1
        ax1 = self.fig.add_subplot(top_gs[0])
        self._panel_axes.append(ax1)
        handle_extra_axes = _render_panel_handle(
            ax1, self.stroke_signal, self.drive_events, page_strokes, t_min, t_max,
        )
        self._panel_axes.extend(handle_extra_axes)

        mean_cum = self.summary.get("mean_abs_cum_catch_err_s")
        parts = [
            f"Page {self.page_index + 1}/{self.total_pages}",
            f"Strokes v{page_strokes[0]}\u2013v{page_strokes[-1]}",
        ]
        if mean_cum is not None:
            parts.append(f"mean |cum err| = {mean_cum:.3f}s")
        ax1.set_title(
            f"Stroke Match Diagnostics \u2014 {self.run_dir.name}\n"
            + "  |  ".join(parts),
            fontsize=12, fontweight="bold",
        )

        # Panel 2
        ax2 = self.fig.add_subplot(top_gs[1], sharex=ax1)
        self._panel_axes.append(ax2)
        _render_panel_durations(
            ax2, self.manifest, self.drive_events, page_strokes,
        )

        # Connecting vertical lines between panels 1 and 2
        page_manifest = self.manifest[
            self.manifest["video_stroke_idx"].isin(page_strokes)
        ]
        for _, mrow in page_manifest.iterrows():
            catch_t = float(mrow["video_catch_time_s"])
            finish_t = float(mrow["video_finish_time_s"])

            for t_val, color in [(catch_t, "tab:green"), (finish_t, "tab:red")]:
                con = ConnectionPatch(
                    xyA=(t_val, 0.0), coordsA=ax1.get_xaxis_transform(),
                    xyB=(t_val, 1.0), coordsB=ax2.get_xaxis_transform(),
                    color=color, linestyle="--", linewidth=0.6, alpha=0.35,
                )
                self.fig.add_artist(con)
                self._connection_patches.append(con)

        # Panel 3
        force_axes = _render_panel_force(
            self.main_gs[1], self.fig, self.manifest, self.segments, page_strokes,
        )
        self._panel_axes.extend(force_axes)

        self.fig.canvas.draw_idle()

    # -- navigation --------------------------------------------------------

    def _set_page(self, new_index: int) -> None:
        new_index = max(0, min(self.total_pages - 1, new_index))
        if new_index == self.page_index and self._panel_axes:
            return
        self.page_index = new_index
        self._draw_page()
        self._sync_widgets()

    def _sync_widgets(self) -> None:
        if self.slider is not None:
            self._ignore_slider = True
            self.slider.set_val(self.page_index + 1)
            self._ignore_slider = False
        if self.textbox is not None:
            self._ignore_text = True
            self.textbox.set_val(str(self.page_index + 1))
            self._ignore_text = False

    def _on_slider_change(self, val: float) -> None:
        if self._ignore_slider:
            return
        self._set_page(int(round(val)) - 1)

    def _on_text_submit(self, text: str) -> None:
        if self._ignore_text:
            return
        try:
            page = int(text.strip())
        except ValueError:
            self._sync_widgets()
            return
        if page < 1 or page > self.total_pages:
            self._sync_widgets()
            return
        self._set_page(page - 1)

    def _on_key(self, event: Any) -> None:
        if event.key in ("right", "d"):
            self._set_page(self.page_index + 1)
        elif event.key in ("left", "a"):
            self._set_page(self.page_index - 1)
        elif event.key == "home":
            self._set_page(0)
        elif event.key == "end":
            self._set_page(self.total_pages - 1)
        elif event.key == "s":
            self._save_current_page()

    def _on_scroll(self, event: Any) -> None:
        if event.button == "up":
            self._set_page(self.page_index + 1)
        elif event.button == "down":
            self._set_page(self.page_index - 1)

    def _save_current_page(self) -> None:
        out = (
            self.run_dir
            / "inference"
            / f"match_diagnostics_page_{self.page_index + 1:02d}.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(str(out), dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    # -- entry point -------------------------------------------------------

    def show(self) -> None:
        self.fig = plt.figure("Match Diagnostics", figsize=(18, 14))

        self.main_gs = self.fig.add_gridspec(
            nrows=2, ncols=1,
            left=0.06, right=0.94,
            top=0.93, bottom=0.12,
            height_ratios=[5, 2],
            hspace=0.25,
        )

        slider_ax = self.fig.add_axes([0.08, 0.03, 0.65, 0.025])
        text_ax = self.fig.add_axes([0.78, 0.03, 0.10, 0.025])

        self.slider = Slider(
            slider_ax, "Page",
            1, max(1, self.total_pages),
            valinit=1, valstep=1,
            color="steelblue",
        )
        self.textbox = TextBox(text_ax, "Go to ", initial="1")

        self.slider.on_changed(self._on_slider_change)
        self.textbox.on_submit(self._on_text_submit)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._draw_page()

        help_ax = self.fig.add_axes([0.0, 0.0, 1.0, 0.018])
        help_ax.axis("off")
        help_ax.text(
            0.5, 0.5,
            "LEFT/RIGHT or scroll: navigate pages  |  HOME/END: first/last  |  S: save PNG",
            transform=help_ax.transAxes,
            ha="center", va="center", fontsize=8, color="gray",
        )

        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive diagnostic viewer for RP3-to-video stroke match alignment.",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help=f"Sports2D runs root (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory to diagnose (skip interactive selection).",
    )
    parser.add_argument(
        "--strokes-per-page", type=int, default=5,
        help="Max strokes per diagnostic page (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir, args.runs_root)

    manifest_path = run_dir / "inference" / "rp3_match_manifest.csv"
    if not manifest_path.exists():
        print(f"No rp3_match_manifest.csv in {run_dir / 'inference'}")
        print("Run inference with --match-rp3 first.")
        return 1

    data = _load_run_data(run_dir)
    if data["manifest"].empty:
        print("No matched strokes in manifest.")
        return 1

    print(f"Run: {run_dir.name}")
    viewer = DiagnosticsViewer(
        run_dir=run_dir,
        data=data,
        strokes_per_page=args.strokes_per_page,
    )
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
