#!/usr/bin/env python3
"""Interactive RP3 stroke viewer.

Flow:
1) Pick a cleaned workout CSV via a curses file selector.
2) Open an interactive matplotlib window with:
   - force-vs-length curve for a selected stroke
   - slider and manual stroke input
   - keyboard/mouse navigation across strokes
"""

from __future__ import annotations

import argparse
import csv
import curses
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

FORCE_COL_RE = re.compile(r"^force_at_([0-9]+(?:\.[0-9]+)?)cm$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a workout CSV from workouts/clean and browse stroke force curves "
            "interactively."
        )
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("rp3-extraction/workouts/clean"),
        help="Directory that contains cleaned workout CSV files (default: workouts/clean).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Open this CSV directly (skip curses picker).",
    )
    parser.add_argument(
        "--start-stroke",
        type=int,
        default=1,
        help="Initial stroke index (1-based, default: 1).",
    )
    return parser.parse_args()


def find_force_columns(fieldnames: list[str]) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for col in fieldnames:
        match = FORCE_COL_RE.match(col)
        if match is None:
            continue
        found.append((col, float(match.group(1))))
    found.sort(key=lambda item: item[1])
    return found


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt_seconds(seconds: float | None, decimals: int = 2) -> str:
    if seconds is None or math.isnan(seconds):
        return "-"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if decimals == 0:
        return f"{minutes:02d}:{int(round(rem)):02d}"
    width = 3 + decimals
    return f"{minutes:02d}:{rem:0{width}.{decimals}f}"


def fmt_number(value: float | None, suffix: str = "", precision: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{precision}f}{suffix}"


def select_csv_with_curses(clean_dir: Path) -> Path | None:
    files = sorted([p for p in clean_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {clean_dir}")

    def _menu(stdscr: Any) -> Path | None:
        curses.curs_set(0)
        index = 0
        top = 0

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            title = f"RP3 workouts in {clean_dir}"
            help_line = "UP/DOWN move   ENTER open   q quit"
            stdscr.addnstr(0, 0, title, max(1, width - 1), curses.A_BOLD)
            stdscr.addnstr(1, 0, help_line, max(1, width - 1), curses.A_DIM)

            visible_rows = max(1, height - 3)
            if index < top:
                top = index
            elif index >= top + visible_rows:
                top = index - visible_rows + 1

            for row in range(visible_rows):
                i = top + row
                if i >= len(files):
                    break
                attr = curses.A_REVERSE if i == index else curses.A_NORMAL
                stdscr.addnstr(3 + row - 1, 0, files[i].name, max(1, width - 1), attr)

            stdscr.refresh()
            key = stdscr.getch()

            if key in (ord("q"), 27):
                return None
            if key in (curses.KEY_UP, ord("k")):
                index = max(0, index - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                index = min(len(files) - 1, index + 1)
                continue
            if key in (curses.KEY_NPAGE,):
                index = min(len(files) - 1, index + visible_rows)
                continue
            if key in (curses.KEY_PPAGE,):
                index = max(0, index - visible_rows)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return files[index]

        return None

    return curses.wrapper(_menu)


def load_workout_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[tuple[str, float]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        force_cols = find_force_columns(reader.fieldnames)
        if not force_cols:
            raise ValueError(
                f"No force_at_*cm columns found in {csv_path}. "
                "Run expand_rp3_curve_data.py first."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows, force_cols


class StrokeViewer:
    def __init__(
        self,
        csv_path: Path,
        rows: list[dict[str, str]],
        force_cols: list[tuple[str, float]],
        start_stroke: int = 1,
    ) -> None:
        self.csv_path = csv_path
        self.rows = rows
        self.force_cols = force_cols
        self.total = len(rows)
        self.index = max(0, min(self.total - 1, start_stroke - 1))
        self._ignore_slider_event = False
        self._ignore_text_event = False

        self.curves = [self._extract_curve(row) for row in rows]
        all_forces = [force for _, y in self.curves for force in y]
        self.y_max = max(all_forces) if all_forces else 1.0
        self.x_max = max(distance for _, distance in force_cols)

        self.fig = None
        self.ax_curve = None
        self.ax_info = None
        self.slider = None
        self.textbox = None
        self.curve_line = None
        self.peak_line = None
        self.avg_pos_line = None
        self.value_artists: dict[str, Any] = {}

    def _extract_curve(self, row: dict[str, str]) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for col, distance_cm in self.force_cols:
            force = to_float(row.get(col))
            if force is None:
                continue
            xs.append(distance_cm)
            ys.append(force)
        return xs, ys

    def _build_info_panel(self) -> None:
        left_specs = [
            ("stroke_no", "Stroke"),
            ("time", "Time"),
            ("distance", "Distance"),
            ("stroke_rate", "Stroke rate"),
            ("split", "Split"),
            ("power", "Power"),
            ("stroke_length", "Stroke length"),
        ]
        right_specs = [
            ("energy_per_stroke", "Energy/stroke"),
            ("peak_force", "Peak force"),
            ("peak_force_pos", "Peak force pos"),
            ("avg_force_pos", "Avg force pos"),
            ("avg_force_pos_rel", "Avg force pos rel"),
            ("rel_peak_pos", "Rel. peak pos"),
            ("drive_time", "Drive time"),
            ("recover_time", "Recover time"),
            ("avg_calc_power", "Avg calc power"),
        ]
        value_color = "#19a9cf"

        self.ax_info.axis("off")
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)

        y0 = 0.95
        dy = 0.105
        for i, (key, label) in enumerate(left_specs):
            y = y0 - i * dy
            self.ax_info.text(0.02, y - 0.05, label, fontsize=12, color="#222222")
            self.value_artists[key] = self.ax_info.text(
                0.02,
                y,
                "-",
                fontsize=18,
                color=value_color,
                weight="bold",
            )

        for i, (key, label) in enumerate(right_specs):
            y = y0 - i * dy
            self.ax_info.text(0.54, y - 0.05, label, fontsize=12, color="#222222")
            self.value_artists[key] = self.ax_info.text(
                0.54,
                y,
                "-",
                fontsize=18,
                color=value_color,
                weight="bold",
            )

    def _stroke_metrics(self, idx: int, avg_force_pos: float | None) -> dict[str, str]:
        row = self.rows[idx]

        stroke_no = to_float(row.get("stroke_number"))
        if stroke_no is None:
            stroke_no_text = f"{idx + 1}/{self.total}"
        else:
            stroke_no_text = f"{int(round(stroke_no))} ({idx + 1}/{self.total})"

        distance = to_float(row.get("distance"))
        stroke_rate = to_float(row.get("stroke_rate"))
        split_s = to_float(row.get("estimated_500m_time"))
        power = to_float(row.get("power"))
        stroke_length_cm = to_float(row.get("stroke_length"))
        energy = to_float(row.get("energy_per_stroke"))
        peak_force = to_float(row.get("peak_force"))
        peak_pos = to_float(row.get("peak_force_pos"))
        rel_peak = to_float(row.get("rel_peak_force_pos"))
        drive = to_float(row.get("drive_time"))
        recover = to_float(row.get("recover_time"))
        avg_calc_power = to_float(row.get("avg_calculated_power"))
        avg_force_pos_rel = None
        if avg_force_pos is not None and stroke_length_cm is not None and stroke_length_cm > 0:
            avg_force_pos_rel = (avg_force_pos / stroke_length_cm) * 100.0

        return {
            "stroke_no": stroke_no_text,
            "time": fmt_seconds(to_float(row.get("time")), decimals=2),
            "distance": fmt_number(distance, " m", precision=1),
            "stroke_rate": fmt_number(stroke_rate, " s/m", precision=1),
            "split": f"{fmt_seconds(split_s, decimals=2)}/500m" if split_s is not None else "-",
            "power": fmt_number(power, " W", precision=0),
            "stroke_length": fmt_number(
                (stroke_length_cm / 100.0) if stroke_length_cm is not None else None,
                " m",
                precision=2,
            ),
            "energy_per_stroke": fmt_number(energy, " J", precision=1),
            "peak_force": fmt_number(peak_force, " N", precision=0),
            "peak_force_pos": fmt_number(peak_pos, " cm", precision=1),
            "avg_force_pos": fmt_number(avg_force_pos, " cm", precision=1),
            "avg_force_pos_rel": fmt_number(avg_force_pos_rel, " %", precision=1),
            "rel_peak_pos": fmt_number(
                (rel_peak * 100.0) if rel_peak is not None else None,
                " %",
                precision=1,
            ),
            "drive_time": fmt_number(drive, " s", precision=2),
            "recover_time": fmt_number(recover, " s", precision=2),
            "avg_calc_power": fmt_number(avg_calc_power, " W", precision=0),
        }

    def _set_index(self, new_index: int) -> None:
        self.index = max(0, min(self.total - 1, new_index))
        self._refresh_view(sync_slider=True, sync_textbox=True)

    def _refresh_view(self, sync_slider: bool, sync_textbox: bool) -> None:
        xs, ys = self.curves[self.index]
        self.curve_line.set_data(xs, ys)

        avg_force_pos: float | None = None
        total_force = sum(ys)
        if ys and total_force > 0:
            avg_force_pos = sum(x * force for x, force in zip(xs, ys)) / total_force

        peak_x = to_float(self.rows[self.index].get("peak_force_pos"))
        if peak_x is None and xs and ys:
            peak_idx = max(range(len(ys)), key=lambda i: ys[i])
            peak_x = xs[peak_idx]
        if self.peak_line is not None:
            if peak_x is None:
                self.peak_line.set_visible(False)
            else:
                self.peak_line.set_xdata([peak_x, peak_x])
                self.peak_line.set_visible(True)
        if self.avg_pos_line is not None:
            if avg_force_pos is None:
                self.avg_pos_line.set_visible(False)
            else:
                self.avg_pos_line.set_xdata([avg_force_pos, avg_force_pos])
                self.avg_pos_line.set_visible(True)

        title = f"{self.csv_path.name}  |  Stroke {self.index + 1}/{self.total}"
        self.ax_curve.set_title(title, fontsize=14, pad=12)

        metrics = self._stroke_metrics(self.index, avg_force_pos)
        for key, artist in self.value_artists.items():
            artist.set_text(metrics.get(key, "-"))

        if sync_slider and self.slider is not None:
            self._ignore_slider_event = True
            self.slider.set_val(self.index + 1)
            self._ignore_slider_event = False
        if sync_textbox and self.textbox is not None:
            self._ignore_text_event = True
            self.textbox.set_val(str(self.index + 1))
            self._ignore_text_event = False

        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value: float) -> None:
        if self._ignore_slider_event:
            return
        idx = int(round(value)) - 1
        self._set_index(idx)

    def _on_text_submit(self, text: str) -> None:
        if self._ignore_text_event:
            return
        try:
            stroke = int(text.strip())
        except ValueError:
            self._ignore_text_event = True
            self.textbox.set_val(str(self.index + 1))
            self._ignore_text_event = False
            return
        if stroke < 1 or stroke > self.total:
            self._ignore_text_event = True
            self.textbox.set_val(str(self.index + 1))
            self._ignore_text_event = False
            return
        self._set_index(stroke - 1)

    def _on_key_press(self, event: Any) -> None:
        if event.key in ("left", "a"):
            self._set_index(self.index - 1)
        elif event.key in ("right", "d"):
            self._set_index(self.index + 1)
        elif event.key in ("up", "w"):
            self._set_index(self.index + 10)
        elif event.key in ("down", "s"):
            self._set_index(self.index - 10)
        elif event.key == "home":
            self._set_index(0)
        elif event.key == "end":
            self._set_index(self.total - 1)

    def _on_scroll(self, event: Any) -> None:
        if event.button == "up":
            self._set_index(self.index + 1)
        elif event.button == "down":
            self._set_index(self.index - 1)

    def show(self) -> None:
        self.fig = plt.figure("RP3 Stroke Viewer", figsize=(14, 8))
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.2,
            width_ratios=[1.55, 1.0],
            wspace=0.15,
        )
        self.ax_curve = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])

        self.ax_curve.set_xlabel("Stroke length (cm)")
        self.ax_curve.set_ylabel("Force (N)")
        self.ax_curve.grid(True, alpha=0.3)
        self.ax_curve.set_xlim(0, self.x_max * 1.03)
        self.ax_curve.set_ylim(0, self.y_max * 1.08)
        (self.curve_line,) = self.ax_curve.plot([], [], color="#12a8d3", linewidth=1.0)
        self.peak_line = self.ax_curve.axvline(
            x=0,
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="Peak position",
            visible=False,
        )
        self.avg_pos_line = self.ax_curve.axvline(
            x=0,
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.9,
            label="Avg force position",
            visible=False,
        )
        self.ax_curve.legend(loc="upper right")

        self._build_info_panel()

        slider_ax = self.fig.add_axes([0.08, 0.08, 0.68, 0.05])
        text_ax = self.fig.add_axes([0.79, 0.08, 0.13, 0.05])

        self.slider = Slider(
            slider_ax,
            "Stroke",
            1,
            self.total,
            valinit=self.index + 1,
            valstep=1,
            color="#12a8d3",
        )
        self.textbox = TextBox(text_ax, "Go", initial=str(self.index + 1))

        self.slider.on_changed(self._on_slider_change)
        self.textbox.on_submit(self._on_text_submit)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._refresh_view(sync_slider=False, sync_textbox=False)
        plt.show()


def main() -> int:
    args = parse_args()

    if args.file is not None:
        csv_path = args.file.expanduser().resolve()
    else:
        clean_dir = args.clean_dir.expanduser().resolve()
        if not clean_dir.exists():
            print(f"Clean directory not found: {clean_dir}", file=sys.stderr)
            return 1
        try:
            csv_path = select_csv_with_curses(clean_dir)
        except curses.error as exc:
            print(
                "Failed to start curses picker. Use --file to open a CSV directly.\n"
                f"Details: {exc}",
                file=sys.stderr,
            )
            return 1
        if csv_path is None:
            print("No file selected.")
            return 0

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        rows, force_cols = load_workout_rows(csv_path)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    viewer = StrokeViewer(
        csv_path=csv_path,
        rows=rows,
        force_cols=force_cols,
        start_stroke=args.start_stroke,
    )
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
