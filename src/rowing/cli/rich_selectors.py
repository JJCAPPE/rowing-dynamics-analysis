"""Rich-powered run selectors with status badges and multi-select.

These replace the curses pickers used by the legacy CLI. They're intentionally
keyboard-friendly: arrow keys navigate, ``space`` toggles selection in
multi-select mode, ``a`` selects all, ``n`` clears, ``enter`` confirms,
``q`` cancels.

We use ``rich.live.Live`` plus low-level keypress reading via ``termios`` so
the experience works in any modern terminal without depending on Textual at
runtime. (Textual remains an optional dep for a future Phase 2.5 if the user
wants a more elaborate UI.)
"""
from __future__ import annotations

import os
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from rowing.cli.status import RunStatus, discover_run_statuses


@dataclass
class PickerCancelled(Exception):
    """Raised when the user aborts a selector with ``q`` / ``ESC`` / ``Ctrl-C``."""


def _read_key() -> str:
    """Read a single keypress (handles arrow-key escape sequences)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            try:
                ch += os.read(fd, 2).decode("utf-8", errors="ignore")
            except Exception:
                pass
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Status-badge rendering
# ---------------------------------------------------------------------------


def _badge(label: str, ok: bool, *, alt: str | None = None) -> Text:
    if ok:
        return Text(f" {label} ", style="black on green")
    return Text(f" {alt or label} ", style="white on grey23")


def _qc_badge(status: RunStatus) -> Text:
    label = status.match_qc_label
    if label is None:
        return Text(" qc ?    ", style="white on grey23")
    style = {
        "excellent": "black on green",
        "good": "black on cyan",
        "acceptable": "black on yellow",
        "poor": "white on red",
    }.get(label, "white on grey23")
    score = status.match_qc_score
    score_txt = f"{score*1000:.0f}ms" if score is not None else ""
    return Text(f" qc:{label[:4]} {score_txt} ", style=style)


def _row_badges(status: RunStatus) -> Text:
    parts = [
        _badge("pose", status.pose),
        Text(" "),
        _badge("strk", status.stroke),
        Text(" "),
        _badge("rp3", status.rp3_clean or status.rp3_dirty),
        Text(" "),
        _badge("drv", status.drive_events),
        Text(" "),
        _badge("match", status.match),
        Text(" "),
        _qc_badge(status),
        Text(" "),
        _badge("seg", status.segments),
        Text(" "),
        _badge("ds", status.dataset),
        Text(" "),
        _badge("rep", status.report),
    ]
    out = Text()
    for part in parts:
        out.append_text(part)
    return out


# ---------------------------------------------------------------------------
# Single-select
# ---------------------------------------------------------------------------


def _build_table(
    statuses: list[RunStatus],
    cursor: int,
    selected: set[int],
    *,
    multi: bool,
    title: str,
) -> Table:
    table = Table(title=title, expand=True, show_lines=False, padding=(0, 1))
    if multi:
        table.add_column("•", justify="center", width=2)
    table.add_column(">", justify="center", width=2)
    table.add_column("run name", overflow="fold", min_width=30)
    table.add_column("status", overflow="fold")
    if statuses and statuses[0].matched_strokes is not None or any(s.matched_strokes for s in statuses):
        table.add_column("strokes", justify="right", width=8)

    for i, status in enumerate(statuses):
        is_cursor = i == cursor
        is_sel = i in selected
        cursor_mark = ">" if is_cursor else " "
        sel_mark = "✔" if is_sel else " " if multi else ""
        name_style = "bold reverse" if is_cursor else ""
        name = Text(status.name, style=name_style)
        cells: list[Text] = []
        if multi:
            cells.append(Text(sel_mark, style="bold cyan" if is_sel else ""))
        cells.append(Text(cursor_mark, style="bold yellow"))
        cells.append(name)
        cells.append(_row_badges(status))
        if "strokes" in [c.header for c in table.columns]:
            cells.append(Text(str(status.matched_strokes) if status.matched_strokes is not None else "-"))
        table.add_row(*cells)
    return table


def _help_line(multi: bool) -> Text:
    if multi:
        return Text(
            "  ↑/↓ move   space toggle   a all   n none   enter confirm   q cancel",
            style="dim",
        )
    return Text("  ↑/↓ move   enter confirm   q cancel", style="dim")


def _interactive_pick(
    statuses: list[RunStatus],
    *,
    multi: bool,
    title: str,
    console: Console,
) -> list[RunStatus]:
    if not statuses:
        raise PickerCancelled("No runs available.")

    cursor = 0
    selected: set[int] = set()

    with Live(console=console, refresh_per_second=20, transient=True) as live:
        while True:
            table = _build_table(statuses, cursor, selected, multi=multi, title=title)
            live.update(Text("\n").join([Text("") , Text("")]) if False else _stack(table, _help_line(multi)))

            try:
                key = _read_key()
            except KeyboardInterrupt:
                raise PickerCancelled("Cancelled.")

            if key in ("q", "\x1b"):
                raise PickerCancelled("Cancelled.")
            if key in ("\r", "\n"):
                if multi:
                    return [statuses[i] for i in sorted(selected)] or [statuses[cursor]]
                return [statuses[cursor]]
            if key in ("\x1b[A", "k"):
                cursor = max(0, cursor - 1)
            elif key in ("\x1b[B", "j"):
                cursor = min(len(statuses) - 1, cursor + 1)
            elif multi and key == " ":
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.add(cursor)
            elif multi and key == "a":
                selected = set(range(len(statuses)))
            elif multi and key == "n":
                selected = set()
            elif key == "g":
                cursor = 0
            elif key == "G":
                cursor = len(statuses) - 1


def _stack(*renderables) -> Text:
    """Stack a Table and a help line via ``rich.console.Group``-like wrapper."""
    from rich.console import Group

    return Group(*renderables)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_run(
    runs_root: Path,
    *,
    title: str = "Select run",
    console: Console | None = None,
) -> RunStatus:
    """Single-select run picker with status badges."""
    statuses = discover_run_statuses(runs_root)
    if not statuses:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    console = console or Console()
    picks = _interactive_pick(statuses, multi=False, title=title, console=console)
    return picks[0]


def select_runs(
    runs_root: Path,
    *,
    title: str = "Select runs (space to toggle)",
    console: Console | None = None,
    filter_fn=None,
) -> list[RunStatus]:
    """Multi-select run picker (used for batch dataset / report operations)."""
    statuses = discover_run_statuses(runs_root)
    if filter_fn is not None:
        statuses = [s for s in statuses if filter_fn(s)]
    if not statuses:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    console = console or Console()
    return _interactive_pick(statuses, multi=True, title=title, console=console)


def render_runs_table(statuses: Iterable[RunStatus], *, title: str = "runs") -> Table:
    """Render a non-interactive overview table for the menu home screen."""
    table = Table(title=title, expand=True, show_lines=False, padding=(0, 1))
    table.add_column("run name", min_width=30, overflow="fold")
    table.add_column("status", overflow="fold")
    table.add_column("strokes", justify="right", width=8)
    for status in statuses:
        cells = [
            Text(status.name),
            _row_badges(status),
            Text(str(status.matched_strokes) if status.matched_strokes is not None else "-"),
        ]
        table.add_row(*cells)
    return table
