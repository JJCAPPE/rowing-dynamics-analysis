"""Interactive selectors used by the legacy curses-based CLIs.

Phase 2 will swap these out for Rich-based selectors with status badges. For
now the new package keeps the existing curses + line-prompt fallbacks so the
behaviour of the inference CLI is unchanged after the Phase 1 refactor.
"""
from __future__ import annotations

import curses
import sys
from pathlib import Path
from typing import Any, Sequence


__all__ = [
    "discover_run_dirs",
    "pick_run_with_curses",
    "pick_run_with_prompt",
    "pick_file_with_curses",
    "pick_file_with_prompt",
    "pick_yes_no_with_curses",
    "pick_yes_no_with_prompt",
    "select_yes_no",
    "prompt_int",
    "prompt_choice",
    "select_run",
    "resolve_run_dir",
    "ensure_path_in_dir",
]


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Return all run directories under *runs_root* with a stroke signal CSV."""
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    if not runs_root.is_dir():
        raise NotADirectoryError(f"Runs path is not a directory: {runs_root}")

    runs: list[Path] = []
    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue
        stroke_csv = candidate / "stroke" / "stroke_signal.csv"
        if stroke_csv.exists():
            runs.append(candidate.resolve())

    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


def pick_run_with_curses(options: Sequence[Path]) -> Path:
    def _inner(stdscr: Any) -> Path:
        idx = 0
        top = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, "Select Sports2D run (Enter to confirm)", width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "UP/DOWN move, ENTER select, q quit", width, curses.A_DIM)

            visible = max(1, height - 3)
            if idx < top:
                top = idx
            elif idx >= top + visible:
                top = idx - visible + 1

            for row in range(visible):
                i = top + row
                if i >= len(options):
                    break
                label = options[i].name
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(row + 2, 0, label, width, attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                idx = min(len(options) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return options[idx]

    return curses.wrapper(_inner)


def pick_run_with_prompt(options: Sequence[Path]) -> Path:
    if not options:
        raise ValueError("No run options available.")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return options[0]

    print("\nAvailable Sports2D runs:")
    for i, option in enumerate(options, start=1):
        print(f"  {i:2d}. {option.name}")

    while True:
        raw = input("Select run number [1]: ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            pick = int(raw) - 1
            if 0 <= pick < len(options):
                return options[pick]
        print("Invalid selection. Enter a listed number.")


def pick_file_with_curses(options: Sequence[Path], title: str) -> Path:
    def _inner(stdscr: Any) -> Path:
        idx = 0
        top = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, title, width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "UP/DOWN move, ENTER select, q quit", width, curses.A_DIM)

            visible = max(1, height - 3)
            if idx < top:
                top = idx
            elif idx >= top + visible:
                top = idx - visible + 1

            for row in range(visible):
                i = top + row
                if i >= len(options):
                    break
                label = options[i].name
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(row + 2, 0, label, width, attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_UP, ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                idx = min(len(options) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return options[idx]

    return curses.wrapper(_inner)


def pick_file_with_prompt(options: Sequence[Path], title: str) -> Path:
    if not options:
        raise ValueError("No options available.")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return options[0]

    print(f"\n{title}")
    for i, option in enumerate(options, start=1):
        print(f"  {i:2d}. {option.name}")

    while True:
        raw = input("Select number [1]: ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            pick = int(raw) - 1
            if 0 <= pick < len(options):
                return options[pick]
        print("Invalid selection.")


def pick_yes_no_with_curses(prompt: str, default_no: bool = True) -> bool:
    labels = ["No", "Yes"]
    idx = 0 if default_no else 1

    def _inner(stdscr: Any) -> bool:
        nonlocal idx
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            width = max(1, width - 1)
            stdscr.addnstr(0, 0, prompt, width, curses.A_BOLD)
            stdscr.addnstr(1, 0, "LEFT/RIGHT or UP/DOWN, ENTER select, q cancel", width, curses.A_DIM)

            for i, label in enumerate(labels):
                x = 2 + i * 12
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(3, x, f"[ {label} ]", max(1, width - x), attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Selection cancelled.")
            if key in (curses.KEY_LEFT, curses.KEY_UP, ord("h"), ord("k")):
                idx = max(0, idx - 1)
                continue
            if key in (curses.KEY_RIGHT, curses.KEY_DOWN, ord("l"), ord("j")):
                idx = min(len(labels) - 1, idx + 1)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                return idx == 1

    return curses.wrapper(_inner)


def pick_yes_no_with_prompt(prompt: str, default_no: bool = True) -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return not default_no

    suffix = "[y/N]" if default_no else "[Y/n]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if raw == "":
            return not default_no
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def select_yes_no(prompt: str, *, default_no: bool = True) -> bool:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return pick_yes_no_with_curses(prompt, default_no=default_no)
        except Exception:
            pass
    return pick_yes_no_with_prompt(prompt, default_no=default_no)


def prompt_int(prompt: str, default: int) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return int(default)
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return int(default)
        if raw.lstrip("-").isdigit():
            return int(raw)
        print("Please enter an integer.")


def prompt_choice(prompt: str, options: Sequence[str], default: str) -> str:
    options_norm = [str(x).strip().lower() for x in options]
    default_norm = str(default).strip().lower()
    if default_norm not in options_norm:
        raise ValueError("default must be one of options")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default_norm

    opt_txt = "/".join(options_norm)
    while True:
        raw = input(f"{prompt} [{default_norm}] ({opt_txt}): ").strip().lower()
        if raw == "":
            return default_norm
        if raw in options_norm:
            return raw
        print(f"Please enter one of: {opt_txt}")


def select_run(runs_root: Path) -> Path:
    options = discover_run_dirs(runs_root)
    if not options:
        raise FileNotFoundError(f"No runs with stroke/stroke_signal.csv found in {runs_root}")
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return pick_run_with_curses(options)
        except Exception:
            pass
    return pick_run_with_prompt(options)


def resolve_run_dir(run_dir: Path | None, runs_root: Path) -> Path:
    if run_dir is None:
        return select_run(runs_root)

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    if not stroke_csv.exists():
        raise FileNotFoundError(f"Missing stroke signal CSV at: {stroke_csv}")
    return run_dir


def ensure_path_in_dir(path: Path, parent_dir: Path, *, label: str) -> None:
    parent_dir = parent_dir.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        path.relative_to(parent_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent_dir}: {path}") from exc
