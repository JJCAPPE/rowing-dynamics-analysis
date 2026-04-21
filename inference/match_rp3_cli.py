#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pair_session import auto_pair_run


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_SESSION_REGISTRY = REPO_ROOT / "session_registry.csv"


@dataclass(frozen=True)
class MatchConfig:
    max_jump_rows: int
    max_interval_error_s: float
    max_cumulative_error_base_s: float
    max_cumulative_error_per_s: float
    max_abs_cum_error_s: float = 2.0
    w_drive: float = 0.4
    w_recover: float = 0.4
    w_interval: float = 1.0
    w_cumulative: float = 1.0
    w_skip: float = 0.08


@dataclass(frozen=True)
class MatchResult:
    manifest: pd.DataFrame
    total_score: float
    matched_rp3_indices: list[int]


def _pick_path_with_curses(options: Sequence[Path], title: str) -> Path:
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
                name = options[i].name
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(row + 2, 0, name, width, attr)

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


def _pick_path_with_prompt(options: Sequence[Path], title: str) -> Path:
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


def _pick_path(options: Sequence[Path], title: str) -> Path:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _pick_path_with_curses(options, title)
        except Exception:
            pass
    return _pick_path_with_prompt(options, title)


def _discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    runs: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        events = p / "inference" / "drive_events.csv"
        if events.exists():
            runs.append(p.resolve())
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _ensure_path_in_dir(path: Path, parent_dir: Path, *, label: str) -> None:
    parent_dir = parent_dir.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        path.relative_to(parent_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent_dir}: {path}") from exc


def _discover_run_rp3_clean(run_dir: Path) -> list[Path]:
    rp3_dir = (run_dir / "rp3").resolve()
    if not rp3_dir.exists() or not rp3_dir.is_dir():
        raise FileNotFoundError(f"Run RP3 directory not found: {rp3_dir}")
    files = sorted(
        [
            p.resolve()
            for p in rp3_dir.glob("*-clean.csv")
            if p.is_file() and not p.name.startswith(".")
        ]
    )
    return files


def _resolve_run_dir(run_dir: Path | None, runs_root: Path) -> Path:
    if run_dir is not None:
        run = run_dir.expanduser().resolve()
        if not run.exists() or not run.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run}")
        if not (run / "inference" / "drive_events.csv").exists():
            raise FileNotFoundError(f"Missing drive_events.csv in: {run / 'inference'}")
        return run
    options = _discover_runs(runs_root)
    if not options:
        raise FileNotFoundError(f"No runs with inference/drive_events.csv found in {runs_root}")
    return _pick_path(options, "Select run (needs inference/drive_events.csv)")


def _resolve_rp3_csv(
    *,
    run_dir: Path,
    rp3_clean_csv: Path | None,
    interactive: bool,
) -> Path:
    rp3_dir = (run_dir / "rp3").resolve()
    if rp3_clean_csv is not None:
        csv_path = rp3_clean_csv.expanduser().resolve()
        if not csv_path.exists() or not csv_path.is_file():
            raise FileNotFoundError(f"RP3 clean CSV not found: {csv_path}")
        _ensure_path_in_dir(csv_path, rp3_dir, label="--rp3-clean-csv")
        if not csv_path.name.lower().endswith("-clean.csv"):
            raise ValueError(f"--rp3-clean-csv must reference a *-clean.csv file: {csv_path.name}")
        return csv_path

    options = _discover_run_rp3_clean(run_dir)
    if not options:
        raise FileNotFoundError(
            f"No RP3 clean CSV files found in {rp3_dir}. "
            "Run inference_cli.py first to generate one from dirty RP3 data."
        )
    if len(options) == 1:
        return options[0]
    if interactive:
        return _pick_path(options, "Select RP3 clean CSV")
    raise ValueError(
        f"Multiple RP3 clean CSV files found in {rp3_dir}. Specify one with --rp3-clean-csv."
    )


def _load_video_events(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "inference" / "drive_events.csv"
    df = pd.read_csv(path)
    required = {
        "stroke_idx",
        "catch_time_s",
        "finish_time_s",
        "drive_duration_s",
        "recover_duration_s",
        "cycle_duration_s",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"drive_events.csv missing columns: {missing}")
    if df.empty:
        raise ValueError(f"No rows in {path}")
    df = df.sort_values("stroke_idx").reset_index(drop=True)
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["catch_time_s", "drive_duration_s", "recover_duration_s", "cycle_duration_s"]].isna().any().any():
        raise ValueError(f"Invalid numeric values in {path}")
    return df


def _load_rp3(clean_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)
    required = {"stroke_number", "time", "drive_time", "recover_time"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RP3 CSV missing columns: {missing}")
    if df.empty:
        raise ValueError(f"No rows in RP3 CSV: {clean_csv}")
    for c in ["stroke_number", "time", "drive_time", "recover_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["time", "drive_time", "recover_time"]].isna().any().any():
        raise ValueError(f"Invalid numeric values in RP3 CSV: {clean_csv}")
    df = df.reset_index(drop=True)
    df["rp3_row_idx"] = np.arange(len(df), dtype=np.int32)
    df["rp3_cycle_s"] = df["drive_time"] + df["recover_time"]
    return df


def _prompt_int(prompt: str, default: int) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.lstrip("-").isdigit():
            return int(raw)
        print("Please enter an integer.")


def _stroke_number_candidates(rp3_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(rp3_df["stroke_number"], errors="coerce").to_numpy(dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        raise ValueError("RP3 CSV has no valid stroke_number values.")
    finite_strokes = np.rint(values[finite_idx]).astype(np.int64)
    return finite_idx, finite_strokes


def _find_rp3_row_by_stroke_number(rp3_df: pd.DataFrame, stroke_no: int) -> int | None:
    finite_idx, finite_strokes = _stroke_number_candidates(rp3_df)
    hit_idx = finite_idx[finite_strokes == int(stroke_no)]
    if hit_idx.size == 0:
        return None
    return int(hit_idx[0])


def _resolve_anchor_rp3_idx(
    rp3_df: pd.DataFrame,
    *,
    anchor_rp3_row_idx: int | None,
    anchor_rp3_stroke_number: int | None,
    interactive: bool,
    video_df: pd.DataFrame | None = None,
    anchor_video_idx: int = 1,
    use_xcorr: bool = True,
    xcorr_max_cost: float = 0.35,
    xcorr_result_ref: list[Any] | None = None,
) -> int:
    if anchor_rp3_row_idx is not None and anchor_rp3_stroke_number is not None:
        raise ValueError("Provide only one of --anchor-rp3-row-idx or --anchor-rp3-stroke-number.")

    if anchor_rp3_row_idx is not None:
        idx = int(anchor_rp3_row_idx)
        if not (0 <= idx < len(rp3_df)):
            raise ValueError(f"anchor_rp3_row_idx out of range: {idx}")
        return idx

    if anchor_rp3_stroke_number is not None:
        stroke_no = int(anchor_rp3_stroke_number)
        row_idx = _find_rp3_row_by_stroke_number(rp3_df, stroke_no)
        if row_idx is None:
            raise ValueError(f"stroke_number {stroke_no} not found in RP3 CSV.")
        return int(row_idx)

    if use_xcorr and video_df is not None:
        from pair_session import coarse_sync_anchor
        try:
            result = coarse_sync_anchor(
                video_df=video_df,
                rp3_df=rp3_df,
                anchor_video_idx=int(anchor_video_idx),
            )
        except ValueError as exc:
            print(f"Cross-correlation coarse sync failed: {exc}")
        else:
            if xcorr_result_ref is not None:
                xcorr_result_ref.append(result)
            stroke_label = (
                f"stroke_number={result.anchor_rp3_stroke_number}"
                if result.anchor_rp3_stroke_number is not None
                else f"row_idx={result.anchor_rp3_idx}"
            )
            print(
                "Coarse xcorr anchor: video stroke "
                f"{anchor_video_idx} -> RP3 {stroke_label} "
                f"(offset={result.offset_strokes}, "
                f"cost={result.best_cost:.3f}s, "
                f"mean|rate diff|={result.mean_rate_diff_spm:.2f} spm, "
                f"overlap={result.overlap_length})"
            )
            if result.best_cost <= xcorr_max_cost:
                return int(result.anchor_rp3_idx)
            print(
                f"Coarse xcorr cost {result.best_cost:.3f}s exceeds threshold "
                f"{xcorr_max_cost:.3f}s; falling back to explicit anchor."
            )

    if not interactive:
        raise ValueError(
            "Missing anchor. Use --anchor-rp3-stroke-number (recommended) or --anchor-rp3-row-idx, "
            "or enable --auto-pair / cross-correlation sync."
        )

    _, finite_strokes = _stroke_number_candidates(rp3_df)
    default_stroke = int(finite_strokes[0])
    min_stroke = int(finite_strokes.min())
    max_stroke = int(finite_strokes.max())
    print(f"\nRP3 stroke_number range: {min_stroke}..{max_stroke}")
    stroke_no = _prompt_int("Enter anchor RP3 stroke_number for the first matched video stroke", default_stroke)
    row_idx = _find_rp3_row_by_stroke_number(rp3_df, stroke_no)
    if row_idx is None:
        raise ValueError(f"stroke_number {stroke_no} not found in RP3 CSV.")
    return int(row_idx)


def _build_match_manifest(
    video_df: pd.DataFrame,
    rp3_df: pd.DataFrame,
    *,
    anchor_video_idx: int,
    anchor_rp3_idx: int,
    cfg: MatchConfig,
) -> MatchResult:
    if not (0 <= anchor_video_idx < len(video_df)):
        raise ValueError(f"anchor_video_stroke_idx out of range: {anchor_video_idx}")
    if not (0 <= anchor_rp3_idx < len(rp3_df)):
        raise ValueError(f"anchor_rp3_idx out of range: {anchor_rp3_idx}")

    v = video_df.iloc[anchor_video_idx:].reset_index(drop=True).copy()
    n = len(v)
    m = len(rp3_df)
    if anchor_rp3_idx + (n - 1) >= m:
        raise ValueError("Not enough RP3 rows after anchor for one-to-one video matches.")

    rp3_cycle = rp3_df["rp3_cycle_s"].to_numpy(dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(rp3_cycle)))
    rp3_rel = prefix[:-1] - prefix[anchor_rp3_idx]

    v_catch = v["catch_time_s"].to_numpy(dtype=np.float64)
    v_finish = v["finish_time_s"].to_numpy(dtype=np.float64)
    v_drive = v["drive_duration_s"].to_numpy(dtype=np.float64)
    v_recover = v["recover_duration_s"].to_numpy(dtype=np.float64)
    v_cycle = v["cycle_duration_s"].to_numpy(dtype=np.float64)

    v_rel_catch = v_catch - v_catch[0]

    inf = float("inf")
    dp = np.full((n, m), inf, dtype=np.float64)
    parent = np.full((n, m), -1, dtype=np.int32)
    dp[0, anchor_rp3_idx] = 0.0

    for i in range(1, n):
        min_row = anchor_rp3_idx + i
        v_rel = float(v_rel_catch[i])
        v_rel_prev = float(v_rel_catch[i - 1])
        v_interval = v_rel - v_rel_prev

        for j in range(min_row, m):
            r_rel = float(rp3_rel[j])
            cum_err = abs(v_rel - r_rel)
            if cum_err > cfg.max_abs_cum_error_s:
                continue
            allowed_cum_err = cfg.max_cumulative_error_base_s + cfg.max_cumulative_error_per_s * max(0.0, v_rel)
            if cum_err > allowed_cum_err:
                continue

            obs = cfg.w_drive * abs(v_drive[i] - float(rp3_df.at[j, "drive_time"])) + cfg.w_recover * abs(
                v_recover[i] - float(rp3_df.at[j, "recover_time"])
            )

            jp_start = max(anchor_rp3_idx + (i - 1), j - cfg.max_jump_rows)
            best = inf
            bestp = -1
            for jp in range(jp_start, j):
                prev = float(dp[i - 1, jp])
                if not math.isfinite(prev):
                    continue

                r_interval = float(r_rel - rp3_rel[jp])
                interval_err = abs(v_interval - r_interval)
                if interval_err > cfg.max_interval_error_s:
                    continue

                skipped = int(j - jp - 1)
                score = (
                    prev
                    + obs
                    + cfg.w_interval * interval_err
                    + cfg.w_cumulative * cum_err
                    + cfg.w_skip * skipped
                )
                if score < best:
                    best = score
                    bestp = jp

            if bestp >= 0:
                dp[i, j] = best
                parent[i, j] = bestp

    end_scores = dp[n - 1]
    j_last = int(np.argmin(end_scores))
    total_score = float(end_scores[j_last])
    if not math.isfinite(total_score):
        raise RuntimeError("Failed to find a feasible RP3 match path. Try a different anchor or looser tolerances.")

    matched_rows = [j_last]
    for i in range(n - 1, 0, -1):
        p = int(parent[i, matched_rows[-1]])
        if p < 0:
            raise RuntimeError("Backtracking failed; incomplete match path.")
        matched_rows.append(p)
    matched_rows.reverse()

    rows: list[dict[str, Any]] = []
    for i, j in enumerate(matched_rows):
        rp3_row = rp3_df.iloc[j]
        v_row = v.iloc[i]

        v_rel = float(v_rel_catch[i])
        r_rel = float(rp3_rel[j])
        cum_err = v_rel - r_rel
        drive_err = float(v_row["drive_duration_s"] - rp3_row["drive_time"])
        recover_err = float(v_row["recover_duration_s"] - rp3_row["recover_time"])

        if i == 0:
            interval_err = 0.0
            skipped = 0
            score_step = float(dp[i, j])
            r_interval = 0.0
            v_interval = 0.0
        else:
            jp = matched_rows[i - 1]
            r_interval = float(rp3_rel[j] - rp3_rel[jp])
            v_interval = float(v_rel_catch[i] - v_rel_catch[i - 1])
            interval_err = v_interval - r_interval
            skipped = int(j - jp - 1)
            score_step = float(dp[i, j] - dp[i - 1, jp])

        rows.append(
            {
                "video_stroke_idx": int(v_row["stroke_idx"]),
                "video_catch_time_s": float(v_row["catch_time_s"]),
                "video_finish_time_s": float(v_row["finish_time_s"]),
                "video_drive_s": float(v_row["drive_duration_s"]),
                "video_recover_s": float(v_row["recover_duration_s"]),
                "video_cycle_s": float(v_row["cycle_duration_s"]),
                "video_catch_rel_s": v_rel,
                "rp3_row_idx": int(j),
                "rp3_stroke_number": int(rp3_row["stroke_number"]),
                "rp3_time_s": float(rp3_row["time"]),
                "rp3_drive_s": float(rp3_row["drive_time"]),
                "rp3_recover_s": float(rp3_row["recover_time"]),
                "rp3_cycle_s": float(rp3_row["rp3_cycle_s"]),
                "rp3_catch_rel_s_from_anchor": r_rel,
                "rp3_rows_skipped_since_prev": int(skipped),
                "cum_catch_err_s": float(cum_err),
                "interval_video_s": float(v_interval),
                "interval_rp3_s": float(r_interval),
                "interval_err_s": float(interval_err),
                "drive_err_s": float(drive_err),
                "recover_err_s": float(recover_err),
                "score_step": float(score_step),
            }
        )

    manifest = pd.DataFrame(rows)

    return MatchResult(
        manifest=manifest,
        total_score=total_score,
        matched_rp3_indices=matched_rows,
    )


def _drift_metrics(manifest: pd.DataFrame) -> dict[str, float]:
    """Compute summary drift statistics from a match manifest."""
    if manifest.empty:
        return {
            "max_abs_cum_err_s": float("nan"),
            "mean_abs_cum_err_s": float("nan"),
            "mean_abs_interval_err_s": float("nan"),
            "max_abs_interval_err_s": float("nan"),
        }
    abs_cum = manifest["cum_catch_err_s"].abs()
    abs_int = manifest["interval_err_s"].abs()
    return {
        "max_abs_cum_err_s": float(abs_cum.max()),
        "mean_abs_cum_err_s": float(abs_cum.mean()),
        "mean_abs_interval_err_s": float(abs_int.mean()),
        "max_abs_interval_err_s": float(abs_int.max()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match video drive events to RP3 clean stroke rows using a manually verified first-stroke anchor "
            "and cumulative timing with allowed RP3 row skips."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run dir with inference/drive_events.csv")
    parser.add_argument("--rp3-clean-csv", type=Path, default=None)
    parser.add_argument("--anchor-video-stroke-idx", type=int, default=1)
    parser.add_argument("--anchor-rp3-row-idx", type=int, default=None)
    parser.add_argument("--anchor-rp3-stroke-number", type=int, default=None)
    parser.add_argument("--max-jump-rows", type=int, default=10)
    parser.add_argument("--max-interval-error-s", type=float, default=2.0)
    parser.add_argument("--max-cumulative-error-base-s", type=float, default=1.5)
    parser.add_argument("--max-cumulative-error-per-s", type=float, default=0.15)
    parser.add_argument("--max-abs-cum-error-s", type=float, default=4.0,
                        help="Hard cap on absolute cumulative timing error (default: 4.0s).")
    parser.add_argument("--w-drive", type=float, default=0.4)
    parser.add_argument("--w-recover", type=float, default=0.4)
    parser.add_argument("--w-interval", type=float, default=1.0)
    parser.add_argument("--w-cumulative", type=float, default=1.0)
    parser.add_argument("--w-skip", type=float, default=0.08)
    parser.add_argument(
        "--auto-pair",
        action="store_true",
        help=(
            "Resolve rp3_clean_csv and anchor_rp3_stroke_number from "
            "session_registry.csv instead of prompting. Falls back to "
            "interactive selection if the registry has no matching row."
        ),
    )
    parser.add_argument(
        "--session-registry",
        type=Path,
        default=None,
        help="Path to session_registry.csv (default: <repo_root>/session_registry.csv).",
    )
    parser.add_argument(
        "--no-xcorr-anchor",
        dest="use_xcorr",
        action="store_false",
        default=True,
        help="Disable cross-correlation based coarse anchor inference.",
    )
    parser.add_argument(
        "--xcorr-max-cost",
        type=float,
        default=0.35,
        help=(
            "Reject cross-correlation anchor candidates whose mean |interval diff| "
            "exceeds this value in seconds (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--reject-on-drift",
        dest="reject_on_drift",
        action="store_true",
        default=True,
        help=(
            "Treat excessive cumulative drift as a hard failure (default). "
            "Manifest files are still emitted for diagnostics, but the CLI "
            "exits with a non-zero status."
        ),
    )
    parser.add_argument(
        "--no-reject-on-drift",
        dest="reject_on_drift",
        action="store_false",
        help="Demote drift hard-reject to a warning (legacy behaviour).",
    )
    parser.add_argument(
        "--drift-reject-max-cum-err-s",
        type=float,
        default=3.0,
        help=(
            "Hard-reject threshold on max |cumulative catch error| in seconds "
            "when --reject-on-drift is enabled (default: 3.0)."
        ),
    )
    parser.add_argument(
        "--drift-reject-mean-interval-err-s",
        type=float,
        default=0.5,
        help=(
            "Hard-reject threshold on mean |interval error| in seconds "
            "when --reject-on-drift is enabled (default: 0.5)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    try:
        run_dir = _resolve_run_dir(args.run_dir, args.runs_root)

        anchor_rp3_stroke_number = args.anchor_rp3_stroke_number
        rp3_clean_csv_arg = args.rp3_clean_csv
        if args.auto_pair:
            try:
                ctx = auto_pair_run(
                    run_dir=run_dir,
                    registry_path=args.session_registry or DEFAULT_SESSION_REGISTRY,
                )
            except (FileNotFoundError, LookupError) as exc:
                print(f"auto-pair failed: {exc}. Falling back to interactive mode.")
            else:
                if rp3_clean_csv_arg is None:
                    rp3_clean_csv_arg = ctx.rp3_clean_csv
                if (
                    anchor_rp3_stroke_number is None
                    and args.anchor_rp3_row_idx is None
                    and ctx.anchor_rp3_stroke_number is not None
                ):
                    anchor_rp3_stroke_number = ctx.anchor_rp3_stroke_number
                print(
                    f"auto-pair: session_id={ctx.session_id} athlete={ctx.athlete_id} "
                    f"active_side={ctx.active_side}"
                )

        rp3_clean_csv = _resolve_rp3_csv(
            run_dir=run_dir,
            rp3_clean_csv=rp3_clean_csv_arg,
            interactive=interactive,
        )
        video_df = _load_video_events(run_dir)
        rp3_df = _load_rp3(rp3_clean_csv)
        xcorr_ref: list[Any] = []
        anchor_rp3_idx = _resolve_anchor_rp3_idx(
            rp3_df,
            anchor_rp3_row_idx=args.anchor_rp3_row_idx,
            anchor_rp3_stroke_number=anchor_rp3_stroke_number,
            interactive=interactive,
            video_df=video_df,
            anchor_video_idx=int(args.anchor_video_stroke_idx),
            use_xcorr=bool(args.use_xcorr),
            xcorr_max_cost=float(args.xcorr_max_cost),
            xcorr_result_ref=xcorr_ref,
        )
        xcorr_result = xcorr_ref[0] if xcorr_ref else None
        cfg = MatchConfig(
            max_jump_rows=int(args.max_jump_rows),
            max_interval_error_s=float(args.max_interval_error_s),
            max_cumulative_error_base_s=float(args.max_cumulative_error_base_s),
            max_cumulative_error_per_s=float(args.max_cumulative_error_per_s),
            max_abs_cum_error_s=float(args.max_abs_cum_error_s),
            w_drive=float(args.w_drive),
            w_recover=float(args.w_recover),
            w_interval=float(args.w_interval),
            w_cumulative=float(args.w_cumulative),
            w_skip=float(args.w_skip),
        )
        result = _build_match_manifest(
            video_df=video_df,
            rp3_df=rp3_df,
            anchor_video_idx=int(args.anchor_video_stroke_idx),
            anchor_rp3_idx=int(anchor_rp3_idx),
            cfg=cfg,
        )
    except Exception as exc:
        print(f"Match failed: {exc}")
        return 1

    out_dir = run_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "rp3_match_manifest.csv"
    summary_json = out_dir / "rp3_match_summary.json"
    aligned_csv = out_dir / "rp3_video_aligned_strokes.csv"
    pairing_manifest_json = out_dir / "pairing_manifest.json"

    result.manifest.to_csv(manifest_csv, index=False)

    selected_rows = rp3_df.iloc[result.matched_rp3_indices].reset_index(drop=True)
    aligned = pd.concat([result.manifest, selected_rows.add_prefix("rp3_row_")], axis=1)
    aligned.to_csv(aligned_csv, index=False)

    skipped_total = int(result.manifest["rp3_rows_skipped_since_prev"].sum())
    drift = _drift_metrics(result.manifest)
    mean_abs_cum_err = drift["mean_abs_cum_err_s"]
    max_abs_cum_err = drift["max_abs_cum_err_s"]
    mean_abs_interval_err = drift["mean_abs_interval_err_s"]
    mean_abs_drive_err = float(result.manifest["drive_err_s"].abs().mean()) if not result.manifest.empty else float("nan")
    mean_abs_recover_err = float(result.manifest["recover_err_s"].abs().mean()) if not result.manifest.empty else float("nan")

    summary = {
        "run_dir": str(run_dir),
        "rp3_clean_csv": str(rp3_clean_csv),
        "anchor_video_stroke_idx": int(args.anchor_video_stroke_idx),
        "anchor_video_stroke_label": int(result.manifest.iloc[0]["video_stroke_idx"]),
        "anchor_rp3_row_idx": int(result.manifest.iloc[0]["rp3_row_idx"]),
        "anchor_rp3_stroke_number": int(result.manifest.iloc[0]["rp3_stroke_number"]),
        "matched_video_strokes": int(len(result.manifest)),
        "matched_rp3_rows": int(len(result.manifest)),
        "total_skipped_rp3_rows": skipped_total,
        "total_score": float(result.total_score),
        "mean_abs_cum_catch_err_s": mean_abs_cum_err,
        "max_abs_cum_catch_err_s": max_abs_cum_err,
        "mean_abs_interval_err_s": mean_abs_interval_err,
        "mean_abs_drive_err_s": mean_abs_drive_err,
        "mean_abs_recover_err_s": mean_abs_recover_err,
        "config": asdict(cfg),
        "outputs": {
            "manifest_csv": str(manifest_csv),
            "aligned_csv": str(aligned_csv),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    # ------------------------------------------------------------------
    # Drift gate + pairing manifest
    # ------------------------------------------------------------------
    drift_reject_reasons: list[str] = []
    cum_threshold = float(args.drift_reject_max_cum_err_s)
    int_threshold = float(args.drift_reject_mean_interval_err_s)
    if np.isfinite(max_abs_cum_err) and max_abs_cum_err > cum_threshold:
        drift_reject_reasons.append(
            f"max |cum_catch_err|={max_abs_cum_err:.3f}s > {cum_threshold:.3f}s"
        )
    if np.isfinite(mean_abs_interval_err) and mean_abs_interval_err > int_threshold:
        drift_reject_reasons.append(
            f"mean |interval_err|={mean_abs_interval_err:.3f}s > {int_threshold:.3f}s"
        )

    pairing_ctx: dict[str, Any] | None = None
    if args.auto_pair:
        try:
            ctx = auto_pair_run(
                run_dir=run_dir,
                registry_path=args.session_registry or DEFAULT_SESSION_REGISTRY,
            )
            pairing_ctx = ctx.to_dict()
        except (FileNotFoundError, LookupError):
            pairing_ctx = None

    accepted_ranges: list[list[int]] = []
    if not result.manifest.empty:
        stroke_ids = result.manifest["video_stroke_idx"].astype(int).tolist()
        if stroke_ids:
            start = stroke_ids[0]
            prev = start
            for s in stroke_ids[1:]:
                if s == prev + 1:
                    prev = s
                    continue
                accepted_ranges.append([int(start), int(prev)])
                start = s
                prev = s
            accepted_ranges.append([int(start), int(prev)])

    pairing_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "session_id": pairing_ctx.get("session_id") if pairing_ctx else None,
        "athlete_id": pairing_ctx.get("athlete_id") if pairing_ctx else None,
        "active_side": pairing_ctx.get("active_side") if pairing_ctx else None,
        "rp3_clean_csv": str(rp3_clean_csv),
        "anchor": {
            "video_stroke_idx": int(args.anchor_video_stroke_idx),
            "video_stroke_label": int(result.manifest.iloc[0]["video_stroke_idx"]),
            "rp3_row_idx": int(result.manifest.iloc[0]["rp3_row_idx"]),
            "rp3_stroke_number": int(result.manifest.iloc[0]["rp3_stroke_number"]),
            "source": (
                "coarse_xcorr"
                if (xcorr_result is not None and args.anchor_rp3_row_idx is None and args.anchor_rp3_stroke_number is None)
                else "explicit"
            ),
        },
        "coarse_xcorr": (xcorr_result.to_dict() if xcorr_result is not None else None),
        "drift_metrics": {
            "mean_abs_cum_catch_err_s": mean_abs_cum_err,
            "max_abs_cum_catch_err_s": max_abs_cum_err,
            "mean_abs_interval_err_s": mean_abs_interval_err,
            "max_abs_interval_err_s": drift["max_abs_interval_err_s"],
        },
        "drift_gate": {
            "reject_on_drift": bool(args.reject_on_drift),
            "max_cum_err_threshold_s": cum_threshold,
            "mean_interval_err_threshold_s": int_threshold,
            "rejected": bool(drift_reject_reasons and args.reject_on_drift),
            "reasons": drift_reject_reasons,
        },
        "accepted_video_stroke_ranges": accepted_ranges,
        "qc_summary": {
            "matched_video_strokes": int(len(result.manifest)),
            "total_skipped_rp3_rows": skipped_total,
        },
        "outputs": {
            "manifest_csv": str(manifest_csv),
            "aligned_csv": str(aligned_csv),
            "summary_json": str(summary_json),
        },
    }

    with pairing_manifest_json.open("w", encoding="utf-8") as f:
        json.dump(pairing_manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Run: {run_dir.name}")
    print(f"RP3 CSV: {rp3_clean_csv.name}")
    print(
        "Anchor: "
        f"video stroke {summary['anchor_video_stroke_label']} -> "
        f"RP3 stroke {summary['anchor_rp3_stroke_number']} (row {summary['anchor_rp3_row_idx']})"
    )
    print(
        "Match quality: "
        f"mean |cum|={mean_abs_cum_err:.3f}s, max |cum|={max_abs_cum_err:.3f}s, "
        f"mean |interval|={mean_abs_interval_err:.3f}s, "
        f"skipped RP3 rows={skipped_total}"
    )
    print("Outputs:")
    print(f"  {manifest_csv}")
    print(f"  {aligned_csv}")
    print(f"  {summary_json}")
    print(f"  {pairing_manifest_json}")

    if drift_reject_reasons:
        if args.reject_on_drift:
            print("ERROR: pairing rejected due to excessive drift:")
            for reason in drift_reject_reasons:
                print(f"  - {reason}")
            print("  (bypass with --no-reject-on-drift if you know what you're doing)")
            return 2
        print("WARNING: drift exceeds hard-reject thresholds (not enforced):")
        for reason in drift_reject_reasons:
            print(f"  - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
