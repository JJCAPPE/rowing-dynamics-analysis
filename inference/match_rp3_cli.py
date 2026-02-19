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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "sports2d_app" / "runs"
DEFAULT_RP3_CLEAN_DIR = REPO_ROOT / "rp3-extraction" / "workouts" / "clean"


@dataclass(frozen=True)
class MatchConfig:
    max_jump_rows: int
    max_interval_error_s: float
    max_cumulative_error_base_s: float
    max_cumulative_error_per_s: float
    w_drive: float
    w_recover: float
    w_interval: float
    w_cumulative: float
    w_skip: float


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


def _discover_rp3_clean(clean_dir: Path) -> list[Path]:
    if not clean_dir.exists():
        raise FileNotFoundError(f"RP3 clean directory not found: {clean_dir}")
    files = sorted([p.resolve() for p in clean_dir.glob("*.csv") if p.is_file()])
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


def _resolve_rp3_csv(rp3_clean_csv: Path | None, clean_dir: Path) -> Path:
    if rp3_clean_csv is not None:
        csv_path = rp3_clean_csv.expanduser().resolve()
        if not csv_path.exists() or not csv_path.is_file():
            raise FileNotFoundError(f"RP3 clean CSV not found: {csv_path}")
        return csv_path
    options = _discover_rp3_clean(clean_dir)
    if not options:
        raise FileNotFoundError(f"No RP3 clean CSV files found in {clean_dir}")
    return _pick_path(options, "Select RP3 clean CSV")


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


def _resolve_anchor_rp3_idx(
    rp3_df: pd.DataFrame,
    *,
    anchor_rp3_row_idx: int | None,
    anchor_rp3_stroke_number: int | None,
    interactive: bool,
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
        rows = rp3_df.index[rp3_df["stroke_number"].astype(int) == stroke_no].to_numpy()
        if rows.size == 0:
            raise ValueError(f"stroke_number {stroke_no} not found in RP3 CSV.")
        return int(rows[0])

    if not interactive:
        raise ValueError(
            "Missing anchor. Use --anchor-rp3-stroke-number (recommended) or --anchor-rp3-row-idx."
        )

    min_stroke = int(rp3_df["stroke_number"].min())
    max_stroke = int(rp3_df["stroke_number"].max())
    print(f"\nRP3 stroke_number range: {min_stroke}..{max_stroke}")
    stroke_no = _prompt_int("Enter anchor RP3 stroke_number for the first matched video stroke", min_stroke)
    rows = rp3_df.index[rp3_df["stroke_number"].astype(int) == stroke_no].to_numpy()
    if rows.size == 0:
        raise ValueError(f"stroke_number {stroke_no} not found in RP3 CSV.")
    return int(rows[0])


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match video drive events to RP3 clean stroke rows using a manually verified first-stroke anchor "
            "and cumulative timing with allowed RP3 row skips."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run dir with inference/drive_events.csv")
    parser.add_argument("--rp3-clean-dir", type=Path, default=DEFAULT_RP3_CLEAN_DIR)
    parser.add_argument("--rp3-clean-csv", type=Path, default=None)
    parser.add_argument("--anchor-video-stroke-idx", type=int, default=0)
    parser.add_argument("--anchor-rp3-row-idx", type=int, default=None)
    parser.add_argument("--anchor-rp3-stroke-number", type=int, default=None)
    parser.add_argument("--max-jump-rows", type=int, default=6)
    parser.add_argument("--max-interval-error-s", type=float, default=1.2)
    parser.add_argument("--max-cumulative-error-base-s", type=float, default=0.8)
    parser.add_argument("--max-cumulative-error-per-s", type=float, default=0.08)
    parser.add_argument("--w-drive", type=float, default=0.4)
    parser.add_argument("--w-recover", type=float, default=0.4)
    parser.add_argument("--w-interval", type=float, default=1.0)
    parser.add_argument("--w-cumulative", type=float, default=1.0)
    parser.add_argument("--w-skip", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    try:
        run_dir = _resolve_run_dir(args.run_dir, args.runs_root)
        rp3_clean_csv = _resolve_rp3_csv(args.rp3_clean_csv, args.rp3_clean_dir)
        video_df = _load_video_events(run_dir)
        rp3_df = _load_rp3(rp3_clean_csv)
        anchor_rp3_idx = _resolve_anchor_rp3_idx(
            rp3_df,
            anchor_rp3_row_idx=args.anchor_rp3_row_idx,
            anchor_rp3_stroke_number=args.anchor_rp3_stroke_number,
            interactive=interactive,
        )
        cfg = MatchConfig(
            max_jump_rows=int(args.max_jump_rows),
            max_interval_error_s=float(args.max_interval_error_s),
            max_cumulative_error_base_s=float(args.max_cumulative_error_base_s),
            max_cumulative_error_per_s=float(args.max_cumulative_error_per_s),
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

    result.manifest.to_csv(manifest_csv, index=False)

    selected_rows = rp3_df.iloc[result.matched_rp3_indices].reset_index(drop=True)
    aligned = pd.concat([result.manifest, selected_rows.add_prefix("rp3_row_")], axis=1)
    aligned.to_csv(aligned_csv, index=False)

    skipped_total = int(result.manifest["rp3_rows_skipped_since_prev"].sum())
    mean_abs_cum_err = float(result.manifest["cum_catch_err_s"].abs().mean())
    mean_abs_interval_err = float(result.manifest["interval_err_s"].abs().mean())
    mean_abs_drive_err = float(result.manifest["drive_err_s"].abs().mean())
    mean_abs_recover_err = float(result.manifest["recover_err_s"].abs().mean())

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

    print(f"Run: {run_dir.name}")
    print(f"RP3 CSV: {rp3_clean_csv.name}")
    print(
        "Anchor: "
        f"video stroke {summary['anchor_video_stroke_label']} -> "
        f"RP3 stroke {summary['anchor_rp3_stroke_number']} (row {summary['anchor_rp3_row_idx']})"
    )
    print(
        "Match quality: "
        f"mean |cum|={mean_abs_cum_err:.3f}s, "
        f"mean |interval|={mean_abs_interval_err:.3f}s, "
        f"skipped RP3 rows={skipped_total}"
    )
    print("Outputs:")
    print(f"  {manifest_csv}")
    print(f"  {aligned_csv}")
    print(f"  {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
