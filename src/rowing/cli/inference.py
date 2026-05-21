#!/usr/bin/env python3
"""Argparse shell for the inference pipeline.

The heavy lifting lives in :mod:`rowing.cli.pipeline`; this module is a thin
``argparse`` adapter so the legacy CLI behaviour (flag names, help text, exit
codes, on-disk artifacts, console output) stays byte-for-byte identical with
the pre-refactor ``inference/inference_cli.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rowing import RUNS_DIR as DEFAULT_RUNS_ROOT
from rowing.cli.pipeline import (
    PipelineOptions,
    options_from_argparse,
    run_inference,
)
from rowing.matching.detect import (
    FINISH_METHOD_VELOCITY_CALIBRATED,
    VALID_FINISH_METHODS,
)


__all__ = ["DEFAULT_RUNS_ROOT", "build_argparser", "parse_args", "main"]


def build_argparser() -> argparse.ArgumentParser:
    """Construct the inference CLI argparser (extracted for re-use by the TUI)."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute drive event timings from stroke_signal.csv using only handle distance "
            "(catch=minima, finish=maxima) for downstream RP3 force-curve pairing."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Sports2D runs root (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory to process directly (skip interactive selection).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/inference).",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=None,
        help=(
            "Smoothing window (seconds) for relative distance signal. "
            "Default: 0.04 for velocity_calibrated, 0.08 otherwise."
        ),
    )
    parser.add_argument(
        "--min-cycle-s",
        type=float,
        default=0.8,
        help="Minimum time between consecutive catches (default: 0.8).",
    )
    parser.add_argument(
        "--min-drive-s",
        type=float,
        default=0.2,
        help="Minimum drive duration from catch to finish (default: 0.2).",
    )
    parser.add_argument(
        "--min-recover-s",
        type=float,
        default=0.2,
        help="Minimum recovery duration from finish to next catch (default: 0.2).",
    )
    parser.add_argument(
        "--min-drive-disp-frac",
        type=float,
        default=0.05,
        help="Minimum drive displacement as fraction of signal span (default: 0.05).",
    )
    parser.add_argument(
        "--slope-tol-frac",
        type=float,
        default=0.05,
        help="Flat-slope tolerance as fraction of slope std (default: 0.05).",
    )
    parser.add_argument(
        "--overlay-opacity",
        type=float,
        default=0.10,
        help="Opacity for full-frame drive overlay video (default: 0.10).",
    )
    parser.add_argument(
        "--overlay-video",
        action="store_true",
        help="Force writing the drive-phase overlay video.",
    )
    parser.add_argument(
        "--no-overlay-video",
        action="store_true",
        help="Skip writing the drive-phase overlay video.",
    )
    parser.add_argument(
        "--match-rp3",
        action="store_true",
        help="Build RP3 stroke matches and export per-2.2cm force/pose segment CSV.",
    )
    parser.add_argument(
        "--no-match-rp3",
        action="store_true",
        help="Skip RP3 matching and segment CSV export.",
    )
    parser.add_argument(
        "--rp3-dirty-csv",
        type=Path,
        default=None,
        help="Optional RP3 dirty CSV in <run>/rp3 to clean and use for matching.",
    )
    parser.add_argument(
        "--anchor-video-stroke-idx",
        type=int,
        default=1,
        help="Video stroke index to anchor matching from (default: 1).",
    )
    parser.add_argument(
        "--anchor-rp3-row-idx",
        type=int,
        default=None,
        help="RP3 row index anchor for the anchor video stroke.",
    )
    parser.add_argument(
        "--anchor-rp3-stroke-number",
        type=int,
        default=None,
        help="RP3 stroke_number anchor for the anchor video stroke (recommended).",
    )
    parser.add_argument(
        "--active-side",
        type=str,
        default=None,
        choices=["left", "right"],
        help="Active side to export canonical one-side features from.",
    )
    parser.add_argument(
        "--finish-velocity-frac",
        type=float,
        default=None,
        help=(
            "Finish velocity threshold as fraction of peak drive velocity. "
            "Default: 0.75 for velocity_calibrated, 0.85 for velocity_threshold."
        ),
    )
    parser.add_argument(
        "--catch-velocity-frac",
        type=float,
        default=None,
        help=(
            "Catch velocity threshold as fraction of peak drive velocity. "
            "Only used with velocity_calibrated. Default: 0.43 (or auto-calibrated from RP3)."
        ),
    )
    parser.add_argument(
        "--finish-method",
        type=str,
        default=FINISH_METHOD_VELOCITY_CALIBRATED,
        choices=sorted(VALID_FINISH_METHODS),
        help=f"Finish detection method (default: {FINISH_METHOD_VELOCITY_CALIBRATED}).",
    )
    parser.add_argument(
        "--use-rp3-finish",
        action="store_true",
        default=True,
        help="Override video finish with catch + rp3_drive_s for segment export (default: enabled).",
    )
    parser.add_argument(
        "--no-use-rp3-finish",
        action="store_true",
        help="Disable RP3 finish override for segment export.",
    )
    parser.add_argument("--max-jump-rows", type=int, default=10, help="Max RP3 row jump between matched strokes.")
    parser.add_argument("--max-interval-error-s", type=float, default=2.0)
    parser.add_argument("--max-cumulative-error-base-s", type=float, default=1.5)
    parser.add_argument("--max-cumulative-error-per-s", type=float, default=0.15)
    parser.add_argument(
        "--max-abs-cum-error-s",
        type=float,
        default=4.0,
        help="Hard cap on absolute cumulative timing error (default: 4.0s).",
    )
    parser.add_argument("--w-drive", type=float, default=0.4)
    parser.add_argument("--w-recover", type=float, default=0.4)
    parser.add_argument("--w-interval", type=float, default=1.0)
    parser.add_argument("--w-cumulative", type=float, default=1.0)
    parser.add_argument("--w-skip", type=float, default=0.08)
    parser.add_argument(
        "--rower-facing",
        type=str,
        default="auto",
        choices=["auto", "left", "right"],
        help=(
            "Canonical rower facing direction in the image. "
            "'auto' detects from catch posture (default). "
            "'right'/'left' forces the convention."
        ),
    )
    parser.add_argument(
        "--include-second-derivatives",
        action="store_true",
        default=False,
        help="Include d2theta/ds2 columns in segment export (ablation-gated, default off).",
    )
    parser.add_argument(
        "--no-build-dataset",
        action="store_true",
        default=False,
        help="Skip training dataset build after segment export.",
    )
    parser.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for training dataset artifacts (default: <output-dir>/training_dataset/).",
    )
    parser.add_argument(
        "--dataset-qc-mode",
        choices=["soft", "hard"],
        default="soft",
        help="QC mode for dataset builder: soft (default) or hard.",
    )
    parser.add_argument(
        "--dataset-n-grid",
        type=int,
        default=64,
        help="Fixed-grid size for resampled force/kinematic sequences (default: 64).",
    )
    parser.add_argument(
        "--dataset-n-pca-components",
        type=int,
        default=20,
        help="Max PCA components for force curve shape decomposition (default: 20).",
    )
    parser.add_argument(
        "--dataset-force-col",
        choices=["force_raw", "force_n"],
        default="force_raw",
        help="Force column used for target representations (default: force_raw).",
    )
    parser.add_argument(
        "--dataset-onset-frac",
        type=float,
        default=0.15,
        help="Onset threshold fraction for phase-lag coordination features (default: 0.15).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argparser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options: PipelineOptions = options_from_argparse(args)
    result = run_inference(options)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
