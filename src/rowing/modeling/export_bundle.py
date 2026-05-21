#!/usr/bin/env python3
"""Export a trained model bundle for use with predict_force_cli.py.

Combines a training_dataset directory (PCA/fPCA + feature names + s_grid)
with a modeling_results directory (Stage A best model + Stage B TCN
state and normalization stats) into a single self-describing
``model_bundle/`` directory.

Example
-------

    .venv/bin/python inference/export_model_bundle.py \
      --dataset-dir runs/<run>/inference/training_dataset \
      --modeling-dir runs/<run>/inference/modeling_results \
      --bundle-dir runs/<run>/inference/model_bundle \
      --active-side-default right
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rowing.modeling.bundle import write_model_bundle


def _detect_git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--modeling-dir", type=Path, required=True)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument(
        "--active-side-default",
        choices=["left", "right"],
        default=None,
        help="Default active_side recorded in manifest.json (optional).",
    )
    p.add_argument(
        "--target-representation",
        choices=["standard", "fpca"],
        default="standard",
        help="Which target decomposition is packaged (default: standard).",
    )
    p.add_argument(
        "--no-head",
        action="store_true",
        help="Record that the training contract excluded head_vs_trunk_deg.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    git_sha = _detect_git_sha()
    out = write_model_bundle(
        bundle_dir=args.bundle_dir,
        dataset_dir=args.dataset_dir,
        modeling_dir=args.modeling_dir,
        active_side_default=args.active_side_default,
        include_head=not args.no_head,
        target_representation=args.target_representation,
        git_sha=git_sha,
    )
    print(f"Wrote model bundle to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
