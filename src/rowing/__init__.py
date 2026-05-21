"""Rowing video analysis pipeline package.

Top-level package providing pose extraction, RP3 matching, dataset construction,
modeling, and unified CLI entry points for rowing biomechanics research.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
SOURCE_VIDEOS_DIR = REPO_ROOT / "source-videos"
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"

__all__ = ["REPO_ROOT", "RUNS_DIR", "SOURCE_VIDEOS_DIR", "TRAINED_MODELS_DIR"]
