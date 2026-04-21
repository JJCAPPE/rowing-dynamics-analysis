"""Ensure the ``inference/`` directory is importable as a flat package during tests."""
from __future__ import annotations

import sys
from pathlib import Path

_INFERENCE_DIR = Path(__file__).resolve().parents[1]
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))
