#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/build_training_dataset.py`."""
from __future__ import annotations

from rowing.dataset.build import main

if __name__ == "__main__":
    raise SystemExit(main())
