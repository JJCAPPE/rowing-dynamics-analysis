#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/predict_force_cli.py`."""
from __future__ import annotations

from rowing.modeling.predict import main

if __name__ == "__main__":
    raise SystemExit(main())
