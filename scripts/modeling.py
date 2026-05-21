#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/modeling.py`."""
from __future__ import annotations

from rowing.modeling.train import main

if __name__ == "__main__":
    raise SystemExit(main())
