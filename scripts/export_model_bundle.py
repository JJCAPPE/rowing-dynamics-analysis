#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/export_model_bundle.py`."""
from __future__ import annotations

from rowing.modeling.export_bundle import main

if __name__ == "__main__":
    raise SystemExit(main())
