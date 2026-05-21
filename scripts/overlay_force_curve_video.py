#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/overlay_force_curve_video.py`."""
from __future__ import annotations

from rowing.cli.overlay import main

if __name__ == "__main__":
    raise SystemExit(main())
