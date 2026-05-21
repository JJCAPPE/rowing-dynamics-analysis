#!/usr/bin/env python3
"""Backwards-compatible shim for `sports2d_app/app_cli.py` (pose extraction wizard)."""
from __future__ import annotations

from rowing.cli.pose import main

if __name__ == "__main__":
    raise SystemExit(main())
