#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/match_rp3_cli.py`."""
from __future__ import annotations

from rowing.matching.match import main

if __name__ == "__main__":
    raise SystemExit(main())
