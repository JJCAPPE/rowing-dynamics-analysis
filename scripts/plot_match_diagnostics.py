#!/usr/bin/env python3
"""Backwards-compatible shim for `inference/plot_match_diagnostics.py`.

The viewer is replaced by the interactive editor at `rowing.matching.editor`
(Phase 4); this shim continues to expose the read-only diagnostic view.
"""
from __future__ import annotations

from rowing.matching.diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())
