#!/usr/bin/env python3
"""Backwards-compatible shim for the old `inference/inference_cli.py` entry point.

Forwards to :mod:`rowing.cli.inference` so existing scripts and notebooks keep
working. Prefer `python -m rowing` (or the `rowing` console script) for new
workflows.
"""
from __future__ import annotations

from rowing.cli.inference import main

if __name__ == "__main__":
    raise SystemExit(main())
