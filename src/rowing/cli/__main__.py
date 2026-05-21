"""Entry point for ``python -m rowing`` and the ``rowing`` console script.

Drives the top-level Rich menu by default. Pass ``--help`` for a brief
overview of which subsystems the menu wires into.
"""
from __future__ import annotations

from rowing.cli.menu import main


if __name__ == "__main__":
    raise SystemExit(main())
