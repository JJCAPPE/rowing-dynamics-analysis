"""Entry point for `python -m rowing` (delegates to :mod:`rowing.cli.__main__`)."""
from __future__ import annotations

from rowing.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
