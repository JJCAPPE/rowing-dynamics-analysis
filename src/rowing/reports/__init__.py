"""HTML report generation for runs and training results.

Phase 5+ of the unified-CLI plan. Each report is a self-contained directory
on disk:

- Per-run report:    ``<run>/inference/report/index.html``
- Per-training report (Phase 6): ``runs/_training/<ts>/report/index.html``

Reports embed PNGs rendered via the matplotlib ``Agg`` backend and Jinja2
templates from :mod:`rowing.reports.templates`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only re-export
    from rowing.reports.run_report import generate_run_report  # noqa: F401
    from rowing.reports.training_report import generate_training_report  # noqa: F401

__all__ = ["generate_run_report", "generate_training_report"]


def __getattr__(name: str):  # PEP 562 lazy attribute lookup
    if name == "generate_run_report":
        from rowing.reports.run_report import generate_run_report

        return generate_run_report
    if name == "generate_training_report":
        from rowing.reports.training_report import generate_training_report

        return generate_training_report
    raise AttributeError(name)
