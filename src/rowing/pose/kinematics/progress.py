from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class ProgressHandle(Protocol):
    def update(self, n: int = 1, *, desc: Optional[str] = None, total: Optional[int] = None) -> None:
        ...

    def close(self, *, status: Optional[str] = None) -> None:
        ...


class ProgressReporter(Protocol):
    def start(self, label: str, total: Optional[int] = None, unit: str = "it") -> ProgressHandle:
        ...


@dataclass
class _NullProgressHandle:
    def update(self, n: int = 1, *, desc: Optional[str] = None, total: Optional[int] = None) -> None:
        return

    def close(self, *, status: Optional[str] = None) -> None:
        return


@dataclass
class _NullProgress:
    def start(self, label: str, total: Optional[int] = None, unit: str = "it") -> ProgressHandle:
        return _NullProgressHandle()


def get_progress(progress: Optional[ProgressReporter]) -> ProgressReporter:
    return progress if progress is not None else _NullProgress()


class TqdmProgressHandle:
    def __init__(self, bar: "tqdm") -> None:
        self._bar = bar

    def update(self, n: int = 1, *, desc: Optional[str] = None, total: Optional[int] = None) -> None:
        if desc:
            self._bar.set_description_str(desc)
        if total is not None:
            self._bar.total = total
            self._bar.refresh()
        self._bar.update(n)

    def close(self, *, status: Optional[str] = None) -> None:
        if status:
            self._bar.set_postfix_str(status)
        self._bar.close()


class TqdmProgress:
    def __init__(self, *, leave: bool = False) -> None:
        self._leave = bool(leave)

    def start(self, label: str, total: Optional[int] = None, unit: str = "it") -> ProgressHandle:
        from tqdm import tqdm

        bar = tqdm(total=total, desc=label, unit=unit, leave=self._leave, dynamic_ncols=True)
        return TqdmProgressHandle(bar)
