from __future__ import annotations

from typing import Callable, Optional


ProgressCallback = Callable[[str, float], None]


def clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def emit_progress(
    callback: Optional[ProgressCallback],
    label: str,
    progress: float,
) -> None:
    if callback is None:
        return
    callback(label, clamp01(progress))


class ProgressMux:
    """Map local sub-stage progress into a shared global callback."""

    def __init__(
        self,
        callback: Optional[ProgressCallback],
        *,
        epsilon: float = 1e-3,
    ) -> None:
        self._callback = callback
        self._epsilon = float(max(0.0, epsilon))
        self._last_progress = 0.0
        self._last_label = ""

    def emit(self, label: str, progress: float) -> None:
        if self._callback is None:
            return
        p = clamp01(progress)
        if p < self._last_progress:
            p = self._last_progress
        if label != self._last_label or (p - self._last_progress) >= self._epsilon:
            self._callback(label, p)
            self._last_label = label
            self._last_progress = p

    def span(
        self,
        start: float,
        end: float,
        *,
        prefix: str = "",
    ) -> ProgressCallback:
        s = clamp01(start)
        e = clamp01(end)
        if e < s:
            s, e = e, s

        def _cb(label: str, progress: float) -> None:
            local = clamp01(progress)
            mapped = s + (e - s) * local
            text = (label or "").strip()
            if prefix:
                text = f"{prefix} | {text}" if text else prefix
            self.emit(text, mapped)

        return _cb
