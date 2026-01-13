from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


def _open_capture(video_path: str | Path) -> cv2.VideoCapture:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    cap = _open_capture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video metadata (width/height not detected).")
        return VideoMetadata(
            path=Path(video_path),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
    finally:
        cap.release()


def read_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Random access a specific frame (BGR uint8)."""
    if frame_idx < 0:
        raise ValueError("frame_idx must be >= 0")
    cap = _open_capture(video_path)
    try:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        if not ok:
            raise RuntimeError(f"Failed to seek to frame {frame_idx}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return frame
    finally:
        cap.release()


def iter_frames(
    video_path: str | Path,
    start: int = 0,
    end: Optional[int] = None,
    stride: int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Iterate frames sequentially (BGR uint8). Yields (frame_idx, frame)."""
    if start < 0:
        raise ValueError("start must be >= 0")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    cap = _open_capture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        i = start
        while True:
            if end is not None and i >= end:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield i, frame
            # advance by stride-1 additional frames
            if stride > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(i + stride))
            i += stride
    finally:
        cap.release()


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v",
        is_color: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))  # (W,H)
        self.fourcc = str(fourcc)
        self.is_color = bool(is_color)
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.fps if self.fps > 0 else 30.0,
            self.frame_size,
            self.is_color,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.path}")
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter is not open (use as a context manager).")
        self._writer.write(frame_bgr)

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._writer is not None:
            self._writer.release()
            self._writer = None
