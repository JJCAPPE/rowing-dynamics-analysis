from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Annotations, RunConfig, VideoInfo, compute_m_per_px, save_run_config
from .io_video import get_video_metadata, read_frame


def _draw_help(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 28
    return out


def _pick_point(frame_bgr: np.ndarray, window: str, prompt: str) -> Tuple[float, float]:
    clicked: list[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, userdata):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((int(x), int(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        disp = frame_bgr.copy()
        disp = _draw_help(disp, [prompt, "LMB: set point | ENTER: confirm | R: reset | ESC: cancel"])
        if clicked:
            x, y = clicked[0]
            cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
            cv2.circle(disp, (x, y), 12, (0, 0, 0), 2)
        cv2.imshow(window, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # Enter
            if clicked:
                x, y = clicked[0]
                cv2.destroyWindow(window)
                return float(x), float(y)
        elif k in (ord("r"), ord("R")):
            clicked.clear()
        elif k == 27:  # Esc
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Annotation cancelled.")


def _select_bbox(
    frame_bgr: np.ndarray, window: str, prompt_lines: Optional[list[str]] = None
) -> Tuple[float, float, float, float]:
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if prompt_lines is None:
        prompt_lines = [
            "Draw a loose athlete bbox, then press ENTER/SPACE to confirm.",
            "Press C to cancel selection.",
        ]
    disp = _draw_help(frame_bgr, prompt_lines)
    x, y, w, h = cv2.selectROI(window, disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window)
    if w <= 1 or h <= 1:
        raise RuntimeError("BBox selection cancelled or invalid (too small).")
    return float(x), float(y), float(w), float(h)


def _prompt_float(prompt: str) -> float:
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
        except ValueError:
            print("Invalid number. Try again.")
            continue
        return v


def _relpath_str(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except Exception:
        return str(path)


def annotate_video(video_path: Path, out_dir: Path, reference_frame_idx: int = 0) -> None:
    """Interactive annotation UX for a single video.

    Creates/overwrites `out_dir/run.json` containing:
    - boat anchor point
    - rigger bbox (for stabilization)
    - initial athlete crop bbox
    - 2-point boat scale + known distance

    This is intentionally simple OpenCV UX so it can be reused later (e.g. in a GUI).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_video_metadata(video_path)
    frame0 = read_frame(video_path, reference_frame_idx)

    try:
        anchor = _pick_point(frame0, "Annotate: anchor", "Click boat anchor point (e.g., oarlock).")
        rigger_bbox = _select_bbox(
            frame0,
            "Annotate: rigger bbox",
            [
                "Draw a tight bbox around the rigger/oarlock hardware.",
                "Press ENTER/SPACE to confirm. Press C to cancel.",
            ],
        )
        bbox = _select_bbox(frame0, "Annotate: athlete bbox")
        s0 = _pick_point(frame0, "Annotate: scale #1", "Click boat scale point #1.")
        s1 = _pick_point(frame0, "Annotate: scale #2", "Click boat scale point #2.")
    finally:
        cv2.destroyAllWindows()

    dist_m = _prompt_float("Known distance between scale points (meters): ")
    m_per_px = compute_m_per_px((s0, s1), dist_m)

    cfg = RunConfig(
        version=1,
        video=VideoInfo(
            path=_relpath_str(Path(video_path), out_dir),
            fps=float(meta.fps),
            width=int(meta.width),
            height=int(meta.height),
            frame_count=int(meta.frame_count),
        ),
        reference_frame_idx=int(reference_frame_idx),
        annotations=Annotations(
            anchor_px=anchor,
            bbox_px=bbox,
            rigger_bbox_px=rigger_bbox,
            scale_points_px=(s0, s1),
            scale_distance_m=float(dist_m),
        ),
        derived={"m_per_px": float(m_per_px)},
        params={
            "stabilization": {
                "lk_win_size": [21, 21],
                "lk_max_level": 3,
                "lk_max_corners": 1,
                "template_half_size": 16,
                "template_search_radius": 50,
                "ema_alpha": 0.8,
                "min_points": 10,
            },
            "crop": {
                "padding": 0.2,
                "ema_alpha": 0.9,
                "min_points": 10,
            },
        },
    )

    save_run_config(out_dir / "run.json", cfg)
    print(f"Wrote {out_dir / 'run.json'}")


