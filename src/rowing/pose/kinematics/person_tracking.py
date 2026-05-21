from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .io_video import VideoWriter, iter_frames
from .stabilize import warp_frame
from .viz import draw_bbox
from .progress import ProgressReporter, get_progress

BBoxXYWH = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonTrackResult:
    boxes_xywh: np.ndarray  # (T,4) float32 in stabilized coords
    status: np.ndarray  # (T,) uint8 (1=tracked, 0=held)
    track_id: Optional[int]


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-6 else 0.0


def track_person_deepsort(
    video_path: Path,
    stabilization_npz: Path,
    bbox0_px: BBoxXYWH,
    out_npz: Path,
    reference_frame_idx: int,
    device: str = "cpu",
    model_name: str = "yolov8n.pt",
    min_conf: float = 0.25,
    debug_video_path: Optional[Path] = None,
    progress: Optional[ProgressReporter] = None,
) -> PersonTrackResult:
    """Track a single person using detector + DeepSORT on stabilized frames."""

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing ultralytics. Install `ultralytics` to enable DeepSORT tracking.") from e

    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing deep_sort_realtime. Install `deep_sort_realtime` to enable DeepSORT tracking."
        ) from e

    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)
    width = int(stab["width"])
    height = int(stab["height"])
    fps = float(stab["fps"]) if "fps" in stab.files else 0.0
    T = int(A.shape[0])
    prog = get_progress(progress)

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    try:
        model.to(device)
    except Exception:
        pass
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    boxes_xywh = np.full((T, 4), np.nan, dtype=np.float32)
    status = np.zeros((T,), dtype=np.uint8)

    x0, y0, w0, h0 = [float(v) for v in bbox0_px]
    target_bbox0 = np.array([x0, y0, x0 + w0, y0 + h0], dtype=np.float32)
    target_id: Optional[int] = None
    last_bbox = None

    writer = None
    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(debug_video_path, fps=fps, frame_size=(width, height))
        writer.__enter__()

    track_stage = prog.start("Stage D: person tracking (DeepSORT)", total=T, unit="frame")
    try:
        for idx, frame in iter_frames(video_path):
            stab_bgr = warp_frame(frame, A[idx], (width, height))

            results = model(stab_bgr, verbose=False)[0]
            detections = []
            if getattr(results, "boxes", None) is not None:
                boxes = results.boxes
                for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    if int(cls) != 0:
                        continue
                    c = float(conf)
                    if c < float(min_conf):
                        continue
                    detections.append([xyxy.cpu().numpy().tolist(), c, "person"])

            tracks = tracker.update_tracks(detections, frame=stab_bgr)

            selected_bbox = None
            selected_id = None
            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 0:
                    continue
                xyxy = np.array(t.to_ltrb(), dtype=np.float32)
                if target_id is None:
                    iou = _bbox_iou_xyxy(xyxy, target_bbox0)
                    if iou >= 0.2:
                        selected_bbox = xyxy
                        selected_id = int(t.track_id)
                        break
                else:
                    if int(t.track_id) == target_id:
                        selected_bbox = xyxy
                        selected_id = int(t.track_id)
                        break

            if selected_bbox is None and target_id is not None and last_bbox is not None:
                best_iou = 0.0
                for t in tracks:
                    if not t.is_confirmed() or t.time_since_update > 0:
                        continue
                    xyxy = np.array(t.to_ltrb(), dtype=np.float32)
                    iou = _bbox_iou_xyxy(xyxy, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        selected_bbox = xyxy
                        selected_id = int(t.track_id)

            if selected_bbox is not None:
                if target_id is None:
                    target_id = selected_id
                x0, y0, x1, y1 = [float(v) for v in selected_bbox]
                boxes_xywh[idx] = np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)
                status[idx] = 1
                last_bbox = selected_bbox
            elif idx > 0 and np.isfinite(boxes_xywh[idx - 1]).all():
                boxes_xywh[idx] = boxes_xywh[idx - 1]
                status[idx] = 0

            if writer is not None:
                out = stab_bgr
                if np.isfinite(boxes_xywh[idx]).all():
                    out = draw_bbox(out, boxes_xywh[idx], color=(255, 0, 0))
                writer.write(out)
            track_stage.update(1)
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
        track_stage.close()

    np.savez_compressed(out_npz, boxes_xywh=boxes_xywh, status=status, track_id=target_id)
    return PersonTrackResult(boxes_xywh=boxes_xywh, status=status, track_id=target_id)
