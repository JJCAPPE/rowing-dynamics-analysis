from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .skeletons import COCO17_JOINT_NAMES


@dataclass(frozen=True)
class Pose2DResult:
    J2d_px: np.ndarray  # (T, J, 3) float32 (x,y,conf) in stabilized full-frame px
    joint_names: Tuple[str, ...]
    fps: float


COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),  # hips
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def _coerce_keypoints_xy(keypoints: Any) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    # common shapes: (J,2) or (1,J,2)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected keypoints shape: {arr.shape}")
    return arr[:, :2].astype(np.float32, copy=False)


def _coerce_scores(scores: Any, J: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.shape[0] != J:
        raise ValueError(f"Unexpected keypoint_scores length: {arr.shape[0]} != {J}")
    return arr.astype(np.float32, copy=False)


def _extract_instances(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    preds = result.get("predictions")
    if preds is None:
        return []
    if isinstance(preds, list) and len(preds) > 0:
        first = preds[0]
        if isinstance(first, list):
            return [p for p in first if isinstance(p, dict)]
        if isinstance(first, dict):
            return [first]
    return []


def _best_instance(instances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    for inst in instances:
        if "keypoint_scores" not in inst:
            continue
        try:
            scores = np.asarray(inst["keypoint_scores"], dtype=np.float32).reshape(-1)
            s = float(np.nanmean(scores)) if scores.size else -1.0
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best = inst
    return best


def _inferencer_joint_names(inferencer: Any) -> Tuple[str, ...]:
    # Best-effort extraction of dataset meta; fall back to COCO-17.
    meta = None
    for obj in (getattr(inferencer, "model", None), getattr(inferencer, "pose_estimator", None)):
        if obj is None:
            continue
        meta = getattr(obj, "dataset_meta", None)
        if isinstance(meta, dict):
            break
    if not isinstance(meta, dict):
        return COCO17_JOINT_NAMES

    if "keypoint_name" in meta and isinstance(meta["keypoint_name"], (list, tuple)):
        names = tuple(str(x) for x in meta["keypoint_name"])
        return names

    # Some configs store keypoints as keypoint_info; keep stable order by id.
    if "keypoint_info" in meta:
        ki = meta["keypoint_info"]
        if isinstance(ki, (list, tuple)):
            # list of dicts containing id/name
            try:
                items = sorted(ki, key=lambda d: int(d.get("id", 0)))
                names = tuple(str(d.get("name", f"kp_{i}")) for i, d in enumerate(items))
                return names
            except Exception:
                pass
        if isinstance(ki, dict):
            try:
                items = sorted(ki.items(), key=lambda kv: int(kv[0]))
                names = tuple(str(v.get("name", f"kp_{k}")) for k, v in items)
                return names
            except Exception:
                pass

    return COCO17_JOINT_NAMES


def infer_pose2d_mmpose(
    video_path: Path,
    stabilization_npz: Path,
    crop_boxes_npy: Path,
    out_npz: Path,
    model: str = "human",
    device: str = "cpu",
    pose2d_weights: Optional[Path] = None,
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    debug_video_path: Optional[Path] = None,
) -> Pose2DResult:
    """Run MMPose on per-frame crops and emit stabilized full-frame keypoints.

    Implements Stage E in `planning.MD`.
    """

    try:
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MMPose is required for the 2D stage. Install `mmpose` (+ mmcv/mmdet) "
            "in your environment, then re-run."
        ) from e

    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)  # (T,2,3)
    width = int(stab["width"])
    height = int(stab["height"])
    fps = float(stab["fps"]) if "fps" in stab.files else 0.0
    T = int(A.shape[0])

    boxes = np.load(crop_boxes_npy).astype(np.float32)
    if boxes.shape[0] != T:
        raise ValueError(f"crop_boxes.npy length {boxes.shape[0]} does not match T={T}")

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(str(config_path))
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(str(checkpoint_path))

    weights_path = checkpoint_path or pose2d_weights
    weights_str = str(weights_path) if weights_path is not None else None
    model_id = str(config_path) if config_path is not None else model

    # Init inferencer (device/weights support varies slightly by version).
    try:
        inferencer = MMPoseInferencer(
            pose2d=model_id,
            pose2d_weights=weights_str,
            device=device,
        )
    except TypeError:
        try:
            inferencer = MMPoseInferencer(model_id, device=device)
        except TypeError:
            inferencer = MMPoseInferencer(model_id)

    joint_names = _inferencer_joint_names(inferencer)
    J = len(joint_names)

    J2d_px = np.full((T, J, 3), np.nan, dtype=np.float32)
    J2d_px[:, :, 2] = 0.0

    from .io_video import VideoWriter, iter_frames
    from .stabilize import warp_frame
    from .viz import draw_keypoints

    writer = None
    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(debug_video_path, fps=fps, frame_size=(width, height))
        writer.__enter__()

    def infer_one(img_bgr: np.ndarray) -> Dict[str, Any]:
        # Try to avoid visualization unless explicitly requested.
        try:
            if weights_str is not None:
                gen = inferencer(
                    img_bgr, model=model_id, weights=weights_str, show=False, return_vis=False
                )
            else:
                gen = inferencer(img_bgr, show=False, return_vis=False)
        except TypeError:
            try:
                if weights_str is not None:
                    gen = inferencer(img_bgr, model=model_id, weights=weights_str, show=False)
                else:
                    gen = inferencer(img_bgr, show=False)
            except TypeError:
                if weights_str is not None:
                    gen = inferencer(img_bgr, model=model_id, weights=weights_str)
                else:
                    gen = inferencer(img_bgr)
        return next(gen)

    try:
        for idx, frame in iter_frames(video_path):
            stab_bgr = warp_frame(frame, A[idx], (width, height))
            x, y, w, h = [float(v) for v in boxes[idx]]
            x0 = int(max(0, round(x)))
            y0 = int(max(0, round(y)))
            x1 = int(min(width, round(x + w)))
            y1 = int(min(height, round(y + h)))
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                if writer is not None:
                    writer.write(stab_bgr)
                continue

            crop = stab_bgr[y0:y1, x0:x1]
            result = infer_one(crop)
            instances = _extract_instances(result)
            inst = _best_instance(instances)
            if inst is not None and "keypoints" in inst and "keypoint_scores" in inst:
                kpts = _coerce_keypoints_xy(inst["keypoints"])
                scores = _coerce_scores(inst["keypoint_scores"], kpts.shape[0])
                if kpts.shape[0] != J:
                    # If the model doesn't match expected joint count, adapt once.
                    if idx == 0:
                        J = int(kpts.shape[0])
                        joint_names = tuple(f"kp_{i}" for i in range(J))
                        J2d_px = np.full((T, J, 3), np.nan, dtype=np.float32)
                        J2d_px[:, :, 2] = 0.0
                    else:
                        # Skip inconsistent frame.
                        inst = None
                if inst is not None:
                    kpts[:, 0] += float(x0)
                    kpts[:, 1] += float(y0)
                    J2d_px[idx, :, 0:2] = kpts[:, 0:2]
                    J2d_px[idx, :, 2] = scores

            if writer is not None:
                edges = COCO17_EDGES if len(joint_names) == 17 else None
                out = draw_keypoints(stab_bgr, J2d_px[idx], edges=edges)
                writer.write(out)
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        J2d_px=J2d_px,
        conf=J2d_px[:, :, 2],
        joint_names=np.array(joint_names, dtype=str),
        fps=float(fps),
    )
    return Pose2DResult(J2d_px=J2d_px, joint_names=joint_names, fps=float(fps))

