from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .skeletons import COCO17_JOINT_NAMES
from .progress import ProgressReporter, get_progress


@dataclass(frozen=True)
class Pose2DResult:
    J2d_px: np.ndarray  # (T, J, 3) float32 (x,y,conf) in stabilized full-frame px
    joint_names: Tuple[str, ...]
    fps: float


@dataclass
class TrackState:
    prev_pelvis: Optional[np.ndarray] = None
    prev_vel: Optional[np.ndarray] = None
    prev_bbox: Optional[np.ndarray] = None
    prev_kpts: Optional[np.ndarray] = None
    ref_hist: Optional[np.ndarray] = None


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


def _bbox_from_keypoints(
    kpts_xy: np.ndarray, scores: np.ndarray, min_conf: float = 0.2
) -> Optional[np.ndarray]:
    valid = (
        (scores >= float(min_conf))
        & np.isfinite(kpts_xy[:, 0])
        & np.isfinite(kpts_xy[:, 1])
    )
    if np.count_nonzero(valid) < 4:
        return None
    xs = kpts_xy[valid, 0]
    ys = kpts_xy[valid, 1]
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _bbox_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a_xyxy]
    bx0, by0, bx1, by1 = [float(v) for v in b_xyxy]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-6 else 0.0


def _pelvis_xy(
    kpts_xy: np.ndarray, scores: np.ndarray, joint_names: Tuple[str, ...]
) -> Optional[np.ndarray]:
    def _idx(name: str) -> Optional[int]:
        try:
            return joint_names.index(name)
        except ValueError:
            return None

    pelvis_idx = _idx("pelvis")
    if pelvis_idx is not None and scores[pelvis_idx] > 0:
        return kpts_xy[pelvis_idx].astype(np.float32)

    lhip = _idx("left_hip")
    rhip = _idx("right_hip")
    if lhip is not None and rhip is not None and scores[lhip] > 0 and scores[rhip] > 0:
        return ((kpts_xy[lhip] + kpts_xy[rhip]) / 2.0).astype(np.float32)

    valid = scores > 0
    if np.count_nonzero(valid) >= 4:
        return kpts_xy[valid].mean(axis=0).astype(np.float32)
    return None


def _appearance_hist(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray, bins: int = 16) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    return hist.astype(np.float32)


def _hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pose_tracking_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "continuity_weight": 4.0,
        "appearance_weight": 2.0,
        "iou_weight": 0.5,
        "conf_weight": 0.3,
        "max_jump_factor": 0.4,
        "motion_sigma_factor": 0.2,
        "smooth_alpha": 0.05,
        "appearance_bins": 16,
        "appearance_update_alpha": 0.1,
        "min_conf": 0.2,
    }
    if isinstance(params, dict):
        defaults.update(params)
    return defaults


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


def _allow_torch_numpy_globals() -> None:
    """Allow numpy globals for PyTorch weights-only loading (trusted checkpoints)."""
    try:
        import os
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        import torch
    except Exception:
        return
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return
    try:
        allow = [
            (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
        ]
        scalar = getattr(np.core.multiarray, "scalar", None)
        if scalar is not None:
            allow.append((scalar, "numpy.core.multiarray.scalar"))
        add_safe_globals(allow)
    except Exception:
        return


def _missing_mmpose_base_files(config_path: Path) -> List[Path]:
    """Return missing _base_ files referenced by a config."""
    try:
        import ast

        text = config_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(config_path))
    except Exception:
        return []

    base_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_base_":
                    try:
                        base_value = ast.literal_eval(node.value)
                    except Exception:
                        base_value = None
                    break
        if base_value is not None:
            break

    if base_value is None:
        return []

    if isinstance(base_value, (str, bytes)):
        base_list = [base_value]
    else:
        try:
            base_list = list(base_value)
        except Exception:
            return []

    missing: List[Path] = []
    for rel in base_list:
        try:
            rel_str = str(rel)
        except Exception:
            continue
        base_path = (config_path.parent / rel_str).resolve()
        if not base_path.exists():
            missing.append(base_path)
    return missing


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
    pose_tracking: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressReporter] = None,
) -> Pose2DResult:
    """Run MMPose on per-frame crops and emit stabilized full-frame keypoints.

    Implements Stage E in `planning.MD`.
    """

    _allow_torch_numpy_globals()
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
    prog = get_progress(progress)

    boxes = np.load(crop_boxes_npy).astype(np.float32)
    if boxes.shape[0] != T:
        raise ValueError(f"crop_boxes.npy length {boxes.shape[0]} does not match T={T}")

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(str(config_path))
        missing = _missing_mmpose_base_files(config_path)
        if missing:
            import warnings

            warnings.warn(
                "MMPose config references missing _base_ files; "
                "falling back to model name resolution.",
                RuntimeWarning,
            )
            config_path = None
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
    from .viz import draw_bbox, draw_keypoints, draw_text_panel

    track_params = _pose_tracking_params(pose_tracking)
    track_state = TrackState()

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

    infer_stage = prog.start("Stage E: pose2d (MMPose)", total=T, unit="frame")
    try:
        for idx, frame in iter_frames(video_path):
            stab_bgr = warp_frame(frame, A[idx], (width, height))
            box = boxes[idx]
            if not np.isfinite(box).all():
                if idx > 0:
                    J2d_px[idx] = J2d_px[idx - 1]
                if writer is not None:
                    writer.write(stab_bgr)
                infer_stage.update(1)
                continue
            x, y, w, h = [float(v) for v in box]
            x0 = int(max(0, round(x)))
            y0 = int(max(0, round(y)))
            x1 = int(min(width, round(x + w)))
            y1 = int(min(height, round(y + h)))
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                if idx > 0:
                    J2d_px[idx] = J2d_px[idx - 1]
                if writer is not None:
                    writer.write(stab_bgr)
                infer_stage.update(1)
                continue

            crop = stab_bgr[y0:y1, x0:x1]
            result = infer_one(crop)
            instances = _extract_instances(result)
            inst = None
            kpts = None
            scores = None
            bbox_xyxy = None
            pelvis = None
            track_score = None
            motion_px = None
            app_score = None
            iou_score = None

            if track_params.get("enabled", True) and instances:
                best_score = -1e9
                best = None
                max_side = float(max(x1 - x0, y1 - y0))
                motion_sigma = max(10.0, float(track_params["motion_sigma_factor"]) * max_side)
                max_jump = max(20.0, float(track_params["max_jump_factor"]) * max_side)
                pred_pelvis = None
                if track_state.prev_pelvis is not None:
                    if track_state.prev_vel is None:
                        pred_pelvis = track_state.prev_pelvis
                    else:
                        pred_pelvis = track_state.prev_pelvis + track_state.prev_vel

                for cand in instances:
                    if "keypoints" not in cand or "keypoint_scores" not in cand:
                        continue
                    cand_kpts = _coerce_keypoints_xy(cand["keypoints"])
                    cand_scores = _coerce_scores(cand["keypoint_scores"], cand_kpts.shape[0])
                    if cand_kpts.shape[0] != J and idx != 0:
                        continue
                    joint_names_use = (
                        joint_names
                        if cand_kpts.shape[0] == J
                        else tuple(f"kp_{i}" for i in range(cand_kpts.shape[0]))
                    )
                    cand_kpts_full = cand_kpts.copy()
                    cand_kpts_full[:, 0] += float(x0)
                    cand_kpts_full[:, 1] += float(y0)

                    cand_bbox = _bbox_from_keypoints(
                        cand_kpts_full, cand_scores, min_conf=float(track_params["min_conf"])
                    )
                    if cand_bbox is None:
                        continue
                    cand_pelvis = _pelvis_xy(cand_kpts_full, cand_scores, joint_names_use)
                    if cand_pelvis is None:
                        continue

                    conf_score = float(np.nanmean(cand_scores)) if cand_scores.size else 0.0

                    motion_score = 0.0
                    d_motion = None
                    if pred_pelvis is not None:
                        d_motion = float(np.linalg.norm(cand_pelvis - pred_pelvis))
                        motion_score = float(np.exp(-((d_motion ** 2) / (2 * motion_sigma**2))))
                        if d_motion > max_jump:
                            motion_score *= 0.1
                    else:
                        motion_score = 1.0

                    cand_hist = None
                    if track_state.ref_hist is not None:
                        cand_hist = _appearance_hist(
                            stab_bgr, cand_bbox, bins=int(track_params["appearance_bins"])
                        )
                    appearance_score = _hist_similarity(track_state.ref_hist, cand_hist)

                    iou = 0.0
                    if track_state.prev_bbox is not None:
                        iou = _bbox_iou(track_state.prev_bbox, cand_bbox)

                    total = (
                        float(track_params["conf_weight"]) * conf_score
                        + float(track_params["continuity_weight"]) * motion_score
                        + float(track_params["appearance_weight"]) * appearance_score
                        + float(track_params["iou_weight"]) * iou
                    )

                    if total > best_score:
                        best_score = total
                        best = (cand, cand_kpts_full, cand_scores, cand_bbox, cand_pelvis, d_motion, appearance_score, iou)

                if best is not None:
                    inst, kpts, scores, bbox_xyxy, pelvis, motion_px, app_score, iou_score = best
                    track_score = best_score
            else:
                inst = _best_instance(instances)
                if inst is not None and "keypoints" in inst and "keypoint_scores" in inst:
                    kpts = _coerce_keypoints_xy(inst["keypoints"])
                    scores = _coerce_scores(inst["keypoint_scores"], kpts.shape[0])
                    kpts[:, 0] += float(x0)
                    kpts[:, 1] += float(y0)

            if inst is not None and kpts is not None and scores is not None:
                if kpts.shape[0] != J:
                    # If the model doesn't match expected joint count, adapt once.
                    if idx == 0:
                        J = int(kpts.shape[0])
                        joint_names = tuple(f"kp_{i}" for i in range(J))
                        J2d_px = np.full((T, J, 3), np.nan, dtype=np.float32)
                        J2d_px[:, :, 2] = 0.0
                    else:
                        inst = None

            if inst is not None and kpts is not None and scores is not None:
                if (
                    track_params.get("enabled", True)
                    and track_state.prev_kpts is not None
                    and float(track_params["smooth_alpha"]) > 0.0
                ):
                    alpha = float(np.clip(track_params["smooth_alpha"], 0.0, 0.999))
                    prev = track_state.prev_kpts
                    cur = kpts.copy()
                    valid = np.isfinite(cur[:, 0]) & np.isfinite(cur[:, 1])
                    cur[valid, 0:2] = alpha * prev[valid, 0:2] + (1.0 - alpha) * cur[valid, 0:2]
                    kpts = cur

                J2d_px[idx, :, 0:2] = kpts[:, 0:2]
                J2d_px[idx, :, 2] = scores

                if bbox_xyxy is None:
                    bbox_xyxy = _bbox_from_keypoints(
                        kpts, scores, min_conf=float(track_params["min_conf"])
                    )
                if pelvis is None:
                    pelvis = _pelvis_xy(kpts, scores, joint_names)

                if pelvis is not None:
                    if track_state.prev_pelvis is not None:
                        vel = pelvis - track_state.prev_pelvis
                        if track_state.prev_vel is None:
                            track_state.prev_vel = vel
                        else:
                            track_state.prev_vel = 0.7 * track_state.prev_vel + 0.3 * vel
                    else:
                        track_state.prev_vel = np.zeros((2,), dtype=np.float32)
                    track_state.prev_pelvis = pelvis

                if bbox_xyxy is not None:
                    track_state.prev_bbox = bbox_xyxy
                    if track_state.ref_hist is None:
                        track_state.ref_hist = _appearance_hist(
                            stab_bgr, bbox_xyxy, bins=int(track_params["appearance_bins"])
                        )
                    else:
                        new_hist = _appearance_hist(
                            stab_bgr, bbox_xyxy, bins=int(track_params["appearance_bins"])
                        )
                        if new_hist is not None:
                            a = float(track_params["appearance_update_alpha"])
                            track_state.ref_hist = (1.0 - a) * track_state.ref_hist + a * new_hist

                track_state.prev_kpts = kpts.copy()
            elif idx > 0:
                # Hold last pose to avoid identity jumps when tracking drops.
                J2d_px[idx] = J2d_px[idx - 1]

            if writer is not None:
                edges = COCO17_EDGES if len(joint_names) == 17 else None
                out = draw_keypoints(stab_bgr, J2d_px[idx], edges=edges)
                if bbox_xyxy is not None:
                    x0b, y0b, x1b, y1b = bbox_xyxy
                    out = draw_bbox(out, (x0b, y0b, x1b - x0b, y1b - y0b), color=(0, 255, 0))
                hud_lines = []
                if track_score is not None:
                    hud_lines.append(f"track_score: {track_score:.2f}")
                if motion_px is not None:
                    hud_lines.append(f"jump_px: {float(motion_px):.1f}")
                if app_score is not None:
                    hud_lines.append(f"app_sim: {float(app_score):.2f}")
                if iou_score is not None:
                    hud_lines.append(f"iou: {float(iou_score):.2f}")
                if hud_lines:
                    out = draw_text_panel(out, hud_lines, origin_xy=(10, 20))
                writer.write(out)
            infer_stage.update(1)
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
        infer_stage.close()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        J2d_px=J2d_px,
        conf=J2d_px[:, :, 2],
        joint_names=np.array(joint_names, dtype=str),
        fps=float(fps),
    )
    return Pose2DResult(J2d_px=J2d_px, joint_names=joint_names, fps=float(fps))
