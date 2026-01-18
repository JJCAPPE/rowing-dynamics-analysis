from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

Point2D = Tuple[float, float]
BBoxXYWH = Tuple[float, float, float, float]


@dataclass
class VideoInfo:
    """Basic metadata needed to make runs reproducible."""

    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "frame_count": int(self.frame_count),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VideoInfo":
        return VideoInfo(
            path=str(d["path"]),
            fps=float(d["fps"]),
            width=int(d["width"]),
            height=int(d["height"]),
            frame_count=int(d["frame_count"]),
        )


@dataclass
class Annotations:
    anchor_px: Point2D
    bbox_px: BBoxXYWH  # (x, y, w, h) in reference frame
    scale_points_px: Tuple[Point2D, Point2D]
    scale_distance_m: float
    rigger_bbox_px: Optional[BBoxXYWH] = None

    def to_dict(self) -> Dict[str, Any]:
        (sx0, sy0), (sx1, sy1) = self.scale_points_px
        out: Dict[str, Any] = {
            "anchor_px": [float(self.anchor_px[0]), float(self.anchor_px[1])],
            "bbox_px": [float(v) for v in self.bbox_px],
            "scale_points_px": [[float(sx0), float(sy0)], [float(sx1), float(sy1)]],
            "scale_distance_m": float(self.scale_distance_m),
        }
        if self.rigger_bbox_px is not None:
            out["rigger_bbox_px"] = [float(v) for v in self.rigger_bbox_px]
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Annotations":
        a = d["anchor_px"]
        b = d["bbox_px"]
        r = d.get("rigger_bbox_px")
        s0, s1 = d["scale_points_px"]
        rigger_bbox = None
        if r is not None:
            rigger_bbox = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        return Annotations(
            anchor_px=(float(a[0]), float(a[1])),
            bbox_px=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            rigger_bbox_px=rigger_bbox,
            scale_points_px=((float(s0[0]), float(s0[1])), (float(s1[0]), float(s1[1]))),
            scale_distance_m=float(d["scale_distance_m"]),
        )


@dataclass
class RunConfig:
    """Serialized to `run.json` (one per processed video)."""

    version: int
    video: VideoInfo
    reference_frame_idx: int
    annotations: Annotations
    derived: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "video": self.video.to_dict(),
            "reference_frame_idx": int(self.reference_frame_idx),
            "annotations": self.annotations.to_dict(),
            "derived": dict(self.derived),
            "params": dict(self.params),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunConfig":
        return RunConfig(
            version=int(d.get("version", 1)),
            video=VideoInfo.from_dict(d["video"]),
            reference_frame_idx=int(d.get("reference_frame_idx", 0)),
            annotations=Annotations.from_dict(d["annotations"]),
            derived=dict(d.get("derived", {})),
            params=dict(d.get("params", {})),
        )

    @property
    def m_per_px(self) -> Optional[float]:
        v = self.derived.get("m_per_px")
        return None if v is None else float(v)

    def resolve_video_path(self, run_json_path: str | Path) -> Path:
        """Resolve self.video.path relative to the directory containing run.json."""
        run_json_path = Path(run_json_path)
        p = Path(self.video.path)
        return p if p.is_absolute() else (run_json_path.parent / p)


POSE_SMOOTHING_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "conf_threshold": 0.3,
    "max_gap": 5,
    "median_window": 5,
}

POSE_TRACKING_DEFAULTS: Dict[str, Any] = {
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
    "strict_id": False,
    "deepsort_model": "yolov8n.pt",
    "deepsort_min_conf": 0.25,
    "deepsort_iou_threshold": 0.3,
    "deepsort_padding": 0.2,
}


def apply_pose_smoothing_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params) if params is not None else {}
    merged = dict(POSE_SMOOTHING_DEFAULTS)
    existing = out.get("pose_smoothing")
    if isinstance(existing, dict):
        merged.update(existing)
    out["pose_smoothing"] = merged
    return out


def apply_pose_tracking_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params) if params is not None else {}
    merged = dict(POSE_TRACKING_DEFAULTS)
    existing = out.get("pose_tracking")
    if isinstance(existing, dict):
        merged.update(existing)
    out["pose_tracking"] = merged
    return out


def compute_m_per_px(scale_points_px: Tuple[Point2D, Point2D], distance_m: float) -> float:
    (x0, y0), (x1, y1) = scale_points_px
    d_px = float(np.hypot(x1 - x0, y1 - y0))
    if d_px <= 1e-9:
        raise ValueError("Scale points are identical (pixel distance ~ 0).")
    if distance_m <= 0:
        raise ValueError("Known distance in meters must be > 0.")
    return float(distance_m / d_px)


def save_run_config(path: str | Path, cfg: RunConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    cfg = RunConfig.from_dict(d)
    cfg.params = apply_pose_smoothing_defaults(cfg.params)
    cfg.params = apply_pose_tracking_defaults(cfg.params)
    return cfg
