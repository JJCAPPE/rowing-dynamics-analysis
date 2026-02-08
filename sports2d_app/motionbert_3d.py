from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from progress_utils import clamp01


APP_ROOT = Path(__file__).resolve().parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from rowing_pose.kinematics import compute_basic_angles_h36m17  # type: ignore
from rowing_pose.model_assets import (  # type: ignore
    DEFAULT_MOTIONBERT_MODEL,
    MOTIONBERT_REPO,
    ensure_asset,
    get_motionbert_model,
)
from rowing_pose.motionbert_format import prepare_motionbert_input_from_coco  # type: ignore
from rowing_pose.motionbert_lift import lift_pose3d_motionbert  # type: ignore
from rowing_pose.skeletons import H36M17_JOINT_NAMES  # type: ignore


@dataclass(frozen=True)
class MotionBertOutputs:
    pose3d_npz: Path
    angles_csv: Path
    metrics_json: Path


ProgressCallback = Callable[[str, float], None]


def _emit_progress(
    callback: Optional[ProgressCallback],
    label: str,
    progress: float,
) -> None:
    if callback is None:
        return
    callback(label, clamp01(progress))


class _MappedProgressHandle:
    def __init__(
        self,
        callback: Optional[ProgressCallback],
        *,
        label: str,
        total: Optional[int],
        start: float,
        end: float,
    ) -> None:
        self._callback = callback
        self._label = label
        self._total = int(total) if total is not None and int(total) > 0 else None
        self._count = 0
        self._start = clamp01(start)
        self._end = clamp01(end)
        if self._end < self._start:
            self._start, self._end = self._end, self._start
        self._emit(label=self._label, local_progress=0.0)

    def _emit(self, *, label: str, local_progress: float) -> None:
        mapped = self._start + (self._end - self._start) * clamp01(local_progress)
        _emit_progress(self._callback, label, mapped)

    def update(
        self,
        n: int = 1,
        *,
        desc: Optional[str] = None,
        total: Optional[int] = None,
    ) -> None:
        if total is not None and int(total) > 0:
            self._total = int(total)
        self._count += max(0, int(n))
        if self._total is None:
            local = 0.0
        else:
            local = self._count / float(max(1, self._total))
        self._emit(label=desc or self._label, local_progress=local)

    def close(self, *, status: Optional[str] = None) -> None:
        self._emit(label=status or self._label, local_progress=1.0)


class _MappedProgressReporter:
    def __init__(
        self,
        callback: Optional[ProgressCallback],
        *,
        start: float,
        end: float,
    ) -> None:
        self._callback = callback
        self._start = start
        self._end = end

    def start(
        self,
        label: str,
        total: Optional[int] = None,
        unit: str = "it",
    ) -> _MappedProgressHandle:
        del unit
        return _MappedProgressHandle(
            self._callback,
            label=label,
            total=total,
            start=self._start,
            end=self._end,
        )


def _resolve_motionbert_assets() -> Tuple[Path, Path, Path]:
    spec = get_motionbert_model(DEFAULT_MOTIONBERT_MODEL)
    if spec is None:
        raise FileNotFoundError("MotionBERT model spec not found.")

    motionbert_root = MOTIONBERT_REPO
    if not motionbert_root.exists():
        raise FileNotFoundError(
            f"MotionBERT repo not found at {motionbert_root}. Ensure submodule is available."
        )

    config_path = spec.config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"MotionBERT config not found at {config_path}. Ensure submodule is available."
        )

    ckpt_path = spec.checkpoint.path
    if not ckpt_path.exists():
        ckpt_path = ensure_asset(
            spec.checkpoint.path,
            spec.checkpoint.url,
            expected_size=spec.checkpoint.size_bytes,
            sha256=spec.checkpoint.sha256,
        )

    return motionbert_root, config_path, ckpt_path


def run_motionbert(
    J2d_px: np.ndarray,
    width: int,
    height: int,
    out_dir: Path,
    fps: Optional[float] = None,
    clip_len: int = 243,
    flip: bool = False,
    rootrel: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> MotionBertOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress_callback, "MotionBERT: resolving model assets", 0.0)
    motionbert_root, config_path, ckpt_path = _resolve_motionbert_assets()

    _emit_progress(progress_callback, "MotionBERT: formatting 2D inputs", 0.08)
    mb_in = prepare_motionbert_input_from_coco(J2d_px, width=width, height=height, mode="pixel")

    _emit_progress(progress_callback, "MotionBERT: running 3D lift", 0.12)
    motionbert_progress = (
        _MappedProgressReporter(progress_callback, start=0.12, end=0.9)
        if progress_callback is not None
        else None
    )
    J3d_raw = lift_pose3d_motionbert(
        mb_in.X_h36m17,
        motionbert_root=motionbert_root,
        checkpoint_path=ckpt_path,
        clip_len=int(clip_len),
        flip=bool(flip),
        rootrel=bool(rootrel),
        config_path=config_path,
        progress=motionbert_progress,
    )

    _emit_progress(progress_callback, "MotionBERT: saving 3D pose", 0.92)
    pose3d_npz = out_dir / "pose3d.npz"
    np.savez_compressed(
        pose3d_npz,
        J3d_raw=J3d_raw,
        J3d_m=np.array([], dtype=np.float32),
        alpha_scale=np.array(np.nan, dtype=np.float32),
        joint_names=np.array(H36M17_JOINT_NAMES, dtype=str),
    )

    _emit_progress(progress_callback, "MotionBERT: computing joint angles", 0.95)
    ang = compute_basic_angles_h36m17(J3d_raw, H36M17_JOINT_NAMES)
    deg = np.degrees(ang.values_rad)
    df = pd.DataFrame(deg, columns=[f"{n}_deg" for n in ang.names])
    df.insert(0, "frame_idx", np.arange(df.shape[0], dtype=int))
    if fps is not None and fps > 0:
        df.insert(1, "time_s", df["frame_idx"] / float(fps))

    angles_csv = out_dir / "angles_h36m.csv"
    df.to_csv(angles_csv, index=False)

    _emit_progress(progress_callback, "MotionBERT: writing metrics", 0.98)
    summary = {}
    for col in df.columns:
        if col in ("frame_idx", "time_s"):
            continue
        v = df[col].to_numpy(dtype=np.float32)
        if np.isfinite(v).any():
            summary[col] = {
                "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v)),
                "rom": float(np.nanmax(v) - np.nanmin(v)),
            }
    summary["video"] = {"fps": float(fps) if fps else None, "frames": int(df.shape[0])}

    metrics_json = out_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    _emit_progress(progress_callback, "MotionBERT: completed", 1.0)

    return MotionBertOutputs(
        pose3d_npz=pose3d_npz,
        angles_csv=angles_csv,
        metrics_json=metrics_json,
    )
