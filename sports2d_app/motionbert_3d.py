from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_ROOT = REPO_ROOT / "pose-extraction-test"

if str(POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(POSE_ROOT))

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
) -> MotionBertOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motionbert_root, config_path, ckpt_path = _resolve_motionbert_assets()

    mb_in = prepare_motionbert_input_from_coco(J2d_px, width=width, height=height, mode="pixel")

    J3d_raw = lift_pose3d_motionbert(
        mb_in,
        motionbert_root=motionbert_root,
        checkpoint_path=ckpt_path,
        clip_len=int(clip_len),
        flip=bool(flip),
        rootrel=bool(rootrel),
        config_path=config_path,
        progress=None,
    )

    pose3d_npz = out_dir / "pose3d.npz"
    np.savez_compressed(
        pose3d_npz,
        J3d_raw=J3d_raw,
        J3d_m=np.array([], dtype=np.float32),
        alpha_scale=np.array(np.nan, dtype=np.float32),
        joint_names=np.array(H36M17_JOINT_NAMES, dtype=str),
    )

    ang = compute_basic_angles_h36m17(J3d_raw, H36M17_JOINT_NAMES)
    deg = np.degrees(ang.values_rad)
    df = pd.DataFrame(deg, columns=[f"{n}_deg" for n in ang.names])
    df.insert(0, "frame_idx", np.arange(df.shape[0], dtype=int))
    if fps is not None and fps > 0:
        df.insert(1, "time_s", df["frame_idx"] / float(fps))

    angles_csv = out_dir / "angles_h36m.csv"
    df.to_csv(angles_csv, index=False)

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

    return MotionBertOutputs(
        pose3d_npz=pose3d_npz,
        angles_csv=angles_csv,
        metrics_json=metrics_json,
    )
