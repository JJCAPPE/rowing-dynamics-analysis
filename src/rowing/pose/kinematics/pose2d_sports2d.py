from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from Sports2D import Sports2D
except ImportError:  # fall back to lowercase package on legacy installs
    from sports2d import Sports2D

from .progress import ProgressReporter, get_progress
from .skeletons import COCO17_JOINT_NAMES

@dataclass(frozen=True)
class Pose2DResult:
    J2d_px: np.ndarray  # (T, J, 3) float32 (x,y,conf)
    joint_names: Tuple[str, ...]
    fps: float

def infer_pose2d_sports2d(
    video_path: Path,
    stabilization_npz: Path,
    out_npz: Path,
    model_name: str = "body",  # Sports2D pose_model (e.g., Body, Body_with_feet, Whole_body)
    mode: str = "balanced",  # lightweight, balanced, performance
    nb_persons: int | str = 1,  # 1 or "all"
    person_ordering: str = "highest_likelihood",
    device: str = "auto",
    det_frequency: int = 1, # Run detection every N frames (tracking in between)
    progress: Optional[ProgressReporter] = None,
) -> Pose2DResult:
    """Run Sports2D on the video and align output to stabilized coordinates.

    Sports2D processes the raw video. We then map the resulting keypoints
    through the stabilization affine transforms to match the pipeline's expectation.
    """
    
    video_path = Path(video_path).resolve()
    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)  # (T,2,3)
    fps = float(stab["fps"]) if "fps" in stab.files else 30.0
    num_frames = A.shape[0]

    prog = get_progress(progress)
    stage = prog.start("Stage E: pose2d (Sports2D)", total=1, unit="run")

    def normalize_nb_persons(value: int | str) -> int | str:
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "all":
                return "all"
            if v.isdigit():
                return int(v)
            return 1
        try:
            v_int = int(value)
        except Exception:
            return 1
        return v_int if v_int > 0 else 1

    def normalize_pose_model(name: str) -> str:
        n = (name or "").strip().lower()
        if n in {"body", "coco_17"}:
            return "Body"
        if n in {"body_with_feet", "halpe_26"}:
            return "Body_with_feet"
        if n in {"whole_body", "coco_133"}:
            return "Whole_body"
        if n in {"whole_body_wrist", "coco_133_wrist"}:
            return "Whole_body_wrist"
        # Default to COCO-17 skeleton to match our parsing assumptions.
        return "Body"

    nb_persons_norm = normalize_nb_persons(nb_persons)
    person_ordering_norm = (person_ordering or "highest_likelihood").strip().lower()
    pose_model = normalize_pose_model(model_name)

    # Sports2D requires an output directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out = Path(tmpdirname)

        # Sports2D configuration (merged with defaults inside Sports2D)
        config = {
            "base": {
                "video_input": str(video_path),
                "video_dir": "",
                "result_dir": str(tmp_out),
                "nb_persons_to_detect": nb_persons_norm,
                "person_ordering_method": person_ordering_norm,
                "show_realtime_results": False,
                "save_vid": False,
                "save_img": False,
                "save_pose": True,
                "calculate_angles": False,
                "save_angles": False,
            },
            "pose": {
                "pose_model": pose_model,
                "mode": str(mode),
                "det_frequency": int(det_frequency),
                "device": str(device),
                "backend": "auto",
                "tracking_mode": "sports2d",
            },
            "post-processing": {
                "show_graphs": False,
                "save_graphs": False,
            },
            "logging": {"use_custom_logging": False},
        }

        # Note: Sports2D might be chatty, so we might want to suppress output or capture it.
        try:
            Sports2D.process(config)
        except Exception as e:
            raise RuntimeError(f"Sports2D failed: {e}") from e

        # Sports2D output structure:
        # {output_dir}/{video_name}_Sports2D/{video_name}_trc.trc
        # or .json files.
        # Let's look for the .trc file or similar results.
        
        # Sports2D usually creates a folder with the video name
        vid_stem = video_path.stem
        # Sports2D sanitizes names sometimes, replacing spaces with underscores etc.
        # check the output directory for a subdirectory ending in Sports2D??
        # Or look for any TRC file.
        trc_files = list(tmp_out.rglob("*.trc"))
        
        if not trc_files:
             raise FileNotFoundError(f"Sports2D did not produce a TRC file in {tmp_out}")
        
        trc_path = trc_files[0]

        # Parse TRC
        # TRC format: header lines, then data
        # Frame#, Time, Name1_X, Name1_Y, Name1_Z, ...
        
        try:
            # Sports2D TRC usually starts data after some metadata.
            # We can find the header line.
            with open(trc_path, 'r') as f:
                lines = f.readlines()
            
            header_idx = 0
            for i, line in enumerate(lines):
                # TRC header usually starts with "Frame#" or "Frame" as the first column
                # Avoid matching "NumFrames" in metadata
                if line.lstrip().startswith("Frame") or "\tFrame" in line or ",Frame" in line:
                    header_idx = i
                    break
            
            # Read first few lines to check separator
            sep = "\t"
            if "," in lines[header_idx]:
                sep = ","
                
            df = pd.read_csv(trc_path, sep=sep, header=header_idx)
        except Exception as e:
             raise ValueError(f"Failed to parse TRC file: {trc_path}") from e

    # Extract keypoints
    keypoint_map = {
        "nose": ["nose"],
        "left_eye": ["l_eye", "left_eye", "lefteye"],
        "right_eye": ["r_eye", "right_eye", "righteye"],
        "left_ear": ["l_ear", "left_ear", "leftear"],
        "right_ear": ["r_ear", "right_ear", "rightear"],
        "left_shoulder": ["l_shoulder", "left_shoulder", "leftshoulder"],
        "right_shoulder": ["r_shoulder", "right_shoulder", "rightshoulder"],
        "left_elbow": ["l_elbow", "left_elbow", "leftelbow"],
        "right_elbow": ["r_elbow", "right_elbow", "rightelbow"],
        "left_wrist": ["l_wrist", "left_wrist", "leftwrist"],
        "right_wrist": ["r_wrist", "right_wrist", "rightwrist"],
        "left_hip": ["l_hip", "left_hip", "lefthip"],
        "right_hip": ["r_hip", "right_hip", "righthip"],
        "left_knee": ["l_knee", "left_knee", "leftknee"],
        "right_knee": ["r_knee", "right_knee", "rightknee"],
        "left_ankle": ["l_ankle", "left_ankle", "leftankle"],
        "right_ankle": ["r_ankle", "right_ankle", "rightankle"],
    }

    trc_frames = len(df)
    n_frames = min(num_frames, trc_frames)
    
    J2d_px = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)
    J2d_px[:, :, 2] = 0.0 # Default confidence 0
    
    def normalize_col_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def split_axis(col_name: str) -> Optional[Tuple[str, str]]:
        # Match suffix axis markers: _X, .X, " X", or just trailing X/Y/Z.
        m = re.match(r"(.+?)(?:[\\s\\._]?)([XYZ])$", col_name.strip(), re.IGNORECASE)
        if not m:
            return None
        base = m.group(1).strip()
        axis = m.group(2).upper()
        # Ignore obvious non-marker columns.
        base_norm = normalize_col_name(base)
        if base_norm in {"frame", "frame#", "time"}:
            return None
        return base, axis

    # Build a map of normalized base name -> axis columns (X/Y/Z).
    axis_cols: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        parsed = split_axis(str(col))
        if parsed is None:
            continue
        base, axis = parsed
        base_norm = normalize_col_name(base)
        axis_cols.setdefault(base_norm, {})[axis] = str(col)
    
    for i, name in enumerate(COCO17_JOINT_NAMES):
        found_base = None
        for candidate in keypoint_map.get(name, []):
            cand_norm = normalize_col_name(candidate)
            if cand_norm in axis_cols and {"X", "Y"}.issubset(axis_cols[cand_norm].keys()):
                found_base = cand_norm
                break

        if found_base is not None:
            try:
                # Try finding Y and Conf
                # Conf might not be there.

                # Column names from axis map.
                x_col = axis_cols[found_base]["X"]
                y_col = axis_cols[found_base]["Y"]

                x_vals = df[x_col].values[:n_frames]
                y_vals = df[y_col].values[:n_frames]

                # Confidence?
                # Some TRC exporters put conf in Z or separate col.
                # Sports2D with RTMPose might not output confidence in TRC.
                # As a fallback, we assume 1.0.
                conf_vals = np.ones_like(x_vals, dtype=np.float32)

                J2d_px[:n_frames, i, 0] = x_vals
                J2d_px[:n_frames, i, 1] = y_vals
                J2d_px[:n_frames, i, 2] = conf_vals

                bad = np.isnan(x_vals) | np.isnan(y_vals) | ((x_vals == 0) & (y_vals == 0))
                J2d_px[:n_frames, i, 2][bad] = 0.0
                J2d_px[:n_frames, i, 0:2][bad] = np.nan
            except KeyError:
                pass

    from .viz import apply_affine_to_keypoints_xyc
    
    for t in range(n_frames):
        J2d_px[t] = apply_affine_to_keypoints_xyc(J2d_px[t], A[t])

    stage.update(1)
    stage.close()
    
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        J2d_px=J2d_px,
        conf=J2d_px[:, :, 2],
        joint_names=np.array(COCO17_JOINT_NAMES, dtype=str),
        fps=float(fps),
    )
    
    return Pose2DResult(J2d_px=J2d_px, joint_names=COCO17_JOINT_NAMES, fps=float(fps))
