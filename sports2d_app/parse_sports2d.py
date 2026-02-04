from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrcData:
    df: pd.DataFrame
    keypoints: List[str]
    fps: Optional[float]


@dataclass(frozen=True)
class MotData:
    df: pd.DataFrame


COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


COCO17_CANDIDATES: Dict[str, List[str]] = {
    "nose": ["nose"],
    "left_eye": ["leye", "l_eye", "left_eye", "left eye"],
    "right_eye": ["reye", "r_eye", "right_eye", "right eye"],
    "left_ear": ["lear", "l_ear", "left_ear", "left ear"],
    "right_ear": ["rear", "r_ear", "right_ear", "right ear"],
    "left_shoulder": ["lshoulder", "left_shoulder", "left shoulder"],
    "right_shoulder": ["rshoulder", "right_shoulder", "right shoulder"],
    "left_elbow": ["lelbow", "left_elbow", "left elbow"],
    "right_elbow": ["relbow", "right_elbow", "right elbow"],
    "left_wrist": ["lwrist", "left_wrist", "left wrist"],
    "right_wrist": ["rwrist", "right_wrist", "right wrist"],
    "left_hip": ["lhip", "left_hip", "left hip"],
    "right_hip": ["rhip", "right_hip", "right hip"],
    "left_knee": ["lknee", "left_knee", "left knee"],
    "right_knee": ["rknee", "right_knee", "right knee"],
    "left_ankle": ["lankle", "left_ankle", "left ankle"],
    "right_ankle": ["rankle", "right_ankle", "right ankle"],
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def parse_trc_file(path: Path) -> TrcData:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame#"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"TRC header not found in {path}")

    header_line = lines[header_idx]
    parts = header_line.split("\t")
    keypoints = [p.strip() for p in parts[2:] if p.strip()]

    col_names = ["frame", "time"]
    for kpt in keypoints:
        col_names.extend([f"{kpt}_x", f"{kpt}_y", f"{kpt}_z"])

    data_lines = lines[header_idx + 2 :]
    if not data_lines:
        raise ValueError(f"No TRC data rows found in {path}")

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep="\t",
        header=None,
        engine="python",
    )

    if df.shape[1] < len(col_names):
        raise ValueError(
            f"TRC column mismatch in {path}: expected at least {len(col_names)} cols, got {df.shape[1]}"
        )
    if df.shape[1] > len(col_names):
        df = df.iloc[:, : len(col_names)]

    df.columns = col_names

    fps = None
    if len(df) > 1:
        dt = pd.Series(df["time"]).diff().dropna()
        if not dt.empty:
            median_dt = float(dt.median())
            if median_dt > 0:
                fps = 1.0 / median_dt

    return TrcData(df=df, keypoints=keypoints, fps=fps)


def parse_mot_file(path: Path) -> MotData:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            header_idx = i
            break
    if header_idx is None:
        # Try to find endheader and use the next line
        for i, line in enumerate(lines):
            if line.strip().lower() == "endheader":
                if i + 1 < len(lines):
                    header_idx = i + 1
                break
    if header_idx is None:
        raise ValueError(f"MOT header not found in {path}")

    header_line = lines[header_idx]
    columns = [c.strip() for c in header_line.split("\t") if c.strip()]
    data_lines = lines[header_idx + 1 :]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep="\t",
        header=None,
        names=columns,
        engine="python",
    )
    return MotData(df=df)


def trc_to_numpy(trc: TrcData) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = trc.df
    time = df["time"].to_numpy(dtype=np.float32)
    keypoints = trc.keypoints
    num_frames = len(df)
    data = np.full((num_frames, len(keypoints), 3), np.nan, dtype=np.float32)
    for i, kpt in enumerate(keypoints):
        data[:, i, 0] = df[f"{kpt}_x"].to_numpy(dtype=np.float32)
        data[:, i, 1] = df[f"{kpt}_y"].to_numpy(dtype=np.float32)
        data[:, i, 2] = df[f"{kpt}_z"].to_numpy(dtype=np.float32)
    return time, data, keypoints


def write_points_csv(trc: TrcData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trc.df.to_csv(out_path, index=False)


def write_points_npz(trc: TrcData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time, data, keypoints = trc_to_numpy(trc)
    np.savez_compressed(
        out_path,
        time=time,
        points=data,
        keypoints=np.array(keypoints, dtype=str),
    )


def write_angles_csv(mot: MotData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mot.df.to_csv(out_path, index=False)


def extract_coco17_from_trc(trc: TrcData) -> Tuple[np.ndarray, List[str]]:
    df = trc.df
    keypoints = trc.keypoints
    norm_map = {_normalize_name(k): k for k in keypoints}

    def find_base(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            norm = _normalize_name(cand)
            if norm in norm_map:
                return norm_map[norm]
        return None

    num_frames = len(df)
    J = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)

    for j, coco_name in enumerate(COCO17_NAMES):
        base = find_base(COCO17_CANDIDATES.get(coco_name, []))
        if base is None:
            continue
        x = df[f"{base}_x"].to_numpy(dtype=np.float32)
        y = df[f"{base}_y"].to_numpy(dtype=np.float32)
        z = df[f"{base}_z"].to_numpy(dtype=np.float32)
        J[:, j, 0] = x
        J[:, j, 1] = y
        J[:, j, 2] = z

    # No confidence in TRC; set 1 where valid
    conf = np.where(np.isfinite(J[:, :, 0]) & np.isfinite(J[:, :, 1]), 1.0, 0.0)
    J2d = np.stack([J[:, :, 0], J[:, :, 1], conf], axis=2)
    return J2d.astype(np.float32), COCO17_NAMES
