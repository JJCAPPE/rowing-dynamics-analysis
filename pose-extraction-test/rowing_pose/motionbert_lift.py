from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from .progress import ProgressReporter, get_progress


@dataclass(frozen=True)
class Pose3DResult:
    J3d_raw: np.ndarray  # (T, 17, 3) float32
    J3d_m: Optional[np.ndarray]  # (T, 17, 3) float32 (scaled convenience)
    alpha_scale: Optional[float]
    joint_names: Tuple[str, ...]


def lift_pose3d_motionbert(
    X_h36m17: np.ndarray,  # (T, 17, 3) (x,y,conf)
    motionbert_root: Path,
    checkpoint_path: Path,
    device: str = "cpu",
    config_path: Optional[Path] = None,
    clip_len: int = 243,
    overlap: Optional[int] = None,
    flip: bool = False,
    rootrel: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> np.ndarray:
    """Lift 2D→3D using MotionBERT.

    This function mirrors MotionBERT's `infer_wild.py` behavior (Stage G),
    but accepts a pre-formatted 2D sequence directly instead of a JSON dataset.

    Requirements:
    - `motionbert_root` points to a MotionBERT repo checkout (vendored or external).
    - `checkpoint_path` is a `.bin`/`.pth` checkpoint containing `checkpoint['model_pos']`.
    """

    import sys
    import torch
    import torch.nn as nn

    motionbert_root = Path(motionbert_root)
    checkpoint_path = Path(checkpoint_path)
    if not motionbert_root.exists():
        raise FileNotFoundError(str(motionbert_root))
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))

    X = np.asarray(X_h36m17, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (17, 3):
        raise ValueError(f"Expected X_h36m17 shape (T,17,3). Got {X.shape}")

    T = int(X.shape[0])
    if T == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)

    if clip_len <= 0:
        raise ValueError("clip_len must be > 0")
    if overlap is None:
        overlap = clip_len // 2
    overlap = int(overlap)
    overlap = max(0, min(overlap, clip_len - 1))
    stride = max(1, clip_len - overlap)

    # Import MotionBERT modules (repo uses absolute `lib.*` imports).
    sys.path.insert(0, str(motionbert_root))
    try:
        from lib.utils.tools import get_config  # type: ignore
        from lib.utils.learning import load_backbone  # type: ignore
        from lib.utils.utils_data import flip_data  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import MotionBERT modules. Ensure motionbert_root is correct and "
            "its dependencies (e.g. pyyaml, easydict) are installed."
        ) from e

    if config_path is None:
        config_path = motionbert_root / "configs/pose3d/MB_ft_h36m_global_lite.yaml"
    args = get_config(str(config_path))

    model_backbone = load_backbone(args)
    model_pos: nn.Module = model_backbone
    model_pos.to(device)

    # MotionBERT checkpoints may require full unpickling (PyTorch 2.6 defaults
    # weights_only=True). These are trusted local assets in this pipeline.
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if "model_pos" not in ckpt:
        raise KeyError("Checkpoint does not contain key 'model_pos' (unexpected format).")

    state = ckpt["model_pos"]
    # Some checkpoints are saved under DataParallel/DistributedDataParallel ("module." prefix).
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_pos.load_state_dict(state, strict=True)
    model_pos.eval()

    # Prepare clip start indices with overlap; ensure last clip covers the end.
    starts = list(range(0, max(1, T - clip_len + 1), stride))
    if starts[-1] != max(0, T - clip_len):
        starts.append(max(0, T - clip_len))

    out = np.zeros((T, 17, 3), dtype=np.float32)
    wsum = np.zeros((T,), dtype=np.float32)

    prog = get_progress(progress)
    stage = prog.start("Stage F/G: pose3d lift (MotionBERT)", total=len(starts), unit="clip")
    with torch.no_grad():
        for st in starts:
            ed = min(T, st + clip_len)
            clip = X[st:ed]  # (L,17,3)
            batch = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1,L,17,3)

            if getattr(args, "no_conf", False):
                batch = batch[:, :, :, :2]

            if flip:
                batch_flip = flip_data(batch)
                pred1 = model_pos(batch)
                pred2 = model_pos(batch_flip)
                pred2 = flip_data(pred2)  # flip back
                pred = (pred1 + pred2) / 2.0
            else:
                pred = model_pos(batch)

            if rootrel:
                pred[:, :, 0, :] = 0

            pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (L,17,3)
            L = int(pred_np.shape[0])

            # Linear blending weights.
            w = np.ones((L,), dtype=np.float32)
            if overlap > 0:
                ramp = min(overlap, L)
                if st > 0 and ramp > 0:
                    w[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                if ed < T and ramp > 0:
                    w[-ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)

            out[st:ed] += pred_np * w[:, None, None]
            wsum[st:ed] += w
            stage.update(1)
    stage.close()

    mask = wsum > 1e-9
    out[mask] /= wsum[mask, None, None]
    out[~mask] = np.nan
    return out
