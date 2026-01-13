from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    device: str = "cpu",
    mmpose_model: str = "human",
    motionbert_root: Optional[Path] = None,
    motionbert_ckpt: Optional[Path] = None,
    clip_len: int = 243,
    flip: bool = False,
    rootrel: bool = False,
    skip_2d: bool = False,
    skip_3d: bool = False,
) -> None:
    """End-to-end pipeline orchestrator.

    This function is intentionally a thin layer that wires together the modular stages:
    annotations → stabilization → crop tracking → pose2d → (optional) pose3d → kinematics → debug.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from .config import load_run_config
    from .ui_annotate import annotate_video

    run_json = out_dir / "run.json"
    if not run_json.exists():
        annotate_video(video_path=video_path, out_dir=out_dir, reference_frame_idx=0)
    cfg = load_run_config(run_json)

    # Stage B: stabilization
    from .stabilize import compute_stabilization

    stab_npz = out_dir / "stabilization.npz"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stabilized_mp4 = debug_dir / "stabilized.mp4"
    if not stab_npz.exists():
        compute_stabilization(
            video_path=video_path,
            anchor0_px=cfg.annotations.anchor_px,
            reference_frame_idx=cfg.reference_frame_idx,
            out_npz=stab_npz,
            debug_video_path=stabilized_mp4,
        )

    # Stage D: crop tracking
    from .crop_track import track_crop_boxes

    crop_npy = out_dir / "crop_boxes.npy"
    crop_mp4 = debug_dir / "crop_boxes.mp4"
    if not crop_npy.exists():
        track_crop_boxes(
            video_path=video_path,
            stabilization_npz=stab_npz,
            bbox0_px=cfg.annotations.bbox_px,
            out_npy=crop_npy,
            debug_video_path=crop_mp4,
            padding=float(cfg.params.get("crop", {}).get("padding", 0.2)),
            ema_alpha=float(cfg.params.get("crop", {}).get("ema_alpha", 0.9)),
            min_points=int(cfg.params.get("crop", {}).get("min_points", 10)),
        )

    # Stage E: 2D pose
    pose2d_npz = out_dir / "pose2d.npz"
    pose2d_mp4 = debug_dir / "pose2d_overlay.mp4"
    if not skip_2d and not pose2d_npz.exists():
        from .pose2d_mmpose import infer_pose2d_mmpose

        infer_pose2d_mmpose(
            video_path=video_path,
            stabilization_npz=stab_npz,
            crop_boxes_npy=crop_npy,
            out_npz=pose2d_npz,
            model=mmpose_model,
            device=device,
            debug_video_path=pose2d_mp4,
        )

    # Load 2D (for 3D + metrics).
    J2d_px = None
    joint_names_2d = None
    if pose2d_npz.exists():
        d = dict(np.load(pose2d_npz, allow_pickle=False))
        J2d_px = d["J2d_px"].astype(np.float32)
        joint_names_2d = tuple(str(x) for x in d["joint_names"].tolist())

    # Stage F/G/H: MotionBERT lift + metric-ish scaling.
    pose3d_npz = out_dir / "pose3d.npz"
    J3d_raw = None
    J3d_m = None
    alpha_scale = None
    joint_names_3d = None

    if (
        not skip_3d
        and motionbert_ckpt is not None
        and J2d_px is not None
        and joint_names_2d is not None
        and not pose3d_npz.exists()
    ):
        import numpy as np

        from .motionbert_format import prepare_motionbert_input_from_coco
        from .motionbert_lift import lift_pose3d_motionbert
        from .scaling import compute_alpha_scale_xy
        from .skeletons import H36M17_JOINT_NAMES

        stab = np.load(stab_npz)
        width = int(stab["width"])
        height = int(stab["height"])

        # For now we assume COCO-17 from MMPose 'human' inferencer.
        mb_in = prepare_motionbert_input_from_coco(J2d_px, width=width, height=height, mode="pixel")

        if motionbert_root is None:
            default_root = Path(__file__).resolve().parent.parent / "third_party" / "MotionBERT"
            motionbert_root = default_root if default_root.exists() else None
        if motionbert_root is None:
            raise SystemExit(
                "MotionBERT requested but no --motionbert-root provided and default vendored "
                "third_party/MotionBERT not found."
            )

        J3d_raw = lift_pose3d_motionbert(
            mb_in.X_h36m17,
            motionbert_root=motionbert_root,
            checkpoint_path=motionbert_ckpt,
            device=device,
            clip_len=clip_len,
            flip=flip,
            rootrel=rootrel,
        )
        joint_names_3d = H36M17_JOINT_NAMES

        # Compute 2D meters (root-relative) using boat scale.
        m_per_px = cfg.m_per_px
        if m_per_px is not None:
            # Map COCO px → H36M px (for comparing XY scale).
            from .skeletons import coco17_to_h36m17

            X_h36m_px = coco17_to_h36m17(J2d_px)
            pelvis_xy = X_h36m_px[:, 0:1, :2]
            J2d_m_xy = (X_h36m_px[:, :, :2] - pelvis_xy) * float(m_per_px)
            conf = X_h36m_px[:, :, 2]
            alpha_scale = compute_alpha_scale_xy(J3d_raw, J2d_m_xy, conf=conf, root_idx=0)
            if alpha_scale is not None:
                J3d_m = (float(alpha_scale) * J3d_raw).astype(np.float32)

        np.savez_compressed(
            pose3d_npz,
            J3d_raw=J3d_raw,
            J3d_m=J3d_m if J3d_m is not None else np.array([], dtype=np.float32),
            alpha_scale=np.array(alpha_scale if alpha_scale is not None else np.nan, dtype=np.float32),
            joint_names=np.array(joint_names_3d, dtype=str),
        )

    # Stage I: angles + metrics
    import numpy as np
    import pandas as pd

    angles_csv = out_dir / "angles.csv"
    metrics_json = out_dir / "metrics.json"

    if J2d_px is not None and joint_names_2d is not None:
        # If 3D exists, prefer it for angles.
        if pose3d_npz.exists():
            d3 = dict(np.load(pose3d_npz, allow_pickle=False))
            if d3.get("J3d_m") is not None and d3["J3d_m"].size:
                J_use = d3["J3d_m"].reshape(-1, 17, 3)
            else:
                J_use = d3["J3d_raw"].reshape(-1, 17, 3)
            joint_names_use = tuple(str(x) for x in d3["joint_names"].tolist())
        else:
            # Fallback: compute angles on 2D (mapped to H36M, in pixels).
            from .skeletons import H36M17_JOINT_NAMES, coco17_to_h36m17

            J_use = coco17_to_h36m17(J2d_px)[:, :, :3]
            joint_names_use = H36M17_JOINT_NAMES

        from .kinematics import compute_basic_angles_h36m17

        ang = compute_basic_angles_h36m17(J_use, joint_names_use)
        deg = np.degrees(ang.values_rad)
        df = pd.DataFrame(deg, columns=[f"{n}_deg" for n in ang.names])
        df.insert(0, "frame_idx", np.arange(df.shape[0], dtype=int))
        if cfg.video.fps > 0:
            df.insert(1, "time_s", df["frame_idx"] / float(cfg.video.fps))

        # Seat travel proxy (root x in meters, if scale available).
        if cfg.m_per_px is not None:
            from .skeletons import coco17_to_h36m17

            X_h36m_px = coco17_to_h36m17(J2d_px)
            pelvis_x_m = (X_h36m_px[:, 0, 0] - X_h36m_px[0, 0, 0]) * float(cfg.m_per_px)
            df["pelvis_x_m"] = pelvis_x_m.astype(np.float32)

        df.to_csv(angles_csv, index=False)

        # Simple summary metrics.
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
        summary["m_per_px"] = float(cfg.m_per_px) if cfg.m_per_px is not None else None
        summary["video"] = {"fps": float(cfg.video.fps), "frame_count": int(cfg.video.frame_count)}
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


def debug_pipeline(run_json_path: Path, out_dir: Optional[Path] = None) -> None:
    """Regenerate debug overlays/videos from saved npz/npy artifacts."""

    import numpy as np

    from .config import load_run_config

    run_json_path = Path(run_json_path)
    cfg = load_run_config(run_json_path)
    if out_dir is None:
        out_dir = run_json_path.parent
    out_dir = Path(out_dir)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    video_path = cfg.resolve_video_path(run_json_path)

    stab_npz = out_dir / "stabilization.npz"
    crop_npy = out_dir / "crop_boxes.npy"
    pose2d_npz = out_dir / "pose2d.npz"

    if stab_npz.exists():
        stab = np.load(stab_npz)
        A = stab["A"].astype(np.float32)
        width = int(stab["width"])
        height = int(stab["height"])
        fps = float(stab["fps"]) if "fps" in stab.files else float(cfg.video.fps)

        from .io_video import VideoWriter, iter_frames
        from .stabilize import warp_frame

        stabilized_mp4 = debug_dir / "stabilized.mp4"
        with VideoWriter(stabilized_mp4, fps=fps, frame_size=(width, height)) as vw:
            for idx, frame in iter_frames(video_path):
                stab_bgr = warp_frame(frame, A[idx], (width, height))
                vw.write(stab_bgr)

        if crop_npy.exists():
            boxes = np.load(crop_npy).astype(np.float32)
            from .viz import draw_bbox

            crop_mp4 = debug_dir / "crop_boxes.mp4"
            with VideoWriter(crop_mp4, fps=fps, frame_size=(width, height)) as vw:
                for idx, frame in iter_frames(video_path):
                    stab_bgr = warp_frame(frame, A[idx], (width, height))
                    vw.write(draw_bbox(stab_bgr, boxes[idx]))

        if pose2d_npz.exists():
            d2 = dict(np.load(pose2d_npz, allow_pickle=False))
            J2d_px = d2["J2d_px"].astype(np.float32)
            joint_names = tuple(str(x) for x in d2["joint_names"].tolist())

            from .pose2d_mmpose import COCO17_EDGES
            from .viz import draw_keypoints

            edges = COCO17_EDGES if len(joint_names) == 17 else None
            pose2d_mp4 = debug_dir / "pose2d_overlay.mp4"
            with VideoWriter(pose2d_mp4, fps=fps, frame_size=(width, height)) as vw:
                for idx, frame in iter_frames(video_path):
                    stab_bgr = warp_frame(frame, A[idx], (width, height))
                    vw.write(draw_keypoints(stab_bgr, J2d_px[idx], edges=edges))

