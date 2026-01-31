from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .model_assets import (
    DEFAULT_MOTIONBERT_MODEL,
    DEFAULT_POSE2D_MODEL,
    ensure_asset,
    get_motionbert_model,
    get_pose2d_model,
)
from .progress import ProgressReporter, get_progress


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    device: str = "cpu",
    mmpose_model: str = DEFAULT_POSE2D_MODEL,
    mmpose_weights: Optional[Path] = None,
    motionbert_root: Optional[Path] = None,
    motionbert_ckpt: Optional[Path] = None,
    motionbert_config: Optional[Path] = None,
    clip_len: int = 243,
    flip: bool = False,
    rootrel: bool = False,
    skip_2d: bool = False,
    skip_3d: bool = False,
    mmpose_config: Optional[Path] = None,
    mmpose_checkpoint: Optional[Path] = None,
    motionbert_model: str = DEFAULT_MOTIONBERT_MODEL,
    pose_tracking_smooth_alpha: Optional[float] = None,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """End-to-end pipeline orchestrator.

    This function is intentionally a thin layer that wires together the modular stages:
    annotations → stabilization → crop tracking → pose2d → (optional) pose3d → kinematics → debug.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prog = get_progress(progress)

    import numpy as np

    from .config import load_run_config
    from .pose_smoothing import (
        pose_smoothing_params_from_dict,
        smooth_joints_3d,
        smooth_keypoints_2d,
    )
    from .ui_annotate import annotate_video

    run_json = out_dir / "run.json"
    if not run_json.exists():
        annotate_video(video_path=video_path, out_dir=out_dir, reference_frame_idx=0)
    cfg = load_run_config(run_json)
    pose_smoothing = pose_smoothing_params_from_dict(cfg.params.get("pose_smoothing"))
    pose_tracking = cfg.params.get("pose_tracking", {})
    if pose_tracking_smooth_alpha is not None:
        smooth_alpha = float(pose_tracking_smooth_alpha)
        smooth_alpha = float(max(0.0, min(0.999, smooth_alpha)))
        pose_tracking = dict(pose_tracking) if isinstance(pose_tracking, dict) else {}
        pose_tracking["smooth_alpha"] = smooth_alpha

    rigger_bbox = getattr(cfg.annotations, "rigger_bbox_px", None)

    # Stage B: rigger tracking (optional) + stabilization
    from .stabilize import compute_stabilization, compute_stabilization_from_rigger_track

    stab_npz = out_dir / "stabilization.npz"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stabilized_mp4 = debug_dir / "stabilized.mp4"
    rigger_npz = out_dir / "rigger_track.npz"
    rigger_debug_mp4 = debug_dir / "rigger_track.mp4"
    use_rigger_track = False
    if rigger_bbox is not None:
        if not rigger_npz.exists():
            from .rig_tracker import track_rigger_bbox

            track_rigger_bbox(
                video_path=video_path,
                bbox0_px=rigger_bbox,
                reference_frame_idx=cfg.reference_frame_idx,
                out_npz=rigger_npz,
                debug_video_path=rigger_debug_mp4,
                ema_alpha=float(cfg.params.get("stabilization", {}).get("ema_alpha", 0.8)),
                min_points=int(cfg.params.get("stabilization", {}).get("min_points", 10)),
                progress=prog,
            )
        try:
            rigger_status = np.load(rigger_npz)["status"].astype(np.float32)
            use_rigger_track = float(np.nanmean(rigger_status)) >= 0.5
        except Exception:
            use_rigger_track = False
    if not stab_npz.exists():
        if use_rigger_track and rigger_bbox is not None:
            rx, ry, rw, rh = [float(v) for v in rigger_bbox]
            rigger_center = (rx + rw / 2.0, ry + rh / 2.0)
            compute_stabilization_from_rigger_track(
                video_path=video_path,
                rigger_track_npz=rigger_npz,
                anchor0_px=rigger_center,
                reference_frame_idx=cfg.reference_frame_idx,
                out_npz=stab_npz,
                debug_video_path=stabilized_mp4,
                progress=prog,
            )
        else:
            compute_stabilization(
                video_path=video_path,
                anchor0_px=cfg.annotations.anchor_px,
                reference_frame_idx=cfg.reference_frame_idx,
                out_npz=stab_npz,
                debug_video_path=stabilized_mp4,
                progress=prog,
            )

    # Stage D: crop tracking (optionally overridden by strict ID tracking)
    from .crop_track import track_crop_boxes

    crop_npy = out_dir / "crop_boxes.npy"
    crop_mp4 = debug_dir / "crop_boxes.mp4"
    strict_id = bool(pose_tracking.get("strict_id", False))
    if strict_id:
        person_track_npz = out_dir / "person_track.npz"
        if not person_track_npz.exists():
            from .person_tracking import track_person_deepsort

            track_person_deepsort(
                video_path=video_path,
                stabilization_npz=stab_npz,
                bbox0_px=cfg.annotations.bbox_px,
                out_npz=person_track_npz,
                reference_frame_idx=cfg.reference_frame_idx,
                device=device,
                model_name=str(pose_tracking.get("deepsort_model", "yolov8n.pt")),
                min_conf=float(pose_tracking.get("deepsort_min_conf", 0.25)),
                debug_video_path=debug_dir / "person_track.mp4",
                progress=prog,
            )
        if not crop_npy.exists():
            dtrack = np.load(person_track_npz)
            boxes = dtrack["boxes_xywh"].astype(np.float32)
            pad = float(pose_tracking.get("deepsort_padding", 0.2))
            if pad > 0:
                boxes[:, 0] -= boxes[:, 2] * pad / 2.0
                boxes[:, 1] -= boxes[:, 3] * pad / 2.0
                boxes[:, 2] *= 1.0 + pad
                boxes[:, 3] *= 1.0 + pad
            np.save(crop_npy, boxes)
    elif not crop_npy.exists():
        track_crop_boxes(
            video_path=video_path,
            stabilization_npz=stab_npz,
            bbox0_px=cfg.annotations.bbox_px,
            out_npy=crop_npy,
            rigger_track_npz=rigger_npz if use_rigger_track else None,
            rigger_bbox0_px=rigger_bbox if use_rigger_track else None,
            debug_video_path=crop_mp4,
            padding=float(cfg.params.get("crop", {}).get("padding", 0.2)),
            ema_alpha=float(cfg.params.get("crop", {}).get("ema_alpha", 0.9)),
            min_points=int(cfg.params.get("crop", {}).get("min_points", 10)),
            progress=prog,
        )

    # Stage E: 2D pose
    pose2d_npz = out_dir / "pose2d.npz"
    pose2d_mp4 = debug_dir / "pose2d_overlay.mp4"
    if not skip_2d and not pose2d_npz.exists():
        from .pose2d_mmpose import infer_pose2d_mmpose

        pose2d_model_id = mmpose_model
        pose2d_config = mmpose_config
        pose2d_checkpoint = mmpose_checkpoint or mmpose_weights
        spec = get_pose2d_model(mmpose_model)
        if spec is not None:
            pose2d_model_id = spec.model_id
            if pose2d_checkpoint is None and spec.checkpoint is not None:
                pose2d_checkpoint = ensure_asset(
                    spec.checkpoint.path,
                    spec.checkpoint.url,
                    expected_size=spec.checkpoint.size_bytes,
                    sha256=spec.checkpoint.sha256,
                )

        infer_pose2d_mmpose(
            video_path=video_path,
            stabilization_npz=stab_npz,
            crop_boxes_npy=crop_npy,
            out_npz=pose2d_npz,
            model=pose2d_model_id,
            device=device,
            pose2d_weights=mmpose_weights,
            config_path=pose2d_config,
            checkpoint_path=pose2d_checkpoint,
            debug_video_path=pose2d_mp4,
            pose_tracking=pose_tracking,
            progress=prog,
        )

    # Load 2D (for 3D + metrics).
    J2d_px = None
    joint_names_2d = None
    pose2d_conf = None
    pose2d_fps = None
    if pose2d_npz.exists():
        d = dict(np.load(pose2d_npz, allow_pickle=False))
        J2d_px = d["J2d_px"].astype(np.float32)
        joint_names_2d = tuple(str(x) for x in d["joint_names"].tolist())
        pose2d_fps = float(d.get("fps", 0.0))
        if "conf" in d:
            pose2d_conf = np.asarray(d["conf"], dtype=np.float32)

    if pose_smoothing.enabled and J2d_px is not None:
        J2d_px = smooth_keypoints_2d(J2d_px, pose_smoothing)
        if pose2d_conf is None:
            pose2d_conf = J2d_px[:, :, 2]
        if pose2d_npz.exists():
            np.savez_compressed(
                pose2d_npz,
                J2d_px=J2d_px,
                conf=pose2d_conf,
                joint_names=(
                    np.array(joint_names_2d, dtype=str)
                    if joint_names_2d is not None
                    else np.array([], dtype=str)
                ),
                fps=float(pose2d_fps) if pose2d_fps is not None else 0.0,
            )

    # Stage F/G/H: MotionBERT lift + metric-ish scaling.
    pose3d_npz = out_dir / "pose3d.npz"
    J3d_raw = None
    J3d_m = None
    alpha_scale = None
    joint_names_3d = None

    resolved_motionbert_ckpt = motionbert_ckpt
    resolved_motionbert_config = motionbert_config
    if not skip_3d and (resolved_motionbert_ckpt is None or resolved_motionbert_config is None):
        spec = get_motionbert_model(motionbert_model)
        if spec is not None:
            if resolved_motionbert_config is None:
                resolved_motionbert_config = spec.config_path
            if resolved_motionbert_config is not None and not resolved_motionbert_config.exists():
                raise FileNotFoundError(
                    f"MotionBERT config not found at {resolved_motionbert_config}. "
                    "Ensure the MotionBERT submodule is available."
                )
            if resolved_motionbert_ckpt is None:
                resolved_motionbert_ckpt = ensure_asset(
                    spec.checkpoint.path,
                    spec.checkpoint.url,
                    expected_size=spec.checkpoint.size_bytes,
                    sha256=spec.checkpoint.sha256,
                )

    if (
        not skip_3d
        and resolved_motionbert_ckpt is not None
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
            checkpoint_path=resolved_motionbert_ckpt,
            device=device,
            config_path=resolved_motionbert_config,
            clip_len=clip_len,
            flip=flip,
            rootrel=rootrel,
            progress=prog,
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

        if pose_smoothing.enabled and J3d_raw is not None:
            J3d_raw = smooth_joints_3d(J3d_raw, pose_smoothing)
            if J3d_m is not None:
                J3d_m = smooth_joints_3d(J3d_m, pose_smoothing)

        np.savez_compressed(
            pose3d_npz,
            J3d_raw=J3d_raw,
            J3d_m=J3d_m if J3d_m is not None else np.array([], dtype=np.float32),
            alpha_scale=np.array(
                alpha_scale if alpha_scale is not None else np.nan, dtype=np.float32
            ),
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

        angles_stage = prog.start("Stage I: angles + metrics", total=1, unit="step")
        ang = compute_basic_angles_h36m17(J_use, joint_names_use)
        deg = np.degrees(ang.values_rad)
        df = pd.DataFrame(deg, columns=[f"{n}_deg" for n in ang.names])
        if {"left_elbow_deg", "right_elbow_deg"}.issubset(df.columns):
            df["elbow_avg_deg"] = df[["left_elbow_deg", "right_elbow_deg"]].mean(
                axis=1, skipna=True
            )
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
        angles_stage.update(1)
        angles_stage.close()

        # Stage J (viz): overlay joints + angles onto the original video.
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        angles_overlay_mp4 = debug_dir / "angles_overlay.mp4"
        try:
            render_angles_overlay_video(
                video_path=video_path,
                stabilization_npz=stab_npz,
                pose2d_npz=pose2d_npz,
                angles_csv=angles_csv,
                out_video_path=angles_overlay_mp4,
                progress=prog,
            )
        except Exception:
            # Keep pipeline robust; overlays are best-effort.
            pass


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

            # Original-frame overlay with joint locations + angle values (if available).
            angles_csv = out_dir / "angles.csv"
            angles_overlay_mp4 = debug_dir / "angles_overlay.mp4"
            if angles_csv.exists():
                try:
                    render_angles_overlay_video(
                        video_path=video_path,
                        stabilization_npz=stab_npz,
                        pose2d_npz=pose2d_npz,
                        angles_csv=angles_csv,
                        out_video_path=angles_overlay_mp4,
                    )
                except Exception:
                    pass


def render_angles_overlay_video(
    video_path: Path,
    stabilization_npz: Path,
    pose2d_npz: Path,
    angles_csv: Path,
    out_video_path: Path,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """Render an overlay video on *original* frames showing joints and computed angles."""

    import cv2
    import numpy as np
    import pandas as pd

    from .io_video import VideoWriter, get_video_metadata, iter_frames
    from .pose2d_mmpose import COCO17_EDGES
    from .skeletons import H36M17_JOINT_NAMES, coco17_to_h36m17
    from .viz import (
        apply_affine_to_keypoints_xyc,
        draw_keypoints,
        draw_text_panel,
        draw_values_at_named_joints,
    )

    meta = get_video_metadata(video_path)
    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)  # (T,2,3)

    d2 = dict(np.load(pose2d_npz, allow_pickle=False))
    J2d_stab = d2["J2d_px"].astype(np.float32)  # stabilized full-frame px
    joint_names_2d = tuple(str(x) for x in d2["joint_names"].tolist())

    # Angles per-frame (degrees).
    df = pd.read_csv(angles_csv)
    if "frame_idx" in df.columns:
        # Ensure direct positional indexing by frame index.
        try:
            df = df.set_index("frame_idx", drop=False)
        except Exception:
            pass

    # Map COCO → H36M (still in stabilized coords), for placing labels at the relevant joints.
    X_h36m_stab = coco17_to_h36m17(J2d_stab)

    # Define which angles to draw and where to place them (at joint B).
    angle_name_to_joint = {
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "trunk_vs_horizontal": "thorax",
    }
    angle_cols = [f"{k}_deg" for k in angle_name_to_joint.keys()]
    angle_cols = [c for c in angle_cols if c in df.columns]

    fps = float(meta.fps) if meta.fps > 0 else float(stab["fps"]) if "fps" in stab.files else 30.0
    prog = get_progress(progress)
    total = int(min(meta.frame_count, J2d_stab.shape[0], A.shape[0]))
    stage = prog.start("Stage J: angles overlay", total=total, unit="frame")
    with VideoWriter(out_video_path, fps=fps, frame_size=(meta.width, meta.height)) as vw:
        for idx, frame_bgr in iter_frames(video_path):
            if idx >= J2d_stab.shape[0]:
                break
            if idx >= A.shape[0]:
                break

            invA = cv2.invertAffineTransform(A[idx])
            J2d_orig = apply_affine_to_keypoints_xyc(J2d_stab[idx], invA)
            X_h36m_orig = apply_affine_to_keypoints_xyc(X_h36m_stab[idx], invA)

            # Draw joints/skeleton.
            edges = COCO17_EDGES if len(joint_names_2d) == 17 else None
            out = draw_keypoints(frame_bgr, J2d_orig, edges=edges)

            # Extract angle values for this frame.
            vals_deg: dict[str, float] = {}
            for col in angle_cols:
                try:
                    v = float(df.loc[idx, col])  # type: ignore[index]
                except Exception:
                    continue
                if np.isfinite(v):
                    vals_deg[col.replace("_deg", "")] = v

            # Label angles near the joints they’re measured at (using H36M joints).
            label_values = {
                angle_name_to_joint[k]: v for k, v in vals_deg.items() if k in angle_name_to_joint
            }
            out = draw_values_at_named_joints(
                out,
                X_h36m_orig,
                joint_names=H36M17_JOINT_NAMES,
                values=label_values,
                fmt="{name}: {value:.1f} deg",
                min_conf=0.2,
                color=(0, 255, 0),
                font_scale=0.5,
                thickness=1,
            )

            # HUD panel with all angles.
            hud_lines = []
            for k in angle_name_to_joint.keys():
                if k in vals_deg:
                    hud_lines.append(f"{k}: {vals_deg[k]:.1f} deg")
            out = draw_text_panel(out, hud_lines, origin_xy=(10, 20))

            vw.write(out)
            stage.update(1)
    stage.close()