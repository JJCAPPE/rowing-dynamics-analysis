from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .model_assets import DEFAULT_MOTIONBERT_MODEL, DEFAULT_POSE2D_MODEL

def _path(p: str) -> Path:
    return Path(p).expanduser()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rowing-pose", description="Rowing video pose pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Create/edit run.json annotations for a video")
    p_ann.add_argument("--video", required=True, type=_path, help="Path to input video")
    p_ann.add_argument("--out", required=True, type=_path, help="Output directory for artifacts")
    p_ann.add_argument("--ref-frame", type=int, default=0, help="Reference frame index (default: 0)")
    p_ann.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run.json in --out if present",
    )

    p_run = sub.add_parser("run", help="Run the full pipeline (uses/creates run.json)")
    p_run.add_argument("--video", required=True, type=_path, help="Path to input video")
    p_run.add_argument("--out", required=True, type=_path, help="Output directory for artifacts")
    p_run.add_argument("--device", default="cpu", help="Torch device for inference (cpu/cuda)")
    p_run.add_argument(
        "--mmpose-model",
        default=DEFAULT_POSE2D_MODEL,
        help="MMPose inferencer alias or preset id (default: vitpose_high)",
    )
    p_run.add_argument(
        "--mmpose-config",
        type=_path,
        default=None,
        help="Path to MMPose config (overrides preset/alias)",
    )
    p_run.add_argument(
        "--mmpose-ckpt",
        type=_path,
        default=None,
        help="Path to MMPose checkpoint (overrides preset/alias)",
    )
    p_run.add_argument(
        "--motionbert-model",
        default=DEFAULT_MOTIONBERT_MODEL,
        help="MotionBERT preset id (default: motionbert_high)",
    )
    p_run.add_argument("--motionbert-root", type=_path, default=None, help="Path to MotionBERT repo")
    p_run.add_argument("--motionbert-ckpt", type=_path, default=None, help="Path to MotionBERT checkpoint")
    p_run.add_argument(
        "--motionbert-config", type=_path, default=None, help="Path to MotionBERT config YAML"
    )
    p_run.add_argument("--clip-len", type=int, default=243, help="MotionBERT clip length (default: 243)")
    p_run.add_argument("--flip", action="store_true", help="Enable MotionBERT flip augmentation")
    p_run.add_argument("--rootrel", action="store_true", help="Enable MotionBERT root-relative output")
    p_run.add_argument("--skip-2d", action="store_true", help="Skip 2D pose stage (use existing pose2d.npz)")
    p_run.add_argument("--skip-3d", action="store_true", help="Skip 3D lift stage")
    p_run.add_argument(
        "--pose-tracking-smooth-alpha",
        type=float,
        default=None,
        help="EMA smoothing alpha for 2D keypoints (0-0.1 recommended)",
    )

    p_dbg = sub.add_parser("debug", help="Regenerate debug overlays from saved artifacts")
    p_dbg.add_argument("--run", required=True, type=_path, help="Path to run.json")
    p_dbg.add_argument("--out", type=_path, default=None, help="Output directory (default: run.json dir)")

    return p


def cmd_annotate(video: Path, out_dir: Path, ref_frame: int, overwrite: bool) -> None:
    from .ui_annotate import annotate_video

    out_dir.mkdir(parents=True, exist_ok=True)
    run_json = out_dir / "run.json"
    if run_json.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing {run_json}. Re-run with --overwrite to replace it."
        )
    annotate_video(video_path=video, out_dir=out_dir, reference_frame_idx=ref_frame)


def cmd_run(
    video: Path,
    out_dir: Path,
    device: str,
    mmpose_model: str,
    mmpose_config: Optional[Path],
    mmpose_ckpt: Optional[Path],
    motionbert_model: str,
    motionbert_root: Optional[Path],
    motionbert_ckpt: Optional[Path],
    motionbert_config: Optional[Path],
    clip_len: int,
    flip: bool,
    rootrel: bool,
    skip_2d: bool,
    skip_3d: bool,
    pose_tracking_smooth_alpha: Optional[float],
) -> None:
    from .pipeline import run_pipeline
    from .progress import TqdmProgress

    run_pipeline(
        video_path=video,
        out_dir=out_dir,
        device=device,
        mmpose_model=mmpose_model,
        mmpose_config=mmpose_config,
        mmpose_checkpoint=mmpose_ckpt,
        motionbert_model=motionbert_model,
        motionbert_root=motionbert_root,
        motionbert_ckpt=motionbert_ckpt,
        motionbert_config=motionbert_config,
        clip_len=clip_len,
        flip=flip,
        rootrel=rootrel,
        skip_2d=skip_2d,
        skip_3d=skip_3d,
        pose_tracking_smooth_alpha=pose_tracking_smooth_alpha,
        progress=TqdmProgress(),
    )


def cmd_debug(run_json: Path, out_dir: Optional[Path]) -> None:
    from .pipeline import debug_pipeline

    debug_pipeline(run_json_path=run_json, out_dir=out_dir)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "annotate":
        cmd_annotate(
            video=args.video,
            out_dir=args.out,
            ref_frame=args.ref_frame,
            overwrite=args.overwrite,
        )
        return

    if args.cmd == "run":
        cmd_run(
            video=args.video,
            out_dir=args.out,
            device=args.device,
            mmpose_model=args.mmpose_model,
            mmpose_config=args.mmpose_config,
            mmpose_ckpt=args.mmpose_ckpt,
            motionbert_model=args.motionbert_model,
            motionbert_root=args.motionbert_root,
            motionbert_ckpt=args.motionbert_ckpt,
            motionbert_config=args.motionbert_config,
            clip_len=args.clip_len,
            flip=args.flip,
            rootrel=args.rootrel,
            skip_2d=args.skip_2d,
            skip_3d=args.skip_3d,
            pose_tracking_smooth_alpha=args.pose_tracking_smooth_alpha,
        )
        return

    if args.cmd == "debug":
        cmd_debug(run_json=args.run, out_dir=args.out)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

