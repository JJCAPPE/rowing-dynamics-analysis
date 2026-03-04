from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from parse_sports2d import (
    extract_coco17_from_trc,
    parse_mot_file,
    parse_trc_file,
    write_angles_csv,
    write_points_csv,
    write_points_npz,
)
from plot_angles import generate_angles_plot
from runner_sports2d import (
    Sports2DOptions,
    Sports2DError,
    build_sports2d_config,
    run_sports2d,
)
from motionbert_3d import run_motionbert
from overlay_3d import generate_pose3d_overlay_video, get_video_metadata
from progress_utils import ProgressMux
from stroke_signal import StrokeTrackingOutputs, run_stroke_signal_tracking


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
RUNS_DIR = REPO_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    input_video: Path
    sports2d_output_dir: Path
    sports2d_annotated_video: Path
    exports_dir: Path
    motionbert_dir: Path
    overlay_dir: Path
    stroke_dir: Optional[Path]
    stroke_signal_csv: Optional[Path]
    stroke_signal_npz: Optional[Path]
    zip_path: Path


@dataclass(frozen=True)
class ExportSummary:
    trc_files: List[Path]
    mot_files: List[Path]
    points_csv: List[Path]
    points_npz: List[Path]
    angles_csv: List[Path]
    angles_plots: List[Path]
    angle_plot_errors: List[str]


@dataclass(frozen=True)
class StrokeTrackingOptions:
    enabled: bool
    handle_source: str = "pose"
    machine_bbox: Optional[Tuple[float, float, float, float]] = None
    machine_cable_point: Optional[Tuple[float, float]] = None
    handle_bbox: Optional[Tuple[float, float, float, float]] = None
    handle_pose_npz: Optional[Path] = None
    m_per_px: Optional[float] = None
    ema_alpha: float = 0.4
    min_points: int = 10
    smooth_window_s: float = 0.2
    min_cycle_s: float = 0.8
    min_drive_s: float = 0.2
    min_recover_s: float = 0.2
    min_drive_disp_frac: float = 0.05
    slope_tol_frac: float = 0.05
    finish_velocity_frac: float = 0.85
    catch_velocity_frac: float = 0.0
    finish_method: str = "velocity_threshold"
    debug_video: bool = True


def _sanitize_stem(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    safe = safe.strip("_")
    return safe or "video"


def _copy_input_video(src_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / f"input{src_path.suffix or '.mp4'}"
    if src_path.resolve() != dst.resolve():
        shutil.copy2(str(src_path), str(dst))
    return dst


def _save_uploaded_video(uploaded, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix or ".mp4"
    dst = dest_dir / f"input{suffix}"
    with dst.open("wb") as f:
        f.write(uploaded.getbuffer())
    return dst


def _person_tag(person_index: int) -> str:
    return f"_person{person_index:02d}"


def _person_pattern(person_index: int) -> re.Pattern[str]:
    tag = _person_tag(person_index)
    return re.compile(rf"{re.escape(tag)}(?!\\d)|_person{person_index}(?!\\d)")


def _filter_person_files(files: List[Path], person_index: int) -> List[Path]:
    pattern = _person_pattern(person_index)
    return sorted(
        [p for p in files if pattern.search(p.stem)],
        key=lambda p: p.name,
    )




def _zip_outputs(zip_path: Path, paths: List[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.is_dir():
                for sub in path.rglob("*"):
                    if sub.is_file():
                        zf.write(sub, sub.relative_to(zip_path.parent))
            elif path.is_file():
                zf.write(path, path.relative_to(zip_path.parent))


POSE_MODEL_DETAILS: Dict[str, Dict[str, Any]] = {
    "Whole_body": {
        "keypoints_set": "COCO_133",
        "includes": ["body", "feet", "hands"],
    },
    "Whole_body_wrist": {
        "keypoints_set": "COCO_133_WRIST",
        "includes": ["body", "feet", "2 hand points"],
    },
    "Body_with_feet": {
        "keypoints_set": "HALPE_26",
        "includes": ["body", "feet"],
    },
    "Body": {
        "keypoints_set": "COCO_17",
        "notes": "Marker augmentation does not apply; kinematics still work.",
    },
    "Hand": {
        "keypoints_set": "HAND_21",
        "notes": "Only supported in lightweight mode (RTMLib).",
    },
    "Face": {"keypoints_set": "FACE_106"},
    "Animal": {"keypoints_set": "ANIMAL2D_17"},
}

MODE_PRESET_DETAILS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "preset": "RTMLib balanced",
        "equivalent_custom_mode_example": {
            "det_class": "YOLOX",
            "det_model": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "det_input_size": [640, 640],
            "pose_class": "RTMPose",
            "pose_model": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            "pose_input_size": [192, 256],
            "source": "Sports2D Demo Config (example equivalent to balanced)",
        },
    }
}

LOG_EXCERPT_PATTERNS = [
    re.compile(r"Estimating pose", re.IGNORECASE),
    re.compile(r"Using .* model .*pose estimation", re.IGNORECASE),
    re.compile(r"Valid .* installation found", re.IGNORECASE),
    re.compile(r"load .*\\.onnx", re.IGNORECASE),
    re.compile(r"Persons detection is run every", re.IGNORECASE),
    re.compile(r"person is analyzed", re.IGNORECASE),
    re.compile(r"keypoint_likelihood_threshold", re.IGNORECASE),
    re.compile(r"Skipping .* angle computation", re.IGNORECASE),
    re.compile(r"Post-processing pose", re.IGNORECASE),
    re.compile(r"Interpolating missing sequences", re.IGNORECASE),
    re.compile(r"Rejecting outliers", re.IGNORECASE),
    re.compile(r"Filtering with", re.IGNORECASE),
    re.compile(r"Converting pose to meters", re.IGNORECASE),
    re.compile(r"Converting from pixels to meters", re.IGNORECASE),
    re.compile(r"Perspective effects corrected", re.IGNORECASE),
    re.compile(r"Camera horizon", re.IGNORECASE),
    re.compile(r"Floor level", re.IGNORECASE),
    re.compile(r"Post-processing angles", re.IGNORECASE),
    re.compile(r"Correcting segment angles", re.IGNORECASE),
    re.compile(r"Pose in pixels saved to", re.IGNORECASE),
    re.compile(r"Pose in meters saved to", re.IGNORECASE),
    re.compile(r"Angles saved to", re.IGNORECASE),
    re.compile(r"Processed video saved to", re.IGNORECASE),
]


def _pose_model_info(pose_model: str) -> Dict[str, Any]:
    base = {"pose_model": pose_model}
    base.update(POSE_MODEL_DETAILS.get(pose_model, {}))
    return base


def _mode_info(mode: str) -> Dict[str, Any]:
    if mode in MODE_PRESET_DETAILS:
        return MODE_PRESET_DETAILS[mode]
    return {"preset": f"RTMLib {mode} (internal preset; model selection handled by RTMLib)"}


def _read_log_lines(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = [line.rstrip() for line in text.splitlines()]
    return [line for line in lines if line.strip()]


def _extract_latest_run_lines(lines: List[str]) -> List[str]:
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if line.startswith("Processing "):
            return lines[idx:]
    return lines


def _merge_lines(*parts: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lines in parts:
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            merged.append(line)
    return merged


def _parse_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox must have 4 comma-separated values: x,y,w,h")
    x, y, w, h = [float(p) for p in parts]
    if w <= 1 or h <= 1:
        raise ValueError("BBox width/height must be > 1")
    return float(x), float(y), float(w), float(h)


def _parse_point(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Point must have 2 comma-separated values: x,y")
    x, y = [float(p) for p in parts]
    return float(x), float(y)


def _select_log_excerpt(lines: List[str]) -> List[str]:
    excerpt: List[str] = []
    for line in lines:
        if any(p.search(line) for p in LOG_EXCERPT_PATTERNS):
            excerpt.append(line)
    return excerpt


def _parse_runtime_info(lines: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    onnx_models: List[str] = []
    warnings: List[str] = []

    for line in lines:
        if match := re.search(
            r"Using (?P<model>[A-Za-z0-9_]+) model(?: \\((?P<desc>[^)]+)\\))? for pose estimation in (?P<mode>[A-Za-z0-9_]+) mode",
            line,
        ):
            info["pose_estimation_model"] = match.group("model")
            if match.group("desc"):
                info["pose_estimation_model_desc"] = match.group("desc")
            info["pose_estimation_mode"] = match.group("mode")
        if match := re.search(r"load (?P<path>.+?\\.onnx) with (?P<backend>\\w+) backend", line):
            onnx_models.append(match.group("path"))
            info.setdefault("onnx_backend", match.group("backend"))
        if match := re.search(r"Valid .* installation found: using (?P<backend>.+) backend", line):
            info["runtime_backend"] = match.group("backend").strip()
        if match := re.search(
            r"Persons detection is run every (?P<freq>\\d+) frames .* Tracking is done with (?P<tracker>\\w+)\\.",
            line,
        ):
            info["det_frequency_frames"] = int(match.group("freq"))
            info["tracking_mode"] = match.group("tracker")
        if match := re.search(
            r"(?P<count>\\d+) person(?:s)? is analyzed\\. Person ordering method is (?P<method>.+)\\.",
            line,
        ):
            info["persons_analyzed"] = int(match.group("count"))
            info["person_ordering_method"] = match.group("method")
        if match := re.search(
            r"keypoint_likelihood_threshold=(?P<kpt>[0-9.]+), average_likelihood_threshold=(?P<avg>[0-9.]+), keypoint_number_threshold=(?P<num>[0-9.]+)",
            line,
        ):
            info["keypoint_likelihood_threshold"] = float(match.group("kpt"))
            info["average_likelihood_threshold"] = float(match.group("avg"))
            info["keypoint_number_threshold"] = float(match.group("num"))
        if match := re.search(
            r"Skipping (?P<angle>.+) angle computation because at least one of the following keypoints is not provided by the pose estimation model: (?P<missing>.+)",
            line,
        ):
            warnings.append(
                f"{match.group('angle')} skipped: missing {match.group('missing')}"
            )
        if match := re.search(
            r"Interpolating missing sequences if they are smaller than (?P<gap>\\d+) frames\\. Large gaps filled with (?P<fill>\\w+)",
            line,
        ):
            info["interpolation_gap_frames"] = int(match.group("gap"))
            info["interpolation_fill"] = match.group("fill")
        if "Rejecting outliers with a Hampel filter" in line:
            info["outlier_rejection"] = "hampel"
        if match := re.search(r"Filtering with (?P<filter>.+)", line):
            info["filtering"] = match.group("filter")
        if match := re.search(
            r"Converting from pixels to meters using a person height of (?P<hm>[0-9.]+) in meters .* and of (?P<hpx>[0-9.]+) in pixels",
            line,
        ):
            info["height_m"] = float(match.group("hm"))
            info["height_px"] = float(match.group("hpx"))
        if match := re.search(
            r"Perspective effects corrected using a camera-to-person distance of (?P<dist>[0-9.]+) m",
            line,
        ):
            info["distance_to_camera_m"] = float(match.group("dist"))
        if match := re.search(r"Camera horizon: (?P<ang>[0-9.]+)", line):
            info["camera_horizon_deg"] = float(match.group("ang"))
        if match := re.search(r"Floor level: (?P<floor>[0-9.]+) px", line):
            info["floor_level_px"] = float(match.group("floor"))
        if match := re.search(
            "Correcting segment angles by removing the (?P<ang>[0-9.]+)(?:\\u00b0)? floor angle",
            line,
        ):
            info["floor_angle_correction_deg"] = float(match.group("ang"))

    if onnx_models:
        info["onnx_models"] = onnx_models
    if warnings:
        info["warnings"] = warnings
    return info


def _estimate_json_preview_height(text: str) -> int:
    lines = max(text.count("\n") + 1, 8)
    return min(800, max(240, lines * 18))


ProgressCallback = Callable[[str, float], None]

STEP_SPANS: Dict[int, Tuple[float, float]] = {
    1: (0.00, 0.52),
    2: (0.52, 0.60),
    3: (0.60, 0.64),
    4: (0.64, 0.80),
    5: (0.80, 0.88),
    6: (0.88, 0.97),
    7: (0.97, 1.00),
}

def _build_sports2d_run_details(
    *,
    input_video: Path,
    run_dir: Path,
    sports2d_out_dir: Path,
    options: Sports2DOptions,
    person_index: int = 0,
) -> Dict[str, Any]:
    config_overrides = build_sports2d_config(
        input_video, sports2d_out_dir, options, include_defaults=False
    )
    config_effective = build_sports2d_config(
        input_video, sports2d_out_dir, options, include_defaults=True, strip_custom=True
    )

    base_cfg = config_effective.get("base", {})
    pose_cfg = config_effective.get("pose", {})
    post_cfg = config_effective.get("post-processing", {})
    px_cfg = config_effective.get("px_to_meters_conversion", {})
    angles_cfg = config_effective.get("angles", {})

    pose_model_name = str(pose_cfg.get("pose_model", options.pose_model))
    mode_name = str(pose_cfg.get("mode", options.mode))

    pipeline_steps = [
        {
            "step": "video_io",
            "algorithm": "OpenCV video reader",
            "input": str(input_video),
        },
        {
            "step": "person_detection",
            "algorithm": "RTMLib top-down detector (per mode preset)",
            "det_frequency_frames": pose_cfg.get("det_frequency"),
        },
        {
            "step": "pose_estimation",
            "algorithm": "RTMLib RTMPose (per mode preset)",
            "pose_model": pose_model_name,
            "mode": mode_name,
        },
        {
            "step": "tracking",
            "algorithm": pose_cfg.get("tracking_mode", "sports2d"),
            "person_ordering_method": base_cfg.get("person_ordering_method"),
            "nb_persons_to_detect": base_cfg.get("nb_persons_to_detect"),
        },
        {
            "step": "person_selection",
            "person_index": person_index,
            "person_file_tag": _person_tag(person_index),
        },
        {
            "step": "pose_post_processing",
            "interpolate": post_cfg.get("interpolate"),
            "interp_gap_smaller_than": post_cfg.get("interp_gap_smaller_than"),
            "fill_large_gaps_with": post_cfg.get("fill_large_gaps_with"),
            "reject_outliers": post_cfg.get("reject_outliers"),
            "filter": post_cfg.get("filter"),
            "filter_type": post_cfg.get("filter_type"),
            "filter_params": post_cfg.get("butterworth"),
        },
        {
            "step": "px_to_meters",
            "enabled": px_cfg.get("to_meters"),
            "perspective_unit": px_cfg.get("perspective_unit"),
            "perspective_value": px_cfg.get("perspective_value"),
            "floor_angle": px_cfg.get("floor_angle"),
            "xy_origin": px_cfg.get("xy_origin"),
        },
        {
            "step": "angles",
            "calculate_angles": base_cfg.get("calculate_angles"),
            "joint_angles": angles_cfg.get("joint_angles"),
            "segment_angles": angles_cfg.get("segment_angles"),
            "flip_left_right": angles_cfg.get("flip_left_right"),
            "correct_segment_angles_with_floor_angle": angles_cfg.get(
                "correct_segment_angles_with_floor_angle"
            ),
        },
        {
            "step": "outputs",
            "save_vid": base_cfg.get("save_vid"),
            "save_img": base_cfg.get("save_img"),
            "save_pose": base_cfg.get("save_pose"),
            "save_angles": base_cfg.get("save_angles"),
        },
    ]

    logs_path = sports2d_out_dir / "logs.txt"
    console_log_path = sports2d_out_dir / "console.log"
    logs_lines = _extract_latest_run_lines(_read_log_lines(logs_path))
    console_lines = _extract_latest_run_lines(_read_log_lines(console_log_path))
    merged_lines = _merge_lines(logs_lines, console_lines)

    runtime_info = _parse_runtime_info(merged_lines)
    log_excerpt = _select_log_excerpt(merged_lines)

    return {
        "run": {
            "run_dir": str(run_dir),
            "input_video": str(input_video),
            "sports2d_result_dir": str(sports2d_out_dir),
            "logs": {"sports2d": str(logs_path), "console": str(console_log_path)},
        },
        "sports2d": {
            "pose_model": _pose_model_info(pose_model_name),
            "mode": _mode_info(mode_name),
            "config_overrides": config_overrides,
            "config_effective": config_effective,
        },
        "pipeline": pipeline_steps,
        "runtime": runtime_info,
        "log_excerpt": log_excerpt,
    }

def _export_sports2d_outputs(
    trc_files: List[Path], mot_files: List[Path], exports_dir: Path
) -> ExportSummary:
    points_csv = []
    points_npz = []
    angles_csv = []

    for trc in trc_files:
        trc_data = parse_trc_file(trc)
        stem = trc.stem
        out_csv = exports_dir / f"{stem}_points.csv"
        out_npz = exports_dir / f"{stem}_points.npz"
        write_points_csv(trc_data, out_csv)
        write_points_npz(trc_data, out_npz)
        points_csv.append(out_csv)
        points_npz.append(out_npz)

    for mot in mot_files:
        mot_data = parse_mot_file(mot)
        stem = mot.stem
        out_csv = exports_dir / f"{stem}_angles.csv"
        write_angles_csv(mot_data, out_csv)
        angles_csv.append(out_csv)

    return ExportSummary(
        trc_files=trc_files,
        mot_files=mot_files,
        points_csv=points_csv,
        points_npz=points_npz,
        angles_csv=angles_csv,
        angles_plots=[],
        angle_plot_errors=[],
    )


def _generate_motionbert_angles_plot(
    angles_csv: Path,
    exports_dir: Path,
    video_path: Optional[Path],
) -> Tuple[List[Path], List[str]]:
    plot_path = exports_dir / f"{angles_csv.stem}_plot.png"
    try:
        generate_angles_plot(
            angles_csv,
            plot_path,
            title="Rowing angles (3D)",
            video_path=video_path,
            include_thumbnails=video_path is not None,
            thumb_max_px=None,
            thumb_zoom=0.045,
            fig_dpi=300,
        )
        return [plot_path], []
    except Exception as exc:
        return [], [f"{angles_csv.name}: {exc}"]


def _run_pipeline(
    *,
    input_video: Path,
    run_dir: Path,
    options: Sports2DOptions,
    stroke_tracking: StrokeTrackingOptions,
    person_index: int = 0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[RunArtifacts, ExportSummary, Optional[Path]]:
    sports2d_out_dir = run_dir / "sports2d"
    exports_dir = run_dir / "exports"
    motionbert_dir = run_dir / "motionbert"
    overlay_dir = run_dir / "overlay"
    stroke_dir = run_dir / "stroke"
    progress = ProgressMux(progress_callback)

    step1 = progress.span(
        *STEP_SPANS[1],
        prefix="Step 1/7: Running Sports2D (pose + tracking)",
    )
    step1("Starting", 0.0)
    result = run_sports2d(
        input_video,
        sports2d_out_dir,
        options,
        progress_callback=step1,
    )
    step1("Completed", 1.0)

    step2 = progress.span(
        *STEP_SPANS[2],
        prefix="Step 2/7: Exporting Sports2D outputs",
    )
    step2("Collecting outputs", 0.05)
    exports_dir.mkdir(parents=True, exist_ok=True)
    person_trc_files = _filter_person_files(result.trc_files, person_index)
    person_mot_files = _filter_person_files(result.mot_files, person_index)
    if not person_trc_files:
        raise RuntimeError(
            f"No TRC files found for person index {person_index}. "
            "Increase the number of persons to detect or choose a different index."
        )
    step2("Parsing TRC and MOT", 0.35)
    summary_base = _export_sports2d_outputs(
        person_trc_files,
        person_mot_files,
        exports_dir,
    )
    step2("Completed", 1.0)

    person_trc = person_trc_files[0]

    step3 = progress.span(
        *STEP_SPANS[3],
        prefix="Step 3/7: Preparing MotionBERT inputs",
    )
    step3("Loading TRC data", 0.05)
    trc_data = parse_trc_file(person_trc)
    step3("Extracting COCO-17 keypoints", 0.6)
    J2d_px, _ = extract_coco17_from_trc(trc_data)
    meta = get_video_metadata(result.annotated_video)
    step3("Completed", 1.0)

    step4 = progress.span(
        *STEP_SPANS[4],
        prefix="Step 4/7: Running MotionBERT 3D lift",
    )
    mb_outputs = run_motionbert(
        J2d_px,
        width=meta.width,
        height=meta.height,
        out_dir=motionbert_dir,
        fps=meta.fps,
        clip_len=243,
        flip=False,
        rootrel=False,
        device=options.device,
        progress_callback=step4,
    )
    step4("Completed", 1.0)

    step5 = progress.span(
        *STEP_SPANS[5],
        prefix="Step 5/7: Tracking handle vs machine",
    )
    stroke_outputs: Optional[StrokeTrackingOutputs] = None
    stroke_error: Optional[str] = None
    if stroke_tracking.enabled:
        step5("Running stroke tracker", 0.05)
        try:
            stroke_dir.mkdir(parents=True, exist_ok=True)
            stroke_outputs = run_stroke_signal_tracking(
                video_path=result.annotated_video,
                out_dir=stroke_dir,
                angles_csv=mb_outputs.angles_csv,
                reference_frame_idx=0,
                machine_bbox=stroke_tracking.machine_bbox,
                machine_cable_point=stroke_tracking.machine_cable_point,
                handle_bbox=stroke_tracking.handle_bbox,
                handle_source=stroke_tracking.handle_source,
                handle_pose_npz=(
                    stroke_tracking.handle_pose_npz
                    if stroke_tracking.handle_pose_npz is not None
                    else (summary_base.points_npz[0] if summary_base.points_npz else None)
                ),
                annotate=False,
                m_per_px=stroke_tracking.m_per_px,
                ema_alpha=stroke_tracking.ema_alpha,
                min_points=stroke_tracking.min_points,
                smooth_window_s=stroke_tracking.smooth_window_s,
                min_cycle_s=stroke_tracking.min_cycle_s,
                min_drive_s=stroke_tracking.min_drive_s,
                min_recover_s=stroke_tracking.min_recover_s,
                min_drive_disp_frac=stroke_tracking.min_drive_disp_frac,
                slope_tol_frac=stroke_tracking.slope_tol_frac,
                finish_velocity_frac=stroke_tracking.finish_velocity_frac,
                catch_velocity_frac=stroke_tracking.catch_velocity_frac,
                finish_method=stroke_tracking.finish_method,
                create_plot=True,
                plot_video_path=result.annotated_video,
                debug_video=stroke_tracking.debug_video,
            )
        except Exception as exc:
            stroke_error = str(exc)
        step5("Completed", 1.0)
    else:
        step5("Skipped (disabled)", 1.0)

    step6 = progress.span(
        *STEP_SPANS[6],
        prefix="Step 6/7: Rendering 3D overlay + plots",
    )
    angles_plots: List[Path] = []
    angle_plot_errors: List[str] = []
    if stroke_outputs is not None and stroke_outputs.merged_angles_plot is not None:
        angles_plots = [stroke_outputs.merged_angles_plot]
        step6("Using stroke-generated angles plot", 0.25)
    else:
        step6("Generating angles plot", 0.05)
        angles_plots, angle_plot_errors = _generate_motionbert_angles_plot(
            mb_outputs.angles_csv,
            exports_dir,
            input_video,
        )
        step6("Angles plot ready", 0.3)
    if stroke_error is not None:
        angle_plot_errors.append(f"stroke tracking: {stroke_error}")

    overlay_video = overlay_dir / "pose3d_overlay.mp4"

    def _overlay_progress(label: str, prog: float) -> None:
        step6(label, 0.3 + 0.7 * max(0.0, min(1.0, prog)))

    generate_pose3d_overlay_video(
        video_path=result.annotated_video,
        pose3d_npz=mb_outputs.pose3d_npz,
        out_video_path=overlay_video,
        stroke_signal_npz=(
            stroke_outputs.stroke_npz
            if stroke_outputs is not None and stroke_outputs.stroke_npz.exists()
            else None
        ),
        progress_callback=_overlay_progress,
    )
    step6("Completed", 1.0)

    step7 = progress.span(
        *STEP_SPANS[7],
        prefix="Step 7/7: Packaging outputs",
    )
    step7("Creating ZIP archive", 0.2)
    zip_path = run_dir / "results.zip"
    zip_inputs = [result.output_dir, exports_dir, motionbert_dir, overlay_dir]
    if stroke_outputs is not None:
        zip_inputs.append(stroke_dir)
    _zip_outputs(zip_path, zip_inputs)
    step7("Completed", 1.0)

    artifacts = RunArtifacts(
        run_dir=run_dir,
        input_video=input_video,
        sports2d_output_dir=result.output_dir,
        sports2d_annotated_video=result.annotated_video,
        exports_dir=exports_dir,
        motionbert_dir=motionbert_dir,
        overlay_dir=overlay_dir,
        stroke_dir=stroke_dir if stroke_outputs is not None else None,
        stroke_signal_csv=stroke_outputs.stroke_csv if stroke_outputs is not None else None,
        stroke_signal_npz=stroke_outputs.stroke_npz if stroke_outputs is not None else None,
        zip_path=zip_path,
    )

    summary = ExportSummary(
        trc_files=summary_base.trc_files,
        mot_files=summary_base.mot_files,
        points_csv=summary_base.points_csv,
        points_npz=summary_base.points_npz,
        angles_csv=summary_base.angles_csv,
        angles_plots=angles_plots,
        angle_plot_errors=angle_plot_errors,
    )

    return artifacts, summary, overlay_video


def main() -> None:
    st.set_page_config(page_title="Sports2D Analysis Demo", layout="wide")

    st.title("Sports2D Analysis Demo")
    st.caption(
        "Sports2D-only pipeline with MotionBERT 3D lift overlay. "
    )

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Video source", ["Upload", "Path"], horizontal=True)
        uploaded = None
        src_path = None
        if mode == "Upload":
            uploaded = st.file_uploader("Upload video", type=None)
            if uploaded is not None:
                src_path = Path(uploaded.name)
        else:
            p = st.text_input("Video path", placeholder="/path/to/video.mp4")
            if p:
                src_path = Path(p).expanduser()

        st.divider()
        st.header("Sports2D Settings")
        nb_persons = st.selectbox("Max persons to detect", [1, 2, 3, "all"], index=0)
        person_index = st.number_input(
            "Person index (0-based)",
            min_value=0,
            value=0,
            step=1,
            help="Select which person to use for MotionBERT + exports. Only one person is processed per run.",
        )
        first_person_height = st.number_input("First person height (m)", 1.2, 2.5, 1.95, 0.01)
        distance_m = st.number_input("Distance to camera (m)", 0.0, 50.0, 5.0, 0.5)
        pose_model = st.selectbox(
            "Pose model",
            ["Whole_body", "Whole_body_wrist", "Body_with_feet", "Body"],
            index=2,
        )
        mode_choice = st.selectbox("Mode", ["lightweight", "balanced", "performance"], index=1)
        det_frequency = st.slider("Detection frequency (frames)", 1, 30, 4, 1)

        device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

        st.divider()
        st.header("Stroke Tracking")
        enable_stroke = st.checkbox("Enable handle/machine stroke tracking", value=False)
        handle_source_choice = st.selectbox(
            "Handle source",
            ["manual bbox", "pose midpoint"],
            index=1,
        )
        machine_bbox_text = st.text_input(
            "Machine reference bbox (x,y,w,h)",
            value="",
            help="Required for stroke tracking.",
        )
        machine_cable_point_text = st.text_input(
            "Machine cable entry point (x,y)",
            value="",
            help="Required for stroke tracking. This is where the cable enters the erg.",
        )
        handle_bbox_text = ""
        if handle_source_choice == "manual bbox":
            handle_bbox_text = st.text_input(
                "Handle bbox (x,y,w,h)",
                value="",
                help="Required for manual handle tracking.",
            )
        m_per_px = st.number_input(
            "Meters-per-pixel scale (optional, 0 = unset)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.001,
        )
        debug_video = st.checkbox("Save stroke tracking debug video", value=True)

        st.divider()
        st.header("Run")
        run_button = st.button("Run Sports2D + 3D", type="primary")

    if run_button:
        if mode == "Path" and (src_path is None or not src_path.exists()):
            st.error("Video path does not exist.")
            st.stop()
        if mode == "Upload" and uploaded is None:
            st.error("Please upload a video.")
            st.stop()
        person_index = int(person_index)
        if isinstance(nb_persons, int) and person_index >= nb_persons:
            st.error(
                f"Person index {person_index} is out of range for max persons {nb_persons}."
            )
            st.stop()

        stroke_tracking = StrokeTrackingOptions(enabled=False)
        if enable_stroke:
            if not machine_bbox_text.strip():
                st.error("Machine bbox is required for stroke tracking.")
                st.stop()
            try:
                machine_bbox = _parse_bbox(machine_bbox_text)
            except ValueError as exc:
                st.error(f"Invalid machine bbox: {exc}")
                st.stop()
            if not machine_cable_point_text.strip():
                st.error("Machine cable entry point is required for stroke tracking.")
                st.stop()
            try:
                machine_cable_point = _parse_point(machine_cable_point_text)
            except ValueError as exc:
                st.error(f"Invalid machine cable entry point: {exc}")
                st.stop()
            handle_source = "pose" if handle_source_choice == "pose midpoint" else "manual"
            handle_bbox = None
            if handle_source == "manual":
                if not handle_bbox_text.strip():
                    st.error("Handle bbox is required for manual handle tracking.")
                    st.stop()
                try:
                    handle_bbox = _parse_bbox(handle_bbox_text)
                except ValueError as exc:
                    st.error(f"Invalid handle bbox: {exc}")
                    st.stop()
            stroke_tracking = StrokeTrackingOptions(
                enabled=True,
                handle_source=handle_source,
                machine_bbox=machine_bbox,
                machine_cable_point=machine_cable_point,
                handle_bbox=handle_bbox,
                m_per_px=float(m_per_px) if m_per_px > 0 else None,
                debug_video=bool(debug_video),
            )

        video_stem = _sanitize_stem(src_path.stem if src_path else "video")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{video_stem}_{timestamp}"
        rp3_dir = run_dir / "rp3"
        rp3_dir.mkdir(parents=True, exist_ok=True)
        input_dir = run_dir / "input"

        if mode == "Upload":
            input_video = _save_uploaded_video(uploaded, input_dir)
        else:
            input_video = _copy_input_video(src_path, input_dir)

        options = Sports2DOptions(
            pose_model=pose_model,
            mode=mode_choice,
            nb_persons=nb_persons,
            person_ordering="highest_likelihood",
            first_person_height_m=float(first_person_height),
            distance_to_camera_m=float(distance_m) if distance_m > 0 else None,
            device=device,
            det_frequency=int(det_frequency),
            slowmo_factor=1.0,
            save_images=False,
            save_graphs=False,
        )

        with st.status("Running Sports2D...", expanded=True) as status:
            progress_bar = st.progress(0, text="Overall progress")
            step_progress_bar = st.progress(0, text="Current step")

            def _on_progress(label: str, progress: float) -> None:
                status.update(label=label, state="running")
                progress_bar.progress(int(progress * 100), text=label)
                match = re.search(r"Step\s+(\d+)/7", label)
                step_idx = int(match.group(1)) if match else None
                if step_idx is None:
                    for idx, (_, end) in STEP_SPANS.items():
                        if progress <= end + 1e-9:
                            step_idx = idx
                            break
                if step_idx is not None and step_idx in STEP_SPANS:
                    start, end = STEP_SPANS[step_idx]
                    if end <= start:
                        local = 1.0
                    else:
                        local = (progress - start) / (end - start)
                    local = max(0.0, min(1.0, local))
                    step_progress_bar.progress(
                        int(local * 100),
                        text=f"Step {step_idx}/7 progress",
                    )

            try:
                artifacts, summary, overlay_video = _run_pipeline(
                    input_video=input_video,
                    run_dir=run_dir,
                    options=options,
                    stroke_tracking=stroke_tracking,
                    person_index=person_index,
                    progress_callback=_on_progress,
                )
            except Sports2DError as exc:
                status.update(label="Sports2D failed", state="error")
                progress_bar.progress(100, text="Failed")
                step_progress_bar.progress(100, text="Failed")
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                status.update(label="Processing failed", state="error")
                progress_bar.progress(100, text="Failed")
                step_progress_bar.progress(100, text="Failed")
                st.exception(exc)
                st.stop()
            progress_bar.progress(100, text="Done")
            step_progress_bar.progress(100, text="Done")
            status.update(label="Done", state="complete")

        st.subheader("Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input video**")
            st.video(str(artifacts.input_video))
        with c2:
            st.markdown("**Sports2D annotated video**")
            st.video(str(artifacts.sports2d_annotated_video))

        st.subheader(f"3D Overlay (Person {person_index})")
        if overlay_video is not None and overlay_video.exists():
            st.video(str(overlay_video))
        else:
            st.info("3D overlay video not available.")

        st.subheader(f"3D Angles Plot (Person {person_index})")
        if summary.angles_plots:
            labels = [p.stem.replace("_plot", "") for p in summary.angles_plots]
            if len(summary.angles_plots) > 1:
                selected = st.selectbox("Angles plot source", labels, index=0)
                plot_idx = labels.index(selected)
            else:
                plot_idx = 0
            st.image(str(summary.angles_plots[plot_idx]))
        else:
            st.info("3D angles plot not available.")
        if summary.angle_plot_errors:
            with st.expander("Angle plot warnings"):
                for msg in summary.angle_plot_errors:
                    st.write(msg)

        st.subheader("Outputs")
        output_lines = [
            "- Sports2D output folder (annotated video, TRC, MOT)",
            f"- Consolidated points CSV/NPZ (person {person_index})",
            f"- Sports2D angles CSV (person {person_index})",
            "- 3D angles plot image (`*_plot.png`)",
            "- MotionBERT 3D pose (`pose3d.npz`)",
            "- H36M angles (`angles_h36m.csv`)",
            "- 3D overlay video (`pose3d_overlay.mp4`)",
        ]
        if artifacts.stroke_signal_csv is not None:
            output_lines.append("- Stroke signal CSV (`stroke_signal.csv`)")
        if artifacts.stroke_signal_npz is not None:
            output_lines.append("- Stroke signal NPZ (`stroke_signal.npz`)")
        st.markdown("\n".join(output_lines))

        st.subheader("Sports2D Run Details")
        run_details = _build_sports2d_run_details(
            input_video=input_video,
            run_dir=run_dir,
            sports2d_out_dir=artifacts.sports2d_output_dir.parent,
            options=options,
            person_index=person_index,
        )
        run_details_text = json.dumps(run_details, indent=2, ensure_ascii=True)
        preview_height = _estimate_json_preview_height(run_details_text)
        with st.container(height=preview_height, border=True):
            st.code(run_details_text, language="json")

        if artifacts.zip_path.exists():
            with artifacts.zip_path.open("rb") as f:
                st.download_button(
                    label="Download Results (ZIP)",
                    data=f,
                    file_name=artifacts.zip_path.name,
                    mime="application/zip",
                )

        st.caption(f"Run folder: {artifacts.run_dir} | RP3 folder: {rp3_dir}")


if __name__ == "__main__":
    main()
