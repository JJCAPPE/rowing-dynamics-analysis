from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SPORTS2D = REPO_ROOT / "pose-extraction-test" / "third_party" / "Sports2D"
MIN_POSE2SIM_VERSION = (0, 10, 40)


@dataclass(frozen=True)
class Sports2DOptions:
    pose_model: str = "Whole_body"
    mode: str = "balanced"
    nb_persons: int | str = 1
    person_ordering: str = "highest_likelihood"
    first_person_height_m: float = 1.7
    distance_to_camera_m: Optional[float] = None
    device: str = "auto"
    det_frequency: int = 4
    save_images: bool = False
    save_graphs: bool = False


@dataclass(frozen=True)
class Sports2DRunResult:
    output_dir: Path
    annotated_video: Path
    trc_files: List[Path]
    mot_files: List[Path]


class Sports2DError(RuntimeError):
    pass


def _import_sports2d() -> object:
    if LOCAL_SPORTS2D.exists():
        if str(LOCAL_SPORTS2D) not in sys.path:
            sys.path.insert(0, str(LOCAL_SPORTS2D))
    try:
        from Sports2D import Sports2D  # type: ignore

        return Sports2D
    except Exception:
        try:
            from sports2d import Sports2D  # type: ignore

            return Sports2D
        except Exception as exc:
            raise Sports2DError(
                "Sports2D is not available. Install it or ensure the local clone exists at "
                f"{LOCAL_SPORTS2D}."
            ) from exc


def _check_pose2sim_version() -> None:
    try:
        from importlib.metadata import version
    except Exception as exc:
        raise Sports2DError("Unable to read Pose2Sim version.") from exc

    try:
        ver = version("Pose2Sim")
    except Exception as exc:
        raise Sports2DError(
            "Pose2Sim is not installed. Install Pose2Sim>=0.10.40 to use Sports2D."
        ) from exc

    def parse(v: str) -> Tuple[int, int, int]:
        parts = v.split("+", 1)[0].split(".")
        nums = [int(p) for p in parts[:3] if p.isdigit() or p.isnumeric()]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums[:3])  # type: ignore[return-value]

    if parse(ver) < MIN_POSE2SIM_VERSION:
        raise Sports2DError(
            f"Pose2Sim {ver} detected. Sports2D requires Pose2Sim >= 0.10.40. "
            "Please upgrade with `pip install --upgrade Pose2Sim>=0.10.40`."
        )


def _normalize_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d in {"cpu", "cuda", "mps", "rocm"}:
        return d.upper()
    return "auto"


def _normalize_nb_persons(value: int | str) -> int | str:
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


def _sanitize_pose_model(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"whole_body", "wholebody", "coco_133"}:
        return "Whole_body"
    if n in {"whole_body_wrist", "wholebody_wrist", "coco_133_wrist"}:
        return "Whole_body_wrist"
    if n in {"body_with_feet", "bodywithfeet", "halpe_26"}:
        return "Body_with_feet"
    if n in {"body", "coco_17"}:
        return "Body"
    return "Whole_body"


def run_sports2d(video_path: Path, out_dir: Path, options: Sports2DOptions) -> Sports2DRunResult:
    Sports2D = _import_sports2d()
    _check_pose2sim_version()

    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_model = _sanitize_pose_model(options.pose_model)
    nb_persons = _normalize_nb_persons(options.nb_persons)
    device = _normalize_device(options.device)

    perspective_value = (
        float(options.distance_to_camera_m)
        if options.distance_to_camera_m is not None
        else 10.0
    )

    config = {
        "base": {
            "video_input": str(video_path),
            "video_dir": "",
            "result_dir": str(out_dir),
            "nb_persons_to_detect": nb_persons,
            "person_ordering_method": options.person_ordering,
            "first_person_height": float(options.first_person_height_m),
            "show_realtime_results": False,
            "save_vid": True,
            "save_img": bool(options.save_images),
            "save_pose": True,
            "calculate_angles": True,
            "save_angles": True,
            "compare": False,
        },
        "pose": {
            "pose_model": pose_model,
            "mode": str(options.mode),
            "det_frequency": int(options.det_frequency),
            "device": device,
            "backend": "auto",
            "tracking_mode": "sports2d",
        },
        "px_to_meters_conversion": {
            "to_meters": True,
            "make_c3d": False,
            "save_calib": False,
            "perspective_value": perspective_value,
            "perspective_unit": "distance_m",
            "floor_angle": "auto",
            "xy_origin": ["auto"],
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "calib_file": "",
        },
        "angles": {
            "display_angle_values_on": ["body", "list"],
            "fontSize": 0.3,
            "joint_angles": [
                "Right ankle",
                "Left ankle",
                "Right knee",
                "Left knee",
                "Right hip",
                "Left hip",
                "Right shoulder",
                "Left shoulder",
                "Right elbow",
                "Left elbow",
                "Right wrist",
                "Left wrist",
            ],
            "segment_angles": [
                "Right foot",
                "Left foot",
                "Right shank",
                "Left shank",
                "Right thigh",
                "Left thigh",
                "Pelvis",
                "Trunk",
                "Shoulders",
                "Head",
                "Right arm",
                "Left arm",
                "Right forearm",
                "Left forearm",
            ],
            "flip_left_right": False,
            "correct_segment_angles_with_floor_angle": True,
        },
        "post-processing": {
            "interpolate": True,
            "interp_gap_smaller_than": 10,
            "fill_large_gaps_with": "last_value",
            "sections_to_keep": "all",
            "min_chunk_size": 10,
            "reject_outliers": True,
            "filter": True,
            "filter_type": "butterworth",
            "show_graphs": bool(options.save_graphs),
            "save_graphs": bool(options.save_graphs),
            "butterworth": {"cut_off_frequency": 6, "order": 4},
        },
        "kinematics": {
            "do_ik": False,
            "use_augmentation": False,
            "feet_on_floor": False,
            "right_left_symmetry": True,
            "participant_mass": [70.0],
        },
        "logging": {"use_custom_logging": False},
    }

    try:
        Sports2D.process(config)
    except Exception as exc:
        raise Sports2DError(f"Sports2D failed: {exc}") from exc

    output_dir = out_dir / f"{video_path.stem}_Sports2D"
    annotated_video = output_dir / f"{output_dir.name}.mp4"
    trc_files = sorted(output_dir.glob("*_px*.trc"))
    mot_files = sorted(output_dir.glob("*_angles*.mot"))

    if not output_dir.exists():
        raise Sports2DError(f"Sports2D output directory not found: {output_dir}")
    if not annotated_video.exists():
        raise Sports2DError(f"Sports2D annotated video not found: {annotated_video}")

    return Sports2DRunResult(
        output_dir=output_dir,
        annotated_video=annotated_video,
        trc_files=trc_files,
        mot_files=mot_files,
    )
