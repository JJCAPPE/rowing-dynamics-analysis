from __future__ import annotations

import copy
import re
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from progress_utils import clamp01


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SPORTS2D = REPO_ROOT / "sports2d_app" / "third_party" / "Sports2D"
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
    slowmo_factor: float = 1.0
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


ProgressCallback = Callable[[str, float], None]


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


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


def build_sports2d_config(
    video_path: Path,
    out_dir: Path,
    options: Sports2DOptions,
    *,
    include_defaults: bool = False,
    strip_custom: bool = False,
) -> Dict[str, Any]:
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()

    pose_model = _sanitize_pose_model(options.pose_model)
    nb_persons = _normalize_nb_persons(options.nb_persons)
    device = _normalize_device(options.device)

    perspective_value = (
        float(options.distance_to_camera_m)
        if options.distance_to_camera_m is not None
        else 10.0
    )

    overrides = {
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
            "slowmo_factor": float(options.slowmo_factor),
            "pose_model": pose_model,
            "mode": str(options.mode),
            "det_frequency": int(options.det_frequency),
            "device": device,
            "backend": "auto",
            "tracking_mode": "sports2d",
            "keypoint_likelihood_threshold": 0.3,
            "average_likelihood_threshold": 0.5,
            "keypoint_number_threshold": 0.3,
            "max_distance": 250,
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

    if not include_defaults:
        return overrides

    Sports2D = _import_sports2d()
    default_config = getattr(Sports2D, "DEFAULT_CONFIG", None)
    if not isinstance(default_config, dict):
        return overrides

    merged = _deep_update(copy.deepcopy(default_config), overrides)
    if strip_custom:
        pose_cfg = merged.get("pose")
        if isinstance(pose_cfg, dict) and "CUSTOM" in pose_cfg:
            pose_cfg = dict(pose_cfg)
            pose_cfg["CUSTOM"] = "<omitted>"
            merged["pose"] = pose_cfg
    return merged


class _Sports2DProgressParser:
    _FRAME_PROGRESS_RE = re.compile(
        r"(?P<pct>\d{1,3})%\|.*?\|\s*(?P<frame>[\d,]+)\s*/\s*(?P<total>[\d,]+)"
    )
    _STAGE_MARKERS: Tuple[Tuple[re.Pattern[str], float, str], ...] = (
        (re.compile(r"Estimating pose", re.IGNORECASE), 0.05, "Sports2D: estimating pose"),
        (
            re.compile(r"Processing video stream", re.IGNORECASE),
            0.12,
            "Sports2D: processing video stream",
        ),
        (
            re.compile(r"Video processing completed", re.IGNORECASE),
            0.62,
            "Sports2D: frame pass complete",
        ),
        (
            re.compile(r"Post-processing pose", re.IGNORECASE),
            0.70,
            "Sports2D: post-processing pose",
        ),
        (
            re.compile(r"Converting pose to meters", re.IGNORECASE),
            0.80,
            "Sports2D: converting pose to meters",
        ),
        (
            re.compile(r"Post-processing angles", re.IGNORECASE),
            0.88,
            "Sports2D: post-processing angles",
        ),
        (
            re.compile(r"Saving images of processed pose and angles", re.IGNORECASE),
            0.92,
            "Sports2D: rendering annotated output video",
        ),
        (
            re.compile(r"Processed video saved to", re.IGNORECASE),
            0.97,
            "Sports2D: output video saved",
        ),
        (
            re.compile(r"Processing .* took [0-9.]+ s", re.IGNORECASE),
            0.99,
            "Sports2D: wrapping up",
        ),
    )

    def __init__(self, callback: Optional[ProgressCallback]) -> None:
        self._callback = callback
        self._progress = 0.0
        self._last_label = ""
        self._last_frame_pct: Optional[int] = None

    @staticmethod
    def _parse_int(value: str) -> int:
        return int(value.replace(",", ""))

    def _emit(self, label: str, progress: float) -> None:
        if self._callback is None:
            return
        p = clamp01(progress)
        if p < self._progress:
            p = self._progress
        # Keep updates readable while still frequent enough for live feedback.
        if label != self._last_label or (p - self._progress) >= 0.002:
            self._callback(label, p)
            self._last_label = label
            self._progress = p

    def _consume_token(self, token: str) -> None:
        line = token.strip()
        if not line:
            return

        if match := self._FRAME_PROGRESS_RE.search(line):
            pct = max(0, min(100, int(match.group("pct"))))
            frame = max(0, self._parse_int(match.group("frame")))
            total = max(0, self._parse_int(match.group("total")))
            if total > 0:
                frame = min(frame, total)
                frame_ratio = frame / float(total)
            else:
                frame_ratio = pct / 100.0

            # Avoid redrawing every single frame while still showing
            # exact frame/total updates at each percentage point.
            if pct == self._last_frame_pct and frame != total:
                return
            self._last_frame_pct = pct

            progress = 0.12 + 0.50 * frame_ratio
            if total > 0:
                label = f"Sports2D: processing frame {frame}/{total} ({pct}%)"
            else:
                label = f"Sports2D: processing frames ({pct}%)"
            self._emit(label, progress)
            return

        for pattern, progress, label in self._STAGE_MARKERS:
            if pattern.search(line):
                self._emit(label, progress)
                return

    def consume_text(self, text: str) -> str:
        chunks = text.replace("\r", "\n").split("\n")
        for token in chunks[:-1]:
            self._consume_token(token)
        return chunks[-1]

    def finalize(self, tail: str) -> None:
        if tail.strip():
            self._consume_token(tail)
        self._emit("Sports2D: completed", 1.0)


def run_sports2d(
    video_path: Path,
    out_dir: Path,
    options: Sports2DOptions,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    poll_interval_s: float = 0.2,
) -> Sports2DRunResult:
    Sports2D = _import_sports2d()
    _check_pose2sim_version()

    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_sports2d_config(video_path, out_dir, options, include_defaults=False)
    console_log = out_dir / "console.log"
    if progress_callback is None:
        try:
            with console_log.open("w", encoding="utf-8") as log_f, redirect_stdout(
                log_f
            ), redirect_stderr(log_f):
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

    parser = _Sports2DProgressParser(progress_callback)
    parser._emit("Sports2D: initializing", 0.0)

    worker_error: List[BaseException] = []

    def _worker() -> None:
        try:
            with console_log.open("w", encoding="utf-8") as log_f, redirect_stdout(
                log_f
            ), redirect_stderr(log_f):
                Sports2D.process(config)
        except BaseException as exc:  # pragma: no cover - defensive rethrow
            worker_error.append(exc)

    worker = threading.Thread(target=_worker, daemon=True, name="sports2d-runner")
    worker.start()

    offset = 0
    tail = ""
    while worker.is_alive():
        if console_log.exists():
            with console_log.open("r", encoding="utf-8", errors="ignore") as log_f:
                log_f.seek(offset)
                chunk = log_f.read()
                offset = log_f.tell()
            if chunk:
                tail = parser.consume_text(tail + chunk)
        worker.join(timeout=max(0.05, float(poll_interval_s)))

    if console_log.exists():
        with console_log.open("r", encoding="utf-8", errors="ignore") as log_f:
            log_f.seek(offset)
            chunk = log_f.read()
            if chunk:
                tail = parser.consume_text(tail + chunk)
    parser.finalize(tail)

    if worker_error:
        exc = worker_error[0]
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
