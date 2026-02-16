from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from motionbert_3d import run_motionbert
from overlay_3d import generate_pose3d_overlay_video, get_video_metadata
from parse_sports2d import (
    extract_coco17_from_trc,
    parse_mot_file,
    parse_trc_file,
    write_angles_csv,
    write_points_csv,
    write_points_npz,
)
from plot_angles import generate_angles_plot
from progress_utils import ProgressMux
from runner_sports2d import Sports2DError, Sports2DOptions, run_sports2d
from stroke_signal import StrokeTrackingOutputs, run_stroke_signal_tracking


APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_VIDEOS_DIR = Path("/Users/giacomo/dev/rowing-video-analysis/source-videos")
ALT_SOURCE_VIDEOS_DIR = Path("/Volumes/T9/rowing-research")
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
    ".mpg",
    ".mpeg",
    ".mts",
    ".m2ts",
    ".wmv",
}


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
    annotate: bool = True
    machine_bbox: Optional[Tuple[float, float, float, float]] = None
    machine_cable_point: Optional[Tuple[float, float]] = None
    handle_bbox: Optional[Tuple[float, float, float, float]] = None
    handle_source: str = "pose"
    handle_pose_npz: Optional[Path] = None
    m_per_px: Optional[float] = None
    ema_alpha: float = 0.4
    min_points: int = 10
    min_stroke_distance_s: float = 0.8
    prominence: Optional[float] = None
    prominence_frac: float = 0.1
    smooth_window_s: float = 0.2
    debug_video: bool = True


@dataclass(frozen=True)
class DebugVideoOptions:
    mode: str = "full"

    def __post_init__(self) -> None:
        if self.mode not in {"full", "first10", "none"}:
            raise ValueError(f"Unsupported debug video mode: {self.mode}")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def max_seconds(self) -> Optional[float]:
        if self.mode == "first10":
            return 10.0
        return None


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


def _person_tag(person_index: int) -> str:
    return f"_person{person_index:02d}"


def _person_pattern(person_index: int) -> re.Pattern[str]:
    tag = _person_tag(person_index)
    return re.compile(rf"{re.escape(tag)}(?!\\d)|_person{person_index}(?!\\d)")


def _filter_person_files(files: List[Path], person_index: int) -> List[Path]:
    pattern = _person_pattern(person_index)
    return sorted([p for p in files if pattern.search(p.stem)], key=lambda p: p.name)


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


def _export_sports2d_outputs(
    trc_files: List[Path], mot_files: List[Path], exports_dir: Path
) -> ExportSummary:
    points_csv: List[Path] = []
    points_npz: List[Path] = []
    angles_csv: List[Path] = []

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


ProgressCallback = Callable[[str, float], None]

STEP_SPANS: dict[int, Tuple[float, float]] = {
    1: (0.00, 0.52),
    2: (0.52, 0.60),
    3: (0.60, 0.64),
    4: (0.64, 0.80),
    5: (0.80, 0.88),
    6: (0.88, 0.97),
    7: (0.97, 1.00),
}

def _run_pipeline(
    *,
    input_video: Path,
    run_dir: Path,
    options: Sports2DOptions,
    stroke_tracking: StrokeTrackingOptions,
    debug_videos: DebugVideoOptions,
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
    j2d_px, _ = extract_coco17_from_trc(trc_data)
    meta = get_video_metadata(result.annotated_video)
    step3("Completed", 1.0)

    step4 = progress.span(
        *STEP_SPANS[4],
        prefix="Step 4/7: Running MotionBERT 3D lift",
    )
    mb_outputs = run_motionbert(
        j2d_px,
        width=meta.width,
        height=meta.height,
        out_dir=motionbert_dir,
        fps=meta.fps,
        clip_len=243,
        flip=False,
        rootrel=False,
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
                annotate=stroke_tracking.annotate,
                m_per_px=stroke_tracking.m_per_px,
                ema_alpha=stroke_tracking.ema_alpha,
                min_points=stroke_tracking.min_points,
                min_stroke_distance_s=stroke_tracking.min_stroke_distance_s,
                prominence=stroke_tracking.prominence,
                prominence_frac=stroke_tracking.prominence_frac,
                smooth_window_s=stroke_tracking.smooth_window_s,
                create_plot=True,
                plot_video_path=result.annotated_video,
                debug_video=(stroke_tracking.debug_video and debug_videos.enabled),
                debug_video_max_seconds=(
                    debug_videos.max_seconds
                    if stroke_tracking.debug_video and debug_videos.enabled
                    else None
                ),
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

    overlay_video: Optional[Path] = None
    if debug_videos.enabled:
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
            max_duration_s=debug_videos.max_seconds,
            progress_callback=_overlay_progress,
        )
    else:
        step6("Overlay video skipped (debug videos disabled)", 1.0)
    step6("Completed", 1.0)

    step7 = progress.span(
        *STEP_SPANS[7],
        prefix="Step 7/7: Packaging outputs",
    )
    step7("Creating ZIP archive", 0.2)
    zip_path = run_dir / "results.zip"
    zip_inputs = [result.output_dir, exports_dir, motionbert_dir]
    if overlay_video is not None and overlay_video.exists():
        zip_inputs.append(overlay_dir)
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


def _fallback_choose_option(prompt: str, options: List[str], default_index: int = 0) -> str:
    print(f"\n{prompt}")
    for idx, item in enumerate(options, start=1):
        marker = " (default)" if idx - 1 == default_index else ""
        print(f"  {idx}. {item}{marker}")
    while True:
        raw = input("Choose option number and press Enter: ").strip()
        if raw == "":
            return options[default_index]
        if raw.isdigit():
            choice = int(raw) - 1
            if 0 <= choice < len(options):
                return options[choice]
        print("Invalid selection. Try again.")


def _curses_choose_option(
    prompt: str,
    options: List[str],
    default_index: int = 0,
) -> str:
    import curses

    def _fit_to_width(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _inner(stdscr: "curses._CursesWindow") -> str:
        selected = max(0, min(default_index, len(options) - 1))
        top = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            max_width = max(0, width - 1)
            visible_rows = max(1, height - 4)

            def _safe_addstr(y: int, x: int, text: str, attr: int = 0) -> None:
                if y < 0 or y >= height or not text:
                    return
                try:
                    stdscr.addstr(y, x, text, attr)
                except curses.error:
                    pass

            if selected < top:
                top = selected
            elif selected >= top + visible_rows:
                top = selected - visible_rows + 1

            _safe_addstr(0, 0, _fit_to_width(prompt, max_width), curses.A_BOLD)
            _safe_addstr(1, 0, _fit_to_width("Use UP/DOWN arrows and Enter.", max_width))

            visible_options = options[top : top + visible_rows]
            for row, option in enumerate(visible_options):
                idx = top + row
                y = row + 3
                prefix = "-> " if idx == selected else "   "
                text = _fit_to_width(f"{prefix}{option}", max_width)
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                _safe_addstr(y, 0, text, attr)

            status = f"Option {selected + 1}/{len(options)}"
            if top > 0 or top + visible_rows < len(options):
                status = f"{status} (scroll for more)"
            _safe_addstr(height - 1, 0, _fit_to_width(status, max_width), curses.A_DIM)

            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(options)
            elif key == curses.KEY_PPAGE:
                selected = max(0, selected - visible_rows)
            elif key == curses.KEY_NPAGE:
                selected = min(len(options) - 1, selected + visible_rows)
            elif key in (10, 13, curses.KEY_ENTER):
                return options[selected]

    return curses.wrapper(_inner)


def _choose_option(prompt: str, options: List[str], default_index: int = 0) -> str:
    if not options:
        raise ValueError("options must not be empty")
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _curses_choose_option(prompt, options, default_index=default_index)
        except Exception:
            pass
    return _fallback_choose_option(prompt, options, default_index=default_index)


def _prompt_existing_file(prompt: str) -> Path:
    while True:
        raw = input(f"{prompt}: ").strip()
        if raw == "":
            print("Please enter a path.")
            continue
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"Path does not exist: {p}")
            continue
        if not p.is_file():
            print(f"Path is not a file: {p}")
            continue
        return p


def _list_source_videos(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Source videos directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Source videos path is not a directory: {directory}")
    videos = [
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    return sorted(videos, key=lambda p: p.name.lower())


def _choose_source_video(directory: Path = SOURCE_VIDEOS_DIR) -> Path:
    videos = _list_source_videos(directory)
    if not videos:
        raise ValueError(f"No video files found in source directory: {directory}")
    labels = [path.name for path in videos]
    selected = _choose_option("Source video", labels, default_index=0)
    selected_path = videos[labels.index(selected)]
    print(f"Selected source video: {selected_path}")
    return selected_path


def _choose_source_video_with_location() -> Path:
    source_locations = [SOURCE_VIDEOS_DIR, ALT_SOURCE_VIDEOS_DIR]
    source_labels = [str(path) for path in source_locations]

    while True:
        selected_label = _choose_option(
            "Source videos directory",
            source_labels,
            default_index=0,
        )
        selected_dir = source_locations[source_labels.index(selected_label)]
        try:
            return _choose_source_video(selected_dir)
        except Exception as exc:
            print(f"Unable to load videos from {selected_dir}: {exc}")
            retry = _choose_option(
                "Choose a different source directory",
                ["yes", "no"],
                default_index=0,
            )
            if retry != "yes":
                raise


def _prompt_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def _prompt_float(prompt: str, default: float, minimum: Optional[float] = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def _prompt_optional_float(prompt: str, default: Optional[float]) -> Optional[float]:
    label = "" if default is None else f" [{default}]"
    while True:
        raw = input(f"{prompt}{label} (blank = default, 'none' = none): ").strip()
        if raw == "":
            return default
        if raw.lower() in {"none", "null", "n"}:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value, or leave blank for none.")
            continue
        if value < 0:
            print("Please enter a non-negative value.")
            continue
        return value


def _prompt_bbox(prompt: str) -> Tuple[float, float, float, float]:
    while True:
        raw = input(f"{prompt} [x,y,w,h]: ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            print("Enter exactly four comma-separated values.")
            continue
        try:
            x, y, w, h = [float(v) for v in parts]
        except ValueError:
            print("Invalid numeric values. Try again.")
            continue
        if w <= 1 or h <= 1:
            print("Width and height must be > 1.")
            continue
        return float(x), float(y), float(w), float(h)


def _prompt_point(prompt: str) -> Tuple[float, float]:
    while True:
        raw = input(f"{prompt} [x,y]: ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            print("Enter exactly two comma-separated values.")
            continue
        try:
            x, y = [float(v) for v in parts]
        except ValueError:
            print("Invalid numeric values. Try again.")
            continue
        return float(x), float(y)


def _open_path(path: Path) -> None:
    if not path.exists():
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"Failed to open {path}: {exc}")


def _collect_options() -> Tuple[Path, int, Sports2DOptions, StrokeTrackingOptions, DebugVideoOptions]:
    print("Sports2D CLI Pipeline")
    print("=====================\n")
    video_path = _choose_source_video_with_location()

    mode_choice = _choose_option(
        "Sports2D mode",
        ["lightweight", "balanced", "performance"],
        default_index=2,
    )
    pose_model = _choose_option(
        "Pose model",
        ["Whole_body", "Whole_body_wrist", "Body_with_feet", "Body"],
        default_index=1,
    )
    nb_persons_raw = _choose_option(
        "Max persons to detect",
        ["1", "2", "3", "all"],
        default_index=0,
    )
    device = _choose_option(
        "Device", 
        ["auto", "cpu", "cuda", "mps"], 
        default_index=0,
    )

    person_index = _prompt_int(
        "Person index (0-based)", 
        default=0, 
        minimum=0,
    )
    if nb_persons_raw.isdigit():
        nb_persons_int = int(nb_persons_raw)
        if person_index >= nb_persons_int:
            raise ValueError(
                f"Person index {person_index} must be smaller than max persons {nb_persons_int}."
            )
        nb_persons: int | str = nb_persons_int
    else:
        nb_persons = nb_persons_raw

    first_person_height = _prompt_float("First person height (m)", default=1.95, minimum=1.0)
    distance_m = _prompt_optional_float("Distance to camera (m)", default=4.0)
    det_frequency = 1
    debug_choice = _choose_option(
        "Debug video output policy",
        [
            "full length",
            "first 10 seconds only",
            "disabled (no debug videos)",
        ],
        default_index=1,
    )
    debug_mode_map = {
        "full length": "full",
        "first 10 seconds only": "first10",
        "disabled (no debug videos)": "none",
    }
    debug_videos = DebugVideoOptions(mode=debug_mode_map[debug_choice])

    enable_stroke = (
        _choose_option(
            "Enable handle/machine stroke tracking",
            ["yes", "no"],
            default_index=0,
        )
        == "yes"
    )
    stroke_tracking = StrokeTrackingOptions(enabled=False)
    if enable_stroke:
        handle_source_choice = _choose_option(
            "Handle source",
            ["manual bbox", "pose midpoint"],
            default_index=1,
        )
        handle_source = "pose" if handle_source_choice == "pose midpoint" else "manual"
        if handle_source == "manual":
            bbox_mode = _choose_option(
                "Stroke ROI source",
                ["annotate interactively", "enter bbox values"],
                default_index=0,
            )
            annotate = bbox_mode == "annotate interactively"
            machine_bbox = None if annotate else _prompt_bbox("Machine reference bbox")
            machine_cable_point = (
                None
                if annotate
                else _prompt_point("Machine cable entry point (where cable enters erg)")
            )
            handle_bbox = None if annotate else _prompt_bbox("Handle bbox")
        else:
            machine_mode = _choose_option(
                "Machine ROI source",
                ["annotate interactively", "enter bbox values"],
                default_index=0,
            )
            annotate = machine_mode == "annotate interactively"
            machine_bbox = None if annotate else _prompt_bbox("Machine reference bbox")
            machine_cable_point = (
                None
                if annotate
                else _prompt_point("Machine cable entry point (where cable enters erg)")
            )
            handle_bbox = None
        m_per_px = _prompt_optional_float(
            "Stroke meters-per-pixel scale",
            default=None,
        )
        if debug_videos.enabled:
            save_debug = (
                _choose_option(
                    "Save stroke tracking debug video",
                    ["yes", "no"],
                    default_index=1,
                )
                == "yes"
            )
        else:
            save_debug = False
        stroke_tracking = StrokeTrackingOptions(
            enabled=True,
            annotate=annotate,
            machine_bbox=machine_bbox,
            machine_cable_point=machine_cable_point,
            handle_bbox=handle_bbox,
            handle_source=handle_source,
            m_per_px=m_per_px,
            debug_video=save_debug,
        )

    options = Sports2DOptions(
        pose_model=pose_model,
        mode=mode_choice,
        nb_persons=nb_persons,
        person_ordering="highest_likelihood",
        first_person_height_m=float(first_person_height),
        distance_to_camera_m=float(distance_m) if distance_m is not None else None,
        device=device,
        det_frequency=int(det_frequency),
        slowmo_factor=1.0,
        save_images=False,
        save_graphs=False,
    )
    return video_path, person_index, options, stroke_tracking, debug_videos


def main() -> int:
    try:
        source_video, person_index, options, stroke_tracking, debug_videos = _collect_options()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\nInput error: {exc}")
        return 1

    video_stem = _sanitize_stem(source_video.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{video_stem}_{timestamp}"
    input_dir = run_dir / "input"
    input_video = _copy_input_video(source_video, input_dir)

    print("\nRunning pipeline...\n")

    progress_state = {"line_len": 0}
    progress_stream = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout

    def _on_progress(label: str, progress: float) -> None:
        pct = int(round(progress * 100))
        width = 30
        filled = int(round(max(0.0, min(1.0, progress)) * width))
        bar = "#" * filled + "-" * (width - filled)
        line = f"[{bar}] {pct:3d}% {label}"
        pad = " " * max(0, progress_state["line_len"] - len(line))
        print(f"\r{line}{pad}", end="", file=progress_stream, flush=True)
        progress_state["line_len"] = len(line)

    try:
        artifacts, summary, overlay_video = _run_pipeline(
            input_video=input_video,
            run_dir=run_dir,
            options=options,
            stroke_tracking=stroke_tracking,
            debug_videos=debug_videos,
            person_index=person_index,
            progress_callback=_on_progress,
        )
    except Sports2DError as exc:
        print(f"\nSports2D failed: {exc}")
        return 2
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 3

    print()
    print("\nDone.\n")
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Sports2D annotated video: {artifacts.sports2d_annotated_video}")
    if artifacts.stroke_signal_csv is not None and artifacts.stroke_signal_csv.exists():
        print(f"Stroke signal CSV: {artifacts.stroke_signal_csv}")
    if overlay_video is not None and overlay_video.exists():
        print(f"3D overlay video: {overlay_video}")
    else:
        print("3D overlay video: not available")
    if summary.angles_plots:
        for plot_path in summary.angles_plots:
            print(f"3D angles plot: {plot_path}")
    else:
        print("3D angles plot: not available")
    if summary.angle_plot_errors:
        for msg in summary.angle_plot_errors:
            print(f"Plot warning: {msg}")
    print(f"Results ZIP: {artifacts.zip_path}")
    """ video_to_open = (
        overlay_video
        if overlay_video is not None and overlay_video.exists()
        else artifacts.sports2d_annotated_video
    )
    _open_path(video_to_open) """

    if summary.angles_plots:
        _open_path(summary.angles_plots[0])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
