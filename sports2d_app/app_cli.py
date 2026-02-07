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
from runner_sports2d import Sports2DError, Sports2DOptions, run_sports2d
from stroke_signal import StrokeTrackingOutputs, run_stroke_signal_tracking


APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "runs"
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
    annotate: bool = True
    machine_bbox: Optional[Tuple[float, float, float, float]] = None
    handle_bbox: Optional[Tuple[float, float, float, float]] = None
    m_per_px: Optional[float] = None
    ema_alpha: float = 0.4
    min_points: int = 10
    min_stroke_distance_s: float = 0.8
    prominence: Optional[float] = None
    prominence_frac: float = 0.1
    smooth_window_s: float = 0.2
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


def _report_progress(
    callback: Optional[ProgressCallback],
    label: str,
    progress: float,
) -> None:
    if callback is None:
        return
    callback(label, max(0.0, min(1.0, progress)))


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

    _report_progress(
        progress_callback,
        "Step 1/7: Running Sports2D (pose + tracking)",
        0.05,
    )
    result = run_sports2d(input_video, sports2d_out_dir, options)

    _report_progress(
        progress_callback,
        "Step 2/7: Exporting Sports2D outputs",
        0.45,
    )
    exports_dir.mkdir(parents=True, exist_ok=True)
    person_trc_files = _filter_person_files(result.trc_files, person_index)
    person_mot_files = _filter_person_files(result.mot_files, person_index)
    if not person_trc_files:
        raise RuntimeError(
            f"No TRC files found for person index {person_index}. "
            "Increase the number of persons to detect or choose a different index."
        )
    summary_base = _export_sports2d_outputs(
        person_trc_files,
        person_mot_files,
        exports_dir,
    )

    person_trc = person_trc_files[0]

    _report_progress(
        progress_callback,
        "Step 3/7: Preparing MotionBERT inputs",
        0.60,
    )
    trc_data = parse_trc_file(person_trc)
    j2d_px, _ = extract_coco17_from_trc(trc_data)

    meta = get_video_metadata(result.annotated_video)
    _report_progress(
        progress_callback,
        "Step 4/7: Running MotionBERT 3D lift",
        0.70,
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
    )

    _report_progress(
        progress_callback,
        "Step 5/7: Tracking handle vs machine",
        0.78,
    )
    stroke_outputs: Optional[StrokeTrackingOutputs] = None
    stroke_error: Optional[str] = None
    if stroke_tracking.enabled:
        try:
            stroke_dir.mkdir(parents=True, exist_ok=True)
            stroke_outputs = run_stroke_signal_tracking(
                video_path=result.annotated_video,
                out_dir=stroke_dir,
                angles_csv=mb_outputs.angles_csv,
                reference_frame_idx=0,
                machine_bbox=stroke_tracking.machine_bbox,
                handle_bbox=stroke_tracking.handle_bbox,
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
                debug_video=stroke_tracking.debug_video,
            )
        except Exception as exc:
            stroke_error = str(exc)

    _report_progress(
        progress_callback,
        "Step 6/7: Rendering 3D overlay + plots",
        0.88,
    )
    angles_plots: List[Path] = []
    angle_plot_errors: List[str] = []
    if stroke_outputs is not None and stroke_outputs.merged_angles_plot is not None:
        angles_plots = [stroke_outputs.merged_angles_plot]
    else:
        angles_plots, angle_plot_errors = _generate_motionbert_angles_plot(
            mb_outputs.angles_csv,
            exports_dir,
            input_video,
        )
    if stroke_error is not None:
        angle_plot_errors.append(f"stroke tracking: {stroke_error}")

    overlay_video = overlay_dir / "pose3d_overlay.mp4"
    generate_pose3d_overlay_video(
        video_path=result.annotated_video,
        pose3d_npz=mb_outputs.pose3d_npz,
        out_video_path=overlay_video,
        stroke_signal_npz=(
            stroke_outputs.stroke_npz
            if stroke_outputs is not None and stroke_outputs.stroke_npz.exists()
            else None
        ),
    )

    _report_progress(
        progress_callback,
        "Step 7/7: Packaging outputs",
        0.95,
    )
    zip_path = run_dir / "results.zip"
    zip_inputs = [result.output_dir, exports_dir, motionbert_dir, overlay_dir]
    if stroke_outputs is not None:
        zip_inputs.append(stroke_dir)
    _zip_outputs(zip_path, zip_inputs)

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

    def _inner(stdscr: "curses._CursesWindow") -> str:
        selected = max(0, min(default_index, len(options) - 1))
        curses.curs_set(0)
        stdscr.keypad(True)
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, prompt, curses.A_BOLD)
            stdscr.addstr(1, 0, "Use UP/DOWN arrows and Enter.")
            for idx, option in enumerate(options):
                y = idx + 3
                prefix = "-> " if idx == selected else "   "
                text = f"{prefix}{option}"
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                stdscr.addstr(y, 0, text, attr)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(options)
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


def _collect_options() -> Tuple[Path, int, Sports2DOptions, StrokeTrackingOptions]:
    print("Sports2D CLI Pipeline")
    print("=====================\n")
    video_path = _prompt_existing_file("Path to input video")

    mode_choice = _choose_option(
        "Sports2D mode",
        ["lightweight", "balanced", "performance"],
        default_index=1,
    )
    pose_model = _choose_option(
        "Pose model",
        ["Whole_body", "Whole_body_wrist", "Body_with_feet", "Body"],
        default_index=0,
    )
    nb_persons_raw = _choose_option(
        "Max persons to detect",
        ["1", "2", "3", "all"],
        default_index=0,
    )
    device = _choose_option("Device", ["auto", "cpu", "cuda", "mps"], default_index=0)

    person_index = _prompt_int("Person index (0-based)", default=0, minimum=0)
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
    distance_m = _prompt_optional_float("Distance to camera (m)", default=5.0)
    det_frequency = _prompt_int("Detection frequency (frames)", default=4, minimum=1)
    slowmo_factor = _prompt_float("Slow-motion factor", default=1.0, minimum=0.1)

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
        bbox_mode = _choose_option(
            "Stroke ROI source",
            ["annotate interactively", "enter bbox values"],
            default_index=0,
        )
        annotate = bbox_mode == "annotate interactively"
        machine_bbox = None if annotate else _prompt_bbox("Machine reference bbox")
        handle_bbox = None if annotate else _prompt_bbox("Handle bbox")
        m_per_px = _prompt_optional_float(
            "Stroke meters-per-pixel scale",
            default=None,
        )
        save_debug = (
            _choose_option(
                "Save stroke tracking debug video",
                ["yes", "no"],
                default_index=0,
            )
            == "yes"
        )
        stroke_tracking = StrokeTrackingOptions(
            enabled=True,
            annotate=annotate,
            machine_bbox=machine_bbox,
            handle_bbox=handle_bbox,
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
        slowmo_factor=float(slowmo_factor),
        save_images=False,
        save_graphs=False,
    )
    return video_path, person_index, options, stroke_tracking


def main() -> int:
    try:
        source_video, person_index, options, stroke_tracking = _collect_options()
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

    def _on_progress(label: str, progress: float) -> None:
        pct = int(round(progress * 100))
        print(f"[{pct:3d}%] {label}")

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
        print(f"\nSports2D failed: {exc}")
        return 2
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 3

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
