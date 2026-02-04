from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from parse_sports2d import (
    extract_coco17_from_trc,
    parse_mot_file,
    parse_trc_file,
    write_angles_csv,
    write_points_csv,
    write_points_npz,
)
from runner_sports2d import Sports2DOptions, Sports2DError, run_sports2d
from motionbert_3d import run_motionbert
from overlay_3d import generate_pose3d_overlay_video, get_video_metadata


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
    zip_path: Path


@dataclass(frozen=True)
class ExportSummary:
    trc_files: List[Path]
    mot_files: List[Path]
    points_csv: List[Path]
    points_npz: List[Path]
    angles_csv: List[Path]


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


def _find_person0_trc(trc_files: List[Path]) -> Optional[Path]:
    for p in trc_files:
        if "_person00" in p.stem:
            return p
    return trc_files[0] if trc_files else None


def _find_person0_mot(mot_files: List[Path]) -> Optional[Path]:
    for p in mot_files:
        if "_person00" in p.stem:
            return p
    return mot_files[0] if mot_files else None


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
    )


def _run_pipeline(
    *,
    input_video: Path,
    run_dir: Path,
    options: Sports2DOptions,
) -> Tuple[RunArtifacts, ExportSummary, Optional[Path]]:
    sports2d_out_dir = run_dir / "sports2d"
    exports_dir = run_dir / "exports"
    motionbert_dir = run_dir / "motionbert"
    overlay_dir = run_dir / "overlay"

    result = run_sports2d(input_video, sports2d_out_dir, options)

    exports_dir.mkdir(parents=True, exist_ok=True)
    summary = _export_sports2d_outputs(result.trc_files, result.mot_files, exports_dir)

    person0_trc = _find_person0_trc(result.trc_files)
    if person0_trc is None:
        raise RuntimeError("No TRC files found for Sports2D output.")

    trc_data = parse_trc_file(person0_trc)
    J2d_px, _ = extract_coco17_from_trc(trc_data)

    meta = get_video_metadata(result.annotated_video)
    mb_outputs = run_motionbert(
        J2d_px,
        width=meta.width,
        height=meta.height,
        out_dir=motionbert_dir,
        fps=meta.fps,
        clip_len=243,
        flip=False,
        rootrel=False,
    )

    overlay_video = overlay_dir / "pose3d_overlay.mp4"
    generate_pose3d_overlay_video(
        video_path=result.annotated_video,
        pose3d_npz=mb_outputs.pose3d_npz,
        out_video_path=overlay_video,
    )

    zip_path = run_dir / "results.zip"
    _zip_outputs(
        zip_path,
        [result.output_dir, exports_dir, motionbert_dir, overlay_dir],
    )

    artifacts = RunArtifacts(
        run_dir=run_dir,
        input_video=input_video,
        sports2d_output_dir=result.output_dir,
        sports2d_annotated_video=result.annotated_video,
        exports_dir=exports_dir,
        motionbert_dir=motionbert_dir,
        overlay_dir=overlay_dir,
        zip_path=zip_path,
    )

    return artifacts, summary, overlay_video


def main() -> None:
    st.set_page_config(page_title="Sports2D Analysis Demo", layout="wide")

    st.title("Sports2D Analysis Demo")
    st.caption(
        "Sports2D-only pipeline (no manual annotation) with MotionBERT 3D overlay. "
        "All Sports2D outputs are preserved."
    )

    st.markdown(
        "**Note:** This app runs Sports2D end-to-end and always performs a 3D lift on Person 0."
    )
    st.warning(
        "Results are most accurate when the athlete moves in a near-2D plane (side view)."
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
        nb_persons = st.selectbox("Number of persons", [1, 2, 3, "all"], index=0)
        first_person_height = st.number_input("First person height (m)", 1.2, 2.5, 1.7, 0.01)
        distance_m = st.number_input("Distance to camera (m)", 0.0, 50.0, 10.0, 0.5)
        pose_model = st.selectbox(
            "Pose model",
            ["Whole_body", "Whole_body_wrist", "Body_with_feet", "Body"],
            index=0,
        )
        mode_choice = st.selectbox("Mode", ["lightweight", "balanced", "performance"], index=1)
        det_frequency = st.slider("Detection frequency (frames)", 1, 30, 4, 1)
        device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

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

        video_stem = _sanitize_stem(src_path.stem if src_path else "video")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{video_stem}_{timestamp}"
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
            save_images=False,
            save_graphs=False,
        )

        with st.status("Running Sports2D...", expanded=True) as status:
            try:
                artifacts, summary, overlay_video = _run_pipeline(
                    input_video=input_video,
                    run_dir=run_dir,
                    options=options,
                )
            except Sports2DError as exc:
                status.update(label="Sports2D failed", state="error")
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                status.update(label="Processing failed", state="error")
                st.exception(exc)
                st.stop()
            status.update(label="Done", state="complete")

        st.subheader("Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input video**")
            st.video(str(artifacts.input_video))
        with c2:
            st.markdown("**Sports2D annotated video**")
            st.video(str(artifacts.sports2d_annotated_video))

        st.subheader("3D Overlay")
        if overlay_video is not None and overlay_video.exists():
            st.video(str(overlay_video))
        else:
            st.info("3D overlay video not available.")

        st.subheader("Outputs")
        st.markdown(
            "- Sports2D output folder (annotated video, TRC, MOT)\n"
            "- Consolidated points CSV/NPZ\n"
            "- Sports2D angles CSV\n"
            "- MotionBERT 3D pose (`pose3d.npz`)\n"
            "- H36M angles (`angles_h36m.csv`)\n"
            "- 3D overlay video (`pose3d_overlay.mp4`)"
        )

        if artifacts.zip_path.exists():
            with artifacts.zip_path.open("rb") as f:
                st.download_button(
                    label="Download Results (ZIP)",
                    data=f,
                    file_name=artifacts.zip_path.name,
                    mime="application/zip",
                )

        st.caption(f"Run folder: {artifacts.run_dir}")


if __name__ == "__main__":
    main()
