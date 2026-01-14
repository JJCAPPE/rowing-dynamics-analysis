from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    streamlit_image_coordinates = None  # type: ignore[assignment]

from rowing_pose.config import Annotations, RunConfig, VideoInfo, compute_m_per_px, save_run_config
from rowing_pose.io_video import VideoWriter, get_video_metadata, iter_frames, read_frame
from rowing_pose.pipeline import run_pipeline
from rowing_pose.pose2d_mmpose import COCO17_EDGES
from rowing_pose.viz import apply_affine_to_keypoints_xyc, draw_keypoints, draw_text_panel


# ---------------------------
# Small utilities
# ---------------------------

H36M17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def _sanitize_stem(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "video"


def _bgr_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    rgb = frame_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


@dataclass(frozen=True)
class DisplayFrame:
    orig_w: int
    orig_h: int
    disp_w: int
    disp_h: int
    pil_disp: Image.Image

    def to_orig_xy(self, x_disp: float, y_disp: float) -> Tuple[float, float]:
        sx = float(self.orig_w) / float(self.disp_w)
        sy = float(self.orig_h) / float(self.disp_h)
        return float(x_disp) * sx, float(y_disp) * sy


def _make_display_frame(frame_bgr: np.ndarray, max_w: int = 900) -> DisplayFrame:
    pil = _bgr_to_pil_rgb(frame_bgr)
    orig_w, orig_h = pil.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid frame size.")
    if orig_w > max_w:
        scale = float(max_w) / float(orig_w)
        disp_w = int(round(orig_w * scale))
        disp_h = int(round(orig_h * scale))
        pil_disp = pil.resize((disp_w, disp_h))
    else:
        disp_w, disp_h = orig_w, orig_h
        pil_disp = pil
    return DisplayFrame(
        orig_w=orig_w, orig_h=orig_h, disp_w=disp_w, disp_h=disp_h, pil_disp=pil_disp
    )


def _draw_overlay(
    disp: DisplayFrame,
    *,
    anchor: Optional[Tuple[float, float]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    scale0: Optional[Tuple[float, float]] = None,
    scale1: Optional[Tuple[float, float]] = None,
) -> Image.Image:
    """Draw existing annotations (in original px) onto the display image."""
    img = disp.pil_disp.copy()
    d = ImageDraw.Draw(img)

    def to_disp(pt: Tuple[float, float]) -> Tuple[int, int]:
        x, y = pt
        x_d = int(round(x * disp.disp_w / disp.orig_w))
        y_d = int(round(y * disp.disp_h / disp.orig_h))
        return x_d, y_d

    # BBox
    if bbox is not None:
        x, y, w, h = bbox
        x0, y0 = to_disp((x, y))
        x1, y1 = to_disp((x + w, y + h))
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        d.text((x0 + 6, y0 + 6), "bbox", fill=(0, 255, 0))

    # Anchor
    if anchor is not None:
        ax, ay = to_disp(anchor)
        r = 7
        d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
        d.text((ax + 10, ay - 10), "anchor", fill=(255, 0, 0))

    # Scale points
    if scale0 is not None:
        sx, sy = to_disp(scale0)
        r = 7
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
        d.text((sx + 10, sy - 10), "scale1", fill=(255, 255, 0))
    if scale1 is not None:
        sx, sy = to_disp(scale1)
        r = 7
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 165, 0), outline=(0, 0, 0), width=2)
        d.text((sx + 10, sy - 10), "scale2", fill=(255, 165, 0))

    return img


def _parse_first_rect(json_data: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Return (x,y,w,h) in canvas coords from st_canvas json_data."""
    if not json_data:
        return None
    objs = json_data.get("objects")
    if not isinstance(objs, list):
        return None
    # Prefer the most recently created rectangle.
    for o in reversed(objs):
        if not isinstance(o, dict):
            continue
        if o.get("type") != "rect":
            continue
        left = float(o.get("left", 0.0))
        top = float(o.get("top", 0.0))
        w = float(o.get("width", 0.0)) * float(o.get("scaleX", 1.0))
        h = float(o.get("height", 0.0)) * float(o.get("scaleY", 1.0))
        if w <= 1 or h <= 1:
            continue
        return left, top, w, h
    return None


# ---------------------------
# 3D overlay video generation
# ---------------------------


def _render_3d_inset(
    J3d: np.ndarray,
    size: Tuple[int, int] = (520, 520),
    pad: int = 18,
    *,
    mirror_x: bool = True,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    """Render a simple 3D stick figure inset (BGR uint8)."""
    import cv2

    W, H = int(size[0]), int(size[1])
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (10, 10, 10)

    J = np.asarray(J3d, dtype=np.float32).reshape(17, 3)
    if not np.isfinite(J).any():
        cv2.putText(canvas, "no 3D", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return canvas

    # Center at pelvis
    J = J - J[0:1, :]

    # Fixed camera rotation for a visible 3D effect.
    yaw = np.deg2rad(30.0)
    pitch = np.deg2rad(-12.0)

    Ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    R = (Rx @ Ry).astype(np.float32)
    Jr = (J @ R.T).astype(np.float32)

    x2 = -Jr[:, 0] if mirror_x else Jr[:, 0]
    # MotionBERT Y axis is typically up; map to image down.
    y2 = -Jr[:, 1] if flip_y else Jr[:, 1]
    z = -Jr[:, 2] if flip_z else Jr[:, 2]
    m = np.isfinite(x2) & np.isfinite(y2)
    if not m.any():
        cv2.putText(canvas, "no 3D", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return canvas

    x2 = x2[m]
    y2 = y2[m]

    x_min, x_max = float(np.min(x2)), float(np.max(x2))
    y_min, y_max = float(np.min(y2)), float(np.max(y2))
    rx = max(1e-6, x_max - x_min)
    ry = max(1e-6, y_max - y_min)

    sx = float(W - 2 * pad) / rx
    sy = float(H - 2 * pad) / ry
    s = float(min(sx, sy))

    x0 = float((x_min + x_max) / 2.0)
    y0 = float((y_min + y_max) / 2.0)

    # Full set again (including NaNs) for drawing edges.
    x_src = -Jr[:, 0] if mirror_x else Jr[:, 0]
    y_src = -Jr[:, 1] if flip_y else Jr[:, 1]
    x_img = (x_src - x0) * s + float(W) / 2.0
    y_img = (y_src - y0) * s + float(H) / 2.0

    def depth_color(zv: float) -> Tuple[int, int, int]:
        # Map depth to a color (near=green, far=blue) for some 3D cue.
        if not np.isfinite(zv):
            return (180, 180, 180)
        t = float(np.clip((zv + 1.0) / 2.0, 0.0, 1.0))
        b = int(round(255 * (1.0 - t)))
        g = int(round(255 * t))
        return (b, g, 60)

    # Draw edges
    for a, b in H36M17_EDGES:
        xa, ya = float(x_img[a]), float(y_img[a])
        xb, yb = float(x_img[b]), float(y_img[b])
        if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            continue
        c = depth_color(float((z[a] + z[b]) / 2.0))
        cv2.line(canvas, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), c, 2)

    # Draw joints
    for j in range(17):
        xj, yj = float(x_img[j]), float(y_img[j])
        if not (np.isfinite(xj) and np.isfinite(yj)):
            continue
        cv2.circle(canvas, (int(round(xj)), int(round(yj))), 3, (255, 255, 255), -1)

    cv2.putText(canvas, "3D", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def generate_pose3d_overlay_video(
    *,
    video_path: Path,
    stabilization_npz: Path,
    pose2d_npz: Path,
    pose3d_npz: Path,
    angles_csv: Optional[Path],
    out_video_path: Path,
    inset_size: Tuple[int, int] = (520, 520),
    mirror_3d: bool = True,
    flip_3d: bool = False,
    flip_3d_depth: bool = False,
) -> None:
    """Create debug/pose3d_overlay.mp4 from existing pipeline artifacts."""
    import cv2
    import pandas as pd

    meta = get_video_metadata(video_path)

    stab = np.load(stabilization_npz)
    A = stab["A"].astype(np.float32)  # (T,2,3)

    d2 = dict(np.load(pose2d_npz, allow_pickle=False))
    J2d_stab = d2["J2d_px"].astype(np.float32)  # (T,J,3)

    d3 = dict(np.load(pose3d_npz, allow_pickle=False))
    if d3.get("J3d_m") is not None and np.asarray(d3["J3d_m"]).size:
        J3d = np.asarray(d3["J3d_m"], dtype=np.float32).reshape(-1, 17, 3)
    else:
        J3d = np.asarray(d3["J3d_raw"], dtype=np.float32).reshape(-1, 17, 3)

    # Optional angles HUD.
    df = None
    if angles_csv is not None and angles_csv.exists():
        try:
            df = pd.read_csv(angles_csv)
            if "frame_idx" in df.columns:
                df = df.set_index("frame_idx", drop=False)
        except Exception:
            df = None

    fps = float(meta.fps) if meta.fps > 0 else 30.0
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    inset_w, inset_h = int(inset_size[0]), int(inset_size[1])
    margin = 12
    x0 = max(0, meta.width - inset_w - margin)
    y0 = margin

    # Prefer H.264 for browser playback; fall back to mp4v if unavailable.
    tmp_path = out_video_path
    writer_fourcc = "avc1"
    try:
        writer = VideoWriter(
            out_video_path, fps=fps, frame_size=(meta.width, meta.height), fourcc="avc1"
        )
    except Exception:
        writer_fourcc = "mp4v"
        tmp_path = out_video_path.with_suffix(".mp4v.mp4")
        writer = VideoWriter(tmp_path, fps=fps, frame_size=(meta.width, meta.height), fourcc="mp4v")

    with writer as vw:
        for idx, frame_bgr in iter_frames(video_path):
            if idx >= J2d_stab.shape[0] or idx >= J3d.shape[0] or idx >= A.shape[0]:
                break

            invA = cv2.invertAffineTransform(A[idx])
            J2d_orig = apply_affine_to_keypoints_xyc(J2d_stab[idx], invA)

            out = draw_keypoints(frame_bgr, J2d_orig, edges=COCO17_EDGES, min_conf=0.2)

            inset = _render_3d_inset(
                J3d[idx],
                size=(inset_w, inset_h),
                mirror_x=mirror_3d,
                flip_y=flip_3d,
                flip_z=flip_3d_depth,
            )

            # Composite inset (opaque) with a small border.
            cv2.rectangle(
                out, (x0 - 2, y0 - 2), (x0 + inset_w + 2, y0 + inset_h + 2), (0, 0, 0), -1
            )
            out[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset

            # HUD with a few angles if available.
            if df is not None:
                hud = []
                for col in df.columns:
                    if not col.endswith("_deg"):
                        continue
                    try:
                        v = float(df.loc[idx, col])  # type: ignore[index]
                    except Exception:
                        continue
                    if np.isfinite(v):
                        hud.append(f"{col.replace('_deg','')}: {v:.1f} deg")
                hud = hud[:10]
                out = draw_text_panel(out, hud, origin_xy=(10, 20))

            vw.write(out)

    # If mp4v was used, try to transcode to H.264 for browser playback.
    if writer_fourcc == "mp4v" and tmp_path != out_video_path:
        try:
            import subprocess

            if shutil.which("ffmpeg") is not None:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_path),
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_video_path),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                tmp_path.unlink()
            else:
                tmp_path.replace(out_video_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.replace(out_video_path)


# ---------------------------
# Streamlit app
# ---------------------------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("video_mode", "Upload")
    ss.setdefault("video_path_input", "")
    ss.setdefault("reference_frame_idx", 0)
    ss.setdefault("annotations_ref_idx", None)

    ss.setdefault("anchor_px", None)
    ss.setdefault("bbox_px", None)
    ss.setdefault("bbox_tl_px", None)
    ss.setdefault("bbox_br_px", None)
    ss.setdefault("scale0_px", None)
    ss.setdefault("scale1_px", None)
    ss.setdefault("scale_dist_m", 2.0)

    ss.setdefault("bbox_click_ver", 0)
    ss.setdefault("last_click_ts", {})
    ss.setdefault("out_dir", None)
    ss.setdefault("video_dest_path", None)

    ss.setdefault("last_run_error", None)
    ss.setdefault("mb_download_error", None)


def main() -> None:
    st.set_page_config(page_title="Rowing Pose UI", layout="wide")
    _init_state()

    repo_root = Path(__file__).resolve().parents[1]  # pose-extraction-test/

    st.title("Rowing Pose — Streamlit UI")
    st.caption("Browser-based annotation → run pipeline → 3D overlay video")

    if streamlit_image_coordinates is None:
        import sys

        st.error(
            "Missing Streamlit UI components. You are likely running Streamlit from the wrong Python "
            "environment.\n\n"
            "Fix:\n"
            "- Activate your venv: `source .venv/bin/activate`\n"
            "- Install UI deps: `python -m pip install -r ui_streamlit/requirements-ui.txt`\n"
            "- Run with venv Streamlit: `.venv/bin/python -m streamlit run ui_streamlit/app.py`\n\n"
            f"Current Python: `{sys.executable}`"
        )
        st.stop()

    with st.sidebar:
        st.header("Inputs")
        mode = st.radio("Video source", options=["Upload", "Path"], key="video_mode")

        uploaded = None
        src_path: Optional[Path] = None
        if mode == "Upload":
            uploaded = st.file_uploader("Upload video", type=None)
            if uploaded is not None:
                src_path = Path(uploaded.name)
        else:
            p = st.text_input(
                "Video path", key="video_path_input", placeholder="/path/to/video.mp4"
            )
            if p:
                src_path = Path(p).expanduser()

        use_timestamp = st.checkbox("Use timestamped output folder", value=False)

        st.divider()
        st.header("Pipeline options")
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
        mmpose_model = st.text_input("MMPose model alias", value="human")

        st.subheader("MotionBERT (required for 3D overlay)")
        motionbert_repo_root = repo_root / "third_party" / "MotionBERT"
        default_ckpt = (
            motionbert_repo_root
            / "checkpoint"
            / "pose3d"
            / "FT_MB_lite_MB_ft_h36m_global_lite"
            / "best_epoch.bin"
        )
        default_ckpt_url = (
            "https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
            "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
        )

        st.write(f"Repo: `{motionbert_repo_root}`")

        # Advanced: custom checkpoint/root (rarely needed)
        with st.expander("Advanced MotionBERT settings"):
            use_custom_ckpt = st.checkbox("Use custom checkpoint path", value=False)
            custom_ckpt_path = st.text_input(
                "Checkpoint path (.bin/.pth)", value="", placeholder="/path/to/best_epoch.bin"
            )
            use_custom_root = st.checkbox("Use custom MotionBERT repo root", value=False)
            custom_root_path = st.text_input(
                "Repo root", value="", placeholder="/path/to/MotionBERT"
            )

        # Resolve which checkpoint/root to use.
        motionbert_root: Optional[Path] = None
        if use_custom_root and custom_root_path.strip():
            motionbert_root = Path(custom_root_path).expanduser()

        motionbert_ckpt: Optional[Path] = None
        if use_custom_ckpt and custom_ckpt_path.strip():
            motionbert_ckpt = Path(custom_ckpt_path).expanduser()
        elif default_ckpt.exists():
            motionbert_ckpt = default_ckpt

        if motionbert_ckpt is not None and motionbert_ckpt.exists():
            st.success(f"Checkpoint found: `{motionbert_ckpt}`")
        else:
            st.warning(
                "MotionBERT checkpoint not found. Download once and it will be reused locally.\n\n"
                f"Expected: `{default_ckpt}`"
            )

            confirm = st.checkbox(
                "I want to download MotionBERT 3D pose checkpoint (~162MB) from HuggingFace",
                value=False,
                key="mb_download_confirm",
            )
            download_btn = st.button("Download checkpoint", disabled=not confirm)

            if download_btn:
                st.session_state["mb_download_error"] = None
                try:
                    import urllib.request

                    default_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    tmp = default_ckpt.with_suffix(default_ckpt.suffix + ".part")
                    if tmp.exists():
                        tmp.unlink()

                    st.info(
                        "Downloading... (saved locally under third_party/MotionBERT/checkpoint/)"
                    )
                    progress = st.progress(0)
                    status = st.empty()

                    with urllib.request.urlopen(default_ckpt_url) as resp:
                        total = resp.headers.get("Content-Length")
                        total_bytes = int(total) if total is not None else None
                        done = 0
                        with tmp.open("wb") as f:
                            while True:
                                chunk = resp.read(1024 * 1024)  # 1MB
                                if not chunk:
                                    break
                                f.write(chunk)
                                done += len(chunk)
                                if total_bytes and total_bytes > 0:
                                    pct = int(min(100, (done * 100) // total_bytes))
                                    progress.progress(pct)
                                    status.write(
                                        f"{done/1024/1024:.1f} / {total_bytes/1024/1024:.1f} MB"
                                    )
                                else:
                                    status.write(f"{done/1024/1024:.1f} MB")

                    tmp.replace(default_ckpt)
                    progress.progress(100)
                    status.write("Download complete.")
                    st.rerun()
                except Exception as e:
                    st.session_state["mb_download_error"] = f"{type(e).__name__}: {e}"
                    try:
                        if "tmp" in locals() and tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass

        if st.session_state.get("mb_download_error"):
            st.error(st.session_state["mb_download_error"])

        clip_len = st.number_input("clip_len", min_value=1, max_value=2000, value=243, step=1)
        flip = st.checkbox("flip augmentation", value=False)
        rootrel = st.checkbox("root-relative output", value=False)

        st.subheader("3D inset orientation")
        mirror_3d = st.checkbox("Mirror 3D left/right", value=True)
        flip_3d = st.checkbox("Flip 3D upside-down", value=False)
        flip_3d_depth = st.checkbox("Flip 3D depth (towards/away)", value=False)

    if src_path is None:
        st.info("Select a video in the sidebar to start.")
        return

    if mode == "Path" and not src_path.exists():
        st.error(f"Video path does not exist: {src_path}")
        return

    video_stem = _sanitize_stem(src_path.stem)
    out_dir_name = f"out_{video_stem}"
    if use_timestamp:
        import datetime as _dt

        out_dir_name = f"{out_dir_name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = repo_root / out_dir_name
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save/copy video into debug/ and use that as the pipeline input (self-contained out folder).
    suffix = src_path.suffix or ".mp4"
    video_dest_path = debug_dir / f"source_video{suffix}"
    if mode == "Upload" and uploaded is not None:
        buf = uploaded.getbuffer()
        need_write = (not video_dest_path.exists()) or (video_dest_path.stat().st_size != len(buf))
        if need_write:
            with video_dest_path.open("wb") as f:
                f.write(buf)
    else:
        # Path mode: copy on first use
        if not video_dest_path.exists():
            shutil.copy2(str(src_path), str(video_dest_path))

    st.session_state["out_dir"] = str(out_dir)
    st.session_state["video_dest_path"] = str(video_dest_path)

    meta = get_video_metadata(video_dest_path)

    st.subheader("Video")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.write(f"**Output folder:** `{out_dir.relative_to(repo_root)}`")
        st.write(f"**Source video (copied):** `{video_dest_path.relative_to(repo_root)}`")
    with c2:
        st.metric("FPS", f"{meta.fps:.2f}")
        st.metric("Frames", f"{meta.frame_count}")
    with c3:
        st.metric("Width", f"{meta.width}")
        st.metric("Height", f"{meta.height}")

    st.subheader("1) Reference frame")
    ref_idx = st.slider(
        "reference_frame_idx",
        min_value=0,
        max_value=max(0, int(meta.frame_count) - 1),
        value=int(st.session_state["reference_frame_idx"]),
        step=1,
        key="reference_frame_idx",
    )
    frame0 = read_frame(video_dest_path, int(ref_idx))
    disp = _make_display_frame(frame0, max_w=900)

    st.subheader("2) Annotation")
    st.caption(
        "Pick: anchor point → bbox → scale point #1 → scale point #2 → enter known distance (m)."
    )

    a_px = st.session_state["anchor_px"]
    b_px = st.session_state["bbox_px"]
    bbox_tl_px = st.session_state["bbox_tl_px"]
    bbox_br_px = st.session_state["bbox_br_px"]
    s0_px = st.session_state["scale0_px"]
    s1_px = st.session_state["scale1_px"]
    ann_ref_idx = st.session_state.get("annotations_ref_idx")

    if (
        a_px is not None or b_px is not None or s0_px is not None or s1_px is not None
    ) and ann_ref_idx is not None:
        if int(ann_ref_idx) != int(ref_idx):
            st.warning(
                "Reference frame changed after you started annotating. "
                "Either set it back or reset annotations (recommended)."
            )

    missing_steps = []
    if a_px is None:
        missing_steps.append("Anchor")
    if b_px is None:
        missing_steps.append("BBox")
    if s0_px is None:
        missing_steps.append("Scale #1")
    if s1_px is None:
        missing_steps.append("Scale #2")
    if not missing_steps:
        missing_steps.append("Review")

    step = st.radio(
        "Annotation step",
        options=["Anchor", "BBox", "Scale #1", "Scale #2", "Review"],
        index=["Anchor", "BBox", "Scale #1", "Scale #2", "Review"].index(missing_steps[0]),
        horizontal=True,
    )

    col_left, col_right = st.columns([3, 2])
    with col_right:
        st.markdown("**Current annotations (original px)**")
        st.write(
            {
                "anchor_px": a_px,
                "bbox_px": b_px,
                "scale_points_px": (s0_px, s1_px),
                "scale_distance_m": float(st.session_state["scale_dist_m"]),
            }
        )
        if st.button("Reset all annotations"):
            st.session_state["anchor_px"] = None
            st.session_state["bbox_px"] = None
            st.session_state["bbox_tl_px"] = None
            st.session_state["bbox_br_px"] = None
            st.session_state["scale0_px"] = None
            st.session_state["scale1_px"] = None
            st.session_state["annotations_ref_idx"] = None
            st.session_state["bbox_click_ver"] += 1
            st.rerun()

        st.session_state["scale_dist_m"] = st.number_input(
            "Known distance between scale points (meters)",
            min_value=0.01,
            max_value=1000.0,
            value=float(st.session_state["scale_dist_m"]),
            step=0.1,
        )

    with col_left:
        overlay_img = _draw_overlay(disp, anchor=a_px, bbox=b_px, scale0=s0_px, scale1=s1_px)

        if step in ("Anchor", "Scale #1", "Scale #2"):
            st.markdown("**Click on the image**")
            click = streamlit_image_coordinates(overlay_img, key=f"click_{step}_{ref_idx}")
            if click is not None:
                ts = click.get("unix_time")
                last_ts = st.session_state["last_click_ts"].get(step)
                if ts is None or ts == last_ts:
                    st.stop()
                st.session_state["last_click_ts"][step] = ts
                x_disp = float(click["x"])
                y_disp = float(click["y"])
                x_orig, y_orig = disp.to_orig_xy(x_disp, y_disp)
                if step == "Anchor":
                    st.session_state["anchor_px"] = (x_orig, y_orig)
                elif step == "Scale #1":
                    st.session_state["scale0_px"] = (x_orig, y_orig)
                elif step == "Scale #2":
                    st.session_state["scale1_px"] = (x_orig, y_orig)
                if st.session_state.get("annotations_ref_idx") is None:
                    st.session_state["annotations_ref_idx"] = int(ref_idx)
                st.rerun()

        elif step == "BBox":
            st.markdown("**BBox via 2 clicks**: click **top-left**, then click **bottom-right**.")
            st.caption(
                "Tip: If you want to redraw, just click again after completing both corners (it will restart)."
            )

            click = streamlit_image_coordinates(
                overlay_img, key=f"click_bbox_{st.session_state['bbox_click_ver']}_{ref_idx}"
            )
            if click is not None:
                ts = click.get("unix_time")
                last_ts = st.session_state["last_click_ts"].get("BBox")
                if ts is not None and ts != last_ts:
                    st.session_state["last_click_ts"]["BBox"] = ts
                    x_disp = float(click["x"])
                    y_disp = float(click["y"])
                    x_orig, y_orig = disp.to_orig_xy(x_disp, y_disp)

                    if bbox_tl_px is None or (bbox_tl_px is not None and bbox_br_px is not None):
                        # Start / restart bbox definition.
                        st.session_state["bbox_tl_px"] = (x_orig, y_orig)
                        st.session_state["bbox_br_px"] = None
                        st.session_state["bbox_px"] = None
                    else:
                        # Finish bbox.
                        st.session_state["bbox_br_px"] = (x_orig, y_orig)
                        x0 = float(min(st.session_state["bbox_tl_px"][0], x_orig))
                        y0 = float(min(st.session_state["bbox_tl_px"][1], y_orig))
                        x1 = float(max(st.session_state["bbox_tl_px"][0], x_orig))
                        y1 = float(max(st.session_state["bbox_tl_px"][1], y_orig))
                        w = float(x1 - x0)
                        h = float(y1 - y0)
                        if w > 1 and h > 1:
                            st.session_state["bbox_px"] = (x0, y0, w, h)

                    if st.session_state.get("annotations_ref_idx") is None:
                        st.session_state["annotations_ref_idx"] = int(ref_idx)
                    st.rerun()

            if st.button("Clear bbox"):
                st.session_state["bbox_px"] = None
                st.session_state["bbox_tl_px"] = None
                st.session_state["bbox_br_px"] = None
                st.session_state["bbox_click_ver"] += 1
                st.rerun()

        else:
            st.image(
                overlay_img, caption="Reference frame with annotations", use_container_width=False
            )

    st.subheader("3) Run + results")

    # Validate readiness.
    ready = (
        st.session_state["anchor_px"] is not None
        and st.session_state["bbox_px"] is not None
        and st.session_state["scale0_px"] is not None
        and st.session_state["scale1_px"] is not None
        and float(st.session_state["scale_dist_m"]) > 0
        and motionbert_ckpt is not None
        and motionbert_ckpt.exists()
        and (
            st.session_state.get("annotations_ref_idx") is None
            or int(st.session_state.get("annotations_ref_idx")) == int(ref_idx)
        )
    )

    if not ready:
        st.warning(
            "Complete annotation and ensure the MotionBERT checkpoint is available "
            "(use the sidebar download button once)."
        )
        return

    run_btn = st.button("Run pipeline + generate 3D overlay", type="primary")

    if run_btn:
        st.session_state["last_run_error"] = None
        try:
            # Write run.json (prevents OpenCV annotation UI from running).
            rel_video = str(Path("debug") / video_dest_path.name)
            dist_m = float(st.session_state["scale_dist_m"])
            m_per_px = compute_m_per_px(
                (st.session_state["scale0_px"], st.session_state["scale1_px"]), dist_m
            )

            cfg = RunConfig(
                version=1,
                video=VideoInfo(
                    path=rel_video,
                    fps=float(meta.fps),
                    width=int(meta.width),
                    height=int(meta.height),
                    frame_count=int(meta.frame_count),
                ),
                reference_frame_idx=int(ref_idx),
                annotations=Annotations(
                    anchor_px=st.session_state["anchor_px"],
                    bbox_px=st.session_state["bbox_px"],
                    scale_points_px=(st.session_state["scale0_px"], st.session_state["scale1_px"]),
                    scale_distance_m=dist_m,
                ),
                derived={"m_per_px": float(m_per_px)},
                params={
                    "stabilization": {
                        "lk_win_size": [21, 21],
                        "lk_max_level": 3,
                        "lk_max_corners": 1,
                        "template_half_size": 16,
                        "template_search_radius": 50,
                    },
                    "crop": {
                        "padding": 0.2,
                        "ema_alpha": 0.9,
                        "min_points": 10,
                    },
                },
            )

            save_run_config(out_dir / "run.json", cfg)

            # Pipeline run.
            if motionbert_ckpt is None or not motionbert_ckpt.exists():
                raise FileNotFoundError(
                    "MotionBERT checkpoint not found. Download it in the sidebar."
                )
            ckpt = motionbert_ckpt

            with st.spinner("Running pipeline (this can take a while)..."):
                run_pipeline(
                    video_path=video_dest_path,
                    out_dir=out_dir,
                    device=device,
                    mmpose_model=mmpose_model,
                    motionbert_root=motionbert_root,
                    motionbert_ckpt=ckpt,
                    clip_len=int(clip_len),
                    flip=bool(flip),
                    rootrel=bool(rootrel),
                    skip_2d=False,
                    skip_3d=False,
                )

            # Generate 3D overlay video.
            stab_npz = out_dir / "stabilization.npz"
            pose2d_npz = out_dir / "pose2d.npz"
            pose3d_npz = out_dir / "pose3d.npz"
            angles_csv = out_dir / "angles.csv"
            out_pose3d_mp4 = out_dir / "debug" / "pose3d_overlay.mp4"

            with st.spinner("Rendering 3D overlay video..."):
                generate_pose3d_overlay_video(
                    video_path=video_dest_path,
                    stabilization_npz=stab_npz,
                    pose2d_npz=pose2d_npz,
                    pose3d_npz=pose3d_npz,
                    angles_csv=angles_csv if angles_csv.exists() else None,
                    out_video_path=out_pose3d_mp4,
                    mirror_3d=bool(mirror_3d),
                    flip_3d=bool(flip_3d),
                    flip_3d_depth=bool(flip_3d_depth),
                )

        except Exception as e:
            st.session_state["last_run_error"] = f"{type(e).__name__}: {e}"

    if st.session_state.get("last_run_error"):
        st.error(st.session_state["last_run_error"])
        return

    # Show results if present.
    pose3d_overlay = out_dir / "debug" / "pose3d_overlay.mp4"
    if pose3d_overlay.exists():
        st.success("Done.")
        st.video(str(pose3d_overlay))

        # Also show existing pipeline output if present.
        angles_overlay = out_dir / "debug" / "angles_overlay.mp4"

        st.markdown("**Downloads**")

        # Zip the whole out_<video> folder (optional; can be large).
        zip_path = out_dir.with_suffix(".zip")
        if st.button("Create/refresh ZIP of output folder"):
            with st.spinner("Creating ZIP..."):
                if zip_path.exists():
                    zip_path.unlink()
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in out_dir.rglob("*"):
                        if p.is_dir():
                            continue
                        zf.write(p, arcname=str(p.relative_to(out_dir.parent)))

        if zip_path.exists():
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            st.write(f"ZIP: `{zip_path.relative_to(repo_root)}` ({size_mb:.1f} MB)")
            # Avoid reading extremely large zips into memory for download_button.
            if size_mb <= 500:
                st.download_button(
                    "Download ZIP",
                    data=zip_path.read_bytes(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )
            else:
                st.info("ZIP is large; download disabled in-app. Use the file path above.")

        # Key file downloads
        key_files = [
            out_dir / "run.json",
            out_dir / "angles.csv",
            out_dir / "metrics.json",
            out_dir / "debug" / "pose3d_overlay.mp4",
            video_dest_path,
        ]
        for p in key_files:
            if not p.exists():
                continue
            label = f"Download {p.relative_to(out_dir)}"
            mime = "application/octet-stream"
            if p.suffix.lower() == ".json":
                mime = "application/json"
            elif p.suffix.lower() == ".csv":
                mime = "text/csv"
            elif p.suffix.lower() == ".mp4":
                mime = "video/mp4"
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > 500:
                st.info(f"`{p.name}` is large ({size_mb:.1f} MB); download disabled in-app.")
                continue
            st.download_button(label, data=p.read_bytes(), file_name=p.name, mime=mime)


if __name__ == "__main__":
    main()
