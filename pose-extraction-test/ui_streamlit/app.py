from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]  # pose-extraction-test/
MODELS_DIR = REPO_ROOT / "models"
os.environ.setdefault("MMENGINE_CACHE_DIR", str(MODELS_DIR / "mmengine"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    streamlit_image_coordinates = None  # type: ignore[assignment]

from rowing_pose.config import (
    POSE_SMOOTHING_DEFAULTS,
    Annotations,
    RunConfig,
    VideoInfo,
    compute_m_per_px,
    load_run_config,
    save_run_config,
)
from rowing_pose.io_video import VideoWriter, get_video_metadata, iter_frames, read_frame
from rowing_pose.model_assets import MODEL_PRESETS, MOTIONBERT_REPO, AssetSpec, ensure_asset
from rowing_pose.pipeline import run_pipeline
from rowing_pose.progress import ProgressReporter, get_progress
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


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _clear_annotations_state(ss: st.session_state.__class__) -> None:
    ss["anchor_px"] = None
    ss["rigger_bbox_px"] = None
    ss["rigger_bbox_tl_px"] = None
    ss["rigger_bbox_br_px"] = None
    ss["bbox_px"] = None
    ss["bbox_tl_px"] = None
    ss["bbox_br_px"] = None
    ss["scale0_px"] = None
    ss["scale1_px"] = None
    ss["annotations_ref_idx"] = None
    ss["bbox_click_ver"] += 1
    ss["rigger_bbox_click_ver"] += 1
    ss["last_click_ts"] = {}


class StreamlitProgressHandle:
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, label: str, total: Optional[int]
    ) -> None:
        self._total = int(total) if total is not None else None
        self._current = 0
        self._label = label
        self._container = container
        self._status = None
        self._status_text = None

        if hasattr(container, "status"):
            self._status = container.status(label, expanded=False)
        else:
            self._status_text = container.empty()
            self._status_text.markdown(label)

        self._bar = container.progress(0.0)

    def update(
        self, n: int = 1, *, desc: Optional[str] = None, total: Optional[int] = None
    ) -> None:
        if total is not None:
            self._total = int(total)
        if desc:
            self._label = desc
            if self._status is not None:
                self._status.update(label=desc, state="running")
            elif self._status_text is not None:
                self._status_text.markdown(desc)

        self._current += int(n)
        if self._total is not None and self._total > 0:
            self._bar.progress(min(self._current / float(self._total), 1.0))

    def close(self, *, status: Optional[str] = None) -> None:
        if status:
            self._label = status
        if self._total is not None and self._total > 0:
            self._bar.progress(1.0)
        if self._status is not None:
            self._status.update(label=self._label, state="complete")
        elif self._status_text is not None:
            self._status_text.markdown(self._label)


class StreamlitProgress(ProgressReporter):
    def __init__(self, parent: st.delta_generator.DeltaGenerator) -> None:
        self._parent = parent

    def start(
        self, label: str, total: Optional[int] = None, unit: str = "it"
    ) -> StreamlitProgressHandle:
        _ = unit
        block = self._parent.container()
        return StreamlitProgressHandle(block, label, total)


def _apply_annotations_from_run(ss: st.session_state.__class__, cfg: RunConfig) -> None:
    ann = cfg.annotations
    ss["anchor_px"] = ann.anchor_px
    ss["rigger_bbox_px"] = ann.rigger_bbox_px
    ss["rigger_bbox_tl_px"] = None
    ss["rigger_bbox_br_px"] = None
    ss["bbox_px"] = ann.bbox_px
    ss["bbox_tl_px"] = None
    ss["bbox_br_px"] = None
    ss["scale0_px"] = ann.scale_points_px[0]
    ss["scale1_px"] = ann.scale_points_px[1]
    ss["scale_dist_m"] = ann.scale_distance_m
    ss["annotations_ref_idx"] = int(cfg.reference_frame_idx)
    ss["reference_frame_idx"] = int(cfg.reference_frame_idx)
    ss["bbox_click_ver"] += 1
    ss["rigger_bbox_click_ver"] += 1
    ss["last_click_ts"] = {}


def _find_latest_run_json(repo_root: Path, video_stem: str) -> Optional[Path]:
    candidates = list(repo_root.glob(f"out_{video_stem}*/run.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _make_video_key(
    *, mode: str, src_path: Optional[Path], uploaded: Optional[Any]
) -> Optional[str]:
    if src_path is None:
        return None
    if mode == "Path":
        try:
            return f"path:{src_path.expanduser().resolve()}"
        except Exception:
            return f"path:{src_path}"
    if uploaded is None:
        return None
    size = getattr(uploaded, "size", None)
    return f"upload:{uploaded.name}:{size}"


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


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    label: str
    asset: AssetSpec


def _draw_overlay(
    disp: DisplayFrame,
    *,
    anchor: Optional[Tuple[float, float]] = None,
    rigger_bbox: Optional[Tuple[float, float, float, float]] = None,
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

    # Rigger bbox
    if rigger_bbox is not None:
        x, y, w, h = rigger_bbox
        x0, y0 = to_disp((x, y))
        x1, y1 = to_disp((x + w, y + h))
        d.rectangle([x0, y0, x1, y1], outline=(0, 128, 255), width=3)
        d.text((x0 + 6, y0 + 6), "rigger", fill=(0, 128, 255))

    # Athlete bbox
    if bbox is not None:
        x, y, w, h = bbox
        x0, y0 = to_disp((x, y))
        x1, y1 = to_disp((x + w, y + h))
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        d.text((x0 + 6, y0 + 6), "athlete", fill=(0, 255, 0))

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
# Model presets / downloads
# ---------------------------

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
    progress: Optional[ProgressReporter] = None,
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
    prog = get_progress(progress)
    total = int(min(meta.frame_count, J2d_stab.shape[0], J3d.shape[0], A.shape[0]))
    stage = prog.start("Stage J: 3D overlay", total=total, unit="frame")

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
            stage.update(1)
    stage.close()

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
    ss.setdefault("active_video_key", None)
    ss.setdefault("preloaded_run_json", None)

    ss.setdefault("anchor_px", None)
    ss.setdefault("rigger_bbox_px", None)
    ss.setdefault("rigger_bbox_tl_px", None)
    ss.setdefault("rigger_bbox_br_px", None)
    ss.setdefault("bbox_px", None)
    ss.setdefault("bbox_tl_px", None)
    ss.setdefault("bbox_br_px", None)
    ss.setdefault("scale0_px", None)
    ss.setdefault("scale1_px", None)
    ss.setdefault("scale_dist_m", 2.0)

    ss.setdefault("bbox_click_ver", 0)
    ss.setdefault("rigger_bbox_click_ver", 0)
    ss.setdefault("last_click_ts", {})
    ss.setdefault("out_dir", None)
    ss.setdefault("video_dest_path", None)

    ss.setdefault("last_run_error", None)
    ss.setdefault("download_errors", {})


def main() -> None:
    st.set_page_config(page_title="Rowing Pose UI", layout="wide")
    _init_state()

    repo_root = REPO_ROOT
    mmpose_available = (
        _module_available("mmpose") and _module_available("mmcv") and _module_available("mmdet")
    )
    motionbert_deps_available = (
        _module_available("torch") and _module_available("yaml") and _module_available("easydict")
    )
    deepsort_available = _module_available("ultralytics") and _module_available(
        "deep_sort_realtime"
    )

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
        st.header("Setup")
        st.subheader("1) Video source")
        mode = st.radio("Source", options=["Upload", "Path"], key="video_mode")

        uploaded = None
        src_path: Optional[Path] = None
        if mode == "Upload":
            uploaded = st.file_uploader("Upload file", type=None)
            if uploaded is not None:
                src_path = Path(uploaded.name)
        else:
            p = st.text_input("File path", key="video_path_input", placeholder="/path/to/video.mp4")
            if p:
                src_path = Path(p).expanduser()

        use_timestamp = st.checkbox("Use timestamped output folder", value=False)

        st.divider()
        st.subheader("2) Pipeline stages")
        st.caption("Always on: stabilization + crop.")
        enable_2d = st.checkbox(
            "Include 2D pose (MMPose)", value=mmpose_available, disabled=not mmpose_available
        )
        if not mmpose_available:
            st.info("MMPose not installed. Install mmpose + mmcv + mmdet to enable 2D.")

        enable_3d = st.checkbox(
            "Include 3D lift (MotionBERT)",
            value=motionbert_deps_available,
            disabled=not motionbert_deps_available,
        )
        if not motionbert_deps_available:
            st.info("MotionBERT deps missing. Install torch, pyyaml, easydict to enable 3D.")
        if enable_3d and not enable_2d:
            st.warning("3D depends on 2D; disabling 3D.")
            enable_3d = False

        st.divider()
        st.subheader("3) Model level")
        preset_keys = list(MODEL_PRESETS.keys())
        preset_key = st.selectbox(
            "Model level (accuracy vs. speed)",
            options=preset_keys,
            index=0,
            format_func=lambda k: MODEL_PRESETS[k].label,
        )
        preset = MODEL_PRESETS[preset_key]

        st.caption(f"2D model: {preset.pose2d.label}")
        st.caption(f"3D model: {preset.motionbert.label}")

        mmpose_model = preset.pose2d.model_id
        pose2d_config_asset = preset.pose2d.config
        pose2d_ckpt_asset = preset.pose2d.checkpoint
        mmpose_config = pose2d_config_asset.path if pose2d_config_asset else None
        mmpose_checkpoint = pose2d_ckpt_asset.path if pose2d_ckpt_asset else None
        motionbert_model = preset.motionbert.key
        motionbert_root = MOTIONBERT_REPO if MOTIONBERT_REPO.exists() else None
        motionbert_ckpt = preset.motionbert.checkpoint.path
        motionbert_config = preset.motionbert.config_path

        use_custom_pose2d = False
        use_custom_motionbert = False
        with st.expander("Advanced model paths"):
            use_custom_pose2d = st.checkbox("Custom 2D model/config/weights", value=False)
            if use_custom_pose2d:
                mmpose_model = st.text_input("MMPose model id/alias", value=mmpose_model)
                custom_cfg_path = st.text_input(
                    "MMPose config path (optional)",
                    value=str(mmpose_config) if mmpose_config else "",
                )
                if custom_cfg_path.strip():
                    mmpose_config = Path(custom_cfg_path).expanduser()
                else:
                    mmpose_config = None

                custom_ckpt_path = st.text_input(
                    "MMPose checkpoint path (optional)",
                    value=str(mmpose_checkpoint) if mmpose_checkpoint else "",
                )
                if custom_ckpt_path.strip():
                    mmpose_checkpoint = Path(custom_ckpt_path).expanduser()
                else:
                    mmpose_checkpoint = None

            use_custom_motionbert = st.checkbox("Custom MotionBERT paths", value=False)
            if use_custom_motionbert:
                custom_root_path = st.text_input(
                    "MotionBERT repo root",
                    value=str(motionbert_root) if motionbert_root else "",
                )
                if custom_root_path.strip():
                    motionbert_root = Path(custom_root_path).expanduser()
                else:
                    motionbert_root = None

                custom_ckpt_path = st.text_input(
                    "MotionBERT checkpoint path",
                    value=str(motionbert_ckpt) if motionbert_ckpt else "",
                )
                if custom_ckpt_path.strip():
                    motionbert_ckpt = Path(custom_ckpt_path).expanduser()
                else:
                    motionbert_ckpt = None

                custom_cfg_path = st.text_input(
                    "MotionBERT config path",
                    value=str(motionbert_config) if motionbert_config else "",
                )
                if custom_cfg_path.strip():
                    motionbert_config = Path(custom_cfg_path).expanduser()
                else:
                    motionbert_config = None

        st.markdown("**Model assets**")
        download_errors: Dict[str, str] = st.session_state["download_errors"]

        def ensure_download(spec: DownloadSpec, enabled: bool) -> bool:
            if not enabled:
                return True
            if spec.asset.path.exists():
                st.caption(f"{spec.label}: ready")
                return True
            if download_errors.get(spec.key):
                st.error(f"{spec.label}: {download_errors[spec.key]}")
                if st.button(f"Retry {spec.label}", key=f"retry_{spec.key}"):
                    download_errors.pop(spec.key, None)
                    st.rerun()
                return False
            try:
                with st.spinner(f"Downloading {spec.label}..."):
                    ensure_asset(
                        spec.asset.path,
                        spec.asset.url,
                        expected_size=spec.asset.size_bytes,
                        sha256=spec.asset.sha256,
                    )
            except Exception as e:
                download_errors[spec.key] = f"{type(e).__name__}: {e}"
                st.error(f"{spec.label}: {download_errors[spec.key]}")
                return False
            st.rerun()
            return False

        pose2d_ready = True
        if enable_2d:
            if not use_custom_pose2d:
                if pose2d_config_asset is not None:
                    cfg_spec = DownloadSpec(
                        key=f"pose2d_cfg_{preset.pose2d.key}",
                        label=f"{preset.pose2d.label} config",
                        asset=pose2d_config_asset,
                    )
                    pose2d_ready = ensure_download(cfg_spec, enabled=True) and pose2d_ready
                if pose2d_ckpt_asset is not None:
                    ckpt_spec = DownloadSpec(
                        key=f"pose2d_ckpt_{preset.pose2d.key}",
                        label=f"{preset.pose2d.label} weights",
                        asset=pose2d_ckpt_asset,
                    )
                    pose2d_ready = ensure_download(ckpt_spec, enabled=True) and pose2d_ready
            else:
                if mmpose_config is not None and not mmpose_config.exists():
                    st.warning("Custom 2D config path not found.")
                    pose2d_ready = False
                if mmpose_checkpoint is not None and not mmpose_checkpoint.exists():
                    st.warning("Custom 2D checkpoint path not found.")
                    pose2d_ready = False

        pose3d_ready = True
        if enable_3d:
            if motionbert_root is None or not motionbert_root.exists():
                st.error("MotionBERT repo not found. Ensure the submodule is available.")
                pose3d_ready = False
            if motionbert_config is None or not motionbert_config.exists():
                st.error("MotionBERT config not found.")
                pose3d_ready = False
            if motionbert_ckpt is None:
                st.error("MotionBERT checkpoint not set.")
                pose3d_ready = False

            if not use_custom_motionbert:
                mb_spec = DownloadSpec(
                    key=f"motionbert_{preset.motionbert.key}",
                    label=f"{preset.motionbert.label} weights",
                    asset=preset.motionbert.checkpoint,
                )
                pose3d_ready = pose3d_ready and ensure_download(mb_spec, enabled=True)
            elif motionbert_ckpt is not None and not motionbert_ckpt.exists():
                st.warning("Custom MotionBERT checkpoint path not found.")
                pose3d_ready = False

        if not enable_3d:
            st.caption("3D stage disabled; MotionBERT settings are inactive.")

        st.divider()
        st.subheader("4) Runtime")
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0)

        st.divider()
        st.subheader("5) Tracking & smoothing")
        st.markdown("**Tracking**")
        pose_tracking_enabled = st.checkbox(
            "Enable pose tracking (2D keypoints)",
            value=True,
            disabled=not enable_2d,
        )
        st.caption("Stabilizes keypoint identity across frames during 2D inference.")

        pose_track_smooth_alpha = st.slider(
            "Tracking smoothing alpha (EMA)",
            min_value=0.0,
            max_value=0.1,
            value=0.05,
            step=0.01,
            disabled=not enable_2d or not pose_tracking_enabled,
        )
        st.caption("Lower values reduce lag; higher values stabilize.")

        strict_id = st.checkbox(
            "Advanced tracking: strict ID (DeepSORT)",
            value=False,
            disabled=not deepsort_available,
        )
        st.caption("Uses full-frame detection + tracking. Slower but most stable.")
        if not deepsort_available:
            st.info("Install `ultralytics` and `deep_sort_realtime` to enable strict ID tracking.")

        deepsort_model = st.text_input(
            "Detector model (YOLO)",
            value="yolov8n.pt",
            disabled=not strict_id,
        )
        deepsort_min_conf = st.slider(
            "Detector min confidence",
            min_value=0.05,
            max_value=0.9,
            value=0.25,
            step=0.05,
            disabled=not strict_id,
        )
        deepsort_padding = st.slider(
            "Tracking bbox padding",
            min_value=0.0,
            max_value=0.6,
            value=0.2,
            step=0.05,
            disabled=not strict_id,
        )

        st.markdown("**Pose smoothing**")
        smooth_pose = st.checkbox(
            "Enable pose smoothing (recommended)",
            value=bool(POSE_SMOOTHING_DEFAULTS.get("enabled", True)),
        )
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(POSE_SMOOTHING_DEFAULTS.get("conf_threshold", 0.3)),
            step=0.05,
            disabled=not smooth_pose,
        )
        max_gap = st.number_input(
            "Max gap (frames)",
            min_value=0,
            max_value=60,
            value=int(POSE_SMOOTHING_DEFAULTS.get("max_gap", 5)),
            step=1,
            disabled=not smooth_pose,
        )
        median_window = st.number_input(
            "Median window (odd)",
            min_value=1,
            max_value=61,
            value=int(POSE_SMOOTHING_DEFAULTS.get("median_window", 5)),
            step=2,
            disabled=not smooth_pose,
        )

        st.divider()
        st.subheader("6) 3D settings")
        clip_len = st.number_input(
            "MotionBERT clip length (frames)",
            min_value=1,
            max_value=2000,
            value=243,
            step=1,
            disabled=not enable_3d,
        )
        flip = st.checkbox("Enable flip augmentation", value=True, disabled=not enable_3d)
        rootrel = st.checkbox("Root-relative 3D output", value=False, disabled=not enable_3d)

        st.subheader("7) 3D overlay")
        mirror_3d = st.checkbox("Mirror left/right", value=True, disabled=not enable_3d)
        flip_3d = st.checkbox("Flip upside-down", value=False, disabled=not enable_3d)
        flip_3d_depth = st.checkbox(
            "Flip depth (towards/away)", value=False, disabled=not enable_3d
        )

        st.divider()
        st.markdown("**Pipeline summary**")
        st.caption(
            "Stages: stabilization (on), crop (on), "
            f"2D pose ({'on' if enable_2d else 'off'}), "
            f"tracking ({'on' if enable_2d and pose_tracking_enabled else 'off'}), "
            f"smoothing ({'on' if enable_2d and smooth_pose else 'off'}), "
            f"3D lift ({'on' if enable_3d else 'off'})."
        )
        extras = []
        if strict_id:
            extras.append("DeepSORT strict ID")
        if enable_3d:
            if flip:
                extras.append("flip augmentation")
            if rootrel:
                extras.append("root-relative 3D")
        if extras:
            st.caption("Extras: " + ", ".join(extras))

    if src_path is None:
        st.info("Select a video in the sidebar to start.")
        return

    if mode == "Path" and not src_path.exists():
        st.error(f"Video path does not exist: {src_path}")
        return

    video_stem = _sanitize_stem(src_path.stem)
    video_key = _make_video_key(mode=mode, src_path=src_path, uploaded=uploaded)
    if video_key != st.session_state.get("active_video_key"):
        st.session_state["active_video_key"] = video_key
        st.session_state["preloaded_run_json"] = None
        _clear_annotations_state(st.session_state)
        run_json = _find_latest_run_json(repo_root, video_stem)
        if run_json is not None and run_json.exists():
            try:
                cfg = load_run_config(run_json)
                _apply_annotations_from_run(st.session_state, cfg)
                st.session_state["preloaded_run_json"] = str(run_json)
            except Exception:
                st.session_state["preloaded_run_json"] = None
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
        "Pick: anchor point → rigger bbox → athlete bbox → scale point #1 → scale point #2 → "
        "enter known distance (m)."
    )
    if st.session_state.get("preloaded_run_json"):
        rel = Path(st.session_state["preloaded_run_json"])
        try:
            rel = rel.relative_to(repo_root)
        except ValueError:
            pass
        st.info(f"Loaded previous annotations from `{rel}`.")

    a_px = st.session_state["anchor_px"]
    rigger_px = st.session_state["rigger_bbox_px"]
    rigger_tl_px = st.session_state["rigger_bbox_tl_px"]
    rigger_br_px = st.session_state["rigger_bbox_br_px"]
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
    if rigger_px is None:
        missing_steps.append("Rigger BBox")
    if b_px is None:
        missing_steps.append("Athlete BBox")
    if s0_px is None:
        missing_steps.append("Scale #1")
    if s1_px is None:
        missing_steps.append("Scale #2")
    if not missing_steps:
        missing_steps.append("Review")

    step = st.radio(
        "Annotation step",
        options=["Anchor", "Rigger BBox", "Athlete BBox", "Scale #1", "Scale #2", "Review"],
        index=[
            "Anchor",
            "Rigger BBox",
            "Athlete BBox",
            "Scale #1",
            "Scale #2",
            "Review",
        ].index(missing_steps[0]),
        horizontal=True,
    )

    col_left, col_right = st.columns([3, 2])
    with col_right:
        st.markdown("**Current annotations (original px)**")
        st.write(
            {
                "anchor_px": a_px,
                "rigger_bbox_px": rigger_px,
                "athlete_bbox_px": b_px,
                "scale_points_px": (s0_px, s1_px),
                "scale_distance_m": float(st.session_state["scale_dist_m"]),
            }
        )
        if st.button("Reset all annotations"):
            st.session_state["anchor_px"] = None
            st.session_state["rigger_bbox_px"] = None
            st.session_state["rigger_bbox_tl_px"] = None
            st.session_state["rigger_bbox_br_px"] = None
            st.session_state["bbox_px"] = None
            st.session_state["bbox_tl_px"] = None
            st.session_state["bbox_br_px"] = None
            st.session_state["scale0_px"] = None
            st.session_state["scale1_px"] = None
            st.session_state["annotations_ref_idx"] = None
            st.session_state["bbox_click_ver"] += 1
            st.session_state["rigger_bbox_click_ver"] += 1
            st.rerun()

        st.session_state["scale_dist_m"] = st.number_input(
            "Known distance between scale points (meters)",
            min_value=0.01,
            max_value=1000.0,
            value=float(st.session_state["scale_dist_m"]),
            step=0.1,
        )

    with col_left:
        overlay_img = _draw_overlay(
            disp,
            anchor=a_px,
            rigger_bbox=rigger_px,
            bbox=b_px,
            scale0=s0_px,
            scale1=s1_px,
        )

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

        elif step == "Rigger BBox":
            st.markdown(
                "**Rigger bbox via 2 clicks**: click **top-left**, then click **bottom-right**."
            )
            st.caption("Tip: Make it tight around the rigger/oarlock hardware for stable tracking.")

            click = streamlit_image_coordinates(
                overlay_img,
                key=f"click_rigger_{st.session_state['rigger_bbox_click_ver']}_{ref_idx}",
            )
            if click is not None:
                ts = click.get("unix_time")
                last_ts = st.session_state["last_click_ts"].get("Rigger BBox")
                if ts is not None and ts != last_ts:
                    st.session_state["last_click_ts"]["Rigger BBox"] = ts
                    x_disp = float(click["x"])
                    y_disp = float(click["y"])
                    x_orig, y_orig = disp.to_orig_xy(x_disp, y_disp)

                    if rigger_tl_px is None or (
                        rigger_tl_px is not None and rigger_br_px is not None
                    ):
                        st.session_state["rigger_bbox_tl_px"] = (x_orig, y_orig)
                        st.session_state["rigger_bbox_br_px"] = None
                        st.session_state["rigger_bbox_px"] = None
                    else:
                        st.session_state["rigger_bbox_br_px"] = (x_orig, y_orig)
                        x0 = float(min(st.session_state["rigger_bbox_tl_px"][0], x_orig))
                        y0 = float(min(st.session_state["rigger_bbox_tl_px"][1], y_orig))
                        x1 = float(max(st.session_state["rigger_bbox_tl_px"][0], x_orig))
                        y1 = float(max(st.session_state["rigger_bbox_tl_px"][1], y_orig))
                        w = float(x1 - x0)
                        h = float(y1 - y0)
                        if w > 1 and h > 1:
                            st.session_state["rigger_bbox_px"] = (x0, y0, w, h)

                    if st.session_state.get("annotations_ref_idx") is None:
                        st.session_state["annotations_ref_idx"] = int(ref_idx)
                    st.rerun()

            if st.button("Clear rigger bbox"):
                st.session_state["rigger_bbox_px"] = None
                st.session_state["rigger_bbox_tl_px"] = None
                st.session_state["rigger_bbox_br_px"] = None
                st.session_state["rigger_bbox_click_ver"] += 1
                st.rerun()

        elif step == "Athlete BBox":
            st.markdown(
                "**Athlete bbox via 2 clicks**: click **top-left**, then click **bottom-right**."
            )
            st.caption(
                "Tip: If you want to redraw, just click again after completing both corners (it will restart)."
            )

            click = streamlit_image_coordinates(
                overlay_img, key=f"click_bbox_{st.session_state['bbox_click_ver']}_{ref_idx}"
            )
            if click is not None:
                ts = click.get("unix_time")
                last_ts = st.session_state["last_click_ts"].get("Athlete BBox")
                if ts is not None and ts != last_ts:
                    st.session_state["last_click_ts"]["Athlete BBox"] = ts
                    x_disp = float(click["x"])
                    y_disp = float(click["y"])
                    x_orig, y_orig = disp.to_orig_xy(x_disp, y_disp)

                    if bbox_tl_px is None or (bbox_tl_px is not None and bbox_br_px is not None):
                        st.session_state["bbox_tl_px"] = (x_orig, y_orig)
                        st.session_state["bbox_br_px"] = None
                        st.session_state["bbox_px"] = None
                    else:
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

            if st.button("Clear athlete bbox"):
                st.session_state["bbox_px"] = None
                st.session_state["bbox_tl_px"] = None
                st.session_state["bbox_br_px"] = None
                st.session_state["bbox_click_ver"] += 1
                st.rerun()

        else:
            st.image(overlay_img, caption="Reference frame with annotations", width="content")

    st.subheader("3) Run + results")

    # Validate readiness.
    ready = (
        st.session_state["anchor_px"] is not None
        and st.session_state["rigger_bbox_px"] is not None
        and st.session_state["bbox_px"] is not None
        and st.session_state["scale0_px"] is not None
        and st.session_state["scale1_px"] is not None
        and float(st.session_state["scale_dist_m"]) > 0
        and (not enable_2d or pose2d_ready)
        and (
            not enable_3d
            or (
                pose3d_ready
                and motionbert_ckpt is not None
                and motionbert_ckpt.exists()
                and motionbert_config is not None
                and motionbert_config.exists()
            )
        )
        and (
            st.session_state.get("annotations_ref_idx") is None
            or int(st.session_state.get("annotations_ref_idx")) == int(ref_idx)
        )
        and (not strict_id or deepsort_available)
    )

    if not ready:
        st.warning("Complete annotation and ensure required model files are ready (see sidebar).")
        return

    progress_slot = st.empty()
    run_btn = st.button("Run pipeline + generate 3D overlay", type="primary")

    if run_btn:
        st.session_state["last_run_error"] = None
        progress_slot.empty()
        progress = StreamlitProgress(progress_slot.container())
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
                    rigger_bbox_px=st.session_state["rigger_bbox_px"],
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
                        "ema_alpha": 0.8,
                        "min_points": 10,
                    },
                    "crop": {
                        "padding": 0.2,
                        "ema_alpha": 0.9,
                        "min_points": 10,
                    },
                    "pose_smoothing": {
                        "enabled": bool(smooth_pose),
                        "conf_threshold": float(conf_threshold),
                        "max_gap": int(max_gap),
                        "median_window": int(median_window),
                    },
                    "pose_tracking": {
                        "enabled": bool(pose_tracking_enabled),
                        "smooth_alpha": float(pose_track_smooth_alpha),
                        "strict_id": bool(strict_id),
                        "deepsort_model": str(deepsort_model).strip() or "yolov8n.pt",
                        "deepsort_min_conf": float(deepsort_min_conf),
                        "deepsort_padding": float(deepsort_padding),
                    },
                },
            )

            save_run_config(out_dir / "run.json", cfg)

            # Pipeline run.
            ckpt = None
            if enable_3d:
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
                    mmpose_config=mmpose_config,
                    mmpose_checkpoint=mmpose_checkpoint,
                    motionbert_model=motionbert_model,
                    motionbert_root=motionbert_root,
                    motionbert_ckpt=ckpt,
                    motionbert_config=motionbert_config,
                    clip_len=int(clip_len),
                    flip=bool(flip),
                    rootrel=bool(rootrel),
                    skip_2d=not enable_2d,
                    skip_3d=not enable_3d,
                    progress=progress,
                )

            # Generate 3D overlay video.
            stab_npz = out_dir / "stabilization.npz"
            pose2d_npz = out_dir / "pose2d.npz"
            pose3d_npz = out_dir / "pose3d.npz"
            angles_csv = out_dir / "angles.csv"
            out_pose3d_mp4 = out_dir / "debug" / "pose3d_overlay.mp4"

            if enable_3d and pose3d_npz.exists() and pose2d_npz.exists():
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
                        progress=progress,
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
