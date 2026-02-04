from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import hashlib

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MOTIONBERT_REPO = REPO_ROOT / "third_party" / "MotionBERT"


@dataclass(frozen=True)
class AssetSpec:
    url: str
    path: Path
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


@dataclass(frozen=True)
class Pose2DModelSpec:
    key: str
    label: str
    model_id: str
    config: Optional[AssetSpec] = None
    checkpoint: Optional[AssetSpec] = None


@dataclass(frozen=True)
class MotionBertModelSpec:
    key: str
    label: str
    config_path: Path
    checkpoint: AssetSpec


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    pose2d: Pose2DModelSpec
    motionbert: MotionBertModelSpec


POSE2D_VITPOSE_HIGH = Pose2DModelSpec(
    key="vitpose_high",
    label="ViTPose-H (COCO 256x192)",
    model_id="td-hm_ViTPose-huge_8xb64-210e_coco-256x192",
    config=AssetSpec(
        url=(
            "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
            "configs/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
        ),
        path=MODELS_DIR / "mmpose" / "vitpose_huge_coco_256x192.py",
    ),
    checkpoint=AssetSpec(
        url=(
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
        ),
        path=MODELS_DIR / "mmpose" / "vitpose_huge_coco_256x192.pth",
    ),
)

POSE2D_DEFAULT = Pose2DModelSpec(
    key="mmpose_default",
    label="MMPose default (auto weights)",
    model_id="human",
)

MOTIONBERT_HIGH = MotionBertModelSpec(
    key="motionbert_high",
    label="MotionBERT full",
    config_path=MOTIONBERT_REPO / "configs" / "pose3d" / "MB_ft_h36m_global.yaml",
    checkpoint=AssetSpec(
        url=(
            "https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
            "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
        ),
        path=MODELS_DIR / "motionbert" / "FT_MB_release_MB_ft_h36m" / "best_epoch.bin",
    ),
)

MOTIONBERT_LITE = MotionBertModelSpec(
    key="motionbert_lite",
    label="MotionBERT lite",
    config_path=MOTIONBERT_REPO / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml",
    checkpoint=AssetSpec(
        url=(
            "https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
            "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
        ),
        path=MODELS_DIR
        / "motionbert"
        / "FT_MB_lite_MB_ft_h36m_global_lite"
        / "best_epoch.bin",
    ),
)

POSE2D_MODELS: Dict[str, Pose2DModelSpec] = {
    POSE2D_VITPOSE_HIGH.key: POSE2D_VITPOSE_HIGH,
    POSE2D_DEFAULT.key: POSE2D_DEFAULT,
}

MOTIONBERT_MODELS: Dict[str, MotionBertModelSpec] = {
    MOTIONBERT_HIGH.key: MOTIONBERT_HIGH,
    MOTIONBERT_LITE.key: MOTIONBERT_LITE,
}

MODEL_PRESETS: Dict[str, ModelPreset] = {
    "high": ModelPreset(
        key="high",
        label="High accuracy (ViTPose-H + MotionBERT full)",
        pose2d=POSE2D_VITPOSE_HIGH,
        motionbert=MOTIONBERT_HIGH,
    ),
    "medium": ModelPreset(
        key="medium",
        label="Medium accuracy (ViTPose-H + MotionBERT lite)",
        pose2d=POSE2D_VITPOSE_HIGH,
        motionbert=MOTIONBERT_LITE,
    ),
    "balanced": ModelPreset(
        key="balanced",
        label="Balanced (MMPose default + MotionBERT lite)",
        pose2d=POSE2D_DEFAULT,
        motionbert=MOTIONBERT_LITE,
    ),
}

DEFAULT_POSE2D_MODEL = POSE2D_VITPOSE_HIGH.key
DEFAULT_MOTIONBERT_MODEL = MOTIONBERT_HIGH.key


def get_pose2d_model(model_id: str) -> Optional[Pose2DModelSpec]:
    return POSE2D_MODELS.get(model_id)


def get_motionbert_model(model_id: str) -> Optional[MotionBertModelSpec]:
    return MOTIONBERT_MODELS.get(model_id)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_size(url: str) -> Optional[int]:
    import urllib.request

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception:
        return None


def ensure_asset(
    path: Path,
    url: str,
    *,
    expected_size: Optional[int] = None,
    sha256: Optional[str] = None,
) -> Path:
    path = Path(path)
    if expected_size is None:
        expected_size = _remote_size(url)

    if path.exists():
        size = path.stat().st_size
        if expected_size is not None and size != expected_size:
            path.unlink()
        elif sha256 is not None and _sha256_file(path) != sha256:
            path.unlink()
        else:
            return path

    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    total_bytes = None
    with urllib.request.urlopen(url) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    tmp.replace(path)

    final_size = path.stat().st_size
    expected = expected_size if expected_size is not None else total_bytes
    if expected is not None and final_size != expected:
        raise ValueError(
            f"Downloaded {path} has size {final_size} bytes, expected {expected} bytes."
        )
    if sha256 is not None:
        digest = _sha256_file(path)
        if digest != sha256:
            raise ValueError(f"Downloaded {path} has sha256 {digest}, expected {sha256}.")
    return path


def ensure_pose2d_assets(spec: Pose2DModelSpec) -> Tuple[Optional[Path], Optional[Path]]:
    config_path = None
    ckpt_path = None
    if spec.config is not None:
        config_path = ensure_asset(
            spec.config.path,
            spec.config.url,
            expected_size=spec.config.size_bytes,
            sha256=spec.config.sha256,
        )
    if spec.checkpoint is not None:
        ckpt_path = ensure_asset(
            spec.checkpoint.path,
            spec.checkpoint.url,
            expected_size=spec.checkpoint.size_bytes,
            sha256=spec.checkpoint.sha256,
        )
    return config_path, ckpt_path


def ensure_motionbert_assets(spec: MotionBertModelSpec) -> Tuple[Path, Path]:
    if not spec.config_path.exists():
        raise FileNotFoundError(
            f"MotionBERT config not found at {spec.config_path}. "
            "Ensure the MotionBERT submodule is available."
        )
    ckpt_path = ensure_asset(
        spec.checkpoint.path,
        spec.checkpoint.url,
        expected_size=spec.checkpoint.size_bytes,
        sha256=spec.checkpoint.sha256,
    )
    return spec.config_path, ckpt_path
