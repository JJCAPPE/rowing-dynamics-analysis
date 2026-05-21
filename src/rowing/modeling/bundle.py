"""Self-describing model bundle for video-only force-curve inference.

A *model bundle* packages everything required to run inference on a new
video run: the fixed ``s`` grid, the (functional) PCA used to decompose
force curves, the Stage A sklearn model + scaler + feature list, the
Stage B torch state + architecture config + feature/target normalization,
and a top-level ``manifest.json`` describing the feature contract.

Layout
------

::

    model_bundle/
      manifest.json
      feature_names.json
      s_grid.npy
      pca_model.joblib                # or fpca_model.joblib
      stageA/
        model.joblib                  # {"model": estimator, "scaler": StandardScaler}
        feature_cols.json
      stageB/
        state.pt
        arch_config.json
        feature_norm.npz              # feat_mean, feat_std
        target_norm.npz               # force_mean, force_std

Every field is optional per stage: if a stage was not trained the sub
directory is simply absent, and ``manifest.json`` declares which stages
are present.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


BUNDLE_MANIFEST_VERSION = 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_model_bundle(
    *,
    bundle_dir: Path,
    dataset_dir: Path,
    modeling_dir: Path,
    active_side_default: str | None = None,
    include_head: bool = True,
    target_representation: str = "standard",
    git_sha: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Assemble a model_bundle directory from training-dataset + modeling outputs.

    Parameters
    ----------
    bundle_dir
        Destination directory.  Created if missing.
    dataset_dir
        Directory produced by ``build_training_dataset.py`` (contains
        ``pca_model.joblib``, ``s_grid.npy``, ``feature_names.json``,
        ``dataset_summary.json``).
    modeling_dir
        Directory produced by ``modeling.py`` (contains
        ``stage0_metadata_model.joblib``, ``stageA_best_model.joblib``,
        ``stageB_tcn_state.pt``, plus the corresponding result JSONs).
    active_side_default
        Default ``active_side`` to assume at inference time when the run
        does not declare one.  Use ``None`` to require explicit
        specification.
    include_head
        Whether the training contract included ``head_vs_trunk_deg``.
    target_representation
        One of ``standard`` or ``fpca``; selects which decomposition
        artifact is shipped.
    git_sha
        Optional SHA recorded in the manifest for reproducibility.
    extra_metadata
        Additional key/value pairs merged into ``manifest.json``.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    modeling_dir = Path(modeling_dir).expanduser().resolve()

    bundle_dir.mkdir(parents=True, exist_ok=True)

    present_stages: dict[str, bool] = {"stageA": False, "stageB": False}

    # ------------------------------------------------------------------
    # Dataset artifacts (feature contract + target decomposition)
    # ------------------------------------------------------------------
    src_s_grid = dataset_dir / "s_grid.npy"
    if not src_s_grid.exists():
        raise FileNotFoundError(f"Missing s_grid.npy in dataset dir: {src_s_grid}")
    s_grid = np.load(src_s_grid)
    np.save(bundle_dir / "s_grid.npy", s_grid)

    src_feature_names = dataset_dir / "feature_names.json"
    if src_feature_names.exists():
        shutil.copy2(src_feature_names, bundle_dir / "feature_names.json")
    else:
        raise FileNotFoundError(f"Missing feature_names.json: {src_feature_names}")

    if target_representation == "fpca":
        src_pca = dataset_dir / "fpca_model.joblib"
        if not src_pca.exists():
            raise FileNotFoundError(
                f"target_representation='fpca' but fpca_model.joblib not found: {src_pca}"
            )
        shutil.copy2(src_pca, bundle_dir / "fpca_model.joblib")
    else:
        src_pca = dataset_dir / "pca_model.joblib"
        if not src_pca.exists():
            raise FileNotFoundError(f"Missing pca_model.joblib: {src_pca}")
        shutil.copy2(src_pca, bundle_dir / "pca_model.joblib")

    dataset_summary: dict[str, Any] = {}
    src_summary = dataset_dir / "dataset_summary.json"
    if src_summary.exists():
        with src_summary.open() as f:
            dataset_summary = json.load(f)

    # ------------------------------------------------------------------
    # Stage A (optional)
    # ------------------------------------------------------------------
    stageA_src = modeling_dir / "stageA_best_model.joblib"
    if stageA_src.exists():
        stageA_dir = bundle_dir / "stageA"
        stageA_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stageA_src, stageA_dir / "model.joblib")

        feature_cols: list[str] = []
        stageA_results_json = modeling_dir / "stageA_results.json"
        if stageA_results_json.exists():
            with stageA_results_json.open() as f:
                sa = json.load(f)
            feature_cols = list(sa.get("feature_cols", []))
        _write_json(stageA_dir / "feature_cols.json", feature_cols)
        present_stages["stageA"] = True

    # ------------------------------------------------------------------
    # Stage B (optional)
    # ------------------------------------------------------------------
    stageB_src = modeling_dir / "stageB_tcn_state.pt"
    stageB_norm_src = modeling_dir / "stageB_tcn_norm.npz"
    stageB_arch_src = modeling_dir / "stageB_tcn_arch.json"
    if stageB_src.exists() and stageB_norm_src.exists() and stageB_arch_src.exists():
        stageB_dir = bundle_dir / "stageB"
        stageB_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stageB_src, stageB_dir / "state.pt")
        shutil.copy2(stageB_arch_src, stageB_dir / "arch_config.json")

        with np.load(stageB_norm_src) as data:
            feat_mean = data["feat_mean"]
            feat_std = data["feat_std"]
            force_mean = float(data["force_mean"])
            force_std = float(data["force_std"])
        np.savez(
            stageB_dir / "feature_norm.npz",
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        np.savez(
            stageB_dir / "target_norm.npz",
            force_mean=np.asarray(force_mean, dtype=np.float64),
            force_std=np.asarray(force_std, dtype=np.float64),
        )
        present_stages["stageB"] = True

    # ------------------------------------------------------------------
    # manifest.json
    # ------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "bundle_manifest_version": BUNDLE_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "target_representation": target_representation,
        "n_grid": int(s_grid.shape[0]),
        "include_head": bool(include_head),
        "active_side_default": active_side_default,
        "stages_present": present_stages,
        "dataset_summary": dataset_summary,
    }
    if extra_metadata:
        manifest.setdefault("extra", {}).update(extra_metadata)
    _write_json(bundle_dir / "manifest.json", manifest)

    return bundle_dir


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


class ModelBundle:
    """In-memory view of a serialized model bundle directory."""

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Not a model bundle (missing manifest.json): {self.bundle_dir}")
        with manifest_path.open() as f:
            self.manifest: dict[str, Any] = json.load(f)

        self.s_grid: np.ndarray = np.load(self.bundle_dir / "s_grid.npy")
        with (self.bundle_dir / "feature_names.json").open() as f:
            self.feature_names: list[str] = json.load(f)

        target_representation = self.manifest.get("target_representation", "standard")
        self.target_representation = target_representation
        if target_representation == "fpca":
            self.pca_model = joblib.load(self.bundle_dir / "fpca_model.joblib")
        else:
            self.pca_model = joblib.load(self.bundle_dir / "pca_model.joblib")

        self.stageA: dict[str, Any] | None = None
        stageA_dir = self.bundle_dir / "stageA"
        if stageA_dir.exists():
            payload = joblib.load(stageA_dir / "model.joblib")
            with (stageA_dir / "feature_cols.json").open() as f:
                feature_cols = json.load(f)
            self.stageA = {
                "model": payload["model"],
                "scaler": payload["scaler"],
                "feature_cols": feature_cols,
            }

        self.stageB: dict[str, Any] | None = None
        stageB_dir = self.bundle_dir / "stageB"
        if stageB_dir.exists():
            with (stageB_dir / "arch_config.json").open() as f:
                arch_config = json.load(f)
            with np.load(stageB_dir / "feature_norm.npz") as data:
                feat_mean = data["feat_mean"].astype(np.float64)
                feat_std = data["feat_std"].astype(np.float64)
            with np.load(stageB_dir / "target_norm.npz") as data:
                force_mean = float(data["force_mean"])
                force_std = float(data["force_std"])
            self.stageB = {
                "state_path": stageB_dir / "state.pt",
                "arch_config": arch_config,
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "force_mean": force_mean,
                "force_std": force_std,
            }

    @property
    def active_side_default(self) -> str | None:
        return self.manifest.get("active_side_default")

    @property
    def include_head(self) -> bool:
        return bool(self.manifest.get("include_head", True))

    @property
    def n_grid(self) -> int:
        return int(self.s_grid.shape[0])

    def has_stage(self, stage: str) -> bool:
        stage = stage.upper()
        if stage == "A":
            return self.stageA is not None
        if stage == "B":
            return self.stageB is not None
        return False
