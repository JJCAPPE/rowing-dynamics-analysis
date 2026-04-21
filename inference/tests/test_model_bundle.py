"""Round-trip smoke test for :mod:`inference.model_bundle`."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA

from model_bundle import ModelBundle, write_model_bundle


def _make_dataset_dir(path: Path, *, with_fpca: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)

    n_grid = 32
    s_grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    np.save(path / "s_grid.npy", s_grid)

    feature_names = [
        "knee_active_deg",
        "hip_active_deg",
        "elbow_active_deg",
        "trunk_vs_horizontal_deg",
        "spine_flexion_deg",
        "head_vs_trunk_deg",
    ]
    (path / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    rng = np.random.default_rng(0)
    curves = rng.normal(size=(20, n_grid))
    pca = PCA(n_components=4).fit(curves)
    joblib.dump(pca, path / "pca_model.joblib")

    if with_fpca:
        # Reuse the same sklearn PCA so we don't import FunctionalPCA in the fixture.
        joblib.dump(pca, path / "fpca_model.joblib")

    summary = {
        "n_strokes": int(curves.shape[0]),
        "n_grid": n_grid,
        "n_pca_components": 4,
        "target_representation": "fpca" if with_fpca else "standard",
    }
    (path / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


def test_bundle_round_trip_dataset_only(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training"
    modeling_dir = tmp_path / "modeling"
    modeling_dir.mkdir()
    _make_dataset_dir(dataset_dir)

    bundle_dir = write_model_bundle(
        bundle_dir=tmp_path / "bundle",
        dataset_dir=dataset_dir,
        modeling_dir=modeling_dir,
        active_side_default="right",
        include_head=True,
        target_representation="standard",
        git_sha="deadbeef",
        extra_metadata={"note": "smoke-test"},
    )

    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "pca_model.joblib").exists()
    assert (bundle_dir / "s_grid.npy").exists()
    assert (bundle_dir / "feature_names.json").exists()
    assert not (bundle_dir / "stageA").exists()
    assert not (bundle_dir / "stageB").exists()

    bundle = ModelBundle(bundle_dir)
    assert bundle.n_grid == 32
    assert bundle.target_representation == "standard"
    assert bundle.active_side_default == "right"
    assert bundle.include_head is True
    assert bundle.feature_names[0] == "knee_active_deg"
    assert bundle.manifest["git_sha"] == "deadbeef"
    assert bundle.manifest["extra"]["note"] == "smoke-test"
    assert bundle.has_stage("A") is False
    assert bundle.has_stage("B") is False
    assert bundle.pca_model is not None


def test_bundle_round_trip_fpca(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training"
    modeling_dir = tmp_path / "modeling"
    modeling_dir.mkdir()
    _make_dataset_dir(dataset_dir, with_fpca=True)

    bundle_dir = write_model_bundle(
        bundle_dir=tmp_path / "bundle",
        dataset_dir=dataset_dir,
        modeling_dir=modeling_dir,
        active_side_default=None,
        include_head=False,
        target_representation="fpca",
    )

    assert (bundle_dir / "fpca_model.joblib").exists()
    bundle = ModelBundle(bundle_dir)
    assert bundle.target_representation == "fpca"
    assert bundle.include_head is False
    assert bundle.active_side_default is None


def test_bundle_missing_pca_raises(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training"
    dataset_dir.mkdir()
    # s_grid and feature_names present but pca missing.
    np.save(dataset_dir / "s_grid.npy", np.linspace(0, 1, 8))
    (dataset_dir / "feature_names.json").write_text(json.dumps(["knee_active_deg"]))

    modeling_dir = tmp_path / "modeling"
    modeling_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        write_model_bundle(
            bundle_dir=tmp_path / "bundle",
            dataset_dir=dataset_dir,
            modeling_dir=modeling_dir,
        )


def test_model_bundle_loads_missing_manifest_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "nobundle"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ModelBundle(empty_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
