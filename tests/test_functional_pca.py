"""Sanity tests for :mod:`inference.functional_pca`.

The functional PCA projects force curves onto a B-spline basis and then runs
standard PCA on the basis coefficients. For smooth synthetic curves the
round-trip should be accurate to within a few percent of the signal RMS.
"""
from __future__ import annotations

import numpy as np
import pytest

from rowing.dataset.functional_pca import FunctionalPCA


def _synthetic_curves(n: int = 48, n_grid: int = 64, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a family of smooth, skew-Gaussian-like force curves on [0, 1]."""
    rng = np.random.default_rng(rng_seed)
    s = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    curves = np.empty((n, n_grid), dtype=np.float64)
    for i in range(n):
        mu = rng.uniform(0.30, 0.55)
        sigma = rng.uniform(0.12, 0.22)
        skew = rng.uniform(-2.0, 2.0)
        base = np.exp(-0.5 * ((s - mu) / sigma) ** 2)
        cdf = 0.5 * (1.0 + np.tanh(skew * (s - mu) / sigma))
        curve = base * (0.5 + cdf)
        curve[curve < 0] = 0.0
        curve = curve / (curve.max() + 1e-12)
        curves[i] = curve
    return curves, s


def test_fpca_roundtrip_smooth_curves() -> None:
    curves, s = _synthetic_curves(n=64, n_grid=64)
    fpca = FunctionalPCA(n_components=6, n_internal_knots=12)
    scores = fpca.fit_transform(curves, s_grid=s)

    assert scores.shape == (curves.shape[0], fpca.n_components_)
    assert fpca.n_components_ <= 6

    recon = fpca.inverse_transform(scores)
    assert recon.shape == curves.shape

    rms_err = float(np.sqrt(np.mean((recon - curves) ** 2)))
    rms_sig = float(np.sqrt(np.mean(curves ** 2)))
    # For 6 components on this smooth family we expect <5% relative RMS.
    assert rms_err / rms_sig < 0.05


def test_fpca_components_explain_most_variance() -> None:
    curves, s = _synthetic_curves(n=96, n_grid=64)
    fpca = FunctionalPCA(n_components=8, n_internal_knots=14).fit(curves, s_grid=s)
    cumulative = float(fpca.explained_variance_ratio_.cumsum()[-1])
    assert cumulative > 0.95


def test_fpca_transform_is_linear() -> None:
    curves, s = _synthetic_curves(n=32, n_grid=64)
    fpca = FunctionalPCA(n_components=5, n_internal_knots=10).fit(curves, s_grid=s)

    scores_full = fpca.transform(curves)
    scores_pair = np.vstack([fpca.transform(curves[:1]), fpca.transform(curves[1:2])])
    np.testing.assert_allclose(scores_pair, scores_full[:2], rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
