"""Functional PCA (fPCA) wrapper compatible with ``sklearn.decomposition.PCA``.

Projects each curve onto a fixed cubic B-spline basis, then runs a
standard PCA on the basis coefficients.  ``transform`` and
``inverse_transform`` round-trip a curve through the basis so that a
downstream model trained against fPCA scores can still reconstruct an
``(N, G)`` curve on the original ``s_grid``.

The object exposes the same attribute/method surface the rest of the
code relies on:

* ``n_components_``      — int, number of retained PCA components
* ``explained_variance_ratio_``
* ``fit(curves)``        — learn basis + PCA
* ``transform(curves)``  — ``(N, G) -> (N, n_components_)``
* ``inverse_transform``  — ``(N, n_components_) -> (N, G)``

Unlike :class:`sklearn.decomposition.PCA`, the basis projection step
operates in curve-space rather than sample-space, so the fPCA scores
smooth out high-frequency noise before fitting the PCA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


def _cubic_bspline_design(
    s_grid: np.ndarray,
    n_internal_knots: int = 10,
    degree: int = 3,
) -> np.ndarray:
    """Build an evaluated B-spline basis matrix ``Phi`` of shape ``(G, n_basis)``.

    Uses open uniform knots on ``[0, 1]``.  We intentionally avoid a
    hard dependency on ``scipy.interpolate.BSpline`` by constructing the
    basis via the Cox-de Boor recursion on each row.
    """
    from scipy.interpolate import BSpline

    s_grid = np.asarray(s_grid, dtype=np.float64)
    # Open uniform knots: repeat boundary knots (degree+1) times.
    if n_internal_knots < 1:
        n_internal_knots = 1
    internal = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    knots = np.concatenate(
        [np.zeros(degree + 1), internal, np.ones(degree + 1)]
    )
    n_basis = len(knots) - degree - 1
    Phi = np.zeros((len(s_grid), n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis)
        c[k] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        Phi[:, k] = np.nan_to_num(spline(s_grid), nan=0.0)
    return Phi


@dataclass
class _FPCAFitState:
    Phi: np.ndarray       # (G, n_basis)
    Phi_pinv: np.ndarray  # (n_basis, G)  -- least-squares pseudo-inverse


class FunctionalPCA:
    """B-spline functional PCA drop-in for :class:`sklearn.decomposition.PCA`."""

    def __init__(
        self,
        n_components: int = 5,
        n_internal_knots: int = 10,
        degree: int = 3,
        whiten: bool = False,
    ):
        self.n_components = int(n_components)
        self.n_internal_knots = int(n_internal_knots)
        self.degree = int(degree)
        self.whiten = bool(whiten)
        self._fit_state: _FPCAFitState | None = None
        self._pca: PCA | None = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def _curves_to_coeffs(self, curves: np.ndarray) -> np.ndarray:
        assert self._fit_state is not None
        # coeffs[i] = (Phi^T Phi)^-1 Phi^T curves[i]
        return curves @ self._fit_state.Phi_pinv.T

    def _coeffs_to_curves(self, coeffs: np.ndarray) -> np.ndarray:
        assert self._fit_state is not None
        return coeffs @ self._fit_state.Phi.T

    def fit(self, curves: np.ndarray, s_grid: np.ndarray | None = None) -> "FunctionalPCA":
        curves = np.asarray(curves, dtype=np.float64)
        if curves.ndim != 2:
            raise ValueError(f"Expected (N, G); got {curves.shape}")
        N, G = curves.shape
        if s_grid is None:
            s_grid = np.linspace(0.0, 1.0, G)
        elif len(s_grid) != G:
            raise ValueError(f"s_grid length {len(s_grid)} != curves columns {G}")

        Phi = _cubic_bspline_design(
            s_grid, n_internal_knots=self.n_internal_knots, degree=self.degree,
        )
        # Least-squares inverse, regularized slightly for numerical safety.
        PhiTPhi = Phi.T @ Phi
        PhiTPhi += 1e-8 * np.eye(PhiTPhi.shape[0])
        Phi_pinv = np.linalg.solve(PhiTPhi, Phi.T)
        self._fit_state = _FPCAFitState(Phi=Phi, Phi_pinv=Phi_pinv)

        coeffs = self._curves_to_coeffs(curves)
        n_components = min(self.n_components, coeffs.shape[0], coeffs.shape[1])
        self._pca = PCA(n_components=n_components, whiten=self.whiten)
        self._pca.fit(coeffs)
        self.n_components_ = int(self._pca.n_components_)
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_.copy()
        return self

    def transform(self, curves: np.ndarray) -> np.ndarray:
        if self._fit_state is None or self._pca is None:
            raise RuntimeError("FunctionalPCA must be fit before transform.")
        coeffs = self._curves_to_coeffs(np.asarray(curves, dtype=np.float64))
        return self._pca.transform(coeffs)

    def fit_transform(self, curves: np.ndarray, s_grid: np.ndarray | None = None) -> np.ndarray:
        self.fit(curves, s_grid=s_grid)
        return self.transform(curves)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        if self._fit_state is None or self._pca is None:
            raise RuntimeError("FunctionalPCA must be fit before inverse_transform.")
        coeffs = self._pca.inverse_transform(np.asarray(scores, dtype=np.float64))
        return self._coeffs_to_curves(coeffs)

    # Pickle-friendly: fields are primitives / numpy / sklearn objects.
