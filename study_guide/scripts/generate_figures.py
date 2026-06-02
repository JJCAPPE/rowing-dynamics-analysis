#!/usr/bin/env python3
"""Generate static figures for the rowing force-curve study guide.

The figures are intentionally schematic and data-independent. They use
synthetic curves with the same shape conventions as the real training
pipeline so the guide can compile even when no full training dataset is
available in the checkout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for 3D projection
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def _curve_family(n: int = 72, g: int = 64, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 1.0, g)
    curves = []
    for _ in range(n):
        peak = rng.normal(520.0, 55.0)
        a = rng.normal(2.6, 0.22)
        b = rng.normal(3.7, 0.28)
        base = (s ** a) * ((1 - s) ** b)
        base = base / base.max()
        shoulder = 0.13 * np.exp(-0.5 * ((s - rng.normal(0.35, 0.035)) / 0.055) ** 2)
        late = 0.04 * np.exp(-0.5 * ((s - rng.normal(0.72, 0.04)) / 0.07) ** 2)
        noise = rng.normal(0.0, 6.0, size=g)
        curves.append(np.maximum(0.0, peak * (base + shoulder + late) + noise))
    return s, np.asarray(curves)


def training_flow() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        ("Matched\nsegments CSV", 0.03, 0.56, "#dbeafe"),
        ("Dataset builder\npivots + grids", 0.23, 0.56, "#e0f2fe"),
        ("Training arrays\nX, Y, mask", 0.43, 0.56, "#dcfce7"),
        ("Stage 0\nsanity baselines", 0.63, 0.72, "#fef3c7"),
        ("Stage A\nPCA regression", 0.63, 0.40, "#fde68a"),
        ("Stage B\nsequence model", 0.63, 0.08, "#fed7aa"),
        ("Evaluation report\nmetrics + gates", 0.83, 0.40, "#ede9fe"),
    ]
    for text, x, y, color in boxes:
        patch = FancyBboxPatch(
            (x, y), 0.15, 0.18,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + 0.075, y + 0.09, text, ha="center", va="center", fontsize=10, weight="semibold")

    arrows = [
        ((0.18, 0.65), (0.23, 0.65)),
        ((0.38, 0.65), (0.43, 0.65)),
        ((0.58, 0.65), (0.63, 0.81)),
        ((0.58, 0.65), (0.63, 0.49)),
        ((0.58, 0.65), (0.63, 0.17)),
        ((0.78, 0.81), (0.83, 0.52)),
        ((0.78, 0.49), (0.83, 0.49)),
        ((0.78, 0.17), (0.83, 0.43)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#475569"))
    ax.text(0.43, 0.23, r"$X_i \in \mathbb{R}^{G\times K}$" "\n" r"$Y_i(s_1),\ldots,Y_i(s_G)$", fontsize=12, color="#166534")
    _save(fig, "training_flow.pdf")


def force_curves() -> None:
    s, curves = _curve_family()
    peak_norm = curves / curves.max(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for c in curves[:16]:
        axes[0].plot(s, c, color="#2563eb", alpha=0.35, lw=1.2)
    axes[0].plot(s, curves.mean(axis=0), color="#0f172a", lw=2.3, label="mean")
    axes[0].set_title("Raw force curves")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_xlabel("drive progress s")
    axes[0].legend(frameon=False)
    for c in peak_norm[:16]:
        axes[1].plot(s, c, color="#059669", alpha=0.35, lw=1.2)
    axes[1].plot(s, peak_norm.mean(axis=0), color="#0f172a", lw=2.3, label="mean")
    axes[1].set_title("Peak-normalized shape")
    axes[1].set_xlabel("drive progress s")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 1)
    _save(fig, "force_curves.png")


def pca_components() -> None:
    s, curves = _curve_family(n=120)
    norm = curves / curves.max(axis=1, keepdims=True)
    pca = PCA(n_components=6).fit(norm)
    recon = pca.inverse_transform(pca.transform(norm[:4]))

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    for j in range(4):
        ax0.plot(s, pca.components_[j], lw=1.9, label=f"PC{j + 1}")
    ax0.axhline(0, color="#64748b", lw=0.8)
    ax0.set_title("Principal component directions")
    ax0.set_xlabel("s")
    ax0.legend(frameon=False, ncol=2)
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(np.arange(1, 7), pca.explained_variance_ratio_, color="#7c3aed")
    ax1.plot(np.arange(1, 7), pca.explained_variance_ratio_.cumsum(), color="#0f172a", marker="o")
    ax1.set_ylim(0, 1.03)
    ax1.set_title("Explained variance")
    ax1.set_xlabel("component")
    ax1.set_ylabel("ratio / cumulative")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, :])
    for i in range(4):
        ax2.plot(s, norm[i], color="#2563eb", alpha=0.55, lw=1.5)
        ax2.plot(s, recon[i], color="#dc2626", alpha=0.8, lw=1.2, ls="--")
    ax2.set_title("PCA reconstruction: true normalized curves (blue) vs reconstructed (red dashed)")
    ax2.set_xlabel("drive progress s")
    ax2.set_ylabel("normalized force")
    ax2.grid(True, alpha=0.25)
    _save(fig, "pca_components.png")


def feature_heatmap() -> None:
    rng = np.random.default_rng(3)
    s = np.linspace(0, 1, 64)
    n = 42
    knee = 58 + 88 * s[None, :] + 7 * np.sin(2 * np.pi * s)[None, :]
    knee = knee + rng.normal(0, 5.0, size=(n, 64)) + np.linspace(-10, 10, n)[:, None] * 0.25

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(knee, aspect="auto", cmap="viridis", origin="lower", extent=[0, 1, 0, n])
    ax.set_title("Kinematic sequence heatmap")
    ax.set_xlabel("drive progress s")
    ax.set_ylabel("stroke index")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("knee_active_deg")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    S, I = np.meshgrid(s, np.arange(n))
    ax3.plot_surface(S, I, knee, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_title("Same tensor as a surface")
    ax3.set_xlabel("s")
    ax3.set_ylabel("stroke")
    ax3.set_zlabel("deg")
    ax3.view_init(elev=28, azim=-135)
    _save(fig, "feature_heatmap.png")


def cv_splits() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True)
    n = 30
    methods = [
        ("time_block", [range(0, 6), range(6, 12), range(12, 18), range(18, 24), range(24, 30)]),
        ("session_held_out", [range(0, 10), range(10, 20), range(20, 30)]),
        ("athlete_held_out", [range(0, 12), range(12, 21), range(21, 30)]),
    ]
    for ax, (name, blocks) in zip(axes, methods):
        ax.add_patch(Rectangle((0, 0.2), n, 0.6, color="#dbeafe"))
        for k, block in enumerate(blocks):
            lo, hi = min(block), max(block) + 1
            ax.add_patch(Rectangle((lo, 0.2), hi - lo, 0.6, color=["#ef4444", "#f97316", "#84cc16", "#14b8a6", "#8b5cf6"][k % 5], alpha=0.8))
            ax.text((lo + hi) / 2, 0.5, f"fold {k}", ha="center", va="center", fontsize=9, color="white", weight="bold")
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=82)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.spines[["left", "right", "top"]].set_visible(False)
    axes[-1].set_xlabel("ordered stroke samples")
    fig.suptitle("Cross-validation: each colored block can become the held-out test fold", y=0.98)
    _save(fig, "cv_splits.pdf")


def metrics_diagram() -> None:
    s, curves = _curve_family(n=2, seed=14)
    y_true = curves[0]
    y_pred = 0.92 * np.roll(y_true, 3) + 32 * np.exp(-0.5 * ((s - 0.28) / 0.05) ** 2)
    phases = [(0, 1 / 3, "#dbeafe"), (1 / 3, 2 / 3, "#dcfce7"), (2 / 3, 1, "#fef3c7")]

    fig, ax = plt.subplots(figsize=(11, 4.7))
    for lo, hi, color in phases:
        ax.axvspan(lo, hi, color=color, alpha=0.65)
    ax.plot(s, y_true, color="#0f172a", lw=2.5, label="true")
    ax.plot(s, y_pred, color="#dc2626", lw=2, ls="--", label="predicted")
    ax.fill_between(s, y_true, y_pred, color="#ef4444", alpha=0.14, label="pointwise error")
    i_t = int(np.argmax(y_true))
    i_p = int(np.argmax(y_pred))
    ax.scatter([s[i_t]], [y_true[i_t]], color="#0f172a", zorder=5)
    ax.scatter([s[i_p]], [y_pred[i_p]], color="#dc2626", zorder=5)
    ax.annotate("peak force error", xy=(s[i_p], y_pred[i_p]), xytext=(0.58, y_pred.max() + 70),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=10)
    ax.annotate("peak position error", xy=((s[i_t] + s[i_p]) / 2, max(y_true[i_t], y_pred[i_p]) + 20),
                xytext=(0.13, y_pred.max() + 70), arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=10)
    ax.text(0.05, 55, "early", fontsize=10, weight="bold")
    ax.text(0.43, 55, "mid", fontsize=10, weight="bold")
    ax.text(0.78, 55, "late", fontsize=10, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("drive progress s")
    ax.set_ylabel("force (N)")
    ax.set_title("Evaluation metrics compare both curve shape and rowing-relevant summaries")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    _save(fig, "metrics_diagram.png")


def stageb_arch() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.text(0.05, 0.88, "Stage B input tensor", fontsize=12, weight="bold")
    ax.text(0.05, 0.78, r"$X \in \mathbb{R}^{N \times G \times K}$", fontsize=14)
    layers = [
        ("normalize\nfeatures", 0.05, 0.42, "#dbeafe"),
        ("TCN blocks\nor Transformer", 0.28, 0.42, "#ede9fe"),
        ("linear head\nper grid point", 0.51, 0.42, "#dcfce7"),
        ("predicted curve\n$\\hat{F}(s)$", 0.74, 0.42, "#fed7aa"),
    ]
    for text, x, y, color in layers:
        ax.add_patch(FancyBboxPatch((x, y), 0.15, 0.2, boxstyle="round,pad=0.018,rounding_size=0.018",
                                    facecolor=color, edgecolor="#334155", linewidth=1.2))
        ax.text(x + 0.075, y + 0.1, text, ha="center", va="center", fontsize=10, weight="semibold")
    for x in [0.20, 0.43, 0.66]:
        ax.add_patch(FancyArrowPatch((x, 0.52), (x + 0.08, 0.52), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#475569"))
    ax.text(0.29, 0.18, "masked MSE ignores invalid bins; optional derivative loss penalizes wrong slope", fontsize=11)
    ax.text(0.29, 0.09, r"$L = \frac{\sum m_{ig}(\hat{y}_{ig}-y_{ig})^2}{\sum m_{ig}} + \lambda\sum(\Delta\hat{y}_{ig}-\Delta y_{ig})^2$", fontsize=14)
    _save(fig, "stageb_architecture.pdf")


def main() -> int:
    training_flow()
    force_curves()
    feature_heatmap()
    pca_components()
    cv_splits()
    metrics_diagram()
    stageb_arch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
