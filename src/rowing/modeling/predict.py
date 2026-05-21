#!/usr/bin/env python3
"""Video-only force-curve inference CLI.

Loads a self-describing model bundle (see :mod:`model_bundle`) and runs
drive-event detection + pose-feature extraction on a Sports2D run,
predicting per-stroke force curves without requiring any RP3 data.

Example
-------

    .venv/bin/python inference/predict_force_cli.py \\
      --run-dir runs/<run> \\
      --model-bundle runs/<run_train>/inference/model_bundle \\
      --stage B \\
      --stroke-length-cm 150

The ``--stroke-length-cm`` flag (or ``--px-per-cm``) is used when
denormalizing the predicted curves from the bundle's normalized ``s``
grid to RP3-style ``force_at_*cm`` columns.  When neither is provided
only the normalized CSV is emitted.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from rowing import REPO_ROOT
from rowing.matching.detect import (
    detect_drive_events,
    FINISH_METHOD_VELOCITY_THRESHOLD,
)
from rowing.rp3.clean import build_force_columns
from rowing.modeling.bundle import ModelBundle
from rowing.dataset.segment_features import (
    build_pose_drive_segments,
    feature_column_names,
    stack_feature_tensor,
)


RP3_CLEAN_MAX_STROKE_LENGTH_CM = 170.0
RP3_CLEAN_STEP_CM = 2.2
FORCE_COL_RE = re.compile(r"^force_at_([0-9]+(?:\.[0-9]+)?)cm$")


# ---------------------------------------------------------------------------
# Event dataframe adapter
# ---------------------------------------------------------------------------


def _events_to_dataframe(events) -> pd.DataFrame:
    rows = [asdict(ev) for ev in events]
    return pd.DataFrame(rows)


def _build_frame_df(df: pd.DataFrame, detection) -> pd.DataFrame:
    """Return the frame-level dataframe consumed by segment_features.

    We rely on the raw ``stroke_signal.csv`` columns (``time_s``,
    ``relative_axis_px``/_smooth, ``velocity_axis_px_s``) for inference;
    the training-side recomputed columns are optional and only used when
    present.
    """
    return df.copy()


# ---------------------------------------------------------------------------
# Stage A and Stage B prediction
# ---------------------------------------------------------------------------


def _predict_stage_a(
    bundle: ModelBundle,
    strokes_df: pd.DataFrame,
) -> np.ndarray:
    """Predict peak-normalized curve shape for each stroke via Stage A.

    Returns an ``(N, G)`` array on the bundle's ``s_grid`` in peak-
    normalized units (max == 1 on the training distribution).  Absolute-
    force reconstruction is not performed here because training does not
    ship a peak-force regressor; callers must treat the output as shape.
    """
    if bundle.stageA is None:
        raise RuntimeError("Bundle does not contain a Stage A model.")
    model = bundle.stageA["model"]
    scaler = bundle.stageA["scaler"]
    feature_cols = bundle.stageA["feature_cols"]

    missing = [c for c in feature_cols if c not in strokes_df.columns]
    if missing:
        raise RuntimeError(f"Missing Stage A feature columns in strokes_df: {missing}")

    X = strokes_df[feature_cols].to_numpy(dtype=np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X_finite = X[finite_mask]
    X_s = scaler.transform(X_finite)
    Y_pred_pca = model.predict(X_s)

    pca = bundle.pca_model
    n_components = int(pca.n_components_)
    if Y_pred_pca.ndim == 1:
        Y_pred_pca = Y_pred_pca.reshape(-1, 1)
    if Y_pred_pca.shape[1] < n_components:
        padded = np.zeros((Y_pred_pca.shape[0], n_components), dtype=np.float64)
        padded[:, : Y_pred_pca.shape[1]] = Y_pred_pca
        Y_pred_pca = padded
    elif Y_pred_pca.shape[1] > n_components:
        Y_pred_pca = Y_pred_pca[:, :n_components]
    curves_finite = pca.inverse_transform(Y_pred_pca)
    curves = np.full((X.shape[0], curves_finite.shape[1]), np.nan, dtype=np.float64)
    curves[finite_mask] = curves_finite
    return curves


def _build_stage_b_model(arch_config: dict[str, Any]):
    """Construct the Stage B torch model described by ``arch_config``."""
    from modeling import _ForceCurveTCN  # lazy import to avoid torch at load-time

    arch = arch_config.get("arch", "tcn")
    if arch == "tcn":
        tcn = _ForceCurveTCN(
            in_channels=int(arch_config["in_channels"]),
            hidden_channels=int(arch_config.get("hidden_channels", 64)),
            n_blocks=int(arch_config.get("n_blocks", 4)),
            kernel_size=int(arch_config.get("kernel_size", 3)),
            dropout=float(arch_config.get("dropout", 0.1)),
        )
        return tcn.build()
    if arch == "transformer":
        from modeling import _ForceCurveTransformer  # lazy import
        tr = _ForceCurveTransformer(
            in_channels=int(arch_config["in_channels"]),
            hidden_channels=int(arch_config.get("hidden_channels", 128)),
            n_layers=int(arch_config.get("n_layers", 4)),
            n_heads=int(arch_config.get("n_heads", 4)),
            dropout=float(arch_config.get("dropout", 0.1)),
        )
        return tr.build()
    raise ValueError(f"Unknown Stage B arch: {arch}")


def _predict_stage_b(
    bundle: ModelBundle,
    feature_tensor: np.ndarray,
) -> np.ndarray:
    if bundle.stageB is None:
        raise RuntimeError("Bundle does not contain a Stage B model.")
    import torch  # lazy import

    stageB = bundle.stageB
    feat_mean = stageB["feat_mean"]
    feat_std = stageB["feat_std"]
    force_mean = stageB["force_mean"]
    force_std = stageB["force_std"]

    X = feature_tensor.astype(np.float64, copy=True)
    finite_mask = np.all(np.isfinite(X.reshape(X.shape[0], -1)), axis=1)
    X_norm = (X - feat_mean) / feat_std
    np.nan_to_num(X_norm, copy=False, nan=0.0)

    model = _build_stage_b_model(stageB["arch_config"])
    state = torch.load(stageB["state_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_n = model(X_t).numpy()
    curves = y_n * force_std + force_mean
    curves[~finite_mask] = np.nan
    return curves


# ---------------------------------------------------------------------------
# Scalar summary builder (for Stage A inputs)
# ---------------------------------------------------------------------------


def _build_scalar_summary(
    feature_tensor: np.ndarray,
    feature_cols: list[str],
    s_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute angle min/max/range/mean/s_at_max mirroring ``_compute_scalar_summary``.

    Only angle columns (ending in ``_deg`` but not ``_ddeg_ds``) contribute.
    """
    angle_indices = [
        (i, col) for i, col in enumerate(feature_cols)
        if col.endswith("_deg") and not col.endswith("_ddeg_ds") and not col.endswith("_d2deg_ds2")
    ]
    N = feature_tensor.shape[0]
    records: list[dict[str, Any]] = []
    for i in range(N):
        row: dict[str, Any] = {}
        for k, col in angle_indices:
            seq = feature_tensor[i, :, k]
            finite = np.isfinite(seq)
            if finite.sum() == 0:
                for suffix in ("_min", "_max", "_range", "_mean", "_s_at_max"):
                    row[f"{col}{suffix}"] = float("nan")
                continue
            vals = seq[finite]
            s_vals = np.asarray(s_grid, dtype=np.float64)[finite]
            argmax = int(np.argmax(vals))
            row[f"{col}_min"] = float(np.min(vals))
            row[f"{col}_max"] = float(np.max(vals))
            row[f"{col}_range"] = float(np.max(vals) - np.min(vals))
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_s_at_max"] = float(s_vals[argmax])
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# cm-indexed RP3-style output
# ---------------------------------------------------------------------------


def _curves_to_cm_table(
    curves_normalized_s: np.ndarray,
    s_grid: np.ndarray,
    stroke_lengths_cm: np.ndarray,
    step_cm: float = RP3_CLEAN_STEP_CM,
    max_stroke_length_cm: float = RP3_CLEAN_MAX_STROKE_LENGTH_CM,
) -> pd.DataFrame:
    """Resample each curve (defined on ``s_grid`` with stroke length cm
    ``stroke_lengths_cm[i]``) onto the RP3 2.2 cm grid.

    Bins beyond the actual stroke length are left as NaN.
    """
    columns = build_force_columns(max_stroke_length_cm, step_cm)
    distances_cm = np.array([float(c.split("_")[2][:-2]) for c in columns], dtype=np.float64)
    N, G = curves_normalized_s.shape
    out = np.full((N, len(columns)), np.nan, dtype=np.float64)
    for i in range(N):
        stroke_len = float(stroke_lengths_cm[i])
        if not np.isfinite(stroke_len) or stroke_len <= 0:
            continue
        curve = curves_normalized_s[i]
        if not np.isfinite(curve).any():
            continue
        finite = np.isfinite(curve)
        s_finite = np.asarray(s_grid, dtype=np.float64)[finite]
        y_finite = curve[finite]
        if s_finite.size < 2:
            continue
        for j, d_cm in enumerate(distances_cm):
            if d_cm > stroke_len + 1e-6:
                continue
            s_target = float(d_cm) / stroke_len
            s_target = min(max(s_target, 0.0), 1.0)
            out[i, j] = float(np.interp(s_target, s_finite, y_finite))
    df = pd.DataFrame(out, columns=columns)
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _derive_metrics(
    curves: np.ndarray,
    s_grid: np.ndarray,
    stroke_lengths_cm: np.ndarray,
) -> pd.DataFrame:
    N, G = curves.shape
    records: list[dict[str, Any]] = []
    for i in range(N):
        curve = curves[i]
        finite = np.isfinite(curve)
        if finite.sum() < 2:
            records.append({
                "peak_force": float("nan"),
                "peak_force_pos_norm": float("nan"),
                "peak_force_pos_cm": float("nan"),
                "impulse_norm": float("nan"),
            })
            continue
        vals = curve[finite]
        s_vals = np.asarray(s_grid, dtype=np.float64)[finite]
        argmax = int(np.argmax(vals))
        stroke_len = float(stroke_lengths_cm[i]) if stroke_lengths_cm is not None else float("nan")
        records.append({
            "peak_force": float(vals[argmax]),
            "peak_force_pos_norm": float(s_vals[argmax]),
            "peak_force_pos_cm": float(s_vals[argmax] * stroke_len) if np.isfinite(stroke_len) else float("nan"),
            "impulse_norm": float(np.trapz(vals, s_vals)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-bundle", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/inference/predictions.",
    )
    p.add_argument(
        "--stage",
        choices=["A", "B"],
        default="B",
        help="Which stage of the bundle to use for prediction (default: B).",
    )
    p.add_argument(
        "--active-side",
        choices=["left", "right"],
        default=None,
        help="Override the bundle's default active side.",
    )
    p.add_argument(
        "--rower-facing",
        choices=["auto", "left", "right"],
        default="auto",
    )
    p.add_argument(
        "--stroke-length-cm",
        type=float,
        default=None,
        help="Override per-stroke length in cm.  When absent, cm-indexed CSV is skipped unless --px-per-cm is given.",
    )
    p.add_argument(
        "--px-per-cm",
        type=float,
        default=None,
        help="Conversion factor from handle displacement (pixels) to cm.",
    )
    p.add_argument(
        "--include-second-derivatives",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--min-cycle-s",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--min-drive-s",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--min-recover-s",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.08,
    )
    p.add_argument(
        "--max-stroke-length-cm",
        type=float,
        default=RP3_CLEAN_MAX_STROKE_LENGTH_CM,
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    bundle = ModelBundle(args.model_bundle.expanduser().resolve())

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "inference" / "predictions").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stroke_csv = run_dir / "stroke" / "stroke_signal.csv"
    if not stroke_csv.exists():
        print(f"Error: stroke_signal.csv not found at {stroke_csv}")
        return 1
    df = pd.read_csv(stroke_csv)

    active_side = args.active_side or bundle.active_side_default or "right"

    print(f"Detecting drive events in {stroke_csv} ...")
    detection = detect_drive_events(
        df,
        smooth_window_s=float(args.smooth_window_s),
        min_cycle_s=float(args.min_cycle_s),
        min_drive_s=float(args.min_drive_s),
        min_recover_s=float(args.min_recover_s),
        finish_method=FINISH_METHOD_VELOCITY_THRESHOLD,
    )
    if not detection.events:
        print("No drive events detected.")
        return 2

    events_df = _events_to_dataframe(detection.events)
    frame_df = _build_frame_df(df, detection)

    print(f"Building pose-only drive segments on s_grid of size {bundle.n_grid} ...")
    segments_df, status_df = build_pose_drive_segments(
        run_dir=run_dir,
        events_df=events_df,
        frame_df=frame_df,
        s_grid=bundle.s_grid,
        active_side=active_side,
        include_head=bundle.include_head,
        include_second_derivatives=args.include_second_derivatives,
        rower_facing=args.rower_facing,
    )
    status_df.to_csv(output_dir / "prediction_segment_status.csv", index=False)

    feature_cols = feature_column_names(
        include_head=bundle.include_head,
        include_second_derivatives=args.include_second_derivatives,
    )
    tensor, seq_order = stack_feature_tensor(
        segments_df, feature_cols=feature_cols, n_grid=bundle.n_grid,
    )
    if tensor.shape[0] == 0:
        print("No valid strokes after pose-feature extraction.")
        return 3

    # ------------------------------------------------------------------
    # Stroke-length resolution for cm-indexed output
    # ------------------------------------------------------------------
    N = tensor.shape[0]
    stroke_lengths_cm = np.full(N, np.nan, dtype=np.float64)
    if args.stroke_length_cm is not None:
        stroke_lengths_cm[:] = float(args.stroke_length_cm)
    elif args.px_per_cm is not None:
        px_per_cm = float(args.px_per_cm)
        if px_per_cm <= 0:
            print("Error: --px-per-cm must be positive.")
            return 1
        for i, seq_idx in enumerate(seq_order):
            ev = events_df.iloc[seq_idx]
            c_px = float(ev.get("catch_distance_px", float("nan")))
            f_px = float(ev.get("finish_distance_px", float("nan")))
            if np.isfinite(c_px) and np.isfinite(f_px):
                stroke_lengths_cm[i] = abs(f_px - c_px) / px_per_cm

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    if args.stage == "A":
        if not bundle.has_stage("A"):
            print("Bundle has no Stage A model; use --stage B.")
            return 1
        scalar_summary = _build_scalar_summary(tensor, feature_cols, bundle.s_grid)
        curves_normalized = _predict_stage_a(bundle, scalar_summary)
        scale_note = "peak_normalized_shape"
    else:
        if not bundle.has_stage("B"):
            print("Bundle has no Stage B model; use --stage A.")
            return 1
        curves_normalized = _predict_stage_b(bundle, tensor)
        scale_note = "newtons_absolute"

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    norm_cols = [f"s_{s:.4f}" for s in bundle.s_grid]
    norm_df = pd.DataFrame(curves_normalized, columns=norm_cols)
    meta_df = pd.DataFrame({
        "seq_idx": seq_order,
        "stroke_idx": [int(events_df.iloc[s]["stroke_idx"]) for s in seq_order],
        "stroke_length_cm": stroke_lengths_cm,
    })
    norm_out = pd.concat([meta_df.reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1)
    norm_out.to_csv(output_dir / "predicted_force_curves_normalized.csv", index=False)

    if np.isfinite(stroke_lengths_cm).any():
        cm_df = _curves_to_cm_table(
            curves_normalized_s=curves_normalized,
            s_grid=bundle.s_grid,
            stroke_lengths_cm=stroke_lengths_cm,
            max_stroke_length_cm=float(args.max_stroke_length_cm),
        )
        cm_out = pd.concat([meta_df.reset_index(drop=True), cm_df.reset_index(drop=True)], axis=1)
        cm_out.to_csv(output_dir / "predicted_force_curves.csv", index=False)
    else:
        print("Note: no stroke_length_cm available; skipped RP3-style force_at_*cm CSV.")

    metrics_df = _derive_metrics(curves_normalized, bundle.s_grid, stroke_lengths_cm)
    metrics_df = pd.concat([meta_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    metrics_df.to_csv(output_dir / "predicted_metrics.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "model_bundle": str(bundle.bundle_dir),
        "stage": args.stage,
        "active_side": active_side,
        "n_strokes_detected": int(len(events_df)),
        "n_strokes_with_features": int(N),
        "s_grid_size": int(bundle.n_grid),
        "output_scale": scale_note,
        "bundle_manifest": bundle.manifest,
    }
    with (output_dir / "prediction_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nPrediction results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
