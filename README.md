# rowing-dynamics-analysis

Research codebase for inferring rowing biomechanics and downstream performance-relevant
quantities directly from standard (unconstrained) video footage, without requiring instrumented
boats or wearable sensors at inference time.

## Objective

Given single- or multi-view video sequences `V(t)` of a rowing athlete, the project targets:

- A temporally consistent 3D human kinematic state `S^(t)`
- Interpretable biomechanical features `F^(t)` (e.g., joint angles, segment kinematics)
- Performance targets `Y^(t)` (e.g., force curve, power) inferred via sequence models

## Core Pipeline (High-Level)

1. Video input (single- or multi-view; non-lab conditions)
2. Preprocessing (stabilization, normalization, view/time alignment)
3. 2D pose estimation to obtain keypoints over time
4. 2D-to-3D lifting to recover a temporally consistent 3D skeleton
5. Kinematic and stroke-structure analysis (biomechanics-first features)
6. Sequence-based inference from motion features to performance quantities
7. Evaluation against ground truth (numerical error) and biomechanical plausibility (structural validity)

## Design Requirements

- Unconstrained input: standard video with variable lighting, background clutter, camera motion,
  compression artifacts, and occlusion.
- No instrumentation at inference: force sensors, ergometer telemetry, or boat telemetry are not
  assumed at inference time. These signals may be used only for supervised training/validation.
- Interpretability-first: intermediate representations should be physically interpretable
  (biomechanics-derived features), not purely end-to-end black-box predictions.
- Generalization: models should generalize across athletes, sessions, and filming conditions.
- Dual evaluation: accuracy is assessed numerically (error vs. ground truth) and structurally
  (kinematics obey rowing biomechanics and realistic timing).

## Current Modules

- `pose-extraction-test/`: offline pipeline for stabilization, crop tracking, 2D pose, optional 3D
  lift, and angle overlays.
- `sports2d_app/`: Sports2D-based app for video processing and 2D pose/keypoint extraction.

## Notes

Generated outputs (videos, `.npz/.npy`, etc.) and local assets (venv, source videos) are intentionally
not committed.
