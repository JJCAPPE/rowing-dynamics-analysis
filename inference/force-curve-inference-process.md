# Force Curve Inference Process (Biomechanics-First)

## 1) Goal
Predict per-stroke force curves from video-only biomechanics at inference time.

Training uses two synchronized supervision sources:
- Pose/kinematics extracted from video via `sports2d_app`.
- Force curves extracted from RP3 exports via `rp3-extraction`.

Inference for a new athlete uses only video.

## 2) Existing Data Assets (Current Code)

### Pose and stroke signals (`sports2d_app`)
Current pipeline already produces the core artifacts needed for modeling:
- `motionbert/angles_h36m.csv`
  - columns: `frame_idx`, `time_s`, `left_knee_deg`, `right_knee_deg`, `left_hip_deg`, `right_hip_deg`, `left_elbow_deg`, `right_elbow_deg`, `trunk_vs_horizontal_deg`, `spine_flexion_deg`, `head_vs_trunk_deg`.
- `stroke/stroke_signal.csv`
  - columns: `frame_idx`, `time_s`, `handle_cx_px`, `handle_cy_px`, `machine_cx_px`, `machine_cy_px`, `relative_axis_px`, `relative_perp_px`, `velocity_axis_px_s`, `stroke_idx`, `stroke_phase`, `is_drive`, `is_catch`, `is_finish`, `axis_x`, `axis_y`.
- `stroke/angles_h36m_with_stroke.csv`
  - merged kinematics + stroke-domain handle signals.

Relevant code paths:
- `sports2d_app/motionbert_3d.py`
- `sports2d_app/stroke_signal.py`
- `sports2d_app/rowing_pose/kinematics.py`

### Force curves (`rp3-extraction`)
Current RP3 cleanup already yields stroke-level force bins:
- `rp3-extraction/workouts/clean/*-clean.csv`
- force columns created by `expand_rp3_curve_data.py`:
  - `force_at_2.2cm`, `force_at_4.4cm`, ..., up to configured max length.
- stroke metadata columns include:
  - `stroke_number`, `time`, `stroke_length`, `peak_force`, `peak_force_pos`, `rel_peak_force_pos`, `drive_time`, `recover_time`, etc.

Relevant code paths:
- `rp3-extraction/expand_rp3_curve_data.py`
- `rp3-extraction/view_rp3_strokes.py`

## 3) Core Representation Decisions

1. Use drive length/progress as the master domain, not wall-clock time.
2. Use interpolation (not nearest-neighbor matching) to map pose features to force bins.
3. Use single-side biomechanics (camera-near side) as the canonical limb chain to reduce occlusion/perspective error.
4. Keep handle kinematics as support features, not primary explanatory variables.
5. Use derivatives with respect to drive progress (`d/ds`), not time (`d/dt`), for force-shape learning.

## 4) Data Contract for Each Stroke
Each supervised sample is one stroke with:

### Inputs `X(s)`
- Kinematic curves over normalized drive progress `s in [0,1]`:
  - knee angle, hip angle, elbow angle, trunk angle.
  - first derivatives `dtheta/ds`.
  - second derivatives `d2theta/ds2`.
- Coordination curves/scalars:
  - phase timing lags (legs->swing, swing->arms).
  - angle ratio features (leg extension vs trunk swing vs arm draw).
- Optional support features:
  - handle velocity/acceleration projected on drive axis.
- Anthropometric context:
  - normalized segment lengths (thigh, shank, upper arm, forearm, trunk).

### Targets `Y(s)`
- Force curve bins aligned to same `s` grid:
  - either direct `F(s_k)` bins.
  - or low-dimensional curve representation (for example PCA coefficients) with reconstructable `F(s)`.

### Metadata
- athlete ID, session ID, stroke index, stroke rate, stroke length, video/source quality flags.

## 5) Session and Stroke Pairing Process
Before modeling, each video session must be paired to the correct RP3 workout export.

1. Pair by recording context:
   - date/time, athlete identity, and piece/workout identity.
2. Resolve absolute stroke ordering:
   - determine which video stroke corresponds to RP3 `stroke_number` start.
3. Validate one-to-one mapping over a calibration window:
   - compare stroke rate trends and drive/recovery timing patterns from video (`stroke_signal.csv`) vs RP3 (`stroke_rate`, `drive_time`, `recover_time`).
4. Lock mapping and store a pairing manifest:
   - this manifest is part of the dataset provenance and reproducibility.

If pairing drift exists, reject the segment rather than weakly aligning noisy labels.

## 6) Single-Side Biomechanics Standardization
Given your choice to use only the camera-near side:

1. Assign `active_side` per session (`left` or `right`).
2. Build canonical feature names:
   - `knee_active_deg`, `hip_active_deg`, `elbow_active_deg`.
   - map from left/right source columns based on `active_side`.
3. Mirror-normalize conventions:
   - ensure increasing/decreasing trajectories have consistent biomechanical meaning regardless of camera side.
4. Keep trunk and spine signals as shared central features.

This keeps symmetry assumptions while avoiding bilateral contamination from occluded far-side joints.

## 7) Drive-Domain Alignment (Critical Step)

### 7.1 Stroke segmentation on video side
Use `stroke_signal.csv` events and phase fields:
- drive starts at catch.
- drive ends at finish.
- only drive portion is used for force-curve supervision.

### 7.2 Stroke segmentation on RP3 side
Each RP3 row is one stroke with force values already sampled along drive length (2.2 cm bins).

### 7.3 Common axis construction
For each paired stroke:
1. Let RP3 force distances be `d_k` (cm bins).
2. Let video stroke have drive progress surrogate from:
   - `stroke_phase` mapped to drive segment, and/or
   - `relative_axis_px` transformed to monotonic drive displacement.
3. Convert video stroke to normalized progress `s_video in [0,1]`.
4. Convert RP3 bins to `s_force = d_k / stroke_length_cm`.
5. Interpolate every input feature from `s_video` onto `s_force`.

Result: no frequency mismatch problem remains (120 Hz vs 90 Hz becomes irrelevant after domain alignment).

## 8) Smoothing and Derivative Policy

1. Base angles:
   - keep current Sports2D/MotionBERT smoothing unless quality checks show high jitter.
2. Derivative computation:
   - apply light smoothing before differentiation (small window).
   - compute derivatives in the progress domain.
3. Avoid heavy post-smoothing:
   - preserve genuine high-frequency biomechanical timing cues.
4. Quality guardrails:
   - reject strokes with non-physiological derivative spikes caused by tracking failures.

## 9) Feature Families to Analyze

### Primary (biomechanics explanatory)
- Active-side knee/hip/elbow curves.
- Trunk and spine curves.
- Their first and second progress derivatives.
- Inter-joint sequencing (onset times, phase lags).

### Secondary (support)
- Handle-axis velocity/acceleration.
- Stroke-level context: stroke rate, drive ratio.

### Anthropometric normalization
- Normalize kinematic magnitudes/timing to athlete body proportions.
- Use limb-length ratios rather than absolute pixel lengths when possible.

## 10) Target Representation Strategy
Use two target representations in parallel during research:

1. **Direct curve bins**
   - model predicts force at each valid bin.
   - use masked loss for bins beyond actual stroke length.

2. **Low-dimensional curve shape**
   - perform PCA on normalized force curves.
   - model predicts PCA coefficients.
   - reconstruct full curve for interpretation and evaluation.

This gives interpretability and robustness in low-data regimes.

## 11) Modeling Path (Temporal + Interpretable)

### Stage A: Baseline interpretable models
- Inputs: stroke-level summary features + phase landmarks.
- Targets: force-curve PCA coefficients.
- Models: regularized linear models, tree ensembles.

Purpose:
- establish whether biomechanics signal is sufficient.
- identify dominant explanatory features.

### Stage B: Sequence models
- Inputs: aligned per-stroke feature sequences on `s` grid.
- Targets: full force curve.
- Models: temporal convolutional network or transformer encoder.

Loss design:
- masked pointwise force loss.
- peak force and peak location auxiliary losses.
- smoothness/shape-consistency penalty.
- optional derivative-of-force loss.

## 12) Evaluation Protocol (Must Match End Goal)
Goal is generalization to unseen athletes.

1. Split by athlete ID only:
   - leave-one-athlete-out or grouped folds.
2. Report curve-level metrics:
   - RMSE/MAE over force bins.
   - correlation with measured curve.
3. Report rowing-relevant metrics:
   - peak force error.
   - peak position (% drive) error.
   - impulse (area under curve) error.
   - phase-specific errors (early drive, mid-drive, late drive).
4. Report reliability under perturbation:
   - dropped keypoints, mild noise, small timing offsets.

## 13) Inference Process for a New Athlete (Video Only)

1. Run the existing video pipeline (`sports2d_app`) to produce:
   - `angles_h36m.csv`
   - `stroke_signal.csv`
   - merged stroke-angle table.
2. Detect/select active side and apply canonical mapping.
3. Segment strokes and isolate drive portions.
4. Build normalized progress grid per stroke.
5. Compute feature sequences (`theta`, `dtheta/ds`, `d2theta/ds2`, coordination, anthropometrics).
6. Apply the trained model to predict force curve per stroke.
7. Output:
   - predicted `force_at_<distance>cm` style table.
   - derived metrics: predicted peak force, peak position, impulse.
   - confidence/quality score per stroke.

No RP3 input is used in this inference stage.

## 14) Quality Control and Failure Modes

### Mandatory QC gates
- Stroke detection confidence (valid catch/finish structure).
- Tracking completeness (minimal missing joint proportion).
- Physical plausibility checks on angles and derivatives.
- Reasonable stroke duration and progression monotonicity.

### Failure handling
- If QC fails, mark stroke as low confidence and skip or down-weight.
- Never silently force predictions from clearly invalid kinematics.

## 15) What We Are Specifically Testing Scientifically
This process evaluates the broader research hypothesis:

- Whether observable biomechanics from monocular video can explain and predict force-curve shape.
- Which kinematic phases (leg drive, trunk swing, arm draw) contribute most to force timing and magnitude.
- How much athlete-specific morphology alters mapping from movement to force.

The process is intentionally biomechanics-first and interpretable, while still enabling temporal ML models that can learn nonlinear stroke dynamics.
