# Force Curve Inference Process (Biomechanics-First)

## 0) Scope Decisions (Current Iteration)
This plan is intentionally scoped to the current phase of work.

1. Data volume expansion is deferred and handled separately.
2. Unilateral biomechanics is the default path (camera-near side only).
3. Anthropometric features are excluded from the initial model.
4. Alignment is done in normalized progress space; no px-to-cm calibration is required in this iteration.

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

### Force curves (run-local `rp3/`)
Current workflow stores RP3 data per run:
- dirty RP3 exports are placed in `runs/<run_name>/rp3/*.csv`
- clean files are generated as `runs/<run_name>/rp3/*-clean.csv` by `inference_cli.py`
- force columns are created by `expand_rp3_curve_data.py`:
  - `force_at_2.2cm`, `force_at_4.4cm`, ..., up to configured max length.
- stroke metadata columns include:
  - `stroke_number`, `time`, `stroke_length`, `peak_force`, `peak_force_pos`, `rel_peak_force_pos`, `drive_time`, `recover_time`, etc.

Relevant code paths:
- `rp3-extraction/expand_rp3_curve_data.py`
- `rp3-extraction/view_rp3_strokes.py`

## 3) Core Representation Decisions

1. Use normalized drive progress as the master domain, not wall-clock time.
2. Use interpolation (not nearest-neighbor matching) to map pose features to force bins.
3. Use single-side biomechanics (camera-near side) as the canonical limb chain.
4. Keep handle kinematics as support features, not primary explanatory variables.
5. Compute derivatives via time-domain differentiation plus chain rule into progress domain.

## 4) Data Contract for Each Stroke
Each supervised sample is one stroke with:

### Inputs `X(s)`
- Kinematic curves over normalized drive progress `s in [0,1]`:
  - knee angle, hip angle, elbow angle, trunk angle.
  - first derivatives `dtheta/ds`.
  - optional second derivatives `d2theta/ds2` (ablation-gated).
- Coordination curves/scalars:
  - phase timing lags (legs->swing, swing->arms).
  - angle ratio features (leg extension vs trunk swing vs arm draw).
- Optional support features:
  - handle velocity/acceleration projected on drive axis.

### Targets `Y(s)`
- Force curve bins aligned to same `s` grid:
  - either direct `F(s_k)` bins.
  - or low-dimensional curve representation with reconstructable `F(s)`.

### Metadata
- athlete ID, session ID, stroke index, stroke rate, stroke length, video/source quality flags.

## 5) Session Pairing and Synchronization Protocol (Priority)
Before modeling, each video session must be paired to the correct RP3 workout export.

1. Pair by recording context:
   - date/time, athlete identity, and piece/workout identity.
2. Use stroke progression signal for boundary anchoring:
   - chain-insert-to-handle progression from `stroke_signal.csv` is the primary video-side stroke structure signal.
3. Coarse synchronization:
   - align video and RP3 stroke streams using stroke-rate/inter-stroke interval cross-correlation.
4. Fine synchronization:
   - align catch/finish sequences to maximize boundary agreement over a calibration window.
5. Acceptance tests:
   - sustained stroke-rate mismatch above threshold (for example ~1 spm) fails pairing.
   - low inter-stroke timing agreement fails pairing.
6. Lock mapping in a pairing manifest:
   - session ID, athlete ID, offset, drift estimate, accepted stroke index ranges, QC metrics.

If pairing drift exists, reject the segment rather than weakly aligning noisy labels.

## 6) Single-Side Biomechanics Standardization
Given the unilateral choice:

1. Assign `active_side` per session (`left` or `right`).
2. Build canonical feature names:
   - `knee_active_deg`, `hip_active_deg`, `elbow_active_deg`.
3. Map from left/right source columns based on `active_side`.
4. Mirror-normalize sign conventions so feature semantics are consistent across camera sides.
5. Keep trunk and spine signals as shared central features.

## 7) Drive-Domain Alignment

### 7.1 Stroke segmentation on video side
Use `stroke_signal.csv` with catch/finish flags and drive indicator:
- drive starts at catch.
- drive ends at finish.
- only drive portion is supervised against RP3 force curves.

### 7.2 Stroke segmentation on RP3 side
Each RP3 row is one stroke with force values already sampled along drive length (2.2 cm bins).

### 7.3 Common axis construction
For each paired stroke:
1. Let RP3 force distances be `d_k` (cm bins).
2. Convert force bins to normalized progress `s_force = d_k / stroke_length_cm`.
3. Build video normalized progress `s_video` from drive progression (`stroke_phase` drive segment as primary).
4. Interpolate video features from `s_video` to `s_force`.

Note: in this iteration, normalized progress is sufficient; absolute px-to-cm calibration is not required.

## 8) Smoothing and Derivative Policy

1. Base angles:
   - keep current Sports2D/MotionBERT smoothing unless quality checks show jitter.
2. Time-domain derivatives first:
   - compute `dtheta/dt` on lightly smoothed angle trajectories.
3. Progress-domain derivatives via chain rule:
   - compute `ds/dt` from drive progression.
   - compute `dtheta/ds = (dtheta/dt) / (ds/dt + eps)`.
4. Second derivatives:
   - include `d2theta/ds2` only if ablation shows clear gain.
5. Explicit smoothing kernel:
   - use a fixed kernel choice (for example Savitzky-Golay, small odd window, low polynomial order), document parameters, and keep them constant during comparison experiments.
6. Quality guardrails:
   - reject strokes with denominator instability (`ds/dt` near zero in drive) or non-physiological derivative spikes.

## 9) Feature Families to Analyze

### Primary (biomechanics explanatory)
- Active-side knee/hip/elbow curves.
- Trunk and spine curves.
- First progress derivatives.
- Optional second derivatives (ablation-gated).
- Inter-joint sequencing (onset times, phase lags).

### Secondary (support)
- Handle-axis velocity/acceleration.
- Stroke-level context: stroke rate, drive ratio.

### Explicitly deferred
- Anthropometric normalization and segment-length features.

## 10) Target Representation Strategy
Use two target representations in parallel during research:

1. **Direct curve bins**
   - model predicts force at each valid bin.
   - use masked loss for bins beyond actual stroke length.

2. **Low-dimensional curve shape**
   - start with standard PCA on normalized force curves.
   - optionally evaluate functional PCA as a follow-up if boundary/smoothness handling becomes limiting.

## 11) Modeling Path (with Stage 0 Sanity Baselines)

### Stage 0: Sanity Baselines (mandatory before kinematic models)
1. Force reproducibility floor:
   - quantify within-condition variability of force curves (same athlete, similar stroke rate/length bins).
   - report median pairwise curve distance and coefficient of variation of key metrics.
2. Metadata-only baseline:
   - regress force-curve PCA coefficients using only scalar metadata (`stroke_rate`, `stroke_length`; optionally `drive_time`).
   - reconstruct predicted curves and report the same metrics as all later models.
3. Baseline gate:
   - any biomechanics model must beat Stage 0 on both curve error and rowing-relevant metrics.

### Stage A: Interpretable kinematic baselines
- Inputs: stroke-level summary kinematic features + phase landmarks.
- Targets: force-curve PCA coefficients.
- Models: regularized linear models and tree ensembles.

Purpose:
- verify added value beyond metadata-only baselines.
- identify dominant explanatory biomechanics features.

### Stage B: Sequence models
- Inputs: aligned per-stroke feature sequences on `s` grid.
- Targets: full force curve.
- Models: temporal convolutional network or transformer encoder.

Loss strategy:
- start with masked pointwise force loss.
- add shape regularizers or derivative losses only if errors show pathological behavior and improvements are verified on held-out data.

## 12) Evaluation Protocol

### Current phase (limited athletes)
- use session-held-out or time-block-held-out splits within athlete.
- label results as provisional and non-generalization claims.

### Generalization phase (once data volume supports it)
- split by athlete ID only (leave-one-athlete-out or grouped folds).

### Common reporting
1. Curve-level metrics:
   - RMSE/MAE over force bins.
   - curve correlation.
2. Rowing-relevant metrics:
   - peak force error.
   - peak position (% drive) error.
   - impulse error.
   - phase-specific errors (early/mid/late drive).
3. Pairing/alignment integrity:
   - stroke-rate consistency between video and RP3.
   - inter-stroke timing agreement statistics.

## 13) Inference Process for a New Athlete (Video Only)

1. Run `sports2d_app` pipeline to produce:
   - `angles_h36m.csv`
   - `stroke_signal.csv`
   - merged stroke-angle table.
2. Determine active side and apply canonical unilateral mapping.
3. Segment strokes and isolate drive portions.
4. Build normalized progress grid per stroke.
5. Compute feature sequences (`theta`, `dtheta/ds`, optional `d2theta/ds2`, coordination).
6. Apply trained model to predict force curve per stroke.
7. Output:
   - predicted `force_at_<distance>cm` style table.
   - derived metrics: predicted peak force, peak position, impulse.
   - confidence/quality score per stroke.

No RP3 input is used in this inference stage.

## 14) Quality Control and Failure Modes

### Mandatory QC gates
- Stroke detection confidence (valid catch/finish structure).
  - Implemented: `qc_weak_detection` flag when min(catch, finish) velocity contrast < 0.15 of peak drive velocity. Continuous metric: `detection_confidence`.
- Tracking completeness (minimal missing joint proportion).
  - Implemented: `qc_tracking_sparse` flag when NaN fraction > 0.3. Continuous metric: `nan_frac_angles`.
- Physical plausibility checks on angles and derivatives.
  - Implemented: `qc_nonphysio_deriv` (max |dθ/dt| > 600 deg/s), `qc_ds_dt_stall` (ds/dt < 5% of median).
- Reasonable stroke duration and progression monotonicity.
  - Implemented: `qc_duration_implausible` flag for drive outside [0.4, 2.0] s or cycle outside [1.5, 4.0] s (hard drop). `qc_progress_nonmonotonic` flag when >15% of frames needed monotonicity repair. Continuous metric: `progress_mono_violation_frac`.
- Post-pairing stroke-rate consistency between video and RP3.
  - Implemented: `qc_rate_mismatch` and `qc_alignment_drift` in `build_training_dataset.py`.

### Failure handling
- If QC fails, mark stroke as low confidence and skip or down-weight.
  - Implemented: `qc_excluded` bool + soft/hard modes in `build_training_dataset.py`. Hard-drop flags: `qc_tracking_sparse`, `qc_nonphysio_deriv`, `qc_duration_implausible`, `qc_alignment_drift`, `qc_rate_mismatch`.
- Never silently force predictions from invalid kinematics/alignment.
  - Implemented: all flags preserved on exports; aggregate `stroke_quality_score` (0–1) available per stroke for inference confidence.

## 15) What We Are Specifically Testing Scientifically
This process evaluates the broader research hypothesis:

- Whether observable unilateral biomechanics from monocular video can explain and predict force-curve shape.
- Which kinematic phases (leg drive, trunk swing, arm draw) contribute most to force timing and magnitude.
- How much predictive value kinematics adds beyond simple metadata baselines.

The process remains biomechanics-first and interpretable while enabling temporal models for nonlinear stroke dynamics.
