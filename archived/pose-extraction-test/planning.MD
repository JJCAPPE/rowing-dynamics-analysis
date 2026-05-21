# Rowing Video → Stabilized Crop → 2D Pose (MMPose) → 3D Lift (MotionBERT) → 3D Skeleton per Frame (with boat scale)

## Objective

Build a Python pipeline that takes a rowing video (single athlete, side-on), stabilizes motion relative to the boat using a user-selected anchor point (e.g., oarlock), applies a smooth crop around the athlete, estimates **2D keypoints** with **MMPose**, lifts to **3D skeleton** with **MotionBERT**, and outputs per-frame 3D joint positions + derived angles/metrics for analysis.

Add a simple **boat scale** step: user clicks two points on the boat and provides their known distance so we can convert pixel distances → meters in the stabilized image plane.

---

## Tech stack (verified APIs to target)

### OpenCV (stabilization + tracking)

Use Lucas–Kanade sparse optical flow:

* `cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold)` (Python wrapper follows OpenCV’s LK API). ([docs.opencv.org][1])

### MMPose (2D keypoints)

Use `MMPoseInferencer` from `mmpose.apis` (generator-based “lazy inference”):

* `inferencer = MMPoseInferencer('human')`
* `result_generator = inferencer(input, ...)`
* `result = next(result_generator)`
* `result` has keys: `visualization`, `predictions`, where predictions include `keypoints` and `keypoint_scores` per instance. ([mmpose.readthedocs.io][2])

### MotionBERT (3D lift)

Follow the repo’s own “in-the-wild” inference pattern (load backbone, feed 2D sequences, clip_len default 243, optional flip augmentation, root-relative option, optional pixel-coordinate alignment). ([Hugging Face][3])

---

## High-level pipeline stages

### Stage A — I/O + user annotations (one-time per video)

1. Read video metadata: FPS, frame size, frame count.
2. On a chosen reference frame (usually frame 0):

   * User clicks **boat anchor** point `P_anchor0 = (x, y)` (oarlock).
   * User defines initial **athlete crop bbox** `B0 = (x, y, w, h)` (loose is fine).
   * User clicks **two boat scale points** `S0=(x0,y0)`, `S1=(x1,y1)` and inputs known distance `D_m` (meters).
3. Persist these annotations to `run.json` so reruns are zero-click.

**Outputs**

* `run.json` containing: anchor point, bbox, scale points, distance, and config.

---

### Stage B — Boat-relative stabilization transform

Goal: make the boat “stationary” in the video (translation-only to start).

1. Track anchor point across frames using LK optical flow:

   * `prevPts` is shape `(N,1,2)` float32 (N=1 for anchor).
   * Each frame produces `P_anchor[t]`, plus `status[t]`.
2. Build per-frame translation that maps `P_anchor[t] → P_anchor0`:

   * `dx = P_anchor0.x - P_anchor[t].x`
   * `dy = P_anchor0.y - P_anchor[t].y`
   * affine matrix `A[t] = [[1,0,dx],[0,1,dy]]`
3. Warp each frame: `frame_stab[t] = cv2.warpAffine(frame[t], A[t], (W,H), ...)`

**Robustness rules**

* If LK tracking fails (`status=0`) at frame t:

  * Reinitialize anchor using a small template match near last good location (agent implements).
  * If still failing: mark frame as “bad” and interpolate anchor path from neighbors.

**Outputs**

* `A[t]` per frame (save to `stabilization.npz`)
* optional debug video: stabilized full frame

---

### Stage C — Boat scale (meters-per-pixel on stabilized frames)

Compute scale on stabilized reference frame (preferred: the same frame used for clicking points).

1. Apply stabilization to the two scale points (or click them directly on stabilized ref frame).
2. Pixel distance:

   * `d_px = ||S1 - S0||`
3. Scale:

   * `m_per_px = D_m / d_px`

**Notes**

* This gives a solid metric scale for **2D stabilized measurements**.
* We will also use it later to put a reasonable metric scale on 3D outputs (with explicit caveats).

**Outputs**

* `m_per_px` stored in `run.json` and `metrics.json`

---

### Stage D — Stabilized crop tracking + smoothing

Goal: produce a per-frame crop `B[t]` that follows the athlete smoothly.

Recommended approach (no detector dependency):

1. Initialize feature points inside initial bbox `B0` on stabilized ref frame:

   * use `cv2.goodFeaturesToTrack` inside bbox mask.
2. Track those points frame-to-frame with LK (on stabilized frames).
3. Compute robust displacement per frame:

   * median of point displacements → `(Δx, Δy)`
4. Update bbox center by displacement.
5. Smooth bbox trajectory with exponential moving average (EMA) or 1D Kalman filter.

Rules:

* Maintain constant aspect ratio (e.g., 4:3 or 16:9).
* Add padding margin (10–25%) to avoid cutting wrists at catch/finish.
* Clamp bbox to frame bounds.

**Outputs**

* `B[t]` for all frames (save to `crop_boxes.npy`)
* optional debug video: stabilized + drawn bbox

---

### Stage E — 2D keypoints with MMPose (single-person)

Goal: per frame: `J2d_px[t, J, 3] = (x_px, y_px, conf)` in **stabilized full-frame pixel coords**.

Implementation:

1. Create `inferencer = MMPoseInferencer('human')` (or a specific top-down config alias/name). ([mmpose.readthedocs.io][2])
2. For each stabilized frame:

   * Crop with `B[t]`, run pose on crop.
   * Because MMPoseInferencer returns predictions per detected instance, pick the “best” instance:

     * highest mean `keypoint_scores` OR
     * instance whose bbox center is closest to crop center
   * Convert crop coords → stabilized full-frame coords by adding crop origin offset.
3. Store results as:

   * `J2d_px[t, j, 0:2]` and `conf[t, j]`

Key API expectations:

* MMPoseInferencer yields dicts with `predictions` and `keypoints`/`keypoint_scores` per instance. ([mmpose.readthedocs.io][2])

**Outputs**

* `pose2d.npz` with `J2d_px`, `conf`, `joint_names`, `fps`

---

### Stage F — Convert 2D joints to MotionBERT input format

MotionBERT expects **17 joints in Human3.6M (H36M) format** and input tensor shaped like `[N, T, 17, C]` where `C=3` (x, y, conf) in their standard setup. ([Hugging Face][4])

Steps:

1. Choose a keypoint convention:

   * If MMPose outputs COCO-17, implement a deterministic mapping to H36M-17 (agent writes a mapping table + docs).
2. Build `X[t, 17, 3] = (x, y, conf)` for all frames.
3. Decide coordinate normalization:

   * Option 1 (**pixel mode**): keep pixel-like relative scaling (similar to `--pixel` in the example inference script). ([Hugging Face][3])
   * Option 2 (**normalized mode**): scale to `[-1, 1]` around image center / crop (like `scale_range=[1,1]` path in the reference script). ([Hugging Face][3])
4. Chunk into clips of length `clip_len` (default 243), with overlap (e.g., 50%). ([Hugging Face][3])
5. Prepare a batching strategy (batch size 1 is fine; this is offline).

---

### Stage G — 3D lift with MotionBERT (offline)

Use MotionBERT’s inference approach shown in `infer_wild.py`:

* load config via `get_config`
* load backbone via `load_backbone`
* load checkpoint (`model_pos` state dict)
* optional flip augmentation and root-relative behavior. ([Hugging Face][3])

Implementation requirements for the agent:

1. Vendor MotionBERT as a submodule or install it editable, and import:

   * `get_config`, `load_backbone`, `flip_data`, model forward call (as in the reference). ([Hugging Face][3])
2. For each clip:

   * `predicted_3d = model_pos(batch_input)`
   * If flip enabled: run flip inference and average (mirrors reference). ([Hugging Face][3])
   * If `rootrel`: set root joint to zero per frame (reference sets `predicted_3d_pos[:,:,0,:]=0`). ([Hugging Face][3])
3. Stitch clips back to a full sequence using overlap blending (linear weight ramp).

**Outputs**

* `J3d_raw[t, 17, 3]` in MotionBERT’s coordinate space.

---

### Stage H — Metric scaling (boat scale) for outputs

We’ll apply boat scale in a conservative, explicitly-defined way:

1. Convert 2D stabilized joints to meters (for 2D analysis):

   * `J2d_m[t, :, 0:2] = (J2d_px[t,:,0:2] - origin_px) * m_per_px`
   * choose `origin_px` as pelvis/root joint in pixels (per frame) or a fixed reference.
2. Scale 3D output for convenience:

   * Compute a single scalar `α` so that MotionBERT XY matches 2D meters “best”:

     * Solve least squares for `α` over all frames/joints:

       * minimize `|| α * J3d_raw_xy - J2d_m_xy ||`
   * Output `J3d_m = α * J3d_raw` (XYZ all scaled by α)

This produces:

* metric-ish XYZ suitable for trajectories and comparing relative ranges,
* angles unaffected (angles are scale-invariant anyway).

**Outputs**

* `pose3d.npz` with `J3d_m`, `J3d_raw`, `alpha_scale`, plus metadata.

---

### Stage I — Angles and metrics extraction

Provide a small library that computes:

* Knee angle: hip–knee–ankle
* Hip angle: shoulder/torso–hip–knee (define trunk vector explicitly)
* Elbow angle: shoulder–elbow–wrist
* Trunk angle vs horizontal (in stabilized frame axes)
* Seat travel proxy (if you track hip root x-translation in meters)

Formula at joint B given A–B–C:

* `θ = arccos( dot(u,v) / (||u|| ||v||) )` where `u=A-B`, `v=C-B`.

**Outputs**

* `angles.csv` (per frame)
* `metrics.json` summary (max flexion, ROM, etc.)

---

## Deliverables (files)

* `run.json` — annotations + settings + m_per_px
* `stabilization.npz` — per-frame affine transforms, tracked anchor path
* `crop_boxes.npy` — per-frame bbox
* `pose2d.npz` — stabilized 2D keypoints (px), confidences, joint names
* `pose3d.npz` — 3D skeleton per frame (raw + scaled), α, joint names
* `angles.csv` — per-frame angles
* `debug/` videos:

  * `stabilized.mp4`
  * `pose2d_overlay.mp4`
  * `pose3d_render.mp4` (optional stick-figure render)

---

## Agent implementation plan (task breakdown)

### Task 1 — Repo scaffold

* Create package `rowing_pose/` with modules:

  * `io_video.py` (ffmpeg/opencv reader, writer)
  * `ui_annotate.py` (click points + bbox)
  * `stabilize.py` (LK tracking + affine warps)
  * `crop_track.py` (bbox tracking + smoothing)
  * `pose2d_mmpose.py` (MMPoseInferencer wrapper)
  * `motionbert_lift.py` (MotionBERT wrapper)
  * `kinematics.py` (angles/metrics)
  * `viz.py` (overlays)
  * `cli.py` (entrypoint)
* Add `pyproject.toml` and a reproducible env doc.

### Task 2 — Annotation UX

* Implement:

  * click 1 point (anchor)
  * draw bbox (drag)
  * click 2 points (scale)
  * prompt for distance in meters
* Save/load `run.json`

### Task 3 — Stabilization

* LK tracking with failure handling
* Warp pipeline
* Debug visualization of tracked anchor

### Task 4 — Crop tracking + smoothing

* goodFeaturesToTrack in bbox + LK
* bbox update + smoothing
* margin + aspect lock

### Task 5 — MMPose 2D inference wrapper

* Initialize `MMPoseInferencer`
* Frame loop:

  * crop frame
  * call inferencer on numpy image array (preferred for speed) or temp image path
  * pick best instance
  * return keypoints + scores in stabilized coords
* Persist to `pose2d.npz`
  (Ensure the wrapper matches the generator + output structure documented by MMPose.) ([mmpose.readthedocs.io][2])

### Task 6 — MotionBERT lift wrapper

* Implement MotionBERT load + inference mirroring `infer_wild.py`:

  * config load
  * backbone + checkpoint load
  * flip option
  * rootrel option
  * clip_len handling
* Accept `X[T,17,3]` and return `J3d_raw[T,17,3]`
  (Keep behavior aligned with the reference script’s flow and flags.) ([Hugging Face][3])

### Task 7 — H36M joint mapping

* Implement explicit mapping from MMPose output format → H36M17
* Unit test the mapping with a dummy skeleton

### Task 8 — Scaling + angles

* Compute `m_per_px`
* Compute `α` (optional toggle)
* Compute and export angles/metrics

### Task 9 — Debug renders

* Stabilized + bbox video
* 2D overlay (skeleton on stabilized frame)
* Optional 3D stick render (simple matplotlib render per frame or lightweight renderer)

### Task 10 — Tests + acceptance criteria

Acceptance checklist:

* Stabilized video keeps oarlock within ±2–5 px typical drift (after smoothing).
* Crop never cuts off wrists through full stroke in a typical clip (with default padding).
* `pose2d.npz` shape is `(T, J, 3)` and confidences are sane.
* MotionBERT output shape `(T, 17, 3)` and is continuous with no clip-boundary jumps.
* Angles are stable across frames (no random spikes unless conf is low).

---

## CLI spec (what the agent should implement)

`python -m rowing_pose.cli run --video path.mp4 --out outdir/ [--device cuda] [--mmpose-model human] [--motionbert-ckpt ...] [--clip-len 243] [--flip] [--rootrel]`

Also:

* `annotate` subcommand (create/edit `run.json`)
* `debug` subcommand (regenerate overlays from saved NPZ)

---

## Notes / constraints (explicit)

* Single camera, side-on: angles are the main truth; “absolute depth” remains fundamentally ambiguous.
* Boat scaling provides strong **2D metric** scale; 3D metric scaling is an informed convenience (and can be disabled).
* Keep all saved joints in a consistent order (documented in each NPZ).

---

## Reference implementations to align with (source-of-truth)

* MMPose inferencer usage + output structure (generator returning `visualization` + `predictions` containing `keypoints` and `keypoint_scores`). ([mmpose.readthedocs.io][2])
* MotionBERT “wild” inference script showing config load, checkpoint load, clip_len default 243, flip averaging, root-relative handling, and pixel-vs-normalized option behavior. ([Hugging Face][3])
* OpenCV LK optical flow function signature and parameters. ([docs.opencv.org][1])

[1]: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html?utm_source=chatgpt.com "OpenCV: Object Tracking"
[2]: https://mmpose.readthedocs.io/en/latest/user_guides/inference.html "Inference with existing models — MMPose 1.3.2 documentation"
[3]: https://huggingface.co/spaces/kzielins/MotionBERT/blob/main/infer_wild.py "infer_wild.py · kzielins/MotionBERT at main"
[4]: https://huggingface.co/walterzhu/MotionBERT/blame/main/README.md "README.md · walterzhu/MotionBERT at main"
