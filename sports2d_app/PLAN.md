# Sports2D-Only App Implementation Plan

Goal: build a completely separate Streamlit app that runs the Sports2D pipeline end-to-end (no manual annotation), preserves all Sports2D outputs (points + angles), and always produces a MotionBERT-based 3D overlay video for Person 0.

## 1. Project Layout (separate folder)
- Create a new top-level folder `sports2d_app/` so nothing touches the existing `pose-extraction-test` UI or pipeline.
- Proposed structure:
  - `sports2d_app/app.py` (Streamlit UI)
  - `sports2d_app/runner_sports2d.py` (wrapper that calls `Sports2D.process`)
  - `sports2d_app/parse_sports2d.py` (TRC/MOT parsing and export)
  - `sports2d_app/motionbert_3d.py` (MotionBERT lift + H36M angles)
  - `sports2d_app/overlay_3d.py` (3D inset compositing on top of Sports2D annotated video)
  - `sports2d_app/requirements.txt`
  - `sports2d_app/README.md`

## 2. Sports2D Runner (maximum reuse)
- Import from the local repo clone: `pose-extraction-test/third_party/Sports2D`.
- Build a config dict and call `Sports2D.process(config_dict)` directly.
- Required config flags:
  - `save_vid = true` (Sports2D annotated video)
  - `save_pose = true` (TRC, one per person)
  - `calculate_angles = true`
  - `save_angles = true` (MOT, one per person)
  - `show_realtime_results = false`
- Output folder:
  - `sports2d_app/runs/<video_stem>_<timestamp>/sports2d/`
  - Keep Sports2D’s output folder intact for downstream use.

## 3. Data Capture (all points + all angles)
- Parse **all** Sports2D TRC files (px coordinates) and write:
  - Raw TRC files unchanged
  - Consolidated CSV/NPZ with all keypoints and metadata
- Parse **all** Sports2D MOT files (angles) and write:
  - Raw MOT files unchanged
  - Consolidated CSV per person for easier analysis
- Keep every Sports2D keypoint the pipeline provides (do not drop extra points).

## 4. 3D Lift (always on, Person 0 only)
- Build a COCO-17 subset from the TRC (by name mapping) for Person 0 only.
- Use the MotionBERT submodule at `pose-extraction-test/third_party/MotionBERT`.
- Run MotionBERT and write:
  - `pose3d.npz` (raw + scaled)
  - `angles_h36m.csv` (H36M angles from 3D)
  - `metrics.json` summary (ROM, min/max, etc.)

## 5. 3D Overlay Video
- Use the Sports2D annotated video as the background.
- Composite a 3D inset (MotionBERT) onto that video.
- Output `pose3d_overlay.mp4` in the run folder.

## 6. Streamlit UI (Sports2D HF-style)
- UI layout mirrors the Hugging Face Sports2D demo:
  - Warnings/notes at top
  - Input/Output video panes side-by-side
  - Controls:
    - `nb_persons_to_detect`
    - `first_person_height`
    - `distance_to_camera`
    - `pose_model` (default Whole_body for max points)
    - `mode` (lightweight/balanced/performance)
- No manual annotation controls.
- Always run 3D.
- Results area:
  - Sports2D annotated video
  - 3D overlay video
  - Download ZIP containing all outputs

## 7. Packaging & Docs
- `requirements.txt` for the new app only.
- `README.md` describing setup, run command, and outputs.

## 8. Validation
- Run with a sample video and confirm:
  - Sports2D outputs exist
  - TRC/MOT parsed correctly
  - 3D overlay generated
  - ZIP export contains all artifacts
