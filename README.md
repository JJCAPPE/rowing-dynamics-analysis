# rowing-video-analysis

`rowing-video-analysis` is a video-first rowing biomechanics pipeline focused on predicting per-stroke force curves from monocular erg videos. During training, it synchronizes two streams: (1) kinematics extracted from video (Sports2D + MotionBERT + handle tracking), and (2) RP3 force exports. At inference time, the same feature contract runs on video only, so no RP3 input is required.

The project is structured as an end-to-end research workflow: pose extraction, stroke segmentation, cross-modal alignment in normalized drive progress (`s in [0,1]`), feature generation, dataset assembly, staged model training (Stage 0/A/B), and portable bundle-based prediction.

The pipeline produces, for each rowing session:

1. Pose and stroke signals from video (Sports2D + MotionBERT + handle/machine tracking).
2. Drive events (catch / finish) inferred from the handle stroke signal.
3. RP3 force-curve matching: video strokes aligned to RP3 export rows in normalized drive-progress space.
4. Per-stroke matched segment CSVs (kinematics + force curves on a shared grid).
5. Training dataset artifacts (PCA / FPCA, fixed-grid kinematic sequences).
6. Stage 0 / A / B modeling and a portable inference bundle for video-only force prediction.

## Pipeline pages

- [Interactive pipeline visualization](https://jjcappe.github.io/rowing-dynamics-analysis/pipeline-visualisation.html)
- [GitHub Pages index](https://jjcappe.github.io/rowing-dynamics-analysis/)
- [Model training stage diagrams](https://jjcappe.github.io/rowing-dynamics-analysis/model-training-stages.html)

## Visual overview

The figures below are the same visuals used in the docs pipeline walkthrough.

![Sports2D pose overlay on erg video](docs/journal/process-pics/erg-sports2d.png)
![Angle traces with stroke events](docs/journal/process-pics/angles_h36m_with_stroke_plot.png)
![Calibrated stroke matching with angle overlay](docs/journal/process-pics/matching-with-angles.png)
![Per-stroke RP3 force curve reconstruction](docs/journal/process-pics/recreation.png)

## Repository layout

```
rowing-video-analysis/
├── pyproject.toml              # `rowing` package + console script
├── README.md
├── session_registry.csv        # video-run ↔ RP3 clean-CSV mapping
├── src/
│   └── rowing/                 # the unified package
│       ├── cli/                # CLI entry points (Phase 2 will add a TUI menu)
│       │   ├── __main__.py     # python -m rowing
│       │   ├── inference.py    # detect → match → export → dataset (legacy main)
│       │   ├── pose.py         # Sports2D + MotionBERT wizard (was app_cli.py)
│       │   └── overlay.py      # force-curve video overlay
│       ├── pose/               # Sports2D, MotionBERT, handle/machine tracking
│       │   ├── runner.py
│       │   ├── motionbert.py
│       │   ├── stroke_signal.py
│       │   ├── parse.py
│       │   ├── plot_angles.py
│       │   ├── overlay_3d.py
│       │   ├── progress_utils.py
│       │   ├── streamlit_app.py  # original Streamlit UI
│       │   └── kinematics/       # rowing_pose subpackage (unchanged content)
│       ├── matching/
│       │   ├── detect.py        # drive event detection
│       │   ├── match.py         # RP3 stroke matcher (DP)
│       │   ├── pair_session.py  # session registry + coarse cross-correlation sync
│       │   └── diagnostics.py   # matplotlib diagnostic viewer (Phase 4 → editor)
│       ├── rp3/
│       │   ├── clean.py         # dirty-CSV cleanup + force-bin expansion
│       │   └── viewer.py        # standalone RP3 stroke viewer
│       ├── dataset/
│       │   ├── feature_contract.py
│       │   ├── segment_features.py
│       │   ├── functional_pca.py
│       │   └── build.py         # multi-run training dataset builder
│       ├── modeling/
│       │   ├── train.py         # Stage 0 / A / B
│       │   ├── eval_metrics.py
│       │   ├── bundle.py
│       │   ├── export_bundle.py
│       │   └── predict.py       # video-only force prediction from bundle
│       └── reports/             # HTML report generators (Phase 5 + 6)
├── scripts/                     # back-compat shims for old script paths
├── tests/                       # pytest suite
├── docs/                        # static pipeline diagrams + design docs
├── runs/                        # per-video pipeline outputs (gitignored)
├── source-videos/               # raw input videos (gitignored)
├── trained_models/              # exported model bundles
├── data/
│   └── rp3-workouts/            # RP3 dirty/clean CSV staging area
├── vendor/                      # vendored Sports2D / MotionBERT
├── models/                      # large model weights (gitignored)
└── archived/                    # legacy code kept for reference
    ├── pose-extraction-test/
    ├── sports2d-PLAN.md
    └── ...
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The editable install registers the `rowing` console script and makes
`import rowing` available everywhere.

## Quick start

The unified CLI is being built up in phases. Today the per-stage
back-compat shims under `scripts/` are still the canonical way to run each
stage. The top-level `python -m rowing` entry point currently forwards to
the legacy inference CLI; Phase 2 will replace it with an interactive menu.

### 1) Pose extraction (Sports2D + MotionBERT)

```bash
.venv/bin/python scripts/app_cli.py
```

Creates `runs/<video_stem>_<timestamp>/{sports2d,motionbert,stroke,exports}/`.

### 2) Drive-event detection only (no RP3)

```bash
.venv/bin/python scripts/inference_cli.py \
  --run-dir runs/<run_name> \
  --no-match-rp3 \
  --overlay-video
```

### 3) RP3 matching + segment export + training dataset

Drop the dirty RP3 CSV in `runs/<run_name>/rp3/`, then:

```bash
.venv/bin/python scripts/inference_cli.py \
  --run-dir runs/<run_name> \
  --anchor-rp3-stroke-number <n> \
  --active-side right
```

### 4) Aggregate dataset across runs

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --segment-csv 'runs/*/inference/rp3_pose_force_matched_segments.csv' \
  --output-dir training_dataset_all_runs/ \
  --qc-mode hard
```

### 5) Train models

```bash
.venv/bin/python scripts/modeling.py \
  --dataset-dir training_dataset_all_runs/ \
  --stages 0 A B \
  --output-dir modeling_results/
```

### 6) Predict force from a new video using a saved bundle

```bash
.venv/bin/python scripts/predict_force_cli.py \
  --run-dir runs/<run_name> \
  --model-bundle trained_models/<bundle_name>
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

## Phasing notes

This repository is mid-migration to a single Rich/Textual TUI plus an
interactive matplotlib match editor. See `docs/inference-cli-reference.md`
for per-flag documentation of the legacy CLI shims and
`docs/force-curve-inference-process.md` for the modeling design doc.
