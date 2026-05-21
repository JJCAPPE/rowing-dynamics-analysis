#!/usr/bin/env python3
"""Embed prefixed model-training SVGs into docs/pipeline-visualisation.html."""
from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
GEN = DOCS / "_model_training_diagrams_generated.html"
PIPE = DOCS / "pipeline-visualisation.html"


def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else "" for line in text.split("\n"))


def wrap_diagram(anchor: str, title: str, subtitle: str, svg_and_caption: str) -> str:
    inner = indent_block(svg_and_caption.strip(), 20)
    return f'''            <div class="model-training-diagram" id="pipeline-{anchor}">
                <div class="model-training-diagram-head">
                    <span class="model-training-diagram-title">{title}</span>
                    <span class="model-training-diagram-sub">{subtitle}</span>
                    <a class="diagram-nav-link" href="model-training-stages.html#{anchor}">Same diagram on model training page</a>
                </div>
                <div class="model-training-diagram-inner">
{inner}
                </div>
            </div>
'''


def parse_generated() -> dict[str, str]:
    raw = GEN.read_text()
    blocks: dict[str, str] = {}
    for m in re.finditer(r"<!-- (mt\w+_) -->\s*\n([\s\S]*?)(?=\n<!-- mt|\Z)", raw):
        blocks[m.group(1)] = m.group(2).strip()
    return blocks


def main() -> None:
    blocks = parse_generated()
    assert "mtT_" in blocks and "mt0_" in blocks and "mtSum_" in blocks, list(blocks.keys())

    target = wrap_diagram(
        "mt-target",
        "Target representation",
        "How force curves are encoded before training",
        blocks["mtT_"],
    )
    s0 = wrap_diagram(
        "mt-stage0",
        "Stage 0",
        "Reproducibility floor and metadata-only baseline",
        blocks["mt0_"],
    )
    sa = wrap_diagram(
        "mt-stage-a",
        "Stage A",
        "Scalar features to PCA coefficients",
        blocks["mtA_"],
    )
    sb = wrap_diagram(
        "mt-stage-b",
        "Stage B",
        "Full kinematic sequences to force curve",
        blocks["mtB_"],
    )
    summary = wrap_diagram(
        "mt-summary",
        "Stage progression",
        "Each stage must beat the previous on the same metrics",
        blocks["mtSum_"],
    )
    ev = wrap_diagram(
        "mt-eval",
        "Evaluation protocol",
        "Splits and metrics (all stages, same held-out set)",
        blocks["mtE_"],
    )

    ptext = PIPE.read_text()

    marker_07 = "        <!-- ═══ 07 MODELING (training) ═══ -->"
    marker_08 = "        <!-- ═══ 08 PREDICTION ═══ -->"
    start07 = ptext.find(marker_07)
    start08 = ptext.find(marker_08)
    if start07 == -1 or start08 == -1:
        raise SystemExit(f"Could not locate step 07/08 markers: 07={start07}, 08={start08}")

    step07_snippet = f'''            <p class="step-nav"><a href="model-training-stages.html">Model training page</a> (full walkthrough: <a href="model-training-stages.html#mt-target">target</a>, <a href="model-training-stages.html#mt-stage0">Stage 0</a>, <a href="model-training-stages.html#mt-stage-a">A</a>, <a href="model-training-stages.html#mt-stage-b">B</a>, <a href="model-training-stages.html#mt-eval">evaluation</a>, <a href="model-training-stages.html#mt-summary">summary</a>)</p>

{target}
            <div class="card-row cols-1">
                <div class="card open t-shared" data-role="training" onclick="toggle(this)" style="border-left: 3px solid var(--text-muted);">
                    <div class="card-head">
                        <span class="card-title" style="color: var(--text-muted);">Stage 0 &mdash; Sanity Baselines</span>
                        <span class="card-chevron">&#9654;</span>
                    </div>
                    <div class="card-brief">Force reproducibility floor + metadata-only regression &mdash; the bar biomechanics models must clear</div>
                    <div class="card-detail">
                        <p><strong>Reproducibility floor:</strong> Quantify within-condition variability of force curves (same athlete, similar stroke rate/length bins). Median pairwise curve distance, coefficient of variation.</p>
                        <p><strong>Metadata baseline:</strong> Regress force-curve PCA coefficients from scalar metadata only (stroke rate, length, drive time). Reconstruct predicted curves. Every biomechanics model must beat this.</p>
                        <div class="schema">
                            <div><span class="hl-shared">Inputs</span>: stroke_rate, stroke_length, drive_time</div>
                            <div><span class="hl-shared">Targets</span>: PCA coefficients &rarr; F&#770;(s)</div>
                            <div><span class="hl-rp3">Gate</span>: biomechanics must improve on this</div>
                        </div>
                    </div>
                </div>
            </div>

{s0}
            <div class="connector" data-role="training">
                <div class="stem"></div>
            </div>

            <div class="card-row cols-1">
                <div class="card open t-shared" data-role="training" onclick="toggle(this)">
                    <div class="card-head">
                        <span class="card-title">Stage A &mdash; Interpretable Kinematic Models</span>
                        <span class="card-chevron">&#9654;</span>
                    </div>
                    <div class="card-brief">Stroke-level summary features + phase landmarks &rarr; regularized linear / tree ensembles &rarr; PCA coefficients</div>
                    <div class="card-detail">
                        <p>Verifies that kinematic features add value beyond metadata. Identifies dominant explanatory biomechanics features through feature importance analysis.</p>
                        <div class="schema">
                            <div><span class="hl-shared">Inputs</span>: summary kinematics + phase landmarks</div>
                            <div><span class="hl-shared">Models</span>: Ridge, Lasso, Random Forest, GBM</div>
                            <div><span class="hl-shared">Targets</span>: force-curve PCA coefficients</div>
                        </div>
                    </div>
                </div>
            </div>

{sa}
            <div class="connector" data-role="training">
                <div class="stem"></div>
            </div>

            <div class="card-row cols-1">
                <div class="card open t-shared" data-role="training" onclick="toggle(this)">
                    <div class="card-head">
                        <span class="card-title">Stage B &mdash; Sequence Models</span>
                        <span class="card-chevron">&#9654;</span>
                    </div>
                    <div class="card-brief">Full progress-aligned kinematic sequences &rarr; TCN / Transformer encoder &rarr; full force curve</div>
                    <div class="card-detail">
                        <p>Temporal convolutional network or transformer encoder operating on the full progress-aligned feature sequences. Captures nonlinear stroke dynamics and cross-joint interactions.</p>
                        <p><strong>Loss:</strong> Masked pointwise force loss. Shape regularizers or derivative losses added only if errors show pathological curve behaviour.</p>
                        <div class="schema">
                            <div><span class="hl-shared">Inputs</span>: X(s) [N_bins &times; N_features]</div>
                            <div><span class="hl-shared">Models</span>: TCN, Transformer encoder</div>
                            <div><span class="hl-shared">Targets</span>: F(s) [N_bins], masked loss</div>
                        </div>
                    </div>
                </div>
            </div>

{sb}
{summary}'''

    new07 = (
        "        <!-- ═══ 07 MODELING (training) ═══ -->\n"
        '        <div class="section" data-role="training">\n'
        '            <div class="step-label">\n'
        '                <span class="step-num">07</span>\n'
        '                <span class="step-title">Model Training</span>\n'
        "            </div>\n\n"
        + step07_snippet
        + "\n        </div>\n\n"
        + "        <div class=\"connector\" data-role=\"shared\">\n            <div class=\"stem\"></div>\n        </div>\n\n"
    )

    ptext = ptext[:start07] + new07 + ptext[start08:]

    step09_snippet = f'''            <p class="step-nav"><a href="model-training-stages.html#mt-eval">Evaluation diagram on model training page</a></p>

{ev}'''

    old09 = re.search(
        r"(<span class=\"step-title\">Evaluation Protocol</span>\s*</div>\s*\n)"
        r'(\s*<div class="card-row cols-2">)',
        ptext,
    )
    if not old09:
        raise SystemExit("Could not find step 09 header")

    ptext = ptext[: old09.end(1)] + "\n" + step09_snippet + "\n" + ptext[old09.start(2) :]

    PIPE.write_text(ptext)
    print("Patched", PIPE)


if __name__ == "__main__":
    main()
