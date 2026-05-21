#!/usr/bin/env python3
"""One-off helper: extract model-training SVGs with unique marker IDs for docs/pipeline-visualisation.html."""
from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

KNOWN_MARKER_IDS = (
    "arrowGrey",
    "arrowGreen",
    "arrowPurple",
    "arrowBlue",
    "arrowOrange",
    "arrowStage0",
)

DEFS_XML = """<defs>
                    <marker id="{p}arrowGrey" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#8a7e6e" stroke-width="1"/></marker>
                    <marker id="{p}arrowGreen" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#047857" stroke-width="1"/></marker>
                    <marker id="{p}arrowPurple" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#6d28d9" stroke-width="1"/></marker>
                    <marker id="{p}arrowBlue" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#2563eb" stroke-width="1"/></marker>
                    <marker id="{p}arrowOrange" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#c2410c" stroke-width="1"/></marker>
                    <marker id="{p}arrowStage0" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="none" stroke="#78716c" stroke-width="1"/></marker>
                </defs>"""


def prefix_svg_markers(svg: str, prefix: str) -> str:
    p = prefix
    for kid in KNOWN_MARKER_IDS:
        svg = svg.replace(f'id="{kid}"', f'id="{p}{kid}"')
        svg = svg.replace(f"url(#{kid})", f"url(#{p}{kid})")
    return svg


def inject_defs_if_missing(svg: str, prefix: str) -> str:
    if "<defs>" in svg:
        return svg
    p = prefix
    defs = DEFS_XML.format(p=p)
    return re.sub(r"(<svg[^>]*>)", r"\1\n                " + defs + "\n", svg, count=1)


def main() -> None:
    src = (DOCS / "model-training-stages.html").read_text()
    pat = re.compile(
        r'<div class="diagram-wrap">\s*(<svg[\s\S]*?</svg>)\s*(?:<p class="diagram-caption">(.*?)</p>)?\s*</div>',
        re.MULTILINE,
    )
    matches = list(pat.finditer(src))
    assert len(matches) == 5, f"expected 5 diagrams, got {len(matches)}"

    prefixes = ("mtT_", "mt0_", "mtA_", "mtB_", "mtE_")
    out: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        svg = m.group(1)
        cap = m.group(2) or ""
        pr = prefixes[i]
        svg = prefix_svg_markers(svg, pr)
        svg = inject_defs_if_missing(svg, pr)
        out.append((pr, svg, cap.strip()))

    # Summary block (last stage div)
    sum_pat = re.search(
        r'<div class="diagram-wrap" style="background: var\(--bg-warm\);">\s*(<svg[\s\S]*?</svg>)\s*</div>',
        src,
    )
    assert sum_pat, "summary svg not found"
    sum_svg = prefix_svg_markers(sum_pat.group(1), "mtSum_")
    sum_svg = inject_defs_if_missing(sum_svg, "mtSum_")

    snippet = DOCS / "_model_training_diagrams_generated.html"
    parts = []
    for pr, svg, cap in out:
        parts.append(f"<!-- {pr} -->\n{svg}\n<p class=\"diagram-caption\">{cap}</p>")
    parts.append("<!-- mtSum_ -->\n" + sum_svg + '\n<p class="diagram-caption">Stage progression: each stage must beat the previous on the same metrics</p>')
    snippet.write_text("\n\n".join(parts))
    print("Wrote", snippet, "lines", len(snippet.read_text().splitlines()))


if __name__ == "__main__":
    main()
