#!/usr/bin/env python3
# Render dtoa-benchmark diagrams for Żmij as standalone SVGs by reusing the
# upstream report renderer (fmtlib/dtoa-benchmark's generate-html.py).
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).

"""Render dtoa-benchmark diagrams for Żmij by reusing the upstream renderer.

Rather than reimplementing the charts, this fetches ``generate-html.py`` and the
result JSON straight from fmtlib/dtoa-benchmark and calls its server-side SVG
renderers (``render_bar_chart`` / ``render_line_chart``). The output is therefore
pixel-identical to https://fmtlib.github.io/dtoa-benchmark/results/ and stays in
sync automatically when the benchmark's styling changes.

Each run produces two standalone, theme-independent ``.svg`` files:
  * ``<machine>-mean.svg``      — mean conversion time per method
  * ``<machine>-by-digits.svg`` — time-per-double vs. significant-digit count

Usage:
    python3 test/render-dtoa-charts.py                     # default runs below
    python3 test/render-dtoa-charts.py <slug> [<slug> ...] # specific stems
    python3 test/render-dtoa-charts.py -o out/ <slug>      # choose output dir

By default the SVGs are written to ``test/charts/`` (the location referenced by
the README) using data downloaded from the benchmark's GitHub Pages site.
"""

import argparse
import re
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

# The generator lives at the repo root (HEAD = default branch); results are
# published to GitHub Pages.
GEN_URL = "https://raw.githubusercontent.com/fmtlib/dtoa-benchmark/HEAD/generate-html.py"
RESULTS_URL = "https://fmtlib.github.io/dtoa-benchmark/results/{slug}.json"

# Where the README expects the charts; also the default output directory.
CHARTS_DIR = Path(__file__).resolve().parent / "charts"

DEFAULT_SLUGS = [
    "apple-m5-max_macos_clang21.0_ab145b9",
    "epyc-7c13_linux_gcc13.3_ee50fc8",
]


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def load_generator():
    """Download dtoa-benchmark's generate-html.py and load it as a module.

    The module's ``__main__`` guard keeps its CLI from running on import.
    """
    mod = types.ModuleType("dtoa_generate_html")
    mod.__file__ = GEN_URL
    exec(compile(_fetch(GEN_URL).decode(), GEN_URL, "exec"), mod.__dict__)
    return mod


def _light_css(css: str) -> str:
    """Drop ``@media (prefers-color-scheme: dark)`` blocks so the SVG always
    renders in the light palette.

    Inside an ``<img>``-embedded SVG, ``prefers-color-scheme`` follows the
    viewer's OS rather than GitHub's theme toggle, which would leave an opaque
    white card mismatched against a dark page. Forcing the light ``:root``
    variables makes the chart render identically everywhere.
    """
    marker = "@media (prefers-color-scheme: dark)"
    out = []
    i = 0
    while True:
        j = css.find(marker, i)
        if j == -1:
            out.append(css[i:])
            break
        out.append(css[i:j])
        depth = 0
        p = css.index("{", j)
        while p < len(css):
            if css[p] == "{":
                depth += 1
            elif css[p] == "}":
                depth -= 1
                if depth == 0:
                    p += 1
                    break
            p += 1
        i = p
    return "".join(out)


def _svg_legend(methods, colors, gen, width, top):
    """Render a color-swatch legend as SVG rows below the plot.

    Upstream's legend is HTML (``render_legend``), so it's absent from the
    isolated ``<svg>``. We recreate it in SVG so the line chart is
    self-explanatory as a standalone image. Returns (markup, extra_height).
    """
    n = len(methods)
    n_cols = min(4, n) or 1
    col_w = width / n_cols
    row_h = 22
    sw = 12
    pad_x = 16
    rows = (n + n_cols - 1) // n_cols

    parts = ['<g class="legend-svg">']
    for i, method in enumerate(methods):
        col, row = i % n_cols, i // n_cols
        x = pad_x + col * col_w
        y = top + row * row_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{sw}" height="{sw}" '
            f'rx="2" ry="2" fill="{colors[method]}"/>'
        )
        parts.append(
            f'<text x="{x + sw + 6:.1f}" y="{y + sw / 2:.1f}" '
            f'dominant-baseline="middle" class="lbl">{gen._esc(method)}</text>'
        )
    parts.append("</g>")
    return "".join(parts), rows * row_h + 8


def standalone_svg(fragment: str, gen, legend=None) -> str:
    """Turn an upstream chart HTML fragment into a self-contained SVG file.

    The renderers return ``<div class="chart-wrap"><svg ...>...</svg>...</div>``;
    we lift out the ``<svg>`` and inject the chart CSS (so classes like
    ``.grid`` resolve) plus a background rect, keeping upstream markup verbatim.
    When ``legend`` (methods, colors) is given, an SVG legend is appended below
    the chart and the viewBox is grown to fit it.
    """
    svg = re.search(r"<svg\b.*?</svg>", fragment, re.S).group(0)
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)

    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    width, height = float(vb.group(1)), float(vb.group(2))

    legend_markup = ""
    if legend is not None:
        methods, colors = legend
        legend_markup, extra = _svg_legend(methods, colors, gen, width,
                                            top=height + 16)
        new_height = height + 16 + extra
        svg = svg.replace(vb.group(0),
                          f'viewBox="0 0 {width:g} {new_height:g}"', 1)

    style = f"<style>{_light_css(gen.PAGE_CSS)}</style>"
    bg = '<rect width="100%" height="100%" fill="var(--bg)"/>'
    open_tag_end = svg.index(">") + 1
    close = svg.rindex("</svg>")
    return (svg[:open_tag_end] + style + bg
            + svg[open_tag_end:close] + legend_markup + svg[close:])


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*",
                        help="result stems (default: latest macOS + Linux)")
    parser.add_argument("-o", "--out-dir", type=Path, default=CHARTS_DIR,
                        help="output directory (default: test/charts)")
    args = parser.parse_args(argv)

    gen = load_generator()
    slugs = args.slugs or DEFAULT_SLUGS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for slug in slugs:
            src = Path(tmp) / f"{slug}.json"
            src.write_bytes(_fetch(RESULTS_URL.format(slug=slug)))

            bucket = gen.aggregate(gen.load_json(src))
            methods = [m for m in bucket["methods"] if m != gen.BASELINE_METHOD]
            colors = gen._palette(bucket["methods"])

            bar = gen.render_bar_chart(methods, bucket["mean"], colors)
            line = gen.render_line_chart(
                methods, bucket["digits"], bucket["times"], colors,
                baseline_method=(gen.BASELINE_METHOD
                                 if gen.BASELINE_METHOD in bucket["methods"]
                                 else None),
            )

            stem = slug.split("_")[0]
            (args.out_dir / f"{stem}-mean.svg").write_text(
                standalone_svg(bar, gen))
            (args.out_dir / f"{stem}-by-digits.svg").write_text(
                standalone_svg(line, gen, legend=(methods, colors)))
            print(f"wrote {stem}-mean.svg and {stem}-by-digits.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
