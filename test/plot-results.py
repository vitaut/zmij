#!/usr/bin/env python3
# Render benchmark CSVs (as produced by benchmark-all-commits.py) as a
# self-contained, zoomable Plotly HTML page.
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path


# Metrics to plot, in display order. Each entry is
# (csv_column, y_axis_label, scale_factor) where scale_factor converts the
# raw CSV value into the displayed unit.
METRICS = [
    ("time_per_double_ns", "Time per double (ns)", 1e9),
    ("throughput", "Throughput (M doubles/s)", 1e-6),
    ("real_time", "Real time per iteration (ms)", 1e-6),
    ("cpu_time", "CPU time per iteration (ms)", 1e-6),
]


def load(csv_path):
    """Read the CSV and return (commits_in_order, series, header_index).

    `commits_in_order` is the list of commit shas in CSV order, deduplicated
    while preserving order. `series[name]` is a dict mapping commit sha to a
    dict of metric_name -> float for that benchmark `name`.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    commits_in_order = []
    seen = set()
    for r in rows:
        sha = r["commit"]
        if sha not in seen:
            seen.add(sha)
            commits_in_order.append(sha)

    series = defaultdict(dict)
    for r in rows:
        name = r["name"]
        if name == "__FAILED__":
            continue
        out = {}
        for col, _label, _scale in METRICS:
            v = r.get(col, "")
            try:
                out[col] = float(v) if v != "" else None
            except ValueError:
                out[col] = None
        series[name][r["commit"]] = out

    return commits_in_order, series


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, sans-serif; background: #fafafa; }}
  header {{ padding: 12px 20px; background: #fff;
            border-bottom: 1px solid #ddd; }}
  header h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
  header .meta {{ color: #666; font-size: 12px; margin-top: 4px; }}
  .controls {{ padding: 8px 20px; background: #fff;
               border-bottom: 1px solid #eee; display: flex; gap: 16px;
               align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 13px; color: #333; }}
  #plot {{ width: 100vw; height: calc(100vh - 110px); }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>
<div class="controls">
  <label>Metric:
    <select id="metric"></select>
  </label>
  <label><input type="checkbox" id="logy"> Log Y</label>
  <label><input type="checkbox" id="markers" checked> Show markers</label>
</div>
<div id="plot"></div>
<script>
const DATA = {data_json};
const METRICS = {metrics_json};
const COMMITS = {commits_json};

const select = document.getElementById('metric');
for (const m of METRICS) {{
  const opt = document.createElement('option');
  opt.value = m.col;
  opt.textContent = m.label;
  select.appendChild(opt);
}}

function buildTraces(metricCol, scale, showMarkers) {{
  const traces = [];
  for (const name of Object.keys(DATA)) {{
    const x = [], y = [], text = [];
    for (let i = 0; i < COMMITS.length; ++i) {{
      const sha = COMMITS[i];
      const v = DATA[name][sha] && DATA[name][sha][metricCol];
      if (v === null || v === undefined) continue;
      x.push(i);
      y.push(v * scale);
      text.push(sha.slice(0, 12));
    }}
    traces.push({{
      type: 'scatter',
      mode: showMarkers ? 'lines+markers' : 'lines',
      name,
      x, y, text,
      hovertemplate:
        '<b>%{{fullData.name}}</b><br>' +
        'commit %{{text}} (#%{{x}})<br>' +
        '%{{y:.4g}}<extra></extra>',
      line: {{ width: 1.5 }},
      marker: {{ size: 4 }},
    }});
  }}
  return traces;
}}

function render() {{
  const m = METRICS.find(x => x.col === select.value) || METRICS[0];
  const logy = document.getElementById('logy').checked;
  const showMarkers = document.getElementById('markers').checked;
  const traces = buildTraces(m.col, m.scale, showMarkers);
  const layout = {{
    margin: {{ l: 70, r: 20, t: 10, b: 50 }},
    xaxis: {{
      title: 'Commit (chronological index)',
      rangeslider: {{ visible: true, thickness: 0.05 }},
      showspikes: true, spikemode: 'across', spikesnap: 'cursor',
      spikedash: 'dot', spikethickness: 1,
    }},
    yaxis: {{
      title: m.label,
      type: logy ? 'log' : 'linear',
      showspikes: true, spikemode: 'across', spikesnap: 'cursor',
      spikedash: 'dot', spikethickness: 1,
    }},
    hovermode: 'x unified',
    legend: {{ orientation: 'h', y: 1.05 }},
    dragmode: 'pan',
  }};
  const config = {{
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    modeBarButtonsToAdd: ['hoverclosest', 'hovercompare'],
  }};
  Plotly.react('plot', traces, layout, config);
}}

select.addEventListener('change', render);
document.getElementById('logy').addEventListener('change', render);
document.getElementById('markers').addEventListener('change', render);
window.addEventListener('resize', () => Plotly.Plots.resize('plot'));
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Convert benchmark CSV into a zoomable HTML plot."
    )
    ap.add_argument("csv", nargs="?", default="results.csv",
                    help="input CSV (default: results.csv)")
    ap.add_argument("-o", "--output",
                    help="output HTML path (default: <csv>.html)")
    ap.add_argument("-t", "--title", default=None,
                    help="page title (default: derived from CSV name)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"{csv_path}: not found")

    out_path = Path(args.output) if args.output else csv_path.with_suffix(
        ".html")
    title = args.title or f"Benchmark — {csv_path.name}"

    commits, series = load(csv_path)
    if not series:
        sys.exit(f"{csv_path}: no successful benchmark rows to plot")

    n_total = len(commits)
    n_ok = sum(1 for sha in commits if any(sha in s for s in series.values()))
    n_failed = n_total - n_ok
    meta = (f"{n_total} commits, {n_ok} succeeded, {n_failed} failed; "
            f"{len(series)} benchmark methods: "
            + ", ".join(sorted(series.keys())))

    metrics_payload = [
        {"col": col, "label": label, "scale": scale}
        for col, label, scale in METRICS
    ]

    html_text = HTML_TEMPLATE.format(
        title=html.escape(title),
        meta=html.escape(meta),
        data_json=json.dumps(series, separators=(",", ":")),
        metrics_json=json.dumps(metrics_payload),
        commits_json=json.dumps(commits),
    )
    out_path.write_text(html_text)
    print(f"Wrote {out_path} ({n_ok}/{n_total} commits plotted)")


if __name__ == "__main__":
    main()
