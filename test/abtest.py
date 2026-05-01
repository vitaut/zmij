#!/usr/bin/env python3
# A/B benchmark driver for https://github.com/vitaut/zmij/.
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).
"""Compare two versions of zmij.{cc,h} using paired, ABBA-interleaved trials
with host sleep/idle prevention where available. With one ref, compares the
working tree against it; with two, compares the second commit against the
first.

    Usage: python3 test/abtest.py [--float] [<base-ref> [<test-ref>]]
                                # defaults: base=HEAD, test=working tree

Exits non-zero if any benchmark regresses with p < 0.01.
"""

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent

TRIALS = 8        # per variant; total runs = 2*TRIALS in ABBA order
REPS = 5          # --benchmark_repetitions per run
MIN_TIME = 0.5    # --benchmark_min_time seconds per repetition
FILTER = ""       # regex like "^zmij$" to narrow benchmarks; "" = all


def nosleep_prefix():
    """Command prefix that keeps the host awake for the duration of a
    wrapped command. Returns [] on platforms without a known inhibitor,
    when the tool isn't installed, or when policy denies it; the bench
    still runs, it just isn't protected from idle sleep."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return ["caffeinate", "-dimsu"]
    if sys.platform.startswith("linux") and shutil.which("systemd-inhibit"):
        prefix = ["systemd-inhibit", "--what=idle:sleep",
                  "--who=abtest", "--why=benchmark"]
        # polkit denies the inhibit-block-{idle,sleep} actions from
        # non-active sessions (e.g. SSH) without interactive auth, so probe
        # once with a no-op command and fall back silently if it fails.
        ok = subprocess.run([*prefix, "true"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
        if ok:
            return prefix
    return []


NOSLEEP = nosleep_prefix()


def run(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    if r.returncode != 0:
        sys.exit(f"$ {' '.join(map(str, cmd))}\n{r.stdout}")
    return r.stdout


def build(label, src, build_dir, deps, zcc, zh, target):
    (src / "zmij.cc").write_text(zcc)
    (src / "zmij.h").write_text(zh)
    args = ["cmake", "-S", src, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DFETCHCONTENT_BASE_DIR={deps}"]
    if (deps / "googlebenchmark-src" / "CMakeLists.txt").exists():
        args.append("-DFETCHCONTENT_FULLY_DISCONNECTED=TRUE")
    print(f"[build:{label}]", file=sys.stderr)
    run(args)
    run(["cmake", "--build", build_dir, "-j", "--target", target])
    return build_dir / "test" / target


def bench_run(exe, json_path):
    cmd = [*NOSLEEP, str(exe),
           f"--benchmark_repetitions={REPS}",
           f"--benchmark_min_time={MIN_TIME}s",
           "--benchmark_enable_random_interleaving=true",
           "--benchmark_report_aggregates_only=false",
           f"--json-out={json_path}"]
    if FILTER:
        cmd.append(f"--benchmark_filter={FILTER}")
    run(cmd)
    out = {}
    for b in json.loads(json_path.read_text()).get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        # The harness exposes per-element time as a Time/double or Time/float
        # counter (kIsIterationInvariantRate | kInvert => seconds per
        # element). Convert to ns. Fall back to real_time (per outer
        # iteration, much larger) if the counter is missing.
        t = b.get("Time/double") or b.get("Time/float")
        t = float(t) * 1e9 if t is not None else float(b["real_time"])
        out.setdefault(b["name"], []).append(t)
    return out


def welch(base, test):
    """Welch's statistic with a normal-approximation tail (erfc + 1.96
    instead of Student's t with Welch-Satterthwaite df, which would need
    a stdlib regularized-incomplete-beta implementation). At n=40 per
    side, this shifts the α=0.01 critical value by ~2.5% (Z=2.576 vs
    t78=2.640), well inside the headroom from using α=0.01 vs the
    textbook 0.05. Don't lower TRIALS below ~5 without revisiting.
    Returns (mean_b, mean_t, pct, ci_lo_pct, ci_hi_pct, p)."""
    mb, mt = mean(base), mean(test)
    if len(base) < 2 or len(test) < 2 or mb == 0:
        return mb, mt, 0.0, 0.0, 0.0, 1.0
    vb = sum((x - mb) ** 2 for x in base) / (len(base) - 1)
    vt = sum((x - mt) ** 2 for x in test) / (len(test) - 1)
    se = math.sqrt(vb / len(base) + vt / len(test))
    pct = (mt - mb) / mb * 100
    if se == 0:
        return mb, mt, pct, pct, pct, (0.0 if mt != mb else 1.0)
    ci = 1.96 * se / mb * 100
    p = math.erfc(abs(mt - mb) / se / math.sqrt(2))
    return mb, mt, pct, pct - ci, pct + ci, p


def read_ref(ref):
    """Resolve `ref` and return (sha, zmij.cc, zmij.h) at that commit."""
    sha = run(["git", "rev-parse", "--verify", ref], cwd=REPO).strip()
    return (sha,
            run(["git", "show", f"{sha}:zmij.cc"], cwd=REPO),
            run(["git", "show", f"{sha}:zmij.h"], cwd=REPO))


def main():
    usage = (f"Usage: {sys.argv[0]} [--float] [<base-ref> [<test-ref>]]"
             "   # defaults: base=HEAD, test=working tree")
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        sys.exit(usage)
    target = "ftoa-benchmark" if "--float" in args else "dtoa-benchmark"
    args = [a for a in args if a != "--float"]
    if len(args) > 2:
        sys.exit(usage)
    base_ref = args[0] if args else "HEAD"
    test_ref = args[1] if len(args) > 1 else None

    base_sha, base_cc, base_h = read_ref(base_ref)
    if test_ref is None:
        test_label = "working tree"
        test_cc = (REPO / "zmij.cc").read_text()
        test_h = (REPO / "zmij.h").read_text()
    else:
        test_sha, test_cc, test_h = read_ref(test_ref)
        test_label = test_sha[:12]
    print(f"comparing base={base_sha[:12]} vs test={test_label}",
          file=sys.stderr)
    if (test_cc, test_h) == (base_cc, base_h):
        print("note: test version of zmij.{cc,h} is identical to base.",
              file=sys.stderr)

    deps = Path.home() / ".cache" / "zmij_bench_deps"
    deps.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zmij_ab_") as t:
        tmp = Path(t)
        src = tmp / "src"
        for rel in run(["git", "ls-files"], cwd=REPO).splitlines():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, src / rel)

        base_exe = build("base", src, tmp / "build/base", deps, base_cc, base_h, target)
        test_exe = build("test", src, tmp / "build/test", deps, test_cc, test_h, target)

        # ABBA-interleaved trials: any monotonic drift (thermal, battery,
        # background work) is split symmetrically between the two variants.
        order = ("A", "B", "B", "A") * ((TRIALS + 1) // 2)
        order = order[: 2 * TRIALS]
        base_d, test_d = {}, {}
        for i, slot in enumerate(order, 1):
            exe, sink = (base_exe, base_d) if slot == "A" else (test_exe, test_d)
            print(f"[{i}/{len(order)}] {slot}", file=sys.stderr)
            for name, samples in bench_run(exe, tmp / f"b{i}.json").items():
                sink.setdefault(name, []).extend(samples)

    rows = []
    for name in sorted(set(base_d) & set(test_d)):
        mb, mt, pct, lo, hi, p = welch(base_d[name], test_d[name])
        # The winner takes it all (else NOISE).
        if p < 0.01 and hi < 0:
            v = "FASTER"
        elif p < 0.01 and lo > 0:
            v = "SLOWER"
        else:
            v = "NOISE"
        rows.append((name, mb, mt, pct, lo, hi, p, v))
    rows.sort(key=lambda r: -abs(r[3]))

    name_w = max(9, max((len(r[0]) for r in rows), default=9))
    print(f"\n{'benchmark':<{name_w}}  {'base':>10} {'test':>10}  {'Δ%':>6}  "
          f"{'95% CI':>13}  {'p':>7}  verdict")
    for name, mb, mt, pct, lo, hi, p, v in rows:
        print(f"{name:<{name_w}}  {mb:>8.2f}ns {mt:>8.2f}ns  "
              f"{pct:+5.1f}%  [{lo:+5.1f},{hi:+5.1f}]  {p:>7.2g}  {v}")

    return 1 if any(r[7] == "SLOWER" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
