#!/usr/bin/env python3
# Benchmark for https://github.com/vitaut/zmij/.
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).
"""Walk a commit range and use abtest.py to compare each commit against
its successor with paired, ABBA-interleaved trials. Commits that fail
to build (or whose comparison run aborts) are skipped so the walk keeps
going, but a row is still written to results.csv so gaps in the chain
are explicit rather than silent."""

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# abtest.py lives next to this script; reuse its build/bench/stat helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abtest import REPO, TRIALS, bench_run, build, welch  # noqa: E402


def run(cmd, cwd=None):
    p = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{p.stdout}")
    return p.stdout


def historical_file(sha: str, path: str):
    p = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.stdout if p.returncode == 0 else None


def build_commit(sha: str, src: Path, builds_dir: Path, deps: Path,
                 target: str):
    """Materialize commit `sha`'s zmij.{cc,h} into `src` and build
    `target` into a per-commit build directory. Returns (exe, error)
    where exactly one of the two is None."""
    zcc = historical_file(sha, "zmij.cc")
    zh = historical_file(sha, "zmij.h")
    if zcc is None or zh is None:
        return None, "zmij.cc/zmij.h missing at this commit"
    # abtest.build prints to stderr and calls sys.exit on cmake/build
    # failure (via abtest.run). Catch SystemExit so one bad commit
    # doesn't take down the whole walk.
    try:
        exe = build(sha[:12], src, builds_dir / sha[:12],
                    deps, zcc, zh, target)
    except SystemExit as e:
        return None, str(e)
    return exe, None


def compare(base_exe: Path, test_exe: Path, tmp: Path):
    """ABBA-interleaved trials, mirroring abtest.main's inner loop.
    Returns dicts of name -> samples (ns) for base and test."""
    order = ("A", "B", "B", "A") * ((TRIALS + 1) // 2)
    order = order[: 2 * TRIALS]
    base_d, test_d = {}, {}
    for i, slot in enumerate(order, 1):
        exe, sink = (base_exe, base_d) if slot == "A" else (test_exe, test_d)
        print(f"  [{i}/{len(order)}] {slot}", file=sys.stderr)
        for name, samples in bench_run(exe, tmp / f"b{i}.json").items():
            sink.setdefault(name, []).extend(samples)
    return base_d, test_d


def verdict(p: float, lo: float, hi: float) -> str:
    if p < 0.01 and hi < 0:
        return "FASTER"
    if p < 0.01 and lo > 0:
        return "SLOWER"
    return "NOISE"


def main():
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        sys.exit(f"Usage: {sys.argv[0]} [--float]")
    target = "ftoa-benchmark" if "--float" in args else "dtoa-benchmark"

    # 7a60b2667c52c328c574fbba0d08e75808e74d2a is a known good commit
    # with all improvements and regressions before it accounted for.
    base = "7a60b2667c52c328c574fbba0d08e75808e74d2a"
    base_sha = run(["git", "rev-parse", "--verify", base], cwd=REPO).strip()
    rest = run(["git", "rev-list", "--reverse", "--topo-order",
                f"{base}..HEAD"], cwd=REPO).split()
    # Include `base` itself so the first pair is (base, base+1).
    commits = [base_sha] + rest
    if len(commits) < 2:
        print("Need at least two commits to compare.")
        return 0

    csv_path = Path("results.csv")
    log_path = Path("results-errors.log")
    log_path.write_text("")
    new_file = not csv_path.exists()

    deps = Path.home() / ".cache" / "zmij_bench_deps"
    deps.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zmij_chain_") as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "src"
        builds_dir = tmp / "build"

        print("Copying tracked files from current workspace...")
        for rel in run(["git", "ls-files"], cwd=REPO).splitlines():
            dst = src / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)

        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow([
                    "base", "test", "benchmark",
                    "base_ns", "test_ns",
                    "delta_pct", "ci_lo_pct", "ci_hi_pct",
                    "p", "verdict",
                ])

            def fail_pair(prev: str, cur: str, tag: str, msg: str):
                print(f"  {tag}: {msg.splitlines()[0] if msg else ''}",
                      file=sys.stderr)
                with log_path.open("a") as lf:
                    lf.write(f"=== {prev} -> {cur} ({tag}) ===\n{msg}\n\n")
                writer.writerow([prev, cur, f"__{tag}__",
                                 "", "", "", "", "", "", tag])
                f.flush()

            prev_sha = commits[0]
            print(f"[build] {prev_sha[:12]}")
            prev_exe, prev_err = build_commit(prev_sha, src, builds_dir,
                                              deps, target)
            if prev_err:
                print(f"  build FAILED: {prev_err.splitlines()[0]}",
                      file=sys.stderr)
                with log_path.open("a") as lf:
                    lf.write(f"=== build {prev_sha} ===\n{prev_err}\n\n")

            for i, sha in enumerate(commits[1:], 1):
                print(f"[{i}/{len(commits) - 1}] "
                      f"{prev_sha[:12]} vs {sha[:12]}")
                cur_exe, cur_err = build_commit(sha, src, builds_dir,
                                                deps, target)
                if cur_err:
                    print(f"  build FAILED: {cur_err.splitlines()[0]}",
                          file=sys.stderr)
                    with log_path.open("a") as lf:
                        lf.write(f"=== build {sha} ===\n{cur_err}\n\n")

                if prev_exe is None or cur_exe is None:
                    fail_pair(prev_sha, sha, "BROKEN",
                              prev_err or cur_err or "unknown")
                else:
                    try:
                        base_d, test_d = compare(prev_exe, cur_exe, tmp)
                    except SystemExit as e:
                        fail_pair(prev_sha, sha, "BENCH_FAILED", str(e))
                    else:
                        for name in sorted(set(base_d) & set(test_d)):
                            mb, mt, pct, lo, hi, p = welch(
                                base_d[name], test_d[name])
                            writer.writerow([
                                prev_sha, sha, name,
                                f"{mb:.6f}", f"{mt:.6f}",
                                f"{pct:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                                f"{p:.6g}", verdict(p, lo, hi),
                            ])
                        f.flush()

                # Free the old build directory; we won't need it again.
                if prev_exe is not None:
                    shutil.rmtree(builds_dir / prev_sha[:12],
                                  ignore_errors=True)
                prev_sha, prev_exe, prev_err = sha, cur_exe, cur_err

    print(f"\nResults written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
