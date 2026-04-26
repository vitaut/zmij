#!/usr/bin/env python3
# Benchmark for https://github.com/vitaut/zmij/.
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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


def historical_file(repo: Path, sha: str, path: str):
    p = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.stdout if p.returncode == 0 else None


def benchmark_commit(sha: str, repo: Path, src: Path, build: Path,
                     deps: Path, writer: csv.writer):
    zcc = historical_file(repo, sha, "zmij.cc")
    zh = historical_file(repo, sha, "zmij.h")
    if zcc is None or zh is None:
        raise RuntimeError("zmij.cc/zmij.h missing at this commit")

    (src / "zmij.cc").write_text(zcc)
    (src / "zmij.h").write_text(zh)

    run(["cmake", "-S", src, "-B", build,
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DFETCHCONTENT_BASE_DIR={deps}"])
    run(["cmake", "--build", build, "-j", "--target", "dtoa-benchmark"])

    exe = build / "test" / "dtoa-benchmark"
    # The binary uses a custom display reporter, so --benchmark_format=json
    # is ineffective. Use the harness's --json-out flag which installs a
    # JSONReporter writing to the given file.
    out_json = build / "bench.json"
    if out_json.exists():
        out_json.unlink()
    run([str(exe), f"--json-out={out_json}"])
    data = json.loads(out_json.read_text())

    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        time_per_double = b.get("Time per double", b.get("Time/double", ""))
        throughput = b.get("Throughput", b.get("Speed", ""))
        writer.writerow([
            sha,
            b.get("name", ""),
            b.get("real_time", ""),
            b.get("cpu_time", ""),
            b.get("time_unit", ""),
            b.get("iterations", ""),
            time_per_double,
            throughput,
        ])


def main():
    with tempfile.TemporaryDirectory(prefix="zmij_bench_") as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        build = tmp / "build"

        # Cache fetched dependencies (e.g. googlebenchmark) across runs to
        # avoid re-downloading and re-building them every time.
        deps = Path.home() / ".cache" / "zmij_bench_deps"
        deps.mkdir(parents=True, exist_ok=True)

        print("Copying tracked files from current workspace...")
        tracked = run(["git", "ls-files"], cwd=REPO).splitlines()
        for rel in tracked:
            dst = src / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)

        # 7a60b2667c52c328c574fbba0d08e75808e74d2a is a known good commit
        # with all improvements and regressions before it accounted for.
        commits = run(
            ["git", "rev-list", "--reverse", "--topo-order",
             "7a60b2667c52c328c574fbba0d08e75808e74d2a..HEAD"],
            cwd=REPO,
        ).split()

        if not commits:
            print("No commits to benchmark.")
            return

        csv_path = Path("results.csv")
        log_path = Path("results-errors.log")
        log_path.write_text("")
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["commit", "name", "real_time", "cpu_time",
                                 "time_unit", "iterations",
                                 "time_per_double_ns", "throughput"])

            for i, sha in enumerate(commits, 1):
                print(f"[{i}/{len(commits)}] {sha[:12]}")
                try:
                    benchmark_commit(sha, REPO, src, build, deps, writer)
                    f.flush()
                except Exception as e:
                    msg = str(e)
                    first = msg.splitlines()[0] if msg else ""
                    print(f"  FAILED: {first}", file=sys.stderr)
                    with log_path.open("a") as lf:
                        lf.write(f"=== {sha} ===\n{msg}\n\n")
                    writer.writerow([sha, "__FAILED__", "", "", "", "", "",
                                     ""])
                    f.flush()

        print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
