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


def run(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{p.stdout}")
    return p.stdout


def benchmark_commit(sha: str, workdir: Path, build: Path, deps: Path,
                     writer: csv.writer):
    run(["git", "checkout", "-q", sha], cwd=workdir)

    if not (workdir / "test" / "benchmark.cc").exists():
        print("  Skipping commit: test/benchmark.cc not found")
        return

    # Reset the build dir between commits to avoid stale state when
    # CMakeLists.txt changes, but reuse cached fetched dependencies.
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    run(["cmake", "-S", str(workdir), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DFETCHCONTENT_BASE_DIR={deps}"])
    run(["cmake", "--build", str(build), "-j",
         "--target", "dtoa-benchmark"])

    exe = build / "test" / "dtoa-benchmark"
    output = run([str(exe), "--benchmark_format=json"])
    data = json.loads(output)

    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":
            continue
        name = b.get("name", "")
        real_time = b.get("real_time", "")
        cpu_time = b.get("cpu_time", "")
        iterations = b.get("iterations", "")
        time_unit = b.get("time_unit", "")
        # Common counters set by run_dtoa / run_dtoa_mixed.
        time_per_double = b.get("Time per double", b.get("Time/double", ""))
        throughput = b.get("Throughput", b.get("Speed", ""))
        writer.writerow([sha, name, real_time, cpu_time, time_unit,
                         iterations, time_per_double, throughput])


def main():
    with tempfile.TemporaryDirectory(prefix="zmij_bench_") as tmp:
        tmp = Path(tmp)
        workdir = tmp / "src"
        build = tmp / "build"
        # Cache fetched dependencies (e.g. googlebenchmark) across commits to
        # avoid re-downloading and re-building them every time.
        deps = Path.home() / ".cache" / "zmij_bench_deps"
        deps.mkdir(parents=True, exist_ok=True)

        print("Cloning repository...")
        run(["git", "clone", "https://github.com/vitaut/zmij.git",
             str(workdir)])

        # 7a60b2667c52c328c574fbba0d08e75808e74d2a is a known good commit
        # with all improvements and regressions before it accounted for.
        commits = run(
            ["git", "rev-list", "--reverse", "--topo-order",
             "7a60b2667c52c328c574fbba0d08e75808e74d2a..HEAD"],
            cwd=workdir
        ).split()

        if not commits:
            print("No commits to benchmark.")
            return

        csv_path = Path("results.csv")
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["commit", "name", "real_time", "cpu_time",
                                 "time_unit", "iterations",
                                 "time_per_double_ns", "throughput_per_s"])

            for i, sha in enumerate(commits, 1):
                print(f"[{i}/{len(commits)}] {sha[:12]}")
                try:
                    benchmark_commit(sha, workdir, build, deps, writer)
                    f.flush()
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
                    writer.writerow([sha, "__FAILED__", "", "", "", "", "",
                                     str(e)])
                    f.flush()

        print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
