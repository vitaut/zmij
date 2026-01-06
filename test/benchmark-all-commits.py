#!/usr/bin/env python3
# Benchmark for https://github.com/vitaut/zmij/.
# Copyright (c) 2025 - present, Victor Zverovich
# Distributed under the MIT license (see LICENSE).

import csv
import os
import subprocess
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
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stdout}")
    return p.stdout


def find_benchmark_exe(build: str):
    for p in build.rglob("*"):
        if p.is_file() and os.access(p, os.X_OK):
            name = p.name.lower()
            if "bench" in name:
                return p
    raise RuntimeError("benchmark executable not found")


def benchmark_commit(sha: str, workdir: Path, writer: csv.writer):
    run(["git", "checkout", "-q", sha], cwd=workdir)

    if not os.path.exists(workdir / "zmij.h"):
        print(f"Skipping commit")
        return

    benchmark = str(workdir / "benchmark")
    run(["c++", "-O3", "-DNDEBUG", "-std=c++20",
         "-I", str(workdir), "-I", ".",
         "benchmark.cc", "zmij-benchmark.cc", str(workdir / "zmij.cc"),
         "fmt/format.cc", "dragonbox/dragonbox_to_chars.cpp",
         "-o", benchmark])

    output = run([benchmark])

    for line in output.splitlines():
        if ":" not in line or "ns" not in line:
            continue
        # Expected: name: Xns (<min>ns - <max>ns) [flags]
        try:
            name, rest = line.split(":", 1)
            agg = rest.split("ns")[0].strip()
            rng = rest.split("(")[1].split(")")[0]
            mn, mx = [x.strip().replace("ns", "") for x in rng.split("-")]
            flags = rest.split(")")[-1].strip()
            writer.writerow([sha, name.strip(), agg, mn, mx, flags])
        except Exception:
            pass

def main():
    with tempfile.TemporaryDirectory(prefix="zmij_bench_") as tmp:
        workdir = Path(tmp)

        print("Cloning repository...")
        run(["git", "clone", "https://github.com/vitaut/zmij.git",
             str(workdir)])

        commits = run(
            ["git", "rev-list", "--reverse", "HEAD"],
            cwd=workdir
        ).split()

        csv_path = Path("results.csv")
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["commit", "method", "aggregated_ns",
                                "min_ns", "max_ns", "flags"])

            for i, sha in enumerate(commits, 1):
                print(f"[{i}/{len(commits)}] {sha[:12]}")
                try:
                    benchmark_commit(sha, workdir, writer)
                    f.flush()
                except Exception as e:
                    print(f"  FAILED: {e}")
                    writer.writerow([sha, "__FAILED__", "", "", "", str(e)])

        print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
