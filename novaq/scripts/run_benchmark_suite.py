"""Run benchmark folders through scripts/run_single.py and append results to a CSV.

Usage:
    python scripts/run_benchmark_suite.py --seed 42
    python scripts/run_benchmark_suite.py --seed 42 --benchmarks max_cut,knapsack
    python scripts/run_benchmark_suite.py --seed 42 --runs 3

Each run appends one row to the CSV. The header is written only when the file is new.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


CSV_FIELDS = [
    "benchmark",
    "seed",
    "status",
    "tests_passed",
    "total_tests",
    "wall_time_s",
    "tokens_used",
    "total_llm_cost",
    "planner_retries",
    "formulizer_retries",
    "coder_retries",
    "debugger_retries",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmarks through run_single.py.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the first run.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per benchmark.")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark subset.")
    parser.add_argument("--config", default="config.yaml", help="Config file passed to run_single.py.")
    parser.add_argument("--name", default="benchmark_suite",
                        help="CSV base name. Output written to results/<name>_<seed>.csv.")
    parser.add_argument("--output", default=None,
                        help="Explicit CSV output path (overrides --name).")
    parser.add_argument("--skip", default="custom", help="Comma-separated benchmark folders to skip.")
    parser.add_argument("--dry-run", action="store_true", help="List commands without running them.")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    benchmarks = _select_benchmarks(root / "legacy" / "benchmarks", args.benchmarks, args.skip)

    if args.dry_run:
        for bench_dir in benchmarks:
            for run_index in range(args.runs):
                seed = args.seed + run_index
                cmd, _ = _build_command(root, bench_dir, seed, args.config)
                print(f"{bench_dir.name} seed={seed}: " + " ".join(cmd))
        return

    output_path = root / (args.output or f"results/{args.name}_{args.seed}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_root = root / "results" / "benchmark_suite_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results" / "runs.jsonl"

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for bench_dir in benchmarks:
            for run_index in range(args.runs):
                seed = args.seed + run_index
                row = _run_one(
                    root=root,
                    bench_dir=bench_dir,
                    seed=seed,
                    config_name=args.config,
                    log_root=log_root,
                    results_path=results_path,
                )
                writer.writerow(row)
                csv_file.flush()
                _print_row(row)

    print(f"\nResults appended to: {output_path.relative_to(root)}")


def _run_one(
    root: Path,
    bench_dir: Path,
    seed: int,
    config_name: str,
    log_root: Path,
    results_path: Path,
) -> dict:
    benchmark_name = bench_dir.name
    cmd, input_mode = _build_command(root, bench_dir, seed, config_name)
    log_path = log_root / f"{benchmark_name}_seed{seed}.log"
    start_offset = results_path.stat().st_size if results_path.exists() else 0

    print(f"\n{'=' * 60}")
    print(f"  {benchmark_name}  seed={seed}  input={input_mode}")
    print(f"{'=' * 60}")

    _run_command(cmd, cwd=root, log_path=log_path)
    result = _read_appended_result(results_path, start_offset, benchmark_name, seed)
    log_metrics = _collect_log_metrics(log_path)

    if not result:
        return {
            "benchmark": benchmark_name,
            "seed": seed,
            "status": "script_failed",
            "tests_passed": "",
            "total_tests": "",
            "wall_time_s": "",
            "tokens_used": "",
            "total_llm_cost": "",
            "planner_retries": "",
            "formulizer_retries": "",
            "coder_retries": "",
            "debugger_retries": "",
        }

    verification = result.get("verification") or {}
    total = int(verification.get("total", 0) or 0)
    passed = int(verification.get("passed", 0) or 0)

    return {
        "benchmark": benchmark_name,
        "seed": seed,
        "status": result.get("status", ""),
        "tests_passed": passed,
        "total_tests": total,
        "wall_time_s": f"{float(result.get('wall_time_s') or 0):.1f}",
        "tokens_used": int(result.get("tokens_used") or 0),
        "total_llm_cost": f"{float(result.get('cost_used') or 0):.4f}",
        "planner_retries": max(0, log_metrics["planner_attempts"] - 1),
        "formulizer_retries": max(0, log_metrics["formulizer_attempts"] - 1),
        "coder_retries": max(0, log_metrics["coder_attempts"] - 1),
        "debugger_retries": max(0, log_metrics["debugger_attempts"] - 1),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _select_benchmarks(benchmarks_dir: Path, subset: str | None, skip: str) -> list[Path]:
    skip_names = {s.strip() for s in skip.split(",") if s.strip()}
    if subset:
        dirs = [benchmarks_dir / s.strip() for s in subset.split(",") if s.strip()]
    else:
        dirs = sorted(p for p in benchmarks_dir.iterdir() if p.is_dir())
    selected = []
    for d in dirs:
        if d.name in skip_names:
            continue
        if not d.exists():
            raise FileNotFoundError(f"Benchmark folder not found: {d}")
        selected.append(d)
    return selected


def _build_command(root: Path, bench_dir: Path, seed: int, config_name: str) -> tuple[list[str], str]:
    sample_path = bench_dir / "sample_cases.txt"
    cmd = [
        sys.executable, "scripts/run_single.py",
        f"legacy/benchmarks/{bench_dir.name}",
        "--seed", str(seed),
        "--config", config_name,
    ]
    if sample_path.exists():
        cmd.extend(["--input", f"legacy/benchmarks/{bench_dir.name}/sample_cases.txt"])
        return cmd, "sample_cases"
    cases_dir = bench_dir / "cases"
    if cases_dir.exists():
        return cmd, "json_cases"
    raise FileNotFoundError(f"No sample_cases.txt or cases/ found in {bench_dir}")


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return proc.wait()


def _read_appended_result(results_path: Path, start_offset: int, benchmark_name: str, seed: int) -> dict:
    if not results_path.exists():
        return {}
    with results_path.open("rb") as f:
        f.seek(start_offset)
        raw = f.read().decode("utf-8", errors="replace")
    parsed = []
    for line in raw.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("benchmark") == benchmark_name and item.get("seed") == seed:
            parsed.append(item)
    return parsed[-1] if parsed else {}


def _collect_log_metrics(log_path: Path) -> Counter:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    m: Counter = Counter()
    m["planner_attempts"]     = len(re.findall(r"\[Stage 1\] Planner .*attempt", text))
    m["formulizer_attempts"]  = len(re.findall(r"\[Stage 2\] Formulizer .*attempt", text))
    m["coder_attempts"]       = len(re.findall(r"\[Stage 3\] Coder .*attempt", text))
    m["debugger_attempts"]    = text.count("[Stage 2.5] Building test cases")
    return m


def _print_row(row: dict) -> None:
    status = row["status"]
    passed = row["tests_passed"]
    total  = row["total_tests"]
    print(f"  → {status}  {passed}/{total} tests  {row['wall_time_s']}s  {row['tokens_used']} tokens")


if __name__ == "__main__":
    main()
