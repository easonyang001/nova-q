"""Parallel version of scripts/run_benchmark_suite.py: same commands, same CSV columns, N benchmarks at a time.

    python scripts/experiments/run_suite_parallel.py --seed 42 --name full_pipeline_local --workers 8
    python scripts/experiments/run_suite_parallel.py --seed 42 --name full_pipeline_local --workers 8 --resume

The original script runs one benchmark after another (about 100 x 5-10 min). Each benchmark is an
independent run_single.py process, so they can share the local vLLM server concurrently. Results are
matched to runs.jsonl by run_id (parsed from each run's log), so parallel appends cannot be mixed up.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark_suite import CSV_FIELDS, _build_command, _collect_log_metrics, _select_benchmarks  # noqa: E402


def run_one(bench_dir: Path, seed: int, config: str, log_root: Path) -> dict:
    cmd, _ = _build_command(ROOT, bench_dir, seed, config)
    log_path = log_root / f"{bench_dir.name}_seed{seed}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"run_id=(\S+)", text)
    result = {}
    if m:
        for line in (ROOT / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("run_id") == m.group(1):
                result = item
    metrics = _collect_log_metrics(log_path)
    if not result:
        return {f: "" for f in CSV_FIELDS} | {"benchmark": bench_dir.name, "seed": seed, "status": "script_failed"}
    ver = result.get("verification") or {}
    return {
        "benchmark": bench_dir.name, "seed": seed, "status": result.get("status", ""),
        "tests_passed": int(ver.get("passed", 0) or 0), "total_tests": int(ver.get("total", 0) or 0),
        "wall_time_s": f"{float(result.get('wall_time_s') or 0):.1f}",
        "tokens_used": int(result.get("tokens_used") or 0),
        "total_llm_cost": f"{float(result.get('cost_used') or 0):.4f}",
        "planner_retries": max(0, metrics["planner_attempts"] - 1),
        "formulizer_retries": max(0, metrics["formulizer_attempts"] - 1),
        "coder_retries": max(0, metrics["coder_attempts"] - 1),
        "debugger_retries": max(0, metrics["debugger_attempts"] - 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benchmarks", default=None)
    ap.add_argument("--skip", default="custom")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default="full_pipeline_local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip benchmarks already in the CSV")
    args = ap.parse_args()

    out = ROOT / "results" / f"{args.name}_{args.seed}.csv"
    log_root = ROOT / "results" / "benchmark_suite_logs" / f"{args.name}_{args.seed}"
    log_root.mkdir(parents=True, exist_ok=True)
    benches = _select_benchmarks(ROOT / "benchmarks", args.benchmarks, args.skip)
    done = set()
    if args.resume and out.exists():
        done = {r["benchmark"] for r in csv.DictReader(out.open(encoding="utf-8")) if r["status"] != "script_failed"}
    todo = [b for b in benches if b.name not in done]
    print(f"{len(benches)} benchmarks, {len(done)} already done, {len(todo)} to run with {args.workers} workers", flush=True)

    new_file = not out.exists() or not args.resume
    t0 = time.time()
    with out.open("w" if new_file else "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        futures = {ex.submit(run_one, b, args.seed, args.config, log_root): b for b in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            writer.writerow(row)
            f.flush()
            print(f"[{k}/{len(todo)} {time.time() - t0:.0f}s] {row['benchmark']}: {row['status']} "
                  f"{row['tests_passed']}/{row['total_tests']}", flush=True)
    print(f"Results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
