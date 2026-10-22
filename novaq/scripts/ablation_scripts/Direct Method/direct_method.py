"""Direct Method ablation — single LLM call directly to build_qubo code.

This is the simplest possible baseline: no Planner, no Formulizer, no Judge,
no Debugger. The raw problem description plus instance structures (no ground
truth) are sent to the LLM in one prompt. The LLM must produce a correct
build_qubo(instance) function from scratch.

Test cases are loaded from past full-pipeline transcript runs — the script
reads results/runs.jsonl to find a prior run's transcripts/<run_id>/cases/
directory for each benchmark. Ground truth is never shown to the LLM.

Usage:
    # All benchmarks (uses past transcripts for cases)
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42

    # Single benchmark
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut

    # Multiple benchmarks
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut,knapsack

    # Custom CSV name
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --name direct_method_run

    # Override model from CLI
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --model qwen/qwen3-coder-next

    # Preview
    python "scripts/ablation_scripts/Direct Method/direct_method.py" --dry-run

Results:
    scripts/ablation_scripts/Direct Method/results/<name>_<seed>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ABL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ABL_DIR / "results"
TRANSCRIPTS_DIR = ABL_DIR / "transcripts"
LOGS_DIR = RESULTS_DIR / "logs"

sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.agents.coder_agent import compile_build_qubo
from qubo_auto.llm_client import LLMClient, TokenBudgetExceeded
from qubo_auto.pipeline import load_config
from qubo_auto.schemas import RunResult
from qubo_auto.verification.test_cases import load_prompt, load_test_cases, TestCase
from qubo_auto.verification.test_runner import run_tests


CSV_FIELDS = [
    "benchmark", "seed", "status",
    "tests_passed", "total_tests",
    "wall_time_s", "tokens_used", "total_llm_cost",
    "failure_modes",
]

ZERO_SHOT_PROMPT = """\
You are an expert in QUBO (Quadratic Unconstrained Binary Optimization).

## Problem Description
{problem_description}

## Your Task
Write a Python function `build_qubo(instance: dict) -> tuple[np.ndarray, float]` that:
- Takes ONE instance dict (keys shown in the examples below)
- Returns (Q, offset) where:
  - Q is a 2D numpy array of shape (n, n), UPPER TRIANGULAR
    (off-diagonal coefficient for x_i*x_j goes in Q[i,j] where i < j; Q[j,i] = 0)
  - offset is a float (constant term)
- The QUBO energy x^T Q x + offset is MINIMIZED at the optimal solution

## Example Instance Structures
{instance_examples}

## Rules
- Import numpy as np inside the function if needed
- Q must be upper triangular: Q[i,j] for i < j holds the full coefficient; Q[j,i] = 0
- n must be computed from the instance data, not hardcoded
- All instance keys must come from the example structures above
- Return ONLY a Python code block — no explanation

```python
import numpy as np

def build_qubo(instance: dict):
    # your implementation
    ...
    return Q, offset
```
"""


# ───────────────────────── helpers ──────────────────────────────────────────

def _rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path)


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", model).strip("_") or "model"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _extract_code(raw: str) -> str:
    import re
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    if "def build_qubo" in cleaned:
        return cleaned
    raise ValueError(f"Could not extract build_qubo from response:\n{raw[:400]}")


# ───────────────────────── case discovery ───────────────────────────────────

def _find_cases_for_benchmark(benchmark_name: str) -> list[TestCase]:
    """Find test cases for a benchmark from past pipeline transcript runs.

    Reads results/runs.jsonl and walks matching run_id transcript dirs.
    Falls back to benchmarks/<name>/cases/ if no transcript cases exist.
    """
    runs_jsonl = ROOT / "results" / "runs.jsonl"
    benchmark_dir = ROOT / "legacy" / "benchmarks" / benchmark_name
    transcripts_root = ROOT / "transcripts"

    # Try transcript directories from past runs (newest first)
    if runs_jsonl.exists():
        runs = []
        for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("benchmark") == benchmark_name:
                    runs.append(r)
            except json.JSONDecodeError:
                continue

        # Prefer successful runs, then any run with cases
        runs.sort(key=lambda r: (r.get("status") != "success", 0))
        for run in runs:
            run_id = run.get("run_id", "")
            cases_dir = transcripts_root / run_id / "cases"
            if cases_dir.exists() and any(cases_dir.glob("*.json")):
                cases = load_test_cases(benchmark_dir, cases_dir=cases_dir)
                if cases:
                    _log(f"  Cases source : transcripts/{run_id}/cases/ "
                         f"({len(cases)} cases, status={run.get('status')})")
                    return cases

    # Fallback: benchmarks/<name>/cases/
    try:
        cases = load_test_cases(benchmark_dir)
        if cases:
            _log(f"  Cases source : benchmarks/{benchmark_name}/cases/ ({len(cases)} cases)")
        return cases
    except FileNotFoundError:
        _log(f"  No cases found for {benchmark_name} — will skip")
        return []


# ───────────────────────── direct-method pipeline ───────────────────────────────

def run_direct_method_pipeline(
    nl_problem: str,
    benchmark_name: str,
    seed: int,
    test_cases: list[TestCase],
    config: dict,
    results_path: Path,
) -> RunResult:
    run_id = f"zs_{benchmark_name}_{seed}_{uuid.uuid4().hex[:8]}"
    start = time.time()
    models = config["models"]
    temps = config["temperatures"]
    verif_cfg = config["verification"]
    limits = config.get("limits", {})
    max_tokens = limits.get("max_tokens_per_run", 50_000)
    use_cache = limits.get("use_cache", True)

    _log(f"=== Direct Method START | run_id={run_id} ===")
    _log(f"  Model : {models['formulizer_agent']}")

    client = LLMClient(
        transcript_dir=TRANSCRIPTS_DIR,
        run_id=run_id,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )

    result = RunResult(
        run_id=run_id,
        benchmark=benchmark_name,
        seed=seed,
        status="failed",
        models_used={"formulizer_agent": models["formulizer_agent"]},
    )

    def _finalize() -> RunResult:
        result.wall_time_s = time.time() - start
        result.tokens_used = client.tokens_used
        result.cost_used = client.cost_used
        _log(f"Finalizing | status={result.status} | tokens={result.tokens_used} | "
             f"cost=${result.cost_used:.4f} | wall={result.wall_time_s:.1f}s")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
        return result

    try:
        if not test_cases:
            _log("  No test cases found — skipping")
            result.error_trace = "No test cases available"
            return _finalize()

        # Build instance examples block — show only instance keys, never ground truth
        instance_examples = ""
        for i, case in enumerate(test_cases[:3]):  # show up to 3 examples
            instance_examples += (
                f"Example {i + 1} (n_variables={case.n_variables}):\n"
                f"{json.dumps(case.instance, indent=2)}\n\n"
            )

        prompt = ZERO_SHOT_PROMPT.format(
            problem_description=nl_problem.strip(),
            instance_examples=instance_examples.strip(),
        )

        _log(f"  Sending direct-method prompt ({len(test_cases)} cases available) ...")
        raw = client.call(
            model=models["formulizer_agent"],
            prompt=prompt,
            temperature=temps["formulizer_agent"],
            step_name="direct_method_code",
        )

        try:
            code = _extract_code(raw)
            build_qubo_fn = compile_build_qubo(code)
            _log("  Code extracted and compiled OK")
        except Exception as e:
            _log(f"  Code extraction/compile FAILED: {e}")
            result.error_trace = traceback.format_exc()
            result.failure_modes = ["coding_error"]
            return _finalize()

        n_cases = len(test_cases)
        _log(f"  Running {n_cases} test cases ...")
        test_results, failure_class = run_tests(
            build_qubo=build_qubo_fn,
            test_cases=test_cases,
        )

        n_pass = sum(1 for r in test_results if r.match)
        result.verification = {"passed": n_pass, "total": n_cases}
        pass_rate = n_pass / n_cases if n_cases else 0
        _log(f"  Tests: {n_pass}/{n_cases} passed ({pass_rate * 100:.0f}%)  "
             f"failure_class={failure_class}")

        for r in test_results:
            if r.error_message:
                _log(f"  Instance {r.instance_id} error: {r.error_message[:200]}")

        if pass_rate >= verif_cfg["pass_threshold"]:
            _log("=== Direct Method SUCCESS ===")
            result.status = "success"
        else:
            result.failure_modes = [failure_class]

        return _finalize()

    except TokenBudgetExceeded as e:
        _log(f"=== Direct Method ABORTED — token budget: {e} ===")
        result.error_trace = str(e)
        result.failure_modes = ["token_budget_exceeded"]
        return _finalize()
    except Exception:
        _log("=== Direct Method EXCEPTION ===")
        result.error_trace = traceback.format_exc()
        return _finalize()


# ───────────────────────── suite driver ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Method ablation runner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default=None,
                        help="Comma-separated benchmark names. Omit to run all.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None,
                        help="Model to use for the direct-method LLM call. Overrides config models.formulizer_agent.")
    parser.add_argument("--name", default="ablation_direct_method",
                        help="CSV base name. Output: results/<name>_<seed>.csv")
    parser.add_argument("--skip", default="custom",
                        help="Comma-separated benchmark names to skip.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    models = config["models"]
    if args.model:
        models["formulizer_agent"] = args.model
    benchmarks = _select_benchmarks(ROOT / "legacy" / "benchmarks", args.benchmarks, args.skip)

    if args.dry_run:
        print("Would run:")
        print(f"  model={models['formulizer_agent']}")
        for bd in benchmarks:
            print(f"  {bd.name}  seed={args.seed}")
        return

    model_slug = _model_slug(models["formulizer_agent"])
    csv_path = RESULTS_DIR / f"{args.name}_{model_slug}_{args.seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_jsonl = RESULTS_DIR / "runs.jsonl"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    print(f"[Direct Method] Ablation run")
    print(f"  Benchmarks : {len(benchmarks)}")
    print(f"  Seed       : {args.seed}")
    print(f"  Model      : {models['formulizer_agent']}")
    print(f"  Config     : {args.config}")
    print(f"  CSV out    : {_rel(csv_path)}")
    print()

    with csv_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for bd in benchmarks:
            row = _run_one(bd, args.seed, config, runs_jsonl)
            writer.writerow(row)
            f_csv.flush()
            _print_row(row)

    print(f"\nDone. CSV: {_rel(csv_path)}")


def _select_benchmarks(bench_dir: Path, subset: str | None, skip: str) -> list[Path]:
    skip_names = {s.strip() for s in skip.split(",") if s.strip()}
    if subset:
        dirs = [bench_dir / s.strip() for s in subset.split(",") if s.strip()]
    else:
        dirs = sorted(p for p in bench_dir.iterdir() if p.is_dir())
    out = []
    for d in dirs:
        if d.name in skip_names:
            continue
        if not d.exists():
            raise FileNotFoundError(f"Benchmark not found: {d}")
        out.append(d)
    return out


def _run_one(benchmark_dir: Path, seed: int, config: dict, runs_jsonl: Path) -> dict:
    bench = benchmark_dir.name
    print(f"\n{'=' * 60}\n  {bench}  seed={seed}\n{'=' * 60}")
    log_path = LOGS_DIR / f"{bench}_seed{seed}.log"

    nl_problem = load_prompt(benchmark_dir)

    _log(f"  Finding test cases for {bench} ...")
    test_cases = _find_cases_for_benchmark(bench)

    log_file = log_path.open("w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)

    try:
        result = run_direct_method_pipeline(
            nl_problem=nl_problem,
            benchmark_name=bench,
            seed=seed,
            test_cases=test_cases,
            config=config,
            results_path=RESULTS_DIR / "runs.jsonl",
        )
    except Exception as exc:
        print(f"[error] {exc}")
        sys.stdout = old_stdout
        log_file.close()
        return _empty_row(bench, seed, error_trace=str(exc))
    finally:
        sys.stdout = old_stdout
        log_file.close()

    verif = result.verification or {}
    return {
        "benchmark": bench,
        "seed": seed,
        "status": result.status,
        "tests_passed": int(verif.get("passed", 0) or 0),
        "total_tests": int(verif.get("total", 0) or 0),
        "wall_time_s": f"{result.wall_time_s:.1f}",
        "tokens_used": int(result.tokens_used or 0),
        "total_llm_cost": f"{float(getattr(result, 'cost_used', 0.0)):.4f}",
        "failure_modes": ";".join(result.failure_modes or []),
    }


def _empty_row(bench: str, seed: int, error_trace: str = "") -> dict:
    return {
        "benchmark": bench, "seed": seed, "status": "script_failed",
        "tests_passed": 0, "total_tests": 0,
        "wall_time_s": "0.0", "tokens_used": 0, "total_llm_cost": "0.0000",
        "failure_modes": error_trace[:200],
    }


def _print_row(row: dict) -> None:
    print(f"  -> {row['status']}  {row['tests_passed']}/{row['total_tests']}  "
          f"{row['wall_time_s']}s  ${row.get('total_llm_cost', '0')}")


if __name__ == "__main__":
    main()
