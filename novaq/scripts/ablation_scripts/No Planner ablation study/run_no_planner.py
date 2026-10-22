"""No-Planner ablation runner — single, multiple, or all benchmarks.

Ablation A3: removes the Planner agent and both Judge gates. The raw
natural-language problem statement is fed directly to the Formulizer.
The Debugger receives the NL input, the test cases, and the Formulizer
output — no StructuredSpec involved at any stage.

What is removed compared to the full pipeline:
- Stage 1 : PlannerAgent (NL -> StructuredSpec).
- Stage 1b: JudgeAgent spec check.
- Stage 2b: JudgeAgent QUBO check.

What is retained:
- Stage 2 : FormulizationAgent — receives raw NL directly.
- Stage 2.5: DebuggerAgent — receives NL + formulation (no spec).
- Stage 3 : CoderAgent + TestRunner — unchanged.
- Test-Runner-driven re-formulation on formulation_error.

Results land under:
    scripts/ablation_scripts/No Planner ablation study/
        results/<name>_<seed>.csv
        transcripts/<run_id>/...

Usage:
    python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42
    python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --benchmarks max_cut
    python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --name no_planner_run
    python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
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

from qubo_auto.agents.coder_agent import compile_build_qubo, generate_code
from qubo_auto.agents.debugger_agent import build_test_cases
from qubo_auto.llm_client import LLMClient, TokenBudgetExceeded
from qubo_auto.pipeline import load_config
from qubo_auto.schemas import (
    Objective, QUBOFormulation, RunResult, StructuredSpec, Variable,
)
from qubo_auto.verification.test_cases import load_prompt, load_test_cases
from qubo_auto.verification.test_runner import run_tests

# Reuse the formulizer prompt — substitute NL for {spec_json}
_FORMULIZER_TEMPLATE = (
    ROOT / "qubo_auto" / "prompts" / "formulizer_agent.txt"
).read_text(encoding="utf-8")


CSV_FIELDS = [
    "benchmark", "seed", "status",
    "tests_passed", "total_tests",
    "wall_time_s", "tokens_used", "total_llm_cost",
    "planner_retries", "formulizer_retries",
    "coder_retries", "debugger_retries",
    "failure_modes",
]


# ───────────────────────── helpers ──────────────────────────────────────────

def _rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path)


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


# ───────────────────────── NL-aware Formulizer ──────────────────────────────

def _sanitize(data: dict) -> dict:
    for pt in data.get("penalty_terms", []):
        if isinstance(pt.get("penalty_weight"), str):
            pt["penalty_weight"] = 1.0
    return data


def formulate_from_nl(
    nl_problem: str,
    client: LLMClient,
    model: str,
    temperature: float,
    previous_issues: list[str] | None = None,
) -> QUBOFormulation:
    """Call Formulizer with raw NL directly — no StructuredSpec."""
    nl_as_spec = json.dumps({
        "problem_description": nl_problem,
        "note": "No structured spec was produced. Infer variables, objective, "
                "and constraints directly from the problem description above.",
    }, indent=2)
    prompt = _FORMULIZER_TEMPLATE.replace("{spec_json}", nl_as_spec)

    if previous_issues:
        issues_block = "\n".join(f"  - {i}" for i in previous_issues)
        prompt += (
            f"\n\n--- FEEDBACK FROM PREVIOUS ATTEMPT [TEST_RUNNER] ---\n"
            f"Your last formulation was rejected. Issues found:\n"
            f"{issues_block}\n"
            "Fix ALL of them. Do not repeat the same mistakes."
        )

    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="formulate_from_nl", json_mode=True)
    data = _sanitize(json.loads(raw))
    return QUBOFormulation.model_validate(data)


def _synthetic_spec(nl_problem: str, formulation: QUBOFormulation) -> StructuredSpec:
    """Minimal StructuredSpec built from NL + formulation for Debugger/Coder compatibility."""
    vars_ = [
        Variable(name=k, type="binary", index=i, description="")
        for i, k in enumerate(formulation.variable_mapping)
    ]
    return StructuredSpec(
        problem_name="(no planner)",
        variables=vars_,
        objective=Objective(direction="minimize", expression="see formulation"),
        constraints=[],
        domain_notes=nl_problem[:800],
        instance_parameters=[],
    )


# ───────────────────────── inline pipeline (no Planner, no Judge) ────────────

def run_no_planner_pipeline(
    nl_problem: str,
    benchmark_name: str,
    seed: int,
    test_cases: list,
    config: dict,
    transcript_root: Path,
    results_path: Path,
    benchmark_dir: Path | None = None,
    raw_test_input: str | None = None,
) -> RunResult:
    run_id = f"{benchmark_name}_{seed}_{uuid.uuid4().hex[:8]}"
    start = time.time()
    models = config["models"]
    temps = config["temperatures"]
    retries = config["retries"]
    verif_cfg = config["verification"]
    limits = config.get("limits", {})
    max_tokens = limits.get("max_tokens_per_run", 50_000)
    use_cache = limits.get("use_cache", True)
    failure_modes: list[str] = []

    _log(f"=== No-Planner Pipeline START | run_id={run_id} ===")
    _log(f"Benchmark : {benchmark_name}  |  seed={seed}")
    _log(f"Models    : formulizer_agent={models['formulizer_agent']}")
    _log(f"            coder_agent={models['coder_agent']}")
    _log(f"Cache     : {'enabled' if use_cache else 'disabled'}")

    client = LLMClient(
        transcript_dir=transcript_root,
        run_id=run_id,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )

    result = RunResult(
        run_id=run_id,
        benchmark=benchmark_name,
        seed=seed,
        status="failed",
        models_used=dict(models),
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
        # ── Stage 1: SKIPPED (Planner + Judge spec check) ────────────────────
        _log(f"[Stage 1] Planner SKIPPED — NL goes directly to Formulizer")

        # ── Stage 2: Formulizer receives raw NL (no Judge gate) ──────────────
        formulation: QUBOFormulation | None = None
        previous_issues: list[str] = []
        for attempt in range(retries["formulate"] + 1):
            _log(f"[Stage 2] Formulizer (from NL) — attempt {attempt + 1}/{retries['formulate'] + 1}")
            try:
                formulation = formulate_from_nl(
                    nl_problem, client,
                    models["formulizer_agent"], temps["formulizer_agent"],
                    previous_issues or None,
                )
                _log(f"[Stage 2] Formulation OK — n_vars={formulation.n_variables}, "
                     f"{len(formulation.objective_terms)} obj terms, "
                     f"{len(formulation.penalty_terms)} penalty groups")
                break
            except Exception as e:
                _log(f"[Stage 2] Formulizer EXCEPTION: {e}")
                previous_issues = [str(e)]
                if attempt == retries["formulate"]:
                    result.error_trace = traceback.format_exc()
                    return _finalize()

        if formulation is None:
            result.error_trace = "Formulizer produced no formulation"
            return _finalize()

        # Stage 2b: SKIPPED (no Judge QUBO check)
        _log(f"[Stage 2b] Judge SKIPPED")
        result.n_iterations += 1

        # Build synthetic spec for Debugger/Coder compatibility
        syn_spec = _synthetic_spec(nl_problem, formulation)

        # ── Stage 2.5: Debugger — NL + test + formulation (no spec) ──────────
        if raw_test_input and benchmark_dir:
            _log(f"[Stage 2.5] Building test cases (NL + formulation, no spec) ...")
            try:
                transcript_cases_dir = transcript_root / run_id / "cases"
                written = build_test_cases(
                    benchmark_dir=benchmark_dir,
                    raw_input=raw_test_input,
                    client=client,
                    model=models.get("debugger_agent", models["formulizer_agent"]),
                    spec=syn_spec,
                    formulation=formulation,
                    temperature=temps.get("debugger_agent", 0.1),
                    cases_dir=transcript_cases_dir,
                )
                test_cases = load_test_cases(benchmark_dir, cases_dir=transcript_cases_dir)
                _log(f"[Stage 2.5] Wrote {len(written)} case(s), loaded {len(test_cases)} total")
            except Exception as e:
                _log(f"[Stage 2.5] Test case building FAILED: {e}")
                result.error_trace = traceback.format_exc()
                return _finalize()

        # ── Stage 3: Coder + Test Runner (intact) ────────────────────────────
        sample_instance = test_cases[0].instance if test_cases else {}
        n_cases = len(test_cases)
        previous_error: str | None = None

        for attempt in range(retries["code"] + 1):
            _log(f"[Stage 3] Coder — attempt {attempt + 1}/{retries['code'] + 1}")
            try:
                code = generate_code(
                    syn_spec, formulation, client,
                    models["coder_agent"], temps["coder_agent"],
                    previous_error, sample_instance=sample_instance,
                )
                build_qubo_fn = compile_build_qubo(code)
                _log(f"[Stage 3] Code generated and compiled OK")
            except Exception as e:
                _log(f"[Stage 3] Coder EXCEPTION: {e}")
                previous_error = traceback.format_exc()
                failure_modes.append("coding_error")
                if attempt == retries["code"]:
                    result.error_trace = previous_error
                    result.failure_modes = failure_modes
                    return _finalize()
                continue

            _log(f"[Stage 3b] Test Runner — running {n_cases} test cases ...")
            test_results, failure_class = run_tests(
                build_qubo=build_qubo_fn,
                test_cases=test_cases,
            )

            n_pass = sum(1 for r in test_results if r.match)
            result.verification = {"passed": n_pass, "total": n_cases}
            pass_rate = n_pass / n_cases if n_cases else 0
            _log(f"[Stage 3b] Tests: {n_pass}/{n_cases} passed "
                 f"({pass_rate * 100:.0f}%)  failure_class={failure_class}")
            for r in test_results:
                if r.error_message:
                    _log(f"[Stage 3b] Instance {r.instance_id} error: {r.error_message[:200]}")

            if pass_rate >= verif_cfg["pass_threshold"]:
                _log(f"=== No-Planner Pipeline SUCCESS ===")
                result.status = "success"
                result.failure_modes = failure_modes
                return _finalize()

            failure_modes.append(failure_class)
            errors = [r.error_message for r in test_results if r.error_message]
            previous_error = "\n".join(errors[:3])

            # Test Runner-driven re-formulation — no Judge re-verification
            if failure_class == "formulation_error" and attempt < retries["code"]:
                _log(f"[Stage 3b] Structural error — re-formulating from NL (no Judge) ...")
                try:
                    formulation = formulate_from_nl(
                        nl_problem, client,
                        models["formulizer_agent"], temps["formulizer_agent"],
                        errors[:3],
                    )
                    syn_spec = _synthetic_spec(nl_problem, formulation)
                    _log(f"[Stage 3b] Re-formulation done")
                except Exception as e:
                    _log(f"[Stage 3b] Re-formulation EXCEPTION: {e}")

            result.n_iterations += 1

        _log(f"=== No-Planner Pipeline FAILED — code exhausted all retries ===")
        result.error_trace = f"Code failed after max retries. Last failure: {failure_class}"
        result.failure_modes = failure_modes
        return _finalize()

    except TokenBudgetExceeded as e:
        _log(f"=== No-Planner Pipeline ABORTED — token budget exceeded: {e} ===")
        result.error_trace = str(e)
        result.failure_modes = failure_modes + ["token_budget_exceeded"]
        return _finalize()
    except Exception:
        _log(f"=== No-Planner Pipeline EXCEPTION ===")
        result.error_trace = traceback.format_exc()
        result.failure_modes = failure_modes
        return _finalize()


# ───────────────────────── suite driver ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="No-Planner ablation runner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default=None,
                        help="Comma-separated folder name(s). Omit to run all.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", default="ablation_no_planner",
                        help="CSV base name. Output: results/<name>_<seed>.csv")
    parser.add_argument("--output", default=None,
                        help="Explicit CSV output path (overrides --name).")
    parser.add_argument("--skip", default="custom",
                        help="Comma-separated benchmark folders to skip.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    benchmarks = _select_benchmarks(ROOT / "legacy" / "benchmarks", args.benchmarks, args.skip)

    if args.dry_run:
        print("Would run:")
        for bd in benchmarks:
            print(f"  {bd.name}  seed={args.seed}")
        return

    csv_path = (Path(args.output) if args.output
                else RESULTS_DIR / f"{args.name}_{args.seed}.csv")
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_jsonl = RESULTS_DIR / "runs.jsonl"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    print(f"[No-Planner] Ablation run")
    print(f"  Benchmarks : {len(benchmarks)}")
    print(f"  Seed       : {args.seed}")
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
    out: list[Path] = []
    for d in dirs:
        if d.name in skip_names:
            continue
        if not d.exists():
            raise FileNotFoundError(f"Benchmark folder not found: {d}")
        out.append(d)
    return out


def _run_one(benchmark_dir: Path, seed: int, config: dict, runs_jsonl: Path) -> dict:
    bench = benchmark_dir.name
    print(f"\n{'=' * 60}\n  {bench}  seed={seed}\n{'=' * 60}")
    log_path = LOGS_DIR / f"{bench}_seed{seed}.log"

    nl_problem = load_prompt(benchmark_dir)

    sample_path = benchmark_dir / "sample_cases.txt"
    raw_test_input: str | None = None
    if sample_path.exists():
        raw_test_input = sample_path.read_text(encoding="utf-8")
        test_cases: list = []
    else:
        test_cases = load_test_cases(benchmark_dir)

    log_file = log_path.open("w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)

    try:
        result = run_no_planner_pipeline(
            nl_problem=nl_problem,
            benchmark_name=bench,
            seed=seed,
            test_cases=test_cases,
            config=config,
            transcript_root=TRANSCRIPTS_DIR,
            results_path=runs_jsonl,
            benchmark_dir=benchmark_dir,
            raw_test_input=raw_test_input,
        )
    except Exception as exc:
        print(f"[error] {exc}")
        sys.stdout = old_stdout
        log_file.close()
        return _empty_row(bench, seed, status="script_failed", error_trace=str(exc))
    finally:
        sys.stdout = old_stdout
        log_file.close()

    log_metrics = _collect_log_metrics(log_path)
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
        "planner_retries": 0,
        "formulizer_retries": max(0, log_metrics["formulizer_attempts"] - 1),
        "coder_retries": max(0, log_metrics["coder_attempts"] - 1),
        "debugger_retries": max(0, log_metrics["debugger_attempts"] - 1),
        "failure_modes": ";".join(result.failure_modes or []),
    }


def _empty_row(bench: str, seed: int, status: str = "script_failed",
               error_trace: str = "") -> dict:
    return {
        "benchmark": bench, "seed": seed, "status": status,
        "tests_passed": 0, "total_tests": 0,
        "wall_time_s": "0.0", "tokens_used": 0, "total_llm_cost": "0.0000",
        "planner_retries": 0, "formulizer_retries": 0,
        "coder_retries": 0, "debugger_retries": 0,
        "failure_modes": error_trace[:200],
    }


def _collect_log_metrics(log_path: Path) -> dict:
    import re
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {
        "planner_attempts": 0,
        "formulizer_attempts": len(re.findall(r"\[Stage 2\] Formulizer.*attempt", text)),
        "coder_attempts": len(re.findall(r"\[Stage 3\] Coder.*attempt", text)),
        "debugger_attempts": text.count("[Stage 2.5] Building test cases"),
    }


def _print_row(row: dict) -> None:
    print(f"  -> {row['status']}  {row['tests_passed']}/{row['total_tests']}  "
          f"{row['wall_time_s']}s  ${row.get('total_llm_cost', '0')}")


if __name__ == "__main__":
    main()
