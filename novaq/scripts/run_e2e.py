"""End to end (spec P5): natural-language problem -> QUBO -> decomposition -> feedback loop -> best solution.

    python scripts/run_e2e.py legacy/benchmarks/knapsack --input legacy/benchmarks/knapsack/sample_cases.txt
    python scripts/run_e2e.py legacy/benchmarks/knapsack --input legacy/benchmarks/knapsack/sample_cases.txt \
        --instance my_instance.json --backend sa --policy agent

Steps:
  1. Front end (original QuantumQUBO Agent pipeline): Planner -> Judge -> Formulizer -> Judge -> Debugger
     -> Coder -> brute-force verification on the small test cases.
  2. Decomposer Agent splits the verified QUBO into objective + one penalty per constraint + original-problem
     checks; deterministic code verifies the split on every test case (retries with the errors).
  3. The chosen instance (--instance JSON, same keys as the test cases; default: largest test case) becomes a
     ProblemInstance and runs through the closed loop (config_loop.yaml).
A report is written to results/e2e/<run_id>.json.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.agents.coder_agent import _extract_code, compile_build_qubo
from qubo_auto.annealing.backends import BACKENDS, get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.bridge import decompose, to_problem_instance
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.tools import verify_small
from qubo_auto.llm_client import LLMClient
from qubo_auto.pipeline import Pipeline, load_config
from qubo_auto.schemas import QUBOFormulation, StructuredSpec
from qubo_auto.verification.test_cases import load_prompt, load_test_cases


def _latest(run_dir: Path, step: str) -> dict:
    files = sorted(glob.glob(str(run_dir / f"{step}_*.json")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no {step} transcript in {run_dir}")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark_dir")
    ap.add_argument("--input", "-i", default=None, help="plain-text test cases (as for run_single.py)")
    ap.add_argument("--instance", default=None, help="JSON file with the instance dict to optimise")
    ap.add_argument("--reference", type=float, default=None, help="known optimum of --instance (optional)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="sa")
    ap.add_argument("--policy", default="agent", choices=["default", "rule", "random", "optuna", "agent"])
    ap.add_argument("--mode", choices=["multi", "single"], default=None)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop-config", default="config_loop.yaml")
    args = ap.parse_args()

    bench = Path(args.benchmark_dir)
    bench = bench if bench.is_absolute() else ROOT / bench
    config = load_config(ROOT / args.config)
    raw_input = Path(args.input).read_text(encoding="utf-8") if args.input else None

    # ---- 1. front end
    print("== 1/3 Front end: natural language -> verified QUBO", flush=True)
    pipeline = Pipeline(config=config, transcript_root=ROOT / "transcripts", results_path=ROOT / "results" / "runs.jsonl")
    fe = pipeline.run(nl_problem=load_prompt(bench), benchmark_name=bench.name, seed=args.seed,
                      test_cases=[] if raw_input else load_test_cases(bench), benchmark_dir=bench,
                      raw_test_input=raw_input)
    if fe.status != "success":
        sys.exit(f"Front end failed ({fe.status}); see transcripts/{fe.run_id}. Nothing to optimise.")
    run_dir = ROOT / "transcripts" / fe.run_id
    spec = StructuredSpec.model_validate(json.loads(_latest(run_dir, "restructure")["response"]))
    formulation = QUBOFormulation.model_validate(json.loads(_latest(run_dir, "formulate")["response"]))
    code = _extract_code(_latest(run_dir, "code")["response"])
    cases = load_test_cases(bench, cases_dir=run_dir / "cases") if raw_input else load_test_cases(bench)

    # ---- 2. decomposition
    print(f"\n== 2/3 Decomposer: objective + per-constraint penalties + checks ({len(cases)} test cases)", flush=True)
    client = LLMClient(transcript_dir=ROOT / "transcripts", run_id=fe.run_id + "_decompose", max_tokens=300_000,
                       use_cache=False)
    model = config["models"].get("formulizer_agent")
    dcode, ns, errors = decompose(spec, formulation, code, compile_build_qubo(code), cases, client, model)
    report = {"run_id": fe.run_id, "benchmark": bench.name, "front_end": fe.model_dump(), "build_qubo": code,
              "decomposition": dcode, "decomposition_errors": errors}
    out_dir = ROOT / "results" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    if errors:
        (out_dir / f"{fe.run_id}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        hint = ""
        if any("cannot reach 0" in e for e in errors):
            hint = ("\nThe front end's QUBO itself encodes a constraint more strictly than the problem states (it passed "
                    "only because the test cases' optima happen to satisfy the stricter form). Re-decomposing cannot fix "
                    "this; the constraint needs a different encoding (planned: Encoding Agent).")
        sys.exit("Decomposition failed verification:\n- " + "\n- ".join(errors) + hint)
    print("   decomposition verified on every test case", flush=True)

    # ---- 3. closed loop
    if args.instance:
        instance, reference, label = json.loads(Path(args.instance).read_text(encoding="utf-8")), args.reference, Path(args.instance).stem
    else:
        case = max(cases, key=lambda c: c.n_variables)
        reference = -case.optimal_value if ns["SENSE"] == "maximize" else case.optimal_value
        instance, label = case.instance, case.name
    problem = to_problem_instance(ns, instance, name=bench.name, instance_id=f"e2e_{bench.name}_{label}", reference=reference)
    check = verify_small(problem) if problem.n_variables <= 20 else {"checked": False}
    print(f"\n== 3/3 Feedback loop on '{label}': n={problem.n_variables}, constraints={list(problem.constraints)}, "
          f"reference={reference}, verify_small={check}", flush=True)
    lcfg = yaml.safe_load((ROOT / args.loop_config).read_text(encoding="utf-8"))
    if args.mode:
        lcfg["agents"]["mode"] = args.mode
    rag = None
    if lcfg["rag"]["enabled"] and not args.no_rag:
        from qubo_auto.annealing.rag import KnowledgeBase
        rag = KnowledgeBase.from_config(lcfg["rag"], ROOT)
    hb = lcfg["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    policy = make_policy(args.policy, seed=args.seed, cfg=lcfg, root=ROOT, rag=rag)
    res = run_loop(problem, get_backend(args.backend), policy, lcfg, simulate=args.simulate, budget=budget,
                   rag=rag, seed=args.seed, root=ROOT)

    report |= {"instance": label, "verify_small": check, "loop": json.loads(res.model_dump_json())}
    (out_dir / f"{fe.run_id}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nBest objective: {res.best_objective} (reference {reference}, gap {res.best_gap}), "
          f"feasibility {res.best_feasibility_rate}, stop: {res.stop_reason}")
    print(f"Best solution: {res.best_solution}")
    print(f"Report: results/e2e/{fe.run_id}.json")


if __name__ == "__main__":
    main()
