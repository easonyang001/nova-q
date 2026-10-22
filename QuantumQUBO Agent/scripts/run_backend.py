"""Run one QUBO on one or more backends and report unified feedback metrics (spec P0).

Usage:
    python scripts/run_backend.py --problem select_k --backend sa
    python scripts/run_backend.py --problem select_k --backend sa --backend dwave_qpu
    python scripts/run_backend.py --problem select_k --backend dwave_qpu --simulate   # local stand-in
    python scripts/run_backend.py --problem select_k --backend sa --param num_sweeps=50 --penalty select_k=2
    python scripts/run_backend.py --backend dwave_qpu --describe

--param name=value applies to every backend; dwave_qpu.name=value to one backend only.
Every execution is appended to results/annealing_runs.jsonl.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing.backends import BACKENDS, get_backend
from qubo_auto.annealing.examples import EXAMPLES
from qubo_auto.annealing.feedback import extract_feedback, log_run
from qubo_auto.annealing.guardrail import GuardrailError, HardwareBudget
import yaml


def _parse_value(text: str) -> float | int:
    return int(text) if text.lstrip("-").isdigit() else float(text)


def _params_for(backend: str, raw: list[str]) -> dict:
    params = {}
    for item in raw:
        key, _, value = item.partition("=")
        target, _, name = key.rpartition(".")
        if not target or target == backend:
            params[name] = _parse_value(value)
    return params


def _problem_kwargs(args) -> dict:
    if args.problem.startswith("qoblib_"):
        if not args.name:
            raise SystemExit("--name is required for QOBLIB problems (e.g. karate, ms_03_050_002)")
        return {"name": args.name}
    if args.problem not in ("select_k_random", "knapsack"):
        return {}
    kw = {"seed": args.instance_seed}
    if args.n is not None:
        kw["n"] = args.n
    if args.k is not None and args.problem == "select_k_random":
        kw["k"] = args.k
    return kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", choices=sorted(EXAMPLES), default="select_k")
    ap.add_argument("--backend", action="append", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--param", action="append", default=[], metavar="[BACKEND.]NAME=VALUE")
    ap.add_argument("--penalty", action="append", default=[], metavar="CONSTRAINT=WEIGHT")
    ap.add_argument("--simulate", action="store_true", help="run real backends on their local stand-in")
    ap.add_argument("--name", default=None, help="QOBLIB instance name for qoblib_* problems")
    ap.add_argument("--n", type=int, default=None, help="size for generated problems (select_k_random, knapsack)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--describe", action="store_true", help="print backend capabilities and exit")
    ap.add_argument("--log", default=str(ROOT / "results" / "annealing_runs.jsonl"))
    args = ap.parse_args()

    if args.describe:
        for name in args.backend:
            print(json.dumps(get_backend(name).describe(), indent=2))
        return

    hb = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    problem = EXAMPLES[args.problem](**_problem_kwargs(args))
    weights = {k: float(v) for k, _, v in (p.partition("=") for p in args.penalty)}
    qubo, offset = problem.to_qubo(weights)
    print(f"Problem   : {problem.name}  n={problem.n_variables}  terms={len(qubo)}  offset={offset:g}")
    print(f"Reference : {problem.sense} objective = {problem.reference_objective}")
    print(f"Penalties : {({**problem.default_penalty_weights, **weights}) or 'none'}\n")

    for name in args.backend:
        backend = get_backend(name)
        params = _params_for(name, args.param)
        real = not args.simulate and not backend.is_simulator
        try:
            if real:
                budget.check(name)
            result = backend.run(qubo, offset, problem.n_variables, params, simulate=args.simulate)
            if real:
                budget.record(name, result.solver_time_s, note=f"run_backend {problem.instance_id}")
        except GuardrailError as e:
            print(f"[{name}] rejected by guardrail:\n  - " + "\n  - ".join(e.errors))
            continue
        fb = extract_feedback(problem, result)
        log_run(Path(args.log), result, fb,
                extra={"problem": problem.name, "penalty_weights": {**problem.default_penalty_weights, **weights}})

        tag = f"{name}{' (simulated)' if fb.simulated else ''}"
        print(f"[{tag}] params={result.params}")
        print(f"  feasibility={fb.feasibility_rate:.1%}  violations={ {k: round(v, 3) for k, v in fb.constraint_violation_rates.items()} }")
        print(f"  best feasible objective={fb.best_feasible_objective}  gap={fb.gap}  "
              f"mean feasible={fb.mean_feasible_objective if fb.mean_feasible_objective is None else round(fb.mean_feasible_objective, 3)}")
        print(f"  samples={fb.n_samples}  unique={fb.n_unique}  diversity={fb.diversity:.3f}  best_energy={fb.best_energy:g}")
        print(f"  wall={fb.wall_time_s:.2f}s  solver={'-' if fb.solver_time_s is None else f'{fb.solver_time_s:.4f}s'}")
        specific = {k: fb.backend_specific[k] for k in
                    ("solver", "chain_strength", "chain_break_fraction_mean", "samples_with_chain_breaks",
                     "num_qubits_used", "max_chain_length") if k in fb.backend_specific}
        if specific:
            print(f"  backend-specific: {specific}")
        print()


if __name__ == "__main__":
    main()
