"""Knowledge-base experiment: how chain_strength affects chain breaks and solution quality on D-Wave.

    python scripts/experiments/exp_dwave_chain_strength.py --n 30 --k 6 --factors 0.25 0.5 2 4

Runs once with Ocean's default (uniform torque compensation), then at multiples of that value.
Every call goes through the hardware budget in config_loop.yaml; results go to results/annealing_runs.jsonl.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing.backends import DWaveQPUBackend
from qubo_auto.annealing.examples import select_k_random
from qubo_auto.annealing.feedback import extract_feedback, log_run
from qubo_auto.annealing.guardrail import HardwareBudget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--num-reads", type=int, default=100)
    ap.add_argument("--factors", type=float, nargs="+", default=[0.25, 0.5, 2.0, 4.0])
    args = ap.parse_args()

    hb = yaml.safe_load((ROOT / "config_loop.yaml").read_text())["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    problem = select_k_random(n=args.n, k=args.k, seed=args.instance_seed)
    qubo, offset = problem.to_qubo()
    backend = DWaveQPUBackend()
    rows = []

    def run(params: dict, label: str) -> dict:
        budget.check(backend.name)
        res = backend.run(qubo, offset, problem.n_variables, params)
        budget.record(backend.name, res.solver_time_s, note=f"chain_strength experiment {problem.instance_id}")
        fb = extract_feedback(problem, res)
        log_run(ROOT / "results" / "annealing_runs.jsonl", res, fb,
                extra={"experiment": "dwave_chain_strength", "instance_id": problem.instance_id, "label": label})
        m = fb.backend_specific
        row = {"label": label, "chain_strength": round(m["chain_strength"], 3),
               "chain_break_frac": round(m["chain_break_fraction_mean"], 4),
               "samples_with_breaks": round(m["samples_with_chain_breaks"], 3),
               "max_chain": m["max_chain_length"], "qubits": m["num_qubits_used"],
               "feasibility": round(fb.feasibility_rate, 3), "best": fb.best_feasible_objective,
               "gap": None if fb.gap is None else round(fb.gap, 4),
               "mean_feasible": None if fb.mean_feasible_objective is None else round(fb.mean_feasible_objective, 2),
               "qpu_ms": round(res.solver_time_s * 1000, 1)}
        print(json.dumps(row), flush=True)
        return row

    print(f"# {problem.instance_id}  n={problem.n_variables}  reference={problem.reference_objective}")
    base = run({"num_reads": args.num_reads}, "default_utc")
    rows.append(base)
    for f in args.factors:
        rows.append(run({"num_reads": args.num_reads, "chain_strength": base["chain_strength"] * f}, f"{f}x"))


if __name__ == "__main__":
    main()
