"""Pilot: does tuning matter more as the QUBO grows? Size-stratified comparison of tuning policies on one frozen,
independently verified, LLM-written model (YouBike tour from the e2e smoke run).

Size is changed only through the station filter of the generated loader (the n stations nearest to MRT Da'an);
the model code is otherwise unchanged. Quality is evaluated outside the model: the delivered tour is decoded via
variable_keys and its closed length measured on the same distance matrix, compared with a multi-start 2-opt tour.

    python scripts/experiments/exp_size_tuning.py --tag size_tuning_pilot_20261011
"""
import argparse
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import yaml

from qubo_auto.annealing.artifacts import atomic_write
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.datakit import load_tables
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.modeler import to_ir

MODEL = ROOT / "results/e2e_smoke3_20261011/direct/youbike_tour_r0/direct.py"
RAW = ROOT / "results/acceptance_truth/raw_20261011/youbike_immediate.json"
SIZES = {100: 10, 300: 17, 500: 22, 1000: 32, 1500: 39, 2000: 45}       # target variables -> stations (n^2)


def instance(n: int):
    code = MODEL.read_text()
    old = "df = df[df['dist_center'] <= 500].copy()"
    assert old in code
    ns = {}
    exec(compile(code.replace(old, f"df = df.nsmallest({n}, 'dist_center').copy()"), "<model>", "exec"), ns)
    data = ns["load_data"](load_tables([RAW]))
    return data, ns["build_ir"](data)


def tour_length(d, order):
    return float(sum(d[order[t]][order[(t + 1) % len(order)]] for t in range(len(order))))


def two_opt_reference(d, restarts=20, seed=0):
    rng, n, best = random.Random(seed), len(d), None
    for _ in range(restarts):
        tour = list(range(n)); rng.shuffle(tour)
        improved = True
        while improved:
            improved = False
            for a, b in itertools.combinations(range(1, n), 2):
                i, j, k, l = tour[a - 1], tour[a], tour[b], tour[(b + 1) % n]
                if d[i][k] + d[j][l] < d[i][j] + d[k][l] - 1e-9:
                    tour[a:b + 1] = reversed(tour[a:b + 1]); improved = True
        length = tour_length(d, tour)
        best = length if best is None or length < best else best
    return best


def decode(raw, x, stations):
    """Closed tour from variable_keys {station, step}; None if x is not a permutation."""
    chosen = [k for k, b in zip(raw["variable_keys"], x) if b and k]
    if len(chosen) != len(stations):
        return None
    by_step = sorted(chosen, key=lambda k: int(k["step"]))
    order = [stations.index(k["station"]) for k in by_step]
    return order if len(set(order)) == len(stations) and len({k["step"] for k in chosen}) == len(stations) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--policies", nargs="*", default=["default", "rule", "penalty_tpe", "penalty_sequential"])
    ap.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--backend", default="cga")
    args = ap.parse_args()
    out = ROOT / "results/experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))
    base["loop"].update(max_solver_calls=args.calls, patience=10, gap_target=-1)
    base["param_caps"] = {"cga": {"timeout": 3}, "sa": {"num_reads": 60, "num_sweeps": 300}}
    base["rag"].update(enabled=False, write_experience=False)
    hb = base["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    atomic_write(out / "manifest.json", json.dumps({"model": str(MODEL), "raw": str(RAW), "sizes": SIZES,
                                                    "args": vars(args), "config": base}, indent=1, default=str))
    rows_path = out / "results.jsonl"
    done = {(r["size"], r["policy"], r["repeat"]) for r in map(json.loads, rows_path.read_text().splitlines())} \
        if rows_path.exists() else set()
    for target in args.sizes:
        n = SIZES[target]
        data, raw = instance(n)
        d = data["dist"]
        ref = two_opt_reference(d)
        ir = to_ir(raw, f"youbike_{n}")
        for rep in range(args.repeats):
            for policy_name in args.policies:
                if (target, policy_name, rep) in done:
                    continue
                cfg = copy.deepcopy(base)
                dest = out / "runs" / f"{target}_{policy_name}_{rep}"
                dest.mkdir(parents=True, exist_ok=True)
                cfg["logging"] = {"loop_log": str(dest / "rounds.jsonl"), "backend_log": str(dest / "backend.jsonl")}
                started = time.time()
                row = {"size": target, "stations": n, "variables": ir.n_variables, "policy": policy_name,
                       "repeat": rep, "reference_2opt": ref}
                try:
                    policy = make_policy(policy_name, seed=rep, cfg=cfg, root=dest, rag=None)
                    res = run_loop(None, get_backend(args.backend), policy, cfg, budget=budget, seed=rep, ir=ir,
                                   root=ROOT, log=lambda m: None)
                    x = res.best_solution[:ir.n_variables] if res.best_solution else None
                    order = decode(raw, x, data["station"]) if x else None
                    length = tour_length(d, order) if order else None
                    row.update(calls=res.solver_calls, feasible=order is not None, length=length,
                               gap=None if length is None else (length - ref) / ref,
                               feasibility_rates=[r.feedback.get("feasibility_rate") if r.feedback else None
                                                  for r in res.rounds])
                except Exception as e:
                    row.update(error=f"{type(e).__name__}: {e}"[:500])
                row["seconds"] = round(time.time() - started, 1)
                with rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{target:>5} vars ({n} st) {policy_name:<18} feasible={row.get('feasible')} "
                      f"gap={row.get('gap') if row.get('gap') is None else round(row['gap'], 3)} "
                      f"calls={row.get('calls')} {row['seconds']}s {row.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
