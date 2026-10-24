"""Reduced real instances for the QAOA (CUDA-Q) comparison: QUBOs small enough for state-vector simulation.

For each case, take a run that passed the independent judge, keep its agent-written model (model.py) and its
prepared real data (data.json) unchanged, shrink the reducible set to n representatives with the pipeline's own
reduce_points tool (weighted k-means, weights summed into representatives), slice every parameter indexed by that
set, then build_ir -> compile_problem -> QUBO exactly as the workflow does. No LLM call happens here.

    python scripts/experiments/qaoa_instances.py --out results/experiments/qaoa_cudaq_20261020/instances
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qubo_auto.annealing.encodings import compile_problem          # noqa: E402
from qubo_auto.annealing.modeler import load_module, to_ir, to_jsonable  # noqa: E402
from qubo_auto.annealing.prep.join import reduce_points            # noqa: E402

R = ROOT / "results"
CASES = {   # case -> judged-passing run (rejudged.json: joint=true)
    "ev_chargers_s": R / "e2e_cfgBt_r1_20261013c/full_rag/ev_chargers_s_r0",
    "aed_groups": R / "e2e_cfgBt_r1_20261013c/full_rag/aed_groups_r0",
    "youbike_tour": R / "e2e_cfgBt_r1_20261013c/full_rag/youbike_tour_r0",
    "london_ambulance_s": R / "e2e_cfgBt_r3_20261013c/full_rag/london_ambulance_s_r0",
    # international replacement for Berlin (judge-passing, 2026-10-21; this run's positions are P0..P10)
    "melbourne_bike_tour": R / "e2e_melb7_g1_20261021/full_rag/melbourne_bike_tour_r0",
    # knapsack (budget inequality -> slack bits): the first problem type with an inequality constraint
    "nyc_pb_budget": R / "e2e_nycpb2_g1_20261022/full_rag/nyc_pb_budget_r0",
}


def reduce_data(schema: dict, data: dict, n: int, seed: int = 0) -> dict:
    """Shrink the reducible set to n members; slice every parameter indexed by it."""
    name, spec = next((k, v) for k, v in schema["sets"].items() if v.get("reducible"))
    members = list(data[name])
    params = schema["params"]
    if not any(v.get("index") == [name] and "lat" in p.lower() for p, v in params.items()):
        return reduce_by_subset(schema, data, name, n, seed)
    lat = next(p for p, v in params.items() if v.get("index") == [name] and "lat" in p.lower())
    lon = next(p for p, v in params.items() if v.get("index") == [name] and "lon" in p.lower())
    weight = spec["reducible"].get("weight")
    df = pd.DataFrame({"_i": range(len(members)), "lat": data[lat], "lon": data[lon]})
    if weight:
        df["w"] = data[weight]
    if seed:
        # repeats: k-means on a random 80% of the real points, so each seed is a genuinely different instance
        df = df.sample(frac=0.8, random_state=seed).reset_index(drop=True)
    keep = reduce_points(df, n, weight="w" if weight else None, seed=seed)
    idx = [int(i) for i in keep["_i"]]
    out = dict(data)
    out[name] = [members[i] for i in idx]
    for p, v in params.items():
        if p not in data or not v.get("index"):
            continue
        ix = v["index"]
        if ix == [name]:
            out[p] = [data[p][i] for i in idx]
        elif ix == [name, name]:
            out[p] = [[data[p][i][j] for j in idx] for i in idx]
    if weight:
        out[weight] = [float(w) for w in keep["w"]]          # cluster weight summed into its representative
    # sets the model ties one-to-one to the reduced set (e.g. tour positions) shrink with it
    for s, v in schema["sets"].items():
        if s != name and v.get("source") != "user" and len(data.get(s, [])) == len(members):
            out[s] = list(data[s])[:n]
    return out


def reduce_by_subset(schema: dict, data: dict, name: str, n: int, seed: int) -> dict:
    """Sets without coordinates (budget ballots): a random subset of n members; a scalar user setting in the same unit
    as a per-member parameter (a budget in USD over costs in USD) is scaled by the subset's share of that parameter's
    total, so the instance keeps the original tightness instead of becoming trivially feasible."""
    rng = np.random.default_rng(seed)
    members = list(data[name])
    idx = sorted(rng.choice(len(members), size=n, replace=False).tolist())
    params = schema["params"]
    out = dict(data)
    out[name] = [members[i] for i in idx]
    for p, v in params.items():
        if p in data and v.get("index") == [name]:
            out[p] = [data[p][i] for i in idx]
    for p, v in params.items():
        if p in data and v.get("index") == [] and isinstance(data[p], (int, float)):
            same = next((q for q, w in params.items() if w.get("index") == [name] and w.get("unit") == v.get("unit")
                         and v.get("unit")), None)
            if same:
                share = sum(out[same]) / sum(data[same])
                out[p] = int(round(data[p] * share / 1000.0)) * 1000 if data[p] >= 1000 else data[p] * share
    return out


def build(case: str, n: int, seed: int = 0) -> dict:
    run = CASES[case]
    ns = load_module((run / "artifacts/model.py").read_text())
    data = json.loads((run / "artifacts/data.json").read_text())
    data = data.get("data", data)
    small = reduce_data(ns["DATA_SCHEMA"], data, n, seed)
    raw = ns["build_ir"](small)
    ir = to_ir(raw, f"{case}_n{n}")
    prob = compile_problem(ir)
    qubo, offset = prob.to_qubo()
    return {"case": case, "n_points": n, "source_run": str(run.relative_to(ROOT)), "sense": ir.sense,
            "n_decision": ir.n_variables, "n_qubits": prob.n_variables,
            "variable_names": [str(v) for v in raw["variable_names"]],
            "qubo": [[int(i), int(j), float(c)] for (i, j), c in qubo.items()], "offset": float(offset),
            "data": small, "ir_raw": to_jsonable(raw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-qubits", type=int, default=26)
    ap.add_argument("--survey", action="store_true", help="only print qubit counts per n")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="repeat instances per size (seed 0 = all points)")
    ap.add_argument("--cases", nargs="*", help="only these cases (default: all)")
    ap.add_argument("--sizes", type=int, nargs="*", help="only these numbers of points (default: 3, 4, ...)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for case in (a.cases or CASES):
        for n in (a.sizes or range(3, 13)):
            stop = False
            for seed in a.seeds:
                try:
                    inst = build(case, n, seed)
                except Exception as e:
                    print(f"{case:<20} n={n:<3} seed={seed} error {type(e).__name__}: {str(e)[:120]}")
                    continue
                q = inst["n_qubits"]
                print(f"{case:<20} n={n:<3} seed={seed} decision={inst['n_decision']:<4} qubits={q}")
                if q > a.max_qubits:
                    stop = True
                    break
                inst["seed"] = seed
                name = f"{case}_n{n}" + (f"_s{seed}" if len(a.seeds) > 1 or seed else "")
                if not a.survey:
                    (out / f"{name}.json").write_text(json.dumps(inst, ensure_ascii=False, default=str))
            if stop:
                break

if __name__ == "__main__":
    main()
