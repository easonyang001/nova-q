"""NVIDIA cuOpt (GPU) on the same verified real-city instances, against the exact optimum (HiGHS MILP on the CPU) and
the pipeline's judged deliveries (SA on the QUBO).

Instances come from the judge's ground truth (the entities the verified runs used), distances are straight-line as in
the judge. Tours use cuOpt routing; budget, coverage and p-median use cuOpt MILP.

    /research/conda-envs/cuopt-env/bin/python scripts/experiments/cuopt_compare.py \
        --out results/experiments/cuopt_20261023/cuopt.jsonl
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "acceptance_truth"


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dist_matrix(rows):
    la = np.array([r["lat"] for r in rows]); lo = np.array([r["lon"] for r in rows])
    return hav(la[:, None], lo[:, None], la[None, :], lo[None, :])


def truth(file, key):
    return json.loads((T / file).read_text())[key]


def pipeline_deliveries(case):
    """True quality of every judged-passing delivery of this case (SA on the QUBO)."""
    vals = []
    for p in glob.glob(str(ROOT / "results" / "*" / "full_rag" / f"{case}_r*" / "record.json")):
        v = json.loads(Path(p).read_text()).get("verdict") or {}
        if all((v.get(g) or {}).get("ok") for g in ("data", "model", "delivery")):
            q = (v["delivery"].get("checks") or {}).get("true_quality")
            if q is not None:
                vals.append(float(q))
    return vals


# ---------------------------------------------------------------- exact references (CPU, HiGHS)

def exact_tsp(D):
    n = len(D)
    idx = {(i, j): k for k, (i, j) in enumerate((i, j) for i in range(n) for j in range(n) if i != j)}
    m = len(idx) + n                                     # x_ij, then MTZ order u_i
    c = np.zeros(m); [c.__setitem__(k, D[i, j]) for (i, j), k in idx.items()]
    A = lil_matrix((2 * n + (n - 1) * (n - 2), m)); lo, hi, r = [], [], 0
    for i in range(n):
        for j in range(n):
            if i != j:
                A[r, idx[i, j]] = 1
        lo.append(1); hi.append(1); r += 1
        for j in range(n):
            if i != j:
                A[r, idx[j, i]] = 1
        lo.append(1); hi.append(1); r += 1
    U = len(idx)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                A[r, U + i] = 1; A[r, U + j] = -1; A[r, idx[i, j]] = n - 1
                lo.append(-np.inf); hi.append(n - 2); r += 1
    lb = np.zeros(m); ub = np.ones(m); lb[U:] = 0; ub[U:] = n - 1
    integ = np.ones(m); integ[U:] = 0
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": 300})
    return float(res.fun)


def exact_knapsack(v, w, B):
    res = milp(-np.asarray(v), constraints=LinearConstraint(np.asarray([w]), -np.inf, B),
               integrality=np.ones(len(v)), bounds=Bounds(0, 1))
    return float(-res.fun)


def mclp_arrays(rows, k, radius):
    D = dist_matrix(rows); n = len(rows); w = np.array([r["pop"] for r in rows], float)
    cover = D <= radius                                     # [site, demand]
    return n, w, cover, k


def exact_mclp(rows, k, radius):
    n, w, cover, k = mclp_arrays(rows, k, radius)
    c = np.concatenate([np.zeros(n), -w])                   # x_i sites, y_j covered
    A = lil_matrix((n + 1, 2 * n))
    for j in range(n):
        A[j, n + j] = 1
        for i in np.where(cover[:, j])[0]:
            A[j, i] = -1
    A[n, :n] = 1
    res = milp(c, constraints=LinearConstraint(A.tocsr(), np.r_[np.full(n, -np.inf), k], np.r_[np.zeros(n), k]),
               integrality=np.ones(2 * n), bounds=Bounds(0, 1))
    return float(-res.fun)


def exact_pmedian(rows, p):
    D = dist_matrix(rows); n = len(rows); w = np.array([r["pop"] for r in rows], float)
    c = np.concatenate([np.zeros(n), (w[:, None] * D).ravel()])        # x_i, y_ji (demand j to site i)
    m = n + n * n
    A = lil_matrix((n + n * n + 1, m)); lo, hi, r = [], [], 0
    for j in range(n):
        for i in range(n):
            A[r, n + j * n + i] = 1
        lo.append(1); hi.append(1); r += 1
    for j in range(n):
        for i in range(n):
            A[r, n + j * n + i] = 1; A[r, i] = -1; lo.append(-np.inf); hi.append(0); r += 1
    A[r, :n] = 1; lo.append(p); hi.append(p); r += 1
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo, hi), integrality=np.ones(m), bounds=Bounds(0, 1))
    return float(res.fun)


# ---------------------------------------------------------------- cuOpt (GPU)

def cuopt_tsp(D, time_limit=5.0):
    import cudf
    from cuopt import routing
    n = len(D)
    dm = routing.DataModel(n, 1)
    dm.add_cost_matrix(cudf.DataFrame(D.astype(np.float32)))
    dm.set_vehicle_locations(cudf.Series([0], dtype="int32"), cudf.Series([0], dtype="int32"))
    ss = routing.SolverSettings()
    ss.set_time_limit(time_limit)
    t = time.time()
    sol = routing.Solve(dm, ss)
    wall = time.time() - t
    route = sol.get_route()["route"].to_arrow().to_pylist()
    length = float(sum(D[a, b] for a, b in zip(route[:-1], route[1:])))
    visits = sorted(set(route))
    return {"objective": length, "seconds": wall, "status": str(sol.get_status()), "covers_all": visits == list(range(n))}


def cuopt_milp(build, sense_max, time_limit=10.0):
    from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE, MINIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings
    prob = Problem("nova-q")
    build(prob, INTEGER)
    s = SolverSettings()
    s.set_parameter("time_limit", time_limit)
    t = time.time()
    prob.solve(s)
    return {"objective": float(prob.ObjValue), "seconds": time.time() - t, "status": str(prob.Status)}


def knapsack_builder(v, w, B):
    def build(prob, INTEGER):
        from cuopt.linear_programming.problem import MAXIMIZE
        x = [prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"x{i}") for i in range(len(v))]
        prob.addConstraint(sum(float(w[i]) * x[i] for i in range(len(v))) <= float(B))
        prob.setObjective(sum(float(v[i]) * x[i] for i in range(len(v))), sense=MAXIMIZE)
    return build


def mclp_builder(rows, k, radius):
    n, w, cover, k = mclp_arrays(rows, k, radius)

    def build(prob, INTEGER):
        from cuopt.linear_programming.problem import MAXIMIZE
        x = [prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"x{i}") for i in range(n)]
        y = [prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"y{j}") for j in range(n)]
        for j in range(n):
            prob.addConstraint(y[j] - sum(x[i] for i in np.where(cover[:, j])[0]) <= 0)
        prob.addConstraint(sum(x) == k)
        prob.setObjective(sum(float(w[j]) * y[j] for j in range(n)), sense=MAXIMIZE)
    return build


def pmedian_builder(rows, p):
    D = dist_matrix(rows); n = len(rows); w = np.array([r["pop"] for r in rows], float)

    def build(prob, INTEGER):
        from cuopt.linear_programming.problem import MINIMIZE
        x = [prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"x{i}") for i in range(n)]
        y = [[prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"y{j}_{i}") for i in range(n)] for j in range(n)]
        for j in range(n):
            prob.addConstraint(sum(y[j]) == 1)
            for i in range(n):
                prob.addConstraint(y[j][i] - x[i] <= 0)
        prob.addConstraint(sum(x) == p)
        prob.setObjective(sum(float(w[j] * D[j, i]) * y[j][i] for j in range(n) for i in range(n)), sense=MINIMIZE)
    return build


def main():
    # the LLM servers hold ~95% of each GPU: managed (unified) memory lets cuOpt's MIP workspace page to host memory
    import rmm
    rmm.reinitialize(managed_memory=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # warm-up: the first GPU call pays CUDA initialisation; it is reported apart, not inside a solve time
    t = time.time(); cuopt_tsp(dist_matrix(truth("truth_melbourne_20261021.json", "melbourne_bike_tour")["stations"]), 1.0)
    rows.append({"case": "_warmup", "seconds": time.time() - t})
    for case, file, kind in (("youbike_tour", "truth_20261011.json", "tour"),
                             ("melbourne_bike_tour", "truth_melbourne_20261021.json", "tour"),
                             ("paris_velib_tour", "truth_test_20261022.json", "tour"),
                             ("nyc_pb_budget", "truth_nyc_pb_20261022.json", "knapsack"),
                             ("cambridge_pb_budget", "truth_test_20261022.json", "knapsack"),
                             ("ev_chargers_s", "truth_20261011.json", "mclp"),
                             ("london_ambulance_s", "truth_intl_20261011.json", "pmedian")):
        tr = truth(file, case)
        t0 = time.time()
        if kind == "tour":
            D = dist_matrix(tr["stations"]); n = len(D)
            opt = exact_tsp(D); t_exact = time.time() - t0
            cu = cuopt_tsp(D)
            better = "min"
        elif kind == "knapsack":
            v = [p["votes"] for p in tr["projects"]]; w = [p["cost"] for p in tr["projects"]]; n = len(v)
            opt = exact_knapsack(v, w, tr["budget"]); t_exact = time.time() - t0
            cu = cuopt_milp(knapsack_builder(v, w, tr["budget"]), True)
            better = "max"
        elif kind == "mclp":
            n = len(tr["villages"])
            opt = exact_mclp(tr["villages"], tr["k"], tr["radius_m"]); t_exact = time.time() - t0
            cu = cuopt_milp(mclp_builder(tr["villages"], tr["k"], tr["radius_m"]), True)
            better = "max"
        else:
            n = len(tr["areas"])
            opt = exact_pmedian(tr["areas"], tr["p"]); t_exact = time.time() - t0
            cu = cuopt_milp(pmedian_builder(tr["areas"], tr["p"]), False)
            better = "min"
        pipe = pipeline_deliveries(case)
        gap = lambda q: (q - opt) / opt if better == "min" else (opt - q) / opt
        row = {"case": case, "kind": kind, "n": n, "optimum": opt, "exact_cpu_s": round(t_exact, 3),
               "cuopt": cu, "cuopt_gap": round(gap(cu["objective"]), 5),
               "pipeline_n": len(pipe), "pipeline_best_gap": round(min(gap(q) for q in pipe), 5) if pipe else None,
               "pipeline_median_gap": round(float(np.median([gap(q) for q in pipe])), 5) if pipe else None}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
