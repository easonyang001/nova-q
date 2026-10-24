"""Where does NVIDIA cuOpt pay off? The same real-city problems grown to city scale, cuOpt (GPU) against HiGHS MILP
(CPU) with the same time limit, and for tours also a classical CPU heuristic (nearest neighbour + 2-opt).

    /research/conda-envs/cuopt-env/bin/python scripts/experiments/cuopt_scale.py --out results/experiments/cuopt_20261023/scale.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuopt_compare import T, hav, dist_matrix, truth, cuopt_tsp, cuopt_milp, mclp_builder, pmedian_builder, \
    mclp_arrays  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def nn_2opt(D, time_limit):
    """Nearest neighbour, then 2-opt until no gain or the time limit (vectorised over one end of the move)."""
    t0 = time.time(); n = len(D)
    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j]); tour.append(nxt); left.remove(nxt)
    tour = np.array(tour)
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            c, d = tour[i:], np.roll(tour, -1)[i:]
            delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
            j = int(np.argmin(delta))
            if delta[j] < -1e-9:
                tour[i:i + j + 1] = tour[i:i + j + 1][::-1]
                improved = True
            if time.time() - t0 > time_limit:
                break
    length = float(sum(D[tour[k], tour[(k + 1) % n]] for k in range(n)))
    return {"objective": length, "seconds": time.time() - t0}


def highs_tsp(D, time_limit):
    """MTZ MILP on the CPU with a time limit: best incumbent (None when it has none)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    n = len(D)
    idx = {(i, j): k for k, (i, j) in enumerate((i, j) for i in range(n) for j in range(n) if i != j)}
    m = len(idx) + n
    c = np.zeros(m)
    for (i, j), k in idx.items():
        c[k] = D[i, j]
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
    lb = np.zeros(m); ub = np.ones(m); ub[U:] = n - 1
    integ = np.ones(m); integ[U:] = 0
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit})
    return {"objective": float(res.fun) if res.x is not None else None, "seconds": time.time() - t0,
            "status": int(res.status)}


def highs_milp_mclp(rows, k, radius, time_limit):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    n, w, cover, k = mclp_arrays(rows, k, radius)
    c = np.concatenate([np.zeros(n), -w])
    A = lil_matrix((n + 1, 2 * n))
    for j in range(n):
        A[j, n + j] = 1
        for i in np.where(cover[:, j])[0]:
            A[j, i] = -1
    A[n, :n] = 1
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), np.r_[np.full(n, -np.inf), k], np.r_[np.zeros(n), k]),
               integrality=np.ones(2 * n), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    return {"objective": float(-res.fun) if res.x is not None else None, "seconds": time.time() - t0, "status": int(res.status)}


def highs_milp_pmedian(rows, p, time_limit):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    D = dist_matrix(rows); n = len(rows); w = np.array([r["pop"] for r in rows], float)
    c = np.concatenate([np.zeros(n), (w[:, None] * D).ravel()])
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
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo, hi), integrality=np.ones(m), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    return {"objective": float(res.fun) if res.x is not None else None, "seconds": time.time() - t0, "status": int(res.status)}


def main():
    import rmm
    rmm.reinitialize(managed_memory=True)            # the LLM servers hold most of each GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=[50, 100, 200, 400, 800])
    a = ap.parse_args()
    out = ROOT / a.out
    rows = []

    def emit(r):
        rows.append(r); print(json.dumps(r), flush=True)
        out.write_text("\n".join(json.dumps(x) for x in rows) + "\n")

    cuopt_tsp(dist_matrix(truth("truth_melbourne_20261021.json", "melbourne_bike_tour")["stations"]), 1.0)   # warm-up
    v = pd.read_csv(T / "raw_test_20261022" / "velib_stations.csv", sep=";", encoding="utf-8-sig")
    xy = v["coordonnees_geo"].str.split(",", expand=True).astype(float)
    d0 = hav(xy[0].to_numpy(), xy[1].to_numpy(), 48.8443, 2.3744)
    order = np.argsort(d0)
    for n in a.sizes:
        pick = order[:n]
        stations = [{"lat": float(xy[0][i]), "lon": float(xy[1][i])} for i in pick]
        D = dist_matrix(stations)
        tl = max(10.0, n / 10)
        cu = cuopt_tsp(D, tl)
        heur = nn_2opt(D, tl)
        hi = highs_tsp(D, tl) if n <= 200 else {"objective": None, "seconds": None, "status": "skipped: MTZ model too large"}
        emit({"problem": "tour (Paris Velib')", "n": n, "time_limit_s": tl, "cuopt": cu, "cpu_2opt": heur, "cpu_highs": hi})
    tl = 60.0
    ev = truth("truth_20261011.json", "ev_chargers")
    emit({"problem": "max coverage (all Taipei villages)", "n": len(ev["villages"]), "time_limit_s": tl,
          "cuopt": cuopt_milp(mclp_builder(ev["villages"], ev["k"], ev["radius_m"]), True, tl),
          "cpu_highs": highs_milp_mclp(ev["villages"], ev["k"], ev["radius_m"], tl)})
    lo = truth("truth_intl_20261011.json", "london_ambulance_l")
    emit({"problem": "p-median (Camden + Islington LSOAs)", "n": len(lo["areas"]), "time_limit_s": tl,
          "cuopt": cuopt_milp(pmedian_builder(lo["areas"], lo["p"]), False, tl),
          "cpu_highs": highs_milp_pmedian(lo["areas"], lo["p"], tl)})


if __name__ == "__main__":
    main()
