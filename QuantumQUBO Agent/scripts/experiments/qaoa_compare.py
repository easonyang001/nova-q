"""QAOA (CUDA-Q) vs simulated annealing vs CGA vs exact on the reduced real instances (qaoa_instances.py).

Per instance: exact optimum by full enumeration of the QUBO (checked against the original problem's constraints and
objective), then each solver's samples are decoded and scored on the ORIGINAL problem:
  best feasible objective, gap to the optimum, P(optimal) and P(feasible) over shots/reads, wall time.
For QAOA also P(optimal) of uniform random guessing (#optimal bitstrings / 2^n), the amplification baseline.

    python scripts/experiments/qaoa_compare.py --dir results/experiments/qaoa_cudaq_20261020 --layers 1 2 3
    python scripts/experiments/qaoa_compare.py --dir ... --layers --solvers sqa dwave_qpu cga
    (cga goes through the normal backend: refused while results/CGA_PAUSED exists)
"""
import argparse
import re
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv                                                                # noqa: E402
load_dotenv(ROOT / ".env")                                                                   # D-Wave token etc.
from qubo_auto.annealing.backends import QAOABackend, SimulatedAnnealingBackend, get_backend  # noqa: E402
from qubo_auto.annealing.encodings import ProblemIR                                         # noqa: E402
from qubo_auto.annealing.modeler import to_ir                                               # noqa: E402


def qmatrix(n, qubo):
    q = np.zeros((n, n))
    for i, j, c in qubo:
        q[int(i), int(j)] += c
    return q


def enumerate_min(n, qubo, offset, chunk=1 << 20):
    """Ground energy and every ground state of x^T Q x + offset, by enumeration in chunks."""
    q = qmatrix(n, qubo)
    best, states = np.inf, []
    shifts = np.arange(n, dtype=np.int64)
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
        x = ((idx[:, None] >> shifts) & 1).astype(np.float64)
        e = np.einsum("ij,ij->i", x @ q, x) + offset
        m = e.min()
        if m < best - 1e-9:
            best, states = m, []
        if m <= best + 1e-9:
            states += [r.astype(int).tolist() for r in x[e <= best + 1e-9]]
    return float(best), states


def score(ir: ProblemIR, samples, counts, opt_value):
    """Decode on the original problem: feasibility + objective of the decision variables."""
    nd, total = ir.n_variables, sum(counts)
    feas = opt = 0
    best = None
    for x, c in zip(samples, counts):
        d = list(x[:nd])
        if all(con.satisfied(d) for con in ir.constraints):
            feas += c
            v = ir.objective(d)
            if best is None or (v > best if ir.sense == "maximize" else v < best):
                best = v
            if abs(v - opt_value) <= 1e-6 * max(1.0, abs(opt_value)):
                opt += c
    gap = None if best is None else abs(best - opt_value) / max(abs(opt_value), 1e-9)
    return {"best_feasible": best, "gap": gap, "found_optimum": gap is not None and gap <= 1e-9,
            "p_feasible": feas / total, "p_optimal": opt / total, "shots": total}


def random_mean_energy(qubo, offset):
    """Exact mean QUBO energy of a uniformly random bit string: E[x_i] = 1/2, E[x_i x_j] = 1/4 (i != j)."""
    return float(offset) + sum(c / 2 if int(i) == int(j) else c / 4 for i, j, c in qubo)


def energy_stats(energies, counts, ground, e_rand):
    """Normalised energy gap g = (E - E_ground) / (E_random - E_ground): 0 = optimum, 1 = random guessing."""
    e, w = np.asarray(energies, float), np.asarray(counts, float)
    g = (e - ground) / max(e_rand - ground, 1e-12)
    mean = float((g * w).sum() / w.sum())
    std = float(np.sqrt(((g - mean) ** 2 * w).sum() / w.sum()))
    order = np.argsort(g)
    cw = np.cumsum(w[order]) / w.sum()
    q = {f"q{int(p * 100)}": float(g[order][np.searchsorted(cw, p)]) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return {"gap_mean": mean, "gap_std": std, "gap_min": float(g.min()), **q,
            "energy_mean": float((e * w).sum() / w.sum()), "energy_min": float(e.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--sa-reads", type=int, default=1000)
    ap.add_argument("--solvers", nargs="*", default=[],
                    help="extra back-ends, one run each: sqa, dwave_qpu, cga (cga refused while results/CGA_PAUSED exists)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out-name", default="compare.jsonl", help="results file inside --dir")
    ap.add_argument("--skip-sa", action="store_true")
    ap.add_argument("--instances", default="instances", help="instance folder inside --dir")
    ap.add_argument("--cvar", type=float, default=1.0, help="QAOA CVaR alpha (<1 records solver qaoa_p{p}_cvar)")
    a = ap.parse_args()
    base = ROOT / a.dir
    out = base / a.out_name
    sample_dir = base / (out.stem + "_samples")
    sample_dir.mkdir(exist_ok=True)
    done = {(r["instance"], r["solver"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    def order(p):
        m = re.match(r"(.+)_n(\d+)(?:_s(\d+))?$", p.stem)
        return (m.group(1), int(m.group(2)), int(m.group(3) or 0))
    files = sorted((base / a.instances).glob("*.json"), key=order)
    if a.only:
        files = [f for f in files if any(o == f.stem or o in f.stem for o in a.only)]

    def emit(rec):
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{rec['instance']:<24} {rec['solver']:<10} q={rec['n_qubits']:<3} opt={rec.get('found_optimum')!s:<5} "
              f"P(opt)={rec.get('p_optimal', 0):.4f} P(feas)={rec.get('p_feasible', 0):.3f} {rec.get('wall_s', 0):7.1f}s",
              flush=True)

    for f in files:
        inst = json.loads(f.read_text())
        ir = to_ir(inst["ir_raw"], f.stem)
        n, qubo, off = inst["n_qubits"], inst["qubo"], inst["offset"]
        head = {"instance": f.stem, "case": inst["case"], "n_points": inst["n_points"], "n_qubits": n,
                "n_decision": inst["n_decision"], "n_terms": len(qubo)}
        t0 = time.time()
        ground, gstates = enumerate_min(n, qubo, off)
        t_exact = time.time() - t0
        feas_g = [s[:ir.n_variables] for s in gstates if all(c.satisfied(s[:ir.n_variables]) for c in ir.constraints)]
        if not feas_g:
            print(f"{f.stem}: QUBO ground state violates a constraint (no feasible point at this size, e.g. equal group sizes with n not divisible by the group count, or penalties too weak); skipped")
            continue
        opt_value = ir.objective(feas_g[0])
        p_random = len(gstates) / 2 ** n
        e_rand = random_mean_energy(qubo, off)

        def run_record(solver, r, ir=ir):
            """Score on the original problem + energy stats; keep the full sample distribution on disk."""
            dec = ir.n_variables
            feas = [all(c.satisfied(list(x[:dec])) for c in ir.constraints) for x in r.samples]
            (sample_dir / f"{f.stem}__{solver}.json").write_text(json.dumps(
                {"energies": r.energies, "counts": r.num_occurrences, "feasible": feas,
                 "ground_energy": ground, "random_mean_energy": e_rand}))
            return {**score(ir, r.samples, r.num_occurrences, opt_value),
                    **energy_stats(r.energies, r.num_occurrences, ground, e_rand)}
        if (f.stem, "exact") not in done:
            emit({**head, "solver": "exact", "optimum": opt_value, "ground_energy": ground,
                  "n_optimal_bitstrings": len(gstates), "p_random": p_random, "random_mean_energy": e_rand,
                  "found_optimum": True, "p_optimal": 1.0, "p_feasible": 1.0, "wall_s": t_exact})
        q = {(int(i), int(j)): float(c) for i, j, c in qubo}
        if not a.skip_sa and (f.stem, "sa") not in done:
            t0 = time.time()
            r = SimulatedAnnealingBackend().run(q, off, n, {"num_reads": a.sa_reads, "seed": 0})
            emit({**head, "solver": "sa", **run_record("sa", r),
                  "optimum": opt_value, "wall_s": time.time() - t0, "solver_s": r.solver_time_s})
        for p in a.layers:
            tag = f"qaoa_p{p}" + ("_cvar" if a.cvar < 1 else "")
            if (f.stem, tag) in done:
                continue
            t0 = time.time()
            r = QAOABackend().run(q, off, n, {"layers": p, "shots": a.shots, "restarts": a.restarts,
                                              "maxiter": a.maxiter, "seed": 0, "cvar_alpha": a.cvar})
            emit({**head, "solver": tag, **run_record(tag, r), "optimum": opt_value,
                  "p_random": p_random, "wall_s": time.time() - t0, "solver_s": r.solver_time_s,
                  "timing": r.metadata["timing"], "theta": r.metadata["theta"]})
        extra = {"sqa": {"num_reads": a.sa_reads, "num_sweeps": 1000, "trotter": 8, "beta": 200.0, "gamma": 1.0},  # beta tuned on 3 instances
                 "dwave_qpu": {"num_reads": a.sa_reads, "annealing_time": 20.0},
                 "cga": {"timeout": 10}}
        for name in a.solvers:
            if (f.stem, name) in done:
                continue
            t0 = time.time()
            try:
                r = get_backend(name).run(q, off, n, extra[name])
            except Exception as e:                                  # one failed back-end must not stop the rest
                print(f"{f.stem} {name} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            emit({**head, "solver": name, **run_record(name, r), "optimum": opt_value,
                  "wall_s": time.time() - t0, "solver_s": r.solver_time_s,
                  "metadata": {k: v for k, v in r.metadata.items() if k not in ("theta",)}})

if __name__ == "__main__":
    main()
