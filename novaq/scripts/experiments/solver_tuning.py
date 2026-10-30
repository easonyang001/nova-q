"""Fair solver comparison: every solver gets the same tuning budget on separate instances, for BOTH its own parameters
and the penalty weights, and is then scored on held-out instances on the ORIGINAL problem.

Why: the earlier comparison (qaoa_compare.py) fed one fixed QUBO to every solver; only SQA's beta had been tuned (on
3 instances that were also scored), and CGA / D-Wave ran with untuned constants. Penalty weights change the energy
scale, so energies of different penalties cannot be compared; this script therefore scores the best FEASIBLE sample on
the original objective (gap to the enumerated optimum) and the feasible fraction, never the QUBO energy.

Protocol
  1. Instances (instances/*.json from qaoa_instances.py) are split once, by a stable hash of the file name, into a
     tuning set (about 1/3, at least one per case) and an evaluation set.
  2. Per solver: Optuna TPE, --trials trials. A trial = (solver parameters, one global penalty multiplier on the
     compiled default weights) run once on every tuning instance. Loss = mean over instances of the relative gap
     (1.0 when no feasible sample was found). Budget parameters (reads / seconds) are NOT tuned: --reads, --cga-s,
     --da-s are the same for every solver.
  3. Evaluation: the solver's defaults and its tuned configuration on every evaluation instance, --eval-seeds seeds.
  4. summary.md: default vs tuned per solver, calls used for tuning, wall time.

    python scripts/experiments/solver_tuning.py --dir results/experiments/qaoa_cudaq_20261020 --solvers sa sqa
    python scripts/experiments/solver_tuning.py --dir ... --solvers cga dwave_qpu fujitsu_da --allow-real --trials 15
    python scripts/experiments/solver_tuning.py --synthetic 9 --solvers sa --trials 8     # smoke test, NOT a result

Real back-ends spend quota (Fujitsu 300 DA-min/month, QPU access time): they need --allow-real, and the number of
calls is printed before anything is sent.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv                                                 # noqa: E402
load_dotenv(ROOT / ".env")
from qaoa_compare import enumerate_min, score                                  # noqa: E402
from qubo_auto.annealing.backends import get_backend                           # noqa: E402
from qubo_auto.annealing.encodings import compile_problem                      # noqa: E402
from qubo_auto.annealing.modeler import to_ir                                  # noqa: E402

REAL = {"cga", "dwave_qpu", "fujitsu_da"}

# (kind, low, high) per tunable. Budget parameters are deliberately absent. CGA exposes only its time limit, so for
# CGA the search is over the penalty multiplier alone.
SPACES = {
    "sa": {"num_sweeps": ("log_int", 100, 20000), "beta_min": ("log", 0.01, 1.0), "beta_max": ("log", 1.0, 100.0)},
    "sqa": {"num_sweeps": ("log_int", 100, 5000), "trotter": ("cat", [4, 8, 16]), "beta": ("log", 0.5, 50.0),
            "gamma": ("log", 0.1, 10.0)},
    "dwave_qpu": {"annealing_time": ("log", 1.0, 200.0), "chain_strength": ("log", 0.3, 5.0)},
    "cga": {},
    "fujitsu_da": {"gs_level": ("int", 0, 100), "num_run": ("cat", [16, 32, 64, 128]), "num_group": ("cat", [1, 2, 4, 8])},
}
PENALTY_MULT = ("log", 1.0, 16.0)


def suggest(trial, name, spec):
    kind = spec[0]
    if kind == "log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "log_int":
        return trial.suggest_int(name, spec[1], spec[2], log=True)
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    return trial.suggest_categorical(name, spec[1])


def budget(solver, a):
    return {"sa": {"num_reads": a.reads}, "sqa": {"num_reads": a.reads}, "dwave_qpu": {"num_reads": a.reads},
            "cga": {"timeout": a.cga_s}, "fujitsu_da": {"time_limit_sec": a.da_s}}[solver]


def synthetic(k):
    """k small knapsack models (12 items). Smoke-test material only: never report results from these."""
    rng = np.random.default_rng(0)
    out = []
    for s in range(k):
        n = 10 + s % 3
        v, w = rng.integers(5, 60, n), rng.integers(3, 30, n)
        raw = {"name": f"syn_knap_{s}", "variable_names": [f"x{i}" for i in range(n)], "sense": "maximize",
               "variable_keys": [{"items": str(i)} for i in range(n)],
               "linear": {i: float(v[i]) for i in range(n)}, "quadratic": [],
               "constraints": [{"name": "budget", "kind": "linear", "coeffs": {i: float(w[i]) for i in range(n)},
                                "sense": "le", "rhs": float(w.sum() // 2)}]}
        out.append((f"syn_knap_n{n}_s{s}", "syn_knap", raw))
    return out


def load(a):
    if a.synthetic:
        return synthetic(a.synthetic)
    rows = []
    for f in sorted((ROOT / a.dir / "instances").glob("*.json")):
        inst = json.loads(f.read_text())
        rows.append((f.stem, inst["case"], inst["ir_raw"]))
    if not rows:
        sys.exit(f"no instances in {ROOT / a.dir / 'instances'} (build them with qaoa_instances.py)")
    return rows


def prepare(stem, case, raw, max_qubits):
    ir = to_ir(raw, stem)
    prob = compile_problem(ir)
    if prob.n_variables > max_qubits:
        return None
    qubo, off = prob.to_qubo()
    q = {(int(i), int(j)): float(c) for (i, j), c in qubo.items()}
    ground, states = enumerate_min(prob.n_variables, [[i, j, c] for (i, j), c in q.items()], off)
    feas = [s[:ir.n_variables] for s in states if all(c.satisfied(s[:ir.n_variables]) for c in ir.constraints)]
    if not feas:
        print(f"{stem}: default QUBO ground state is infeasible; skipped")
        return None
    return {"stem": stem, "case": case, "ir": ir, "prob": prob, "opt": ir.objective(feas[0]), "n": prob.n_variables}


def split(insts):
    tune, evaluate, seen = [], [], set()
    for x in sorted(insts, key=lambda x: x["stem"]):
        h = int(hashlib.md5(x["stem"].encode()).hexdigest(), 16) % 3
        if h == 0 or x["case"] not in seen:
            tune.append(x)
            seen.add(x["case"])
        else:
            evaluate.append(x)
    return tune, evaluate


def solve(backend, inst, params, mult, seed):
    """One run; the QUBO is rescaled to max|coef| = 1 (the argmin is unchanged) so absolute betas mean the same thing."""
    prob = inst["prob"]
    weights = {k: v * mult for k, v in prob.default_penalty_weights.items()}
    qubo, off = prob.to_qubo(weights)
    scale = max([abs(c) for c in qubo.values()] + [1e-12])
    q = {(int(i), int(j)): float(c) / scale for (i, j), c in qubo.items()}
    p = dict(params)
    if backend.name == "sa":
        p["seed"] = seed
    t0 = time.time()
    r = backend.run(q, off / scale, prob.n_variables, p)
    s = score(inst["ir"], r.samples, r.num_occurrences, inst["opt"])
    s["wall_s"] = time.time() - t0
    return s


def loss(rec):
    return 1.0 if rec["gap"] is None else min(1.0, rec["gap"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/experiments/qaoa_cudaq_20261020")
    ap.add_argument("--solvers", nargs="+", default=["sa", "sqa"], choices=sorted(SPACES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--reads", type=int, default=1000, help="same read budget for sa / sqa / dwave_qpu")
    ap.add_argument("--cga-s", type=int, default=10)
    ap.add_argument("--da-s", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--max-qubits", type=int, default=26)
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0, help="smoke test on k synthetic knapsacks (not a result)")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    insts = [x for x in (prepare(*r, a.max_qubits) for r in load(a)) if x]
    tune, evaluate = split(insts)
    print(f"{len(insts)} instances: {len(tune)} for tuning, {len(evaluate)} for evaluation")
    if not evaluate:
        sys.exit("need at least one evaluation instance")
    out = Path(a.out) if a.out else ROOT / "results" / "experiments" / ("solver_tuning_smoke" if a.synthetic else "solver_tuning")
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"tune": [x["stem"] for x in tune], "eval": [x["stem"] for x in evaluate],
                                                "synthetic": bool(a.synthetic)}, indent=1))
    calls = {s: a.trials * len(tune) + len(evaluate) * 2 * (1 if s in REAL else a.eval_seeds) for s in a.solvers}
    print("solver calls planned:", calls)
    real = [s for s in a.solvers if s in REAL]
    if real and not a.allow_real:
        sys.exit(f"{real} spend quota; rerun with --allow-real after checking the call counts above")

    rows = []
    for solver in a.solvers:
        backend = get_backend(solver)
        space = SPACES[solver]
        fixed = budget(solver, a)
        log = (out / f"tuning_{solver}.jsonl").open("w")

        def objective(trial):
            params = {k: suggest(trial, k, spec) for k, spec in space.items()}
            mult = suggest(trial, "penalty_mult", PENALTY_MULT)
            recs = [solve(backend, x, {**fixed, **params}, mult, a.seed) for x in tune]
            value = statistics.mean(loss(r) for r in recs)
            log.write(json.dumps({"trial": trial.number, "params": params, "penalty_mult": mult, "loss": value,
                                  "p_feasible": statistics.mean(r["p_feasible"] for r in recs)}) + "\n")
            log.flush()
            return value

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed))
        t0 = time.time()
        study.optimize(objective, n_trials=a.trials)
        best = {**study.best_params}
        mult = best.pop("penalty_mult")
        (out / f"best_{solver}.json").write_text(json.dumps({"params": best, "penalty_mult": mult, "fixed": fixed,
                                                             "tuning_loss": study.best_value}, indent=1))
        seeds = [a.seed] if solver in REAL else list(range(a.eval_seeds))
        for label, params, m in (("default", {}, 1.0), ("tuned", best, mult)):
            for x in evaluate:
                for sd in seeds:
                    r = solve(backend, x, {**fixed, **params}, m, sd)
                    rows.append({"solver": solver, "config": label, "instance": x["stem"], "case": x["case"],
                                 "n_qubits": x["n"], "seed": sd, **r})
        (out / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"{solver}: tuned in {time.time() - t0:.0f}s, best loss {study.best_value:.4f}, penalty x{mult:.2f}")

    lines = ["# Solver tuning summary", "",
             "SYNTHETIC SMOKE TEST: not a result." if a.synthetic else "",
             f"{len(tune)} tuning / {len(evaluate)} evaluation instances; {a.trials} TPE trials per solver; equal budgets: "
             f"{a.reads} reads (sa, sqa, dwave_qpu), CGA {a.cga_s} s, Fujitsu DA {a.da_s} s. Loss = relative gap of the best "
             "feasible sample on the original objective (1.0 when none was feasible).", "",
             "| Solver | Config | Mean gap | Found optimum | Mean P(feasible) | Median wall s | Tuning calls |", "|---|---|--:|--:|--:|--:|--:|"]
    for solver in a.solvers:
        for label in ("default", "tuned"):
            rs = [r for r in rows if r["solver"] == solver and r["config"] == label]
            lines.append(f"| {solver} | {label} | {statistics.mean(loss(r) for r in rs):.4f} | "
                         f"{statistics.mean(1.0 if r['found_optimum'] else 0.0 for r in rs):.2f} | "
                         f"{statistics.mean(r['p_feasible'] for r in rs):.3f} | {statistics.median(r['wall_s'] for r in rs):.1f} | "
                         f"{a.trials * len(tune) if label == 'tuned' else 0} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
