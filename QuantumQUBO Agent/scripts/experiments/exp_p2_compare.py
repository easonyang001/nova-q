"""P2 experiment (spec §10): budget-matched comparison of agents vs. baselines on the SA stand-in.

    python scripts/experiments/exp_p2_compare.py --tag p2_sa_v1 --seeds 10 --workers 8
    python scripts/experiments/exp_p2_compare.py --tag smoke --seeds 1 --workers 4 --policies default rule agent-multi-rag

Design:
  * Oracle-QUBO condition: generated instances with exact references.
  * Train/test split (§7.3): agents first run on TRAIN instances and write the experience store;
    the store is then frozen and every TEST run reads it (write disabled). Test ids are excluded.
  * Same loop config (solver-call budget, patience, caps) for every policy.
Outputs in results/experiments/<tag>/: runs.csv, summary.md, loops/*.jsonl, experience.jsonl
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.examples import knapsack_random, select_k_random
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.rag import KnowledgeBase

POLICIES = {  # name -> (policy, agent mode, rag on)
    "default": ("default", None, False), "rule": ("rule", None, False), "random": ("random", None, False),
    "optuna": ("optuna", None, False),
    "penalty_tpe": ("penalty_tpe", None, False), "penalty_grid": ("penalty_grid", None, False),
    "penalty_cem": ("penalty_cem", None, False), "penalty_sequential": ("penalty_sequential", None, False),
    "agent-multi-rag": ("agent", "multi", True), "agent-single-rag": ("agent", "single", True),
    "agent-multi-norag": ("agent", "multi", False),
}
SPECS = [  # (generator, kwargs) per problem family / size; calibrated so SA defaults do NOT already
           # reach the optimum (knapsack gap ~15-21%, select_k ~0-2%). Graph colouring is solved by SA defaults
           # at these sizes, so it is kept for hardware experiments rather than this SA comparison.
    (knapsack_random, {"n": 40}), (knapsack_random, {"n": 60}),
    (select_k_random, {"n": 60, "k": 10}), (select_k_random, {"n": 80, "k": 12}),
]
INSTANCES: list = []          # filled before forking so workers inherit them (instances hold lambdas)


def build_instances(per_spec: int, seed0: int) -> list:
    return [gen(seed=seed0 + i, **kw) for gen, kw in SPECS for i in range(per_spec)]


def _cfg(base: dict, out: Path, run_id: str, rag_on: bool, mode: str | None, write_exp: bool, exclude: list[str]) -> dict:
    cfg = copy.deepcopy(base)
    cfg["logging"] = {"loop_log": str(out / "loops" / f"{run_id}.jsonl"),
                      "backend_log": str(out / "backend" / f"{run_id}.jsonl")}
    cfg["rag"].update(enabled=rag_on, write_experience=write_exp, experience_path=str(out / "experience.jsonl"),
                      exclude_instance_ids=exclude)
    if mode:
        cfg["agents"]["mode"] = mode
    return cfg


def run_one(job: dict) -> dict:
    inst = INSTANCES[job["instance"]]
    pol_name, mode, rag_on = POLICIES[job["policy"]]
    cfg = _cfg(job["cfg"], Path(job["out"]), job["run_id"], rag_on, mode, job["write_exp"], job["exclude"])
    rag = KnowledgeBase.from_config(cfg["rag"], ROOT) if rag_on else None
    t0 = time.time()
    try:
        policy = make_policy(pol_name, seed=job["seed"], cfg=cfg, root=ROOT, rag=rag)
        res = run_loop(inst, get_backend("sa"), policy, cfg, rag=rag, seed=job["seed"], root=ROOT, verbose=False)
        row = {"best_objective": res.best_objective, "gap": res.best_gap, "feasibility": res.best_feasibility_rate,
               "solver_calls": res.solver_calls, "stop_reason": res.stop_reason, "llm_tokens": res.llm_tokens,
               "fallback_rounds": sum(r.source.startswith("fallback") for r in res.rounds),
               "guardrail_rejections": sum(len(r.rejected) for r in res.rounds), "error": ""}
    except Exception as e:
        row = {"best_objective": None, "gap": None, "feasibility": None, "solver_calls": 0, "stop_reason": "crash",
               "llm_tokens": 0, "fallback_rounds": 0, "guardrail_rejections": 0, "error": f"{type(e).__name__}: {e}"}
    return {"phase": job["phase"], "policy": job["policy"], "instance_id": inst.instance_id,
            "problem": inst.name, "n_variables": inst.n_variables, "reference": inst.reference_objective,
            "seed": job["seed"], "wall_s": round(time.time() - t0, 1), **row}


def summarize(rows: list[dict], out: Path) -> str:
    from scipy.stats import wilcoxon
    test = [r for r in rows if r["phase"] == "test"]
    policies = [p for p in POLICIES if any(r["policy"] == p for r in test)]
    lines = ["| Policy | Runs | Mean gap | Optimal (gap=0) | Mean solver calls | Mean LLM tokens | Fallback rounds | Crashes |",
             "|---|---|---|---|---|---|---|---|"]
    by = {p: [r for r in test if r["policy"] == p] for p in policies}
    for p in policies:
        rs = by[p]
        gaps = [r["gap"] if r["gap"] is not None else 1.0 for r in rs]     # no feasible solution counts as gap 1
        lines.append(f"| {p} | {len(rs)} | {statistics.mean(gaps):.4f} | {sum(g == 0 for g in gaps)}/{len(rs)} | "
                     f"{statistics.mean(r['solver_calls'] for r in rs):.2f} | {statistics.mean(r['llm_tokens'] for r in rs):.0f} | "
                     f"{sum(r['fallback_rounds'] for r in rs)} | {sum(r['stop_reason'] == 'crash' for r in rs)} |")
    lines += ["", "Paired Wilcoxon signed-rank on gap (same instance and seed), two-sided:", "",
              "| Comparison | n pairs | Mean gap A | Mean gap B | p-value |", "|---|---|---|---|---|"]
    key = lambda r: (r["instance_id"], r["seed"])
    done = set()
    for a in [p for p in policies if p.startswith("agent")]:
        for b in [p for p in policies if p != a and frozenset((a, p)) not in done]:
            done.add(frozenset((a, b)))
            A = {key(r): (r["gap"] if r["gap"] is not None else 1.0) for r in by[a]}
            B = {key(r): (r["gap"] if r["gap"] is not None else 1.0) for r in by[b]}
            ks = sorted(set(A) & set(B))
            ga, gb = [A[k] for k in ks], [B[k] for k in ks]
            try:
                pval = wilcoxon(ga, gb).pvalue if any(x != y for x, y in zip(ga, gb)) else 1.0
            except ValueError:
                pval = float("nan")
            lines.append(f"| {a} vs {b} | {len(ks)} | {statistics.mean(ga):.4f} | {statistics.mean(gb):.4f} | {pval:.4g} |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--per-spec", type=int, default=2, help="test instances per (family, size)")
    ap.add_argument("--train-per-spec", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", type=Path, default=ROOT / "config_loop.yaml")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--resume", action="store_true", help="keep rows in runs.csv and skip finished jobs")
    args = ap.parse_args()

    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    (out / "config_used.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    train = build_instances(args.train_per_spec, seed0=1000)
    test = build_instances(args.per_spec, seed0=0)
    INSTANCES.extend(train + test)
    test_ids = [i.instance_id for i in test]
    print(f"train instances: {len(train)}  test instances: {len(test)}  seeds: {args.seeds}  policies: {args.policies}")

    ctx = mp.get_context("fork")
    rows: list[dict] = []
    # Phase 1: agents with RAG build the experience store on TRAIN instances only.
    train_jobs = [{"phase": "train", "policy": "agent-multi-rag", "instance": i, "seed": 0, "out": str(out),
                   "run_id": f"train_{INSTANCES[i].instance_id}", "cfg": base_cfg, "write_exp": True, "exclude": test_ids}
                  for i in range(len(train))] if any(p.endswith("-rag") for p in args.policies) else []
    # Phase 2: every policy on TEST instances, experience frozen.
    test_jobs = [{"phase": "test", "policy": p, "instance": len(train) + j, "seed": s, "out": str(out),
                  "run_id": f"test_{p}_{INSTANCES[len(train) + j].instance_id}_s{s}", "cfg": base_cfg,
                  "write_exp": False, "exclude": test_ids}
                 for p in args.policies for j in range(len(test)) for s in range(args.seeds)]
    csv_path = out / "runs.csv"
    if args.resume and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in ("gap", "best_objective", "feasibility", "reference"):
                    r[k] = float(r[k]) if r[k] not in ("", "None") else None
                for k in ("seed", "solver_calls", "llm_tokens", "fallback_rounds", "guardrail_rejections", "n_variables"):
                    r[k] = int(r[k])
                rows.append(r)
        finished = {(r["phase"], r["policy"], r["instance_id"], r["seed"]) for r in rows if r["stop_reason"] != "crash"}
        rows = [r for r in rows if r["stop_reason"] != "crash"]
        skip = lambda j: (j["phase"], j["policy"], INSTANCES[j["instance"]].instance_id, j["seed"]) in finished
        train_jobs = [j for j in train_jobs if not skip(j)]
        test_jobs = [j for j in test_jobs if not skip(j)]
        print(f"resume: {len(rows)} finished rows kept; {len(train_jobs)} train + {len(test_jobs)} test jobs left")
    with csv_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as f:
        writer = None
        if args.resume and rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        for phase, jobs in (("train", train_jobs), ("test", test_jobs)):
            if not jobs:
                continue
            t0 = time.time()
            with ctx.Pool(args.workers) as pool:
                for k, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    if k % 20 == 0 or k == len(jobs):
                        print(f"[{phase}] {k}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    print(summarize(rows, out))


if __name__ == "__main__":
    main()
