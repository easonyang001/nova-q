"""QAOA on CUDA-Q on the agents' own verified models, where the QUBO fits one GPU: every judged-passing run of the
given cases is rebuilt from its saved model and data, solved by the qaoa_cudaq backend (same loop and policy as the
end-to-end runs), and QAOA's own best sample (no classical polish) is judged by the same frozen judge.

    python scripts/experiments/agent_models_qaoa.py --cases cambridge_pb_budget \
        --out results/experiments/qaoa_agent_20261024/runs.jsonl
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acceptance"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qubo_auto.annealing import modeler as M  # noqa: E402
from qubo_auto.annealing.backends import get_backend  # noqa: E402
from qubo_auto.annealing.baselines import make_policy  # noqa: E402
from qubo_auto.annealing.loop import run_loop  # noqa: E402
from judge import judge  # noqa: E402
from agent_models_cuopt import passed, truth_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--max-solver-calls", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["run"] for l in out.read_text().splitlines()} if out.exists() else set()
    truth = truth_all()
    cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))
    cfg["loop"]["max_solver_calls"] = a.max_solver_calls
    for case in a.cases:
        for rec_path in sorted(glob.glob(str(ROOT / "results" / "e2e_*" / "full_rag" / f"{case}_r*" / "record.json"))):
            run = Path(rec_path).parent
            key = str(run.relative_to(ROOT / "results"))
            if key in done or not passed(run):
                continue
            code_path = next(p for p in (run / "model_final.py", run / "model_repaired.py", run / "model.py") if p.exists())
            data_path = next(p for p in (run / "instance_final.json", run / "instance.json") if p.exists())
            ns = M.load_module(code_path.read_text(encoding="utf-8"))
            data = json.loads(data_path.read_text(encoding="utf-8"))
            raw = ns["build_ir"](data)
            ir = M.to_ir(raw, case)
            work = out.parent / key.replace("/", "__")
            res = run_loop(None, get_backend("qaoa_cudaq"), make_policy("default"), cfg, ir=ir, root=work,
                           log=lambda m: print("   ", m, flush=True), seed=0)
            x = res.best_solution[:ir.n_variables] if res.best_solution else None
            feasible = x is not None and all(c.satisfied(x) for c in ir.constraints)
            row = {"run": key, "case": case, "n_variables": ir.n_variables, "qaoa_objective": ir.objective(x) if feasible else None,
                   "feasible": feasible, "best_feasibility_rate": res.best_feasibility_rate, "solver_calls": res.solver_calls,
                   "wall_s": res.wall_time_s, "rounds": [r.model_dump() for r in res.rounds][:3]}
            if feasible:
                v = judge(case, truth, ns["DATA_SCHEMA"], data, raw, x, {})
                row["judged"] = {g: (v.get(g) or {}).get("ok") for g in ("data", "model", "delivery")}
                row["true_quality"] = ((v.get("delivery") or {}).get("checks") or {}).get("true_quality")
                row["gap_vs_reference"] = ((v.get("delivery") or {}).get("checks") or {}).get("gap_vs_reference")
            print(json.dumps({k: row[k] for k in row if k != "rounds"}), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
