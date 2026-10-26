"""NVIDIA cuOpt on the agents' own verified models: every judged-passing end-to-end run is rebuilt from its saved
model and data (the version the judge saw), solved with cuOpt MILP (cuopt_ir_worker.py, in the cuOpt env), and cuOpt's
answer is judged by the same frozen judge as the SA delivery.

    python scripts/experiments/agent_models_cuopt.py --out results/experiments/cuopt_agent_20261024/runs.jsonl
"""
import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acceptance"))
from qubo_auto.annealing import modeler as M  # noqa: E402
from judge import judge  # noqa: E402

CUOPT_PY = "/research/conda-envs/cuopt-env/bin/python"
WORKER = Path(__file__).resolve().parent / "cuopt_ir_worker.py"
CASES = ("ev_chargers_s", "youbike_tour", "aed_groups", "london_ambulance_s", "melbourne_bike_tour", "nyc_pb_budget",
         "sg_hawker_groups", "paris_velib_tour", "cambridge_pb_budget",
         "basel_aed_groups", "dublin_bike_tour", "somerville_pb_budget")


def passed(run: Path) -> bool:
    """Judged pass: the batch's rejudged.json when it has one (the published verdict), else record.json."""
    rejudged = run.parent.parent / "rejudged.json"
    if rejudged.exists():
        rows = json.loads(rejudged.read_text())
        rows = rows if isinstance(rows, list) else rows.get("runs", [])
        for r in rows:
            if r.get("run") == run.name:
                return bool(r.get("joint"))
    v = json.loads((run / "record.json").read_text()).get("verdict") or {}
    return all((v.get(g) or {}).get("ok") for g in ("data", "model", "delivery"))


def truth_all() -> dict:
    t = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "acceptance_truth" / "truth_*.json"))):
        t.update(json.loads(Path(f).read_text()))
    return t


def ir_json(ir) -> dict:
    return {"n_variables": ir.n_variables, "sense": ir.sense, "constant": ir.constant,
            "linear": [[i, c] for i, c in ir.linear.items()],
            "quadratic": [[i, j, c] for (i, j), c in ir.quadratic.items()],
            "constraints": [c.model_dump() for c in ir.constraints]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["run"] for l in out.read_text().splitlines()} if out.exists() else set()
    truth = truth_all()
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
            rec = json.loads((run / "record.json").read_text())
            row = {"run": key, "case": case, "n_variables": ir.n_variables, "n_quadratic": len(ir.quadratic),
                   "sa_objective": (rec.get("delivery") or {}).get("objective"),
                   "sa_true_quality": ((rec.get("verdict") or {}).get("delivery", {}).get("checks") or {}).get("true_quality")}
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / "ir.json", Path(tmp) / "sol.json"
                src.write_text(json.dumps(ir_json(ir)))
                p = subprocess.run([CUOPT_PY, str(WORKER), str(src), str(dst), str(a.time_limit)],
                                   capture_output=True, text=True, timeout=a.time_limit * 10 + 300)
                if p.returncode or not dst.exists():
                    row["error"] = (p.stderr or p.stdout)[-800:]
                else:
                    got = json.loads(dst.read_text())
                    x = got.pop("solution")
                    row["cuopt"] = got
                    row["cuopt"]["feasible"] = all(c.satisfied(x) for c in ir.constraints)
                    row["cuopt"]["ir_objective"] = ir.objective(x)
                    snap = run / "truth_snapshot.json"
                    t = dict(truth)
                    if snap.exists():
                        t[case] = json.loads(snap.read_text())
                    v = judge(case, t, ns["DATA_SCHEMA"], data, raw, x, {})
                    row["cuopt"]["judged"] = {g: (v.get(g) or {}).get("ok") for g in ("data", "model", "delivery")}
                    row["cuopt"]["true_quality"] = ((v.get("delivery") or {}).get("checks") or {}).get("true_quality")
            print(json.dumps(row), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
