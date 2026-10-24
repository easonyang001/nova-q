"""Re-judge a finished batch with the current judge and truth (runs keep the verdict of the judge they started with).

    python scripts/rejudge.py e2e_smoke13_20261012
Writes results/<tag>/rejudged.json and prints one line per run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "acceptance")); sys.path.insert(0, str(ROOT / "scripts"))
from judge import judge                                                     # noqa: E402
from qubo_auto.annealing import modeler as M                               # noqa: E402
import run_e2e_batch as B                                                   # noqa: E402

TRUTHS = [ROOT / "results/acceptance_truth/truth_20261011.json", ROOT / "results/acceptance_truth/truth_intl_20261011.json",
          ROOT / "results/acceptance_truth/truth_melbourne_20261021.json",
          ROOT / "results/acceptance_truth/truth_basel_20261021.json",
          ROOT / "results/acceptance_truth/truth_nyc_pb_20261022.json",
          ROOT / "results/acceptance_truth/truth_test_20261022.json",
          ROOT / "results/acceptance_truth/truth_test2_20261024.json"]


def main(tag: str):
    truth = {}
    for t in TRUTHS:
        truth.update(json.loads(t.read_text()))
    cases = {c["id"]: c for c in json.loads((ROOT / "acceptance/cases.json").read_text())["cases"]}
    out = []
    for rec_path in sorted((ROOT / "results" / tag).glob("*/*_r*/record.json")):
        d = rec_path.parent
        rec = json.loads(rec_path.read_text())
        row = {"arm": rec.get("arm"), "case": rec["case"], "run": d.name, "failed_stage": rec.get("failed_stage"),
               "delivery_source": (rec.get("delivery") or {}).get("solution_source"), "cost": rec.get("cost")}
        try:
            if rec.get("arm") == "direct":
                v = rec.get("verdict") or {}
            elif (d / "workflow.json").exists():
                wf = json.loads((d / "workflow.json").read_text())
                ns = M.load_module(wf["_final_code"])
                data = wf["_final_data"]
                jt, row["truth_used"] = B.live_truth(cases[rec["case"]], truth, d, B.system_downloads(d / "data_files"))
                v = judge(rec["case"], jt, ns["DATA_SCHEMA"], data, ns["build_ir"](data), wf.get("solution"), {})
            else:
                v = {}
        except Exception as e:
            v, row["rejudge_error"] = {}, f"{type(e).__name__}: {e}"[:200]
        for g in ("data", "model", "delivery"):
            row[g] = (v.get(g) or {}).get("ok")
        row["joint"] = bool(row["data"] and row["model"] and row["delivery"])
        out.append(row)
        print(f"{row['arm']:<10} {row['case']:<22} failed={str(row['failed_stage']):<8} data={row['data']} "
              f"model={row['model']} delivery={row['delivery']} joint={row['joint']} source={row['delivery_source']}")
    (ROOT / "results" / tag / "rejudged.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"joint success {sum(r['joint'] for r in out)}/{len(out)}")


if __name__ == "__main__":
    main(sys.argv[1])
