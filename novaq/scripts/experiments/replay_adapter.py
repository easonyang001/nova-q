"""Replay only the data preparation of finished runs with another adapter mode, and judge the data again.

Uses each run's own downloaded files, confirmed specification and model; nothing is searched or solved. A quick
development check of a data-preparation change (e.g. QUBO_ADAPTER_MODE=agent) on the cases where it failed before.

    QUBO_ADAPTER_MODE=agent python scripts/experiments/replay_adapter.py results/e2e_smoke20_20261012/full_rag/aed_groups_l_r0 ...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "acceptance", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from dotenv import load_dotenv                                              # noqa: E402
load_dotenv(ROOT / ".env")
from judge import judge                                                    # noqa: E402
from qubo_auto.annealing import adapter, clarify as C, modeler as M        # noqa: E402
from qubo_auto.annealing.datakit import load_tables                        # noqa: E402
from qubo_auto.llm_client import LLMClient                                 # noqa: E402
import run_e2e_batch as B                                                  # noqa: E402

TRUTHS = [ROOT / "results/acceptance_truth/truth_20261011.json", ROOT / "results/acceptance_truth/truth_intl_20261011.json"]


def main(run_dirs):
    truth = {}
    for t in TRUTHS:
        truth.update(json.loads(t.read_text()))
    cases = {c["id"]: c for c in __import__("caselist").load()["cases"]}
    for d in map(Path, run_dirs):
        rec = json.loads((d / "record.json").read_text())
        code = (d / "model_repaired.py").read_text() if (d / "model_repaired.py").exists() else (d / "model.py").read_text()
        ns = M.load_module(code)
        draft = C.DraftSpec.model_validate(json.loads((d / "session.json").read_text())["draft"])
        files = [f for f in (d / "data_files").iterdir() if not f.name.startswith("_")]
        sources = list(json.loads((d / "data_files/_sources.json").read_text()).values()) \
            if (d / "data_files/_sources.json").exists() else []
        client = LLMClient(ROOT / "results/experiments/replay_transcripts", d.name, max_tokens=3_000_000, use_cache=False)
        _, data, errors, report = adapter.build_adapter(ns, load_tables(files), C.spec_for_data(draft), client, B.MODEL,
                                                       sources=sources, data_check=True)
        verdict = {}
        if data:
            tr, _ = B.live_truth(cases[rec["case"]], truth, d, B.system_downloads(d / "data_files"))
            verdict = judge(rec["case"], tr, ns["DATA_SCHEMA"], data, ns["build_ir"](data), None, {}).get("data", {})
        before = (rec.get("verdict") or {}).get("data", {}).get("ok")
        print(f"{rec['case']:<22} before data={before}  now data={verdict.get('ok')}  mode={report.get('adapter_mode')} "
              f"agent_steps={len(report.get('agent_steps', []))} errors={'; '.join(errors)[:160]}", flush=True)
        out = d.parent.parent.parent / "replay"
        out.mkdir(exist_ok=True)
        (out / f"{d.parent.parent.name}_{d.name}_{adapter.ADAPTER_MODE}.json").write_text(json.dumps(
            {"case": rec["case"], "before": before, "now": verdict, "errors": errors,
             "agent_steps": report.get("agent_steps", []), "mode": report.get("adapter_mode")},
            ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
