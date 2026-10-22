"""Data preparation INTO MODEL FORMATS on the 100-dataset sweep's own downloads (no search, no solving).

Each dataset gets the model input format of its problem type (format_templates.py): distance matrices as derived
parameters, user-defined sets, reducible sets, matrices over two sets. The adapter mode comes from
QUBO_ADAPTER_MODE (agent / plan / code), one mode per process. The 100 items are split once, by a fixed hash, into
70 'dev' items (look at failures, fix generic tool bugs) and 30 'holdout' items (only measured).

    QUBO_ADAPTER_MODE=agent python scripts/experiments/format_sweep.py --tag fmt_20261013 --split dev --workers 6
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from dotenv import load_dotenv                                             # noqa: E402
load_dotenv(ROOT / ".env")
os.environ.setdefault("LLM_REQUEST_TIMEOUT_S", "1200")

import format_templates as F                                               # noqa: E402
from data_robustness import BBOX                                           # noqa: E402
from data_sweep_100 import ROWS                                            # noqa: E402
from qubo_auto.annealing import adapter                                    # noqa: E402
from qubo_auto.annealing.datakit import load_tables                        # noqa: E402
from qubo_auto.annealing.modeler import load_module                        # noqa: E402
from qubo_auto.llm_client import LLMClient                                 # noqa: E402

SOURCE = ROOT / "results/experiments/data_sweep100_system_v2_20261012"
MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")


def folder(row) -> Path:
    country, _, _, query = row[:4]
    return SOURCE / re.sub(r"\s+", "_", f"{country} {query}".strip())


def split_of(row) -> str:
    h = int(hashlib.sha256(f"{row[0]}|{row[3]}".encode()).hexdigest(), 16) % 100
    return "dev" if h < 70 else "holdout"


def run_item(row, out: Path) -> dict:
    country, _, _, query, kind, place, thing, ptype = row
    name = F.template_for(kind, ptype)
    rec = {"country": country, "query": query, "problem_type": ptype, "template": name,
           "mode": adapter.ADAPTER_MODE, "split": split_of(row)}
    t0 = time.time()
    files_dir = folder(row) / "files"
    files = [p for p in files_dir.glob("*") if p.is_file() and not p.name.startswith("_")] if files_dir.exists() else []
    if not files:
        rec["status"] = "no_downloaded_data"
        return rec
    try:
        tables = {}
        for f in files:
            try:
                tables.update(load_tables([f]))
            except Exception:
                pass
        if not tables:
            rec["status"] = "unreadable"
            return rec
        ns = load_module(F.module_source(name))
        client = LLMClient(out / "transcripts", re.sub(r"\W+", "_", f"{country}_{query}")[:60], max_tokens=3_000_000,
                           use_cache=False)
        src = json.loads((files_dir / "_sources.json").read_text()) if (files_dir / "_sources.json").exists() else {}
        sources = [{"title": v.get("title", ""), "url": v.get("resource_url") or v.get("url", "")} for v in src.values()]
        code, data, errors, report = adapter.build_adapter(ns, tables, F.spec_text(name, thing, place), client, MODEL,
                                                           attempts=3, data_check=True, sources=sources)
        rec.update(adapter_ok=not errors and bool(data), errors=[str(e)[:400] for e in errors][:3],
                   adapter_mode_used=report.get("adapter_mode"), agent_steps=len(report.get("agent_steps", [])),
                   tools=report.get("tools_used", []), tokens=client.tokens_used)
        if data and not errors:
            rec["sanity"] = F.sanity(name, data, BBOX.get(country))
        rec["status"] = "ok" if rec.get("sanity", {}).get("ok") else ("sanity_failed" if data and not errors else "adapter_failed")
    except Exception as e:
        rec["status"], rec["errors"] = "crash", [f"{type(e).__name__}: {str(e)[:300]}"]
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
    return rec


def failure_key(rec: dict) -> str:
    """Group failures by cause: the first error line with numbers, names and quotes stripped."""
    if rec["status"] == "sanity_failed":
        return "sanity: " + ",".join(k for k, v in rec["sanity"].items() if v is False)
    e = (rec.get("errors") or [""])[0]
    e = re.sub(r"Traceback.*", "", e, flags=re.S)
    e = re.sub(r"'[^']*'|\"[^\"]*\"|\d+(\.\d+)?", "_", e)
    return f"{rec['status']}: " + re.sub(r"\s+", " ", e)[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = ROOT / "results/experiments" / a.tag / adapter.ADAPTER_MODE
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in ROWS if a.split == "all" or split_of(r) == a.split]
    if a.only:
        rows = [r for r in rows if any(o in r[3] for o in a.only)]
    path = out / f"{a.split}.jsonl"
    done = {json.loads(l)["query"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [r for r in rows if r[3] not in done]
    print(f"{len(rows)} items ({len(todo)} to run), mode {adapter.ADAPTER_MODE}, {a.workers} workers", flush=True)
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(a.workers) as pool:
        # write each result as soon as it finishes: in-order output lost a whole slow run when the session ended
        for fut in as_completed([pool.submit(run_item, r, out) for r in todo]):
            rec = fut.result()
            with path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"[{rec['country']}] {rec['query'][:34]:<34} {rec['template']:<15} {rec['status']:<18} "
                  f"{rec.get('seconds', 0):>6}s", flush=True)
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    have = [r for r in recs if r["status"] not in ("no_downloaded_data", "unreadable")]
    ok = sum(r["status"] == "ok" for r in have)
    print(f"\nwith data: {len(have)}/{len(recs)}; prepared correctly: {ok}/{len(have)}")
    groups = collections.Counter(failure_key(r) for r in have if r["status"] != "ok")
    print("failure groups (most frequent first):")
    for k, n in groups.most_common(25):
        print(f"  {n:>3}  {k}")
    by_t = collections.defaultdict(lambda: [0, 0])
    for r in have:
        by_t[r["template"]][0] += r["status"] == "ok"; by_t[r["template"]][1] += 1
    print("by template:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_t.items())})


if __name__ == "__main__":
    main()
