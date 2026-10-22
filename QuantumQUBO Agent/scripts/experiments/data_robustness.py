"""Data robustness sweep: many real public datasets from around the world through the system's data stage.

For each (country, platform, query) the platform connectors return candidates; the first readable files are
downloaded and handed to the Data Adapter with one of two fixed, problem-independent instrument tasks:

  points  "every <thing> in <place> as one point per real place, with a name and WGS84 coordinates"
  stats   "per area, the latest-period total of <measure>, at the lowest administrative level"

Each dataset records: found, downloaded, readable, detected situations, adapter success / attempts / tools, and
independent sanity checks that do not use the system's own judgement (coordinates inside the country, no duplicate
places, not more rows than the raw table, no total/aggregate rows). This is a DEVELOPMENT diagnostic: datasets
used here to find toolkit gaps cannot later serve as evidence of generalisation.

    python scripts/experiments/data_robustness.py --tag data_robustness_20261011
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
os.environ.setdefault("LLM_REQUEST_TIMEOUT_S", "900")

from qubo_auto.annealing import opendata, platforms
from qubo_auto.annealing.adapter import build_adapter
from qubo_auto.annealing.artifacts import atomic_write
from qubo_auto.annealing.datakit import load_tables
from qubo_auto.annealing.modeler import load_module
from qubo_auto.annealing.prep import situations
from qubo_auto.llm_client import LLMClient

MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
BBOX = {"TW": (21.8, 119.3, 25.4, 122.1), "UK": (49.8, -8.7, 60.9, 1.8), "DE": (47.2, 5.8, 55.1, 15.1),
        "FR": (41.3, -5.2, 51.1, 9.6), "US": (24.4, -125.0, 49.5, -66.9), "CA": (41.6, -141.1, 83.2, -52.6),
        "AU": (-43.7, 113.1, -10.6, 153.7)}
# (country, platform, portal, query, task, place, thing)
QUERIES = [
    ("TW", "govtw", "", "臺北市 公園", "points", "臺北市", "公園"),
    ("TW", "govtw", "", "臺北市 停車場", "points", "臺北市", "停車場"),
    ("TW", "govtw", "", "新北市 圖書館", "points", "新北市", "圖書館"),
    ("TW", "govtw", "", "高雄市 各里人口數", "stats", "高雄市", "人口"),
    ("TW", "govtw", "", "臺中市 各區人口", "stats", "臺中市", "人口"),
    ("UK", "ckan", "data.gov.uk", "GP practices locations", "points", "England", "GP practice"),
    ("UK", "ckan", "data.gov.uk", "public toilets", "points", "the United Kingdom", "public toilet"),
    ("UK", "arcgis", "", "London schools", "points", "London", "school"),
    ("UK", "ckan", "data.gov.uk", "population estimates local authority", "stats", "the United Kingdom", "population"),
    ("DE", "ckan", "govdata.de", "Spielplätze", "points", "Germany", "Spielplatz (playground)"),
    ("DE", "europa", "", "Spielplatz Berlin", "points", "Berlin", "Spielplatz (playground)"),
    ("DE", "ckan", "govdata.de", "Einwohner Bezirke", "stats", "Germany", "Einwohner (residents)"),
    ("DE", "ckan", "govdata.de", "Defibrillator Standorte", "points", "Germany", "AED / Defibrillator"),
    ("FR", "opendatasoft", "opendata.paris.fr", "défibrillateurs", "points", "Paris", "défibrillateur (AED)"),
    ("FR", "opendatasoft", "opendata.paris.fr", "fontaines à boire", "points", "Paris", "fontaine à boire"),
    ("FR", "europa", "", "musées France", "points", "France", "musée"),
    ("FR", "opendatasoft", "opendata.paris.fr", "population arrondissement", "stats", "Paris", "population"),
    ("US", "socrata", "data.cityofnewyork.us", "public libraries", "points", "New York City", "library"),
    ("US", "socrata", "data.cityofchicago.org", "Divvy bicycle stations", "points", "Chicago", "bike station"),
    ("US", "socrata", "data.cityofnewyork.us", "parks properties", "points", "New York City", "park"),
    ("US", "socrata", "data.cityofchicago.org", "population census tract", "stats", "Chicago", "population"),
    ("CA", "ckan", "open.canada.ca", "hospitals", "points", "Canada", "hospital"),
    ("CA", "ckan", "open.canada.ca", "population by census division", "stats", "Canada", "population"),
    ("AU", "ckan", "data.gov.au", "public toilets", "points", "Australia", "public toilet"),
    ("AU", "ckan", "data.gov.au", "playgrounds", "points", "Australia", "playground"),
]

POINTS_MODEL = '''
DATA_SCHEMA = {"sets": {"places": {"meaning": "one real place", "source": "public"}},
  "params": {"name": {"meaning": "name of the place", "index": ["places"], "type": "text", "source": "public"},
             "lat": {"meaning": "latitude WGS84", "index": ["places"], "type": "number", "source": "public", "unit": "deg_lat"},
             "lon": {"meaning": "longitude WGS84", "index": ["places"], "type": "number", "source": "public", "unit": "deg_lon"}}}
def make_example_data(size, seed):
    return {"places": [f"P{i}" for i in range(size)], "name": [f"place {i}" for i in range(size)],
            "lat": [25.0 + i / 100 for i in range(size)], "lon": [121.5 + i / 100 for i in range(size)]}
def build_ir(data):
    n = len(data["places"])
    return {"name": "points", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "variable_keys": [{"places": p} for p in data["places"]], "linear": {str(i): 1 for i in range(n)},
            "quadratic": [], "constant": 0, "constraints": []}
def describe_solution(data, x):
    return {"places": [n for n, b in zip(data["name"], x) if b], "lat": data["lat"], "lon": data["lon"]}
'''
STATS_MODEL = '''
DATA_SCHEMA = {"sets": {"areas": {"meaning": "an area at the lowest administrative level in the table", "source": "public"}},
  "params": {"name": {"meaning": "name of the area", "index": ["areas"], "type": "text", "source": "public"},
             "value": {"meaning": "the measure for the latest period, all categories combined", "index": ["areas"],
                       "type": "number", "source": "public", "min": 0}}}
def make_example_data(size, seed):
    return {"areas": [f"A{i}" for i in range(size)], "name": [f"area {i}" for i in range(size)],
            "value": [100.0 + i for i in range(size)]}
def build_ir(data):
    n = len(data["areas"])
    return {"name": "stats", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "variable_keys": [{"areas": a} for a in data["areas"]],
            "linear": {str(i): float(v) for i, v in enumerate(data["value"])}, "quadratic": [], "constant": 0,
            "constraints": []}
def describe_solution(data, x):
    return {"areas": [n for n, b in zip(data["name"], x) if b], "value": data["value"]}
'''
TOTAL_WORDS = re.compile(r"^(總計|合計|小計|全市|total|totals|gesamt|insgesamt|ensemble|all\b|sum\b)", re.I)


def candidates(platform, portal, query):
    if platform == "govtw":
        out = []
        for h in opendata.search_govtw(query, limit=4):
            for r in opendata.resources_govtw(h["nid"])[:1]:
                out.append({"id": f"gov-{h['nid']}", "title": h["title"], "format": r["format"], "url": r["url"],
                            "page": h["url"], "source": "data.gov.tw"})
        return out
    return platforms.search(platform, portal, query)


def sanity(task, country, data, raw_rows):
    """Independent checks; never use the system's own verdicts."""
    checks = {}
    key = "places" if task == "points" else "areas"
    n = len(data.get(key, []))
    checks["rows_out"], checks["rows_raw"] = n, raw_rows
    checks["not_more_than_raw"] = n <= raw_rows
    checks["nonempty"] = n > 0
    names = [str(v).strip().lower() for v in data.get("name", [])]
    checks["duplicate_names"] = len(names) - len(set(names))
    if task == "points":
        b = BBOX[country]
        inside = sum(1 for a, o in zip(data.get("lat", []), data.get("lon", []))
                     if isinstance(a, (int, float)) and isinstance(o, (int, float)) and b[0] <= a <= b[2] and b[1] <= o <= b[3])
        checks["inside_country"] = f"{inside}/{n}"
        checks["ok"] = n > 0 and inside == n and checks["not_more_than_raw"]
    else:
        totals = [v for v in data.get("name", []) if TOTAL_WORDS.match(str(v).strip())]
        checks["total_rows_left"] = totals[:5]
        numeric = sum(isinstance(v, (int, float)) for v in data.get("value", []))
        checks["numeric"] = f"{numeric}/{n}"
        checks["ok"] = n > 0 and not totals and numeric == n and checks["not_more_than_raw"]
    return checks


def run_system(item, out: Path, client):
    """The system's own acquisition: Scout queries + platform connectors + previews + ranking + retries, then the
    toolkit-enforced adapter with the data double-check. Only the request is given; no platform or query."""
    from qubo_auto.annealing.acquisition import prepare_real_data
    country, platform, portal, query, task, place, thing = item
    rec = {"country": country, "platform": "system", "portal": "", "query": query, "task": task, "stage": "search"}
    started = time.time()
    spec = (f"List every {thing} in {place} as one point per real place (rows describing the same place count "
            f"once), with its name and WGS84 latitude/longitude; drop rows without a usable location. Use official "
            f"open data of {place}." if task == "points" else
            f"For each area at the LOWEST administrative level available for {place}, the {thing} for the LATEST "
            f"period, all categories (sex, age, ...) combined; no total or aggregate rows. Use official open data.")
    code = POINTS_MODEL if task == "points" else STATS_MODEL
    try:
        data, errors, report = prepare_real_data(code, spec, client, MODEL, out / "files", search_rounds=2,
                                                 max_downloads=4, log=lambda m: None, data_check=True)
        files = [p for p in (out / "files").glob("*") if p.is_file() and not p.name.startswith("_")]
        rec["found"] = len(files)
        tables = {}
        for f in files:
            try:
                tables.update(load_tables([f]))
            except Exception:
                pass
        rec["relevant"] = len(files)
        if tables:
            rec["tables"] = {k: list(v.shape) for k, v in tables.items()}
            rec["situations"] = [f"{s.kind} ({s.column})" for s in situations.detect(tables)]
            rec["stage"] = "adapter"
        else:
            rec["stage"] = "download" if files else "search"
        src = json.loads((out / "files" / "_sources.json").read_text()) if (out / "files" / "_sources.json").exists() else {}
        rec["sources"] = [{"title": v.get("title"), "url": v.get("resource_url") or v.get("url")} for v in src.values()]
        rec["title"] = "; ".join(v.get("title", "")[:40] for v in src.values())
        rec.update(adapter_ok=not errors and bool(data), adapter_errors=[str(e)[:300] for e in errors][:4],
                   tools=report.get("tools_used", []), flags=report.get("flags", [])[:4],
                   data_check=report.get("data_check"), review=report.get("review_required", [])[:4])
        if data and not errors:
            raw_rows = sum(len(v) for v in tables.values())
            rec["sanity"] = sanity(task, country, data, raw_rows)
            rec["stage"] = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        (out / "traceback.txt").write_text(traceback.format_exc())
    finally:
        rec["seconds"] = round(time.time() - started, 1)
    return rec


def run(item, out: Path, client):
    country, platform, portal, query, task, place, thing = item
    rec = {"country": country, "platform": platform, "portal": portal, "query": query, "task": task, "stage": "search"}
    started = time.time()
    try:
        cands = [c for c in candidates(platform, portal, query) if c.get("format", "").upper() in platforms.READABLE]
        rec["found"] = len(cands)
        if not cands:
            return rec
        # the system's own relevance ranking chooses what to download (the first search hit was a budget table)
        ns = load_module(POINTS_MODEL if task == "points" else STATS_MODEL)
        spec = (f"List every {thing} in {place} as one point per real place (rows describing the same place count "
                f"once), with its name and WGS84 latitude/longitude; drop rows without a usable location."
                if task == "points" else
                f"For each area at the LOWEST administrative level in the table for {place}, the {thing} for the "
                f"LATEST period, all categories (sex, age, ...) combined; no total or aggregate rows.")
        for c in cands:
            c.setdefault("detail", "")
        ranked = opendata.rank_candidates(spec, ns["DATA_SCHEMA"], cands, client, MODEL)
        cands = [c for c in ranked if c.get("relevant")]
        rec["relevant"] = len(cands)
        if not cands:
            return rec
        rec["stage"] = "download"
        tables, files = {}, []
        for c in cands[:3]:
            try:
                path = opendata.download(c["url"], out / "files", c.get("format", ""), c.get("title", "data")[:40])
                t = load_tables([path])
                if t and max(len(df) for df in t.values()) > 0:
                    tables, files = t, [path]
                    rec.update(title=c.get("title"), url=c["url"], format=c.get("format"))
                    break
            except Exception as e:
                rec.setdefault("download_errors", []).append(f"{c.get('title', '')[:40]}: {type(e).__name__}: {str(e)[:160]}")
        if not tables:
            return rec
        rec["stage"] = "adapter"
        rec["tables"] = {k: list(v.shape) for k, v in tables.items()}
        rec["situations"] = [f"{s.kind} ({s.column})" for s in situations.detect(tables)]
        raw_rows = sum(len(v) for v in tables.values())
        code, data, errors, report = build_adapter(ns, tables, spec, client, MODEL, attempts=3, data_check=True,
                                                   sources=[{"title": rec.get("title", ""), "url": rec.get("url", "")}])
        atomic_write(out / "adapter.py", code)
        rec.update(adapter_ok=not errors, adapter_errors=[e[:300] for e in errors][:4],
                   tools=report.get("tools_used", []), flags=report.get("flags", [])[:4],
                   data_check=report.get("data_check"), review=report.get("review_required", [])[:4])
        if not errors:
            rec["stage"] = "sanity"
            rec["sanity"] = sanity(task, country, data, raw_rows)
            rec["stage"] = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        (out / "traceback.txt").write_text(traceback.format_exc())
    finally:
        rec["seconds"] = round(time.time() - started, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", nargs="*", help="countries to run")
    args = ap.parse_args()
    base = ROOT / "results/experiments" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    rows_path = base / "results.jsonl"
    done = {json.loads(l)["query"] for l in rows_path.read_text().splitlines()} if rows_path.exists() else set()
    client = LLMClient(base / "transcripts", "robust", max_tokens=5_000_000, use_cache=False)
    for item in QUERIES:
        if (args.only and item[0] not in args.only) or item[3] in done:
            continue
        out = base / re.sub(r"\W+", "_", f"{item[0]}_{item[3]}")[:60]
        out.mkdir(parents=True, exist_ok=True)
        rec = run(item, out, client)
        with rows_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        s = rec.get("sanity", {})
        print(f"{item[0]} {item[3][:34]:<34} found={rec.get('found')} relevant={rec.get('relevant')} stage={rec['stage']} adapter={rec.get('adapter_ok')} "
              f"sane={s.get('ok')} {rec['seconds']}s {rec.get('error', '')[:80]}", flush=True)


if __name__ == "__main__":
    main()
