"""The 100-dataset world sweep: data needs of 12 optimisation problem types in 20 countries.

Each row: (country, platform, portal, query, task, place, thing, problem). The data stage runs exactly as in
data_robustness.py (system search + relevance ranking, download, loader, situations, toolkit-enforced adapter with the
data double-check, independent sanity checks). A Markdown report is written at the end.

    python scripts/experiments/data_sweep_100.py --tag data_sweep100_20261011
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import data_robustness as R                                     # noqa: E402

R.BBOX.update({"NL": (50.7, 3.3, 53.6, 7.3), "ES": (27.6, -18.2, 43.8, 4.4), "IT": (35.4, 6.6, 47.1, 18.6),
               "AT": (46.3, 9.5, 49.1, 17.2), "BE": (49.5, 2.5, 51.6, 6.4), "DK": (54.5, 8.0, 57.8, 15.2),
               "FI": (59.7, 20.5, 70.1, 31.6), "IE": (51.4, -10.7, 55.4, -5.9), "CH": (45.8, 5.9, 47.9, 10.5),
               "JP": (24.0, 122.9, 45.6, 146.0), "NZ": (-47.4, 166.4, -34.3, 178.6), "SG": (1.15, 103.6, 1.48, 104.1),
               "KR": (33.1, 124.6, 38.7, 131.0)})

P, S = "points", "stats"
ROWS = [
    # ---------------- location / coverage (population + candidate sites)
    ("TW", "govtw", "", "臺北市 各里人口數", S, "臺北市", "人口", "location"),
    ("TW", "govtw", "", "新北市 各里人口", S, "新北市", "人口", "location"),
    ("TW", "govtw", "", "臺中市 電動車充電站", P, "臺中市", "電動車充電站", "location"),
    ("UK", "ckan", "data.gov.uk", "LSOA population weighted centroids", P, "England and Wales", "LSOA centroid", "location"),
    ("UK", "ckan", "data.gov.uk", "electric vehicle charging points", P, "the United Kingdom", "EV charging point", "location"),
    ("DE", "ckan", "govdata.de", "Ladesäulen Elektromobilität", P, "Germany", "Ladesäule (EV charger)", "location"),
    ("DE", "ckan", "govdata.de", "Einwohner Ortsteile Berlin", S, "Berlin", "Einwohner (residents)", "location"),
    ("FR", "opendatasoft", "opendata.paris.fr", "bornes de recharge", P, "Paris", "borne de recharge (EV charger)", "location"),
    ("FR", "europa", "", "population communes France", S, "France", "population", "location"),
    ("US", "socrata", "data.cityofnewyork.us", "population by neighborhood tabulation area", S, "New York City", "population", "location"),
    ("US", "socrata", "data.cityofchicago.org", "EV charging stations", P, "Chicago", "EV charging station", "location"),
    ("CA", "ckan", "open.canada.ca", "electric vehicle charging stations", P, "Canada", "EV charging station", "location"),
    ("AU", "ckan", "data.gov.au", "electric vehicle charging stations", P, "Australia", "EV charging station", "location"),
    ("NL", "europa", "", "laadpalen", P, "the Netherlands", "laadpaal (EV charger)", "location"),
    ("ES", "europa", "", "puntos de recarga vehículo eléctrico", P, "Spain", "punto de recarga (EV charger)", "location"),
    ("NL", "europa", "", "bevolking per gemeente", S, "the Netherlands", "bevolking (population)", "location"),
    # ---------------- set cover / emergency services
    ("TW", "govtw", "", "臺北市 消防隊", P, "臺北市", "消防隊", "set_cover"),
    ("TW", "govtw", "", "AED 設置場所", P, "臺灣", "AED", "set_cover"),
    ("UK", "arcgis", "", "fire stations England", P, "England", "fire station", "set_cover"),
    ("DE", "ckan", "govdata.de", "Defibrillatoren Standorte", P, "Germany", "Defibrillator (AED)", "set_cover"),
    ("FR", "opendatasoft", "opendata.paris.fr", "défibrillateurs", P, "Paris", "défibrillateur (AED)", "set_cover"),
    ("US", "socrata", "data.cityofnewyork.us", "FDNY firehouse listing", P, "New York City", "firehouse", "set_cover"),
    ("US", "socrata", "data.sfgov.org", "fire stations", P, "San Francisco", "fire station", "set_cover"),
    ("CA", "ckan", "open.canada.ca", "fire stations", P, "Canada", "fire station", "set_cover"),
    ("AU", "ckan", "data.gov.au", "defibrillators", P, "Australia", "defibrillator (AED)", "set_cover"),
    ("IE", "europa", "", "fire stations Ireland", P, "Ireland", "fire station", "set_cover"),
    # ---------------- routing / tours (service points)
    ("TW", "govtw", "", "YouBike 站點", P, "臺北市", "YouBike 站點", "routing"),
    ("TW", "govtw", "", "臺北市 公廁", P, "臺北市", "公廁", "routing"),
    ("UK", "ckan", "data.gov.uk", "public toilets", P, "the United Kingdom", "public toilet", "routing"),
    ("UK", "arcgis", "", "Santander cycles docking stations London", P, "London", "cycle docking station", "routing"),
    ("DE", "europa", "", "Briefkästen Standorte", P, "Germany", "Briefkasten (post box)", "routing"),
    ("FR", "opendatasoft", "opendata.paris.fr", "vélib stations", P, "Paris", "Vélib' station", "routing"),
    ("FR", "opendatasoft", "opendata.paris.fr", "sanisettes", P, "Paris", "sanisette (public toilet)", "routing"),
    ("US", "socrata", "data.cityofchicago.org", "Divvy bicycle stations", P, "Chicago", "bike station", "routing"),
    ("US", "socrata", "data.cityofnewyork.us", "public libraries", P, "New York City", "library", "routing"),
    ("AU", "ckan", "data.gov.au", "public toilets", P, "Australia", "public toilet", "routing"),
    ("FI", "europa", "", "kaupunkipyöräasemat", P, "Finland", "city bike station", "routing"),
    ("DK", "europa", "", "bycykel stationer", P, "Denmark", "city bike station", "routing"),
    # ---------------- assignment with capacity (schools, shelters, hospitals)
    ("TW", "govtw", "", "避難收容處所", P, "臺灣", "避難收容處所", "assignment"),
    ("TW", "govtw", "", "臺北市 國民小學", P, "臺北市", "國民小學", "assignment"),
    ("UK", "arcgis", "", "London schools", P, "London", "school", "assignment"),
    ("UK", "ckan", "data.gov.uk", "NHS hospitals", P, "England", "hospital", "assignment"),
    ("DE", "ckan", "govdata.de", "Schulen Standorte", P, "Germany", "Schule (school)", "assignment"),
    ("FR", "europa", "", "établissements scolaires", P, "France", "école (school)", "assignment"),
    ("US", "socrata", "data.cityofchicago.org", "Chicago public schools locations", P, "Chicago", "public school", "assignment"),
    ("US", "socrata", "data.cityofnewyork.us", "school locations", P, "New York City", "school", "assignment"),
    ("CA", "ckan", "open.canada.ca", "hospitals", P, "Canada", "hospital", "assignment"),
    ("AU", "ckan", "data.gov.au", "schools locations", P, "Australia", "school", "assignment"),
    ("NZ", "arcgis", "", "New Zealand schools", P, "New Zealand", "school", "assignment"),
    ("CH", "europa", "", "Schulen Schweiz", P, "Switzerland", "Schule (school)", "assignment"),
    # ---------------- clustering (many points)
    ("TW", "govtw", "", "臺北市 行道樹", P, "臺北市", "行道樹", "clustering"),
    ("UK", "ckan", "data.gov.uk", "street trees", P, "the United Kingdom", "street tree", "clustering"),
    ("DE", "europa", "", "Spielplatz Berlin", P, "Berlin", "Spielplatz (playground)", "clustering"),
    ("FR", "opendatasoft", "opendata.paris.fr", "arbres", P, "Paris", "arbre (tree)", "clustering"),
    ("US", "socrata", "data.cityofnewyork.us", "parks properties", P, "New York City", "park", "clustering"),
    ("AU", "ckan", "data.gov.au", "playgrounds", P, "Australia", "playground", "clustering"),
    ("JP", "arcgis", "", "Japan evacuation sites", P, "Japan", "evacuation site", "clustering"),
    ("SG", "arcgis", "", "Singapore hawker centres", P, "Singapore", "hawker centre", "clustering"),
    ("KR", "arcgis", "", "Seoul public wifi", P, "Seoul", "public wifi point", "clustering"),
    # ---------------- scheduling (demand over time)
    ("TW", "govtw", "", "臺北市 119 救護 出勤 時段", S, "臺北市", "救護出勤次數", "scheduling"),
    ("TW", "govtw", "", "醫院 急診 人次 月別", S, "臺灣", "急診人次", "scheduling"),
    ("UK", "ckan", "data.gov.uk", "A&E attendances", S, "England", "A&E attendances", "scheduling"),
    ("US", "socrata", "data.cityofnewyork.us", "EMS incident dispatch", S, "New York City", "EMS incidents", "scheduling"),
    ("US", "socrata", "data.cityofchicago.org", "311 service requests", S, "Chicago", "service requests", "scheduling"),
    ("CA", "ckan", "open.canada.ca", "hospital emergency department visits", S, "Canada", "emergency visits", "scheduling"),
    ("FR", "opendatasoft", "opendata.paris.fr", "comptage vélo", S, "Paris", "bicycle counts", "scheduling"),
    ("DE", "europa", "", "Notaufnahme Fallzahlen", S, "Germany", "Notaufnahme cases", "scheduling"),
    # ---------------- resource allocation / knapsack (costs, budgets, projects)
    ("TW", "govtw", "", "臺北市 預算 各機關", S, "臺北市", "預算", "resource_allocation"),
    ("TW", "govtw", "", "公共工程 標案 金額", S, "臺灣", "標案金額", "knapsack"),
    ("UK", "ckan", "data.gov.uk", "local authority capital expenditure", S, "England", "capital expenditure", "knapsack"),
    ("US", "socrata", "data.cityofnewyork.us", "capital projects", S, "New York City", "project budget", "knapsack"),
    ("US", "socrata", "data.cityofchicago.org", "budget appropriations", S, "Chicago", "appropriation", "resource_allocation"),
    ("CA", "ckan", "open.canada.ca", "government expenditures by department", S, "Canada", "expenditure", "resource_allocation"),
    ("AU", "ckan", "data.gov.au", "budget by agency", S, "Australia", "budget", "resource_allocation"),
    ("FR", "europa", "", "budget primitif ville de Paris", S, "Paris", "budget", "resource_allocation"),
    ("DE", "ckan", "govdata.de", "Haushalt Ausgaben", S, "Germany", "Ausgaben (expenditure)", "resource_allocation"),
    ("IT", "europa", "", "bilancio comune spese", S, "Italy", "spese (expenditure)", "resource_allocation"),
    # ---------------- portfolio (prices / returns)
    ("TW", "govtw", "", "基金 淨值", S, "臺灣", "基金淨值", "portfolio"),
    ("TW", "govtw", "", "股價 指數 月", S, "臺灣", "股價指數", "portfolio"),
    ("UK", "ckan", "data.gov.uk", "house price index", S, "the United Kingdom", "house price index", "portfolio"),
    ("US", "socrata", "", "electricity prices", S, "the United States", "electricity price", "portfolio"),
    ("CA", "ckan", "open.canada.ca", "fuel prices", S, "Canada", "fuel price", "portfolio"),
    ("AU", "ckan", "data.gov.au", "fuel prices", S, "Australia", "fuel price", "portfolio"),
    ("FR", "europa", "", "prix des carburants", S, "France", "prix du carburant (fuel price)", "portfolio"),
    # ---------------- feature selection (wide measurement tables)
    ("TW", "govtw", "", "空氣品質 監測 小時值", S, "臺灣", "空氣品質", "feature_selection"),
    ("UK", "ckan", "data.gov.uk", "air quality monitoring", S, "the United Kingdom", "air quality", "feature_selection"),
    ("US", "socrata", "data.cityofnewyork.us", "air quality", S, "New York City", "air quality", "feature_selection"),
    ("DE", "europa", "", "Luftqualität Messwerte", S, "Germany", "Luftqualität (air quality)", "feature_selection"),
    ("AT", "europa", "", "Luftgüte Messstellen", S, "Austria", "Luftgüte (air quality)", "feature_selection"),
    ("ES", "europa", "", "calidad del aire", S, "Spain", "calidad del aire (air quality)", "feature_selection"),
    # ---------------- graph / network (stops, links)
    ("TW", "govtw", "", "公車站位", P, "臺北市", "公車站", "graph"),
    ("TW", "govtw", "", "臺北市 自行車道", P, "臺北市", "自行車道", "network_design"),
    ("UK", "ckan", "data.gov.uk", "NaPTAN bus stops", P, "Great Britain", "bus stop", "graph"),
    ("US", "socrata", "data.cityofchicago.org", "CTA bus stops", P, "Chicago", "bus stop", "graph"),
    ("US", "socrata", "data.cityofnewyork.us", "bicycle routes", P, "New York City", "bicycle route", "network_design"),
    ("FR", "opendatasoft", "opendata.paris.fr", "aménagements cyclables", P, "Paris", "aménagement cyclable (bike lane)", "network_design"),
    ("DE", "europa", "", "Haltestellen ÖPNV", P, "Germany", "Haltestelle (stop)", "graph"),
    ("BE", "europa", "", "arrêts transports en commun", P, "Belgium", "arrêt (transit stop)", "graph"),
    ("CA", "ckan", "open.canada.ca", "transit stops", P, "Canada", "transit stop", "graph"),
    ("AU", "ckan", "data.gov.au", "bus stops", P, "Australia", "bus stop", "graph"),
]


def classify(r: dict) -> str:
    """One failure category per dataset (or 'passed')."""
    if r.get("sanity", {}).get("ok"):
        return "passed"
    if r.get("error"):
        return "crash: " + r["error"].split(":")[0]
    if not r.get("found"):
        return "search: no candidates"
    if not r.get("relevant"):
        return "search: none relevant"
    if r.get("stage") == "download":
        return "download / unreadable"
    if r.get("adapter_ok") is False:
        e = " ".join(r.get("adapter_errors", []))
        for key, label in (("DATA CHECK", "adapter: double-check disagreement"), ("REQUIRED TOOL", "adapter: required tool not used"),
                           ("FORBIDDEN", "adapter: forbidden shortcut"), ("(0, 0)", "adapter: (0,0) coordinates"),
                           ("outside", "adapter: coordinates out of range"), ("ids repeat", "adapter: duplicate ids"),
                           ("DATA PROBLEM FOUND BY A TOOL", "adapter: tool flag"), ("JSON", "adapter: NaN / non-JSON output"),
                           ("must be a non-empty", "adapter: empty result"), ("Timeout", "adapter: timeout")):
            if key in e:
                return label
        return "adapter: other error"
    s = r.get("sanity", {})
    if s.get("inside_country") and s["inside_country"].split("/")[0] != s["inside_country"].split("/")[1]:
        return "sanity: points outside the country"
    if s.get("total_rows_left"):
        return "sanity: total rows left"
    if s.get("not_more_than_raw") is False:
        return "sanity: more rows than the raw table"
    return "sanity: other"


def report(rows: list[dict], path: Path):
    n = len(rows)
    cat = collections.Counter(classify(r) for r in rows)
    lines = [f"# 100-dataset data sweep ({n} datasets)", "",
             "Development diagnostic of the data stage (search, ranking, download, loader, situations, adapter with "
             "double-check, independent sanity checks). Not evidence of generalisation.", "",
             "## Outcome", "", "| Outcome | Datasets |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in cat.most_common()]
    for key, title in (("country", "By country"), ("problem", "By problem type"), ("platform", "By platform"),
                       ("task", "By task")):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r.get(key, "?")].append(r)
        lines += ["", f"## {title}", "", "| | n | found | relevant | readable | adapter ok | passed |", "|---|---:|---:|---:|---:|---:|---:|"]
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {g} | {len(rs)} | {sum(bool(r.get('found')) for r in rs)} | {sum(bool(r.get('relevant')) for r in rs)} | "
                         f"{sum('tables' in r for r in rs)} | {sum(bool(r.get('adapter_ok')) for r in rs)} | "
                         f"{sum(bool(r.get('sanity', {}).get('ok')) for r in rs)} |")
    sit = collections.Counter(re.sub(r" \(.*", "", s) for r in rows for s in r.get("situations", []))
    lines += ["", "## Data situations detected", "", "| Situation | Datasets |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sit.most_common()]
    tools = collections.Counter(t for r in rows for t in r.get("tools", []))
    lines += ["", "## Toolkit functions used by successful and failed adapters", "", "| Tool | Datasets |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in tools.most_common()]
    lines += ["", "## Every dataset", "", "| Country | Problem | Query | Outcome | Title | Seconds |", "|---|---|---|---|---|---:|"]
    for r in rows:
        lines.append(f"| {r['country']} | {r.get('problem', '')} | {r['query']} | {classify(r)} | "
                     f"{str(r.get('title', ''))[:50].replace('|', '/')} | {r.get('seconds')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--system", action="store_true", help="let the system's Scout search (only the request is given)")
    args = ap.parse_args()
    base = ROOT / "results/experiments" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    rows_path = base / "results.jsonl"
    if not args.report_only:
        done = {f"{r['country']}|{r['query']}" for r in map(json.loads, rows_path.read_text().splitlines())} \
            if rows_path.exists() else set()
        client = R.LLMClient(base / "transcripts", "sweep", max_tokens=50_000_000, use_cache=False)
        for row in ROWS:
            country, platform, portal, query, task, place, thing, problem = row
            if f"{country}|{query}" in done:
                continue
            out = base / re.sub(r"\W+", "_", f"{country}_{query}")[:60]
            out.mkdir(parents=True, exist_ok=True)
            rec = (R.run_system if args.system else R.run)((country, platform, portal, query, task, place, thing), out, client)
            rec["problem"] = problem
            with rows_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"[{sum(1 for _ in rows_path.open())}/{len(ROWS)}] {country} {query[:34]:<34} {classify(rec)} "
                  f"{rec['seconds']}s", flush=True)
            report([json.loads(l) for l in rows_path.read_text().splitlines()], base / "report.md")
    report([json.loads(l) for l in rows_path.read_text().splitlines()], base / "report.md")


if __name__ == "__main__":
    main()
