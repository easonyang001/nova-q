"""Download a corpus of real open-data files to find where the data layer breaks (no labels needed).

One thread per portal, at most one request per second per host, an identifying User-Agent, robots.txt respected,
files capped in size. Output (never committed: results/** is ignored):

    results/data_cache/corpus/<portal>/<n>_<id>.<ext>      the files
    results/data_cache/corpus/manifest.jsonl               one line per file: portal, title, url, format, bytes, sha256, licence

Portals of the evaluation's held-out cities are refused (EXCLUDED): their datasets must not feed any tuning, or the
held-out tests stop being held out. Taipei, London, Melbourne, New York and Berlin are development cities and allowed.

    python scripts/crawl_open_data.py --per-portal 120
    python scripts/crawl_open_data.py --portals data.gov.uk --per-portal 3        # quick check
"""
from __future__ import annotations
import argparse
import hashlib
import json
import threading
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_cache" / "corpus"
UA = "QuantumQUBOAgent research (open-data catalogue index)"       # the same identity catalog.py uses
MAX_BYTES = 15 * 1024 * 1024

# held-out test cities (README test sets 1 and 2): never crawled
EXCLUDED = {"data.gov.sg", "opendata.paris.fr", "data.cambridgema.gov", "opendata.swiss", "ckan.opendata.swiss",
            "data.gov.ie", "data.somervillema.gov"}

CKAN = {"data.gov.uk": "https://data.gov.uk/api/action", "govdata.de": "https://ckan.govdata.de/api/3/action",
        "open.canada.ca": "https://open.canada.ca/data/api/action", "data.gov.au": "https://data.gov.au/data/api/3/action",
        "data.go.jp": "https://www.data.go.jp/data/api/3/action", "data.overheid.nl": "https://data.overheid.nl/data/api/3/action",
        "datenregister.berlin.de": "https://datenregister.berlin.de/api/3/action"}
SOCRATA = {"data.cityofnewyork.us"}
FORMATS = {"CSV": "csv", "GEOJSON": "geojson", "JSON": "json", "XLSX": "xlsx", "ZIP": "zip", "TSV": "tsv"}
QUERIES = ["stations", "population", "parks", "schools", "hospitals", "transport", "budget", "facilities", "locations",
           "census", "bicycle", "public toilets", "libraries", "air quality", "buildings", "bus stops", "health", "trees"]


class Host:
    """Politeness: one request per `gap` seconds, robots.txt honoured."""
    def __init__(self, base: str, gap: float = 1.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.gap, self.last = gap, 0.0
        self.robots = urllib.robotparser.RobotFileParser()
        try:
            r = self.session.get(f"{urlparse(base).scheme}://{urlparse(base).netloc}/robots.txt", timeout=20)
            self.robots.parse(r.text.splitlines() if r.ok else [])
        except requests.RequestException:
            self.robots.parse([])

    def get(self, url, **kw):
        if not self.robots.can_fetch(UA, url):
            raise PermissionError(f"robots.txt disallows {url}")
        wait = self.gap - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        return self.session.get(url, timeout=kw.pop("timeout", 30), **kw)


def candidates_ckan(host: Host, base: str, portal: str):
    seen = set()
    for q in QUERIES:
        try:
            r = host.get(f"{base}/package_search", params={"q": q, "rows": 25})
            pkgs = r.json().get("result", {}).get("results", [])
        except Exception as e:
            print(f"[{portal}] search {q!r} failed: {type(e).__name__}", flush=True)
            continue
        for p in pkgs:
            for res in p.get("resources", []):
                fmt = str(res.get("format", "")).upper().strip().lstrip(".")
                url = res.get("url")
                if fmt in FORMATS and url and url not in seen:
                    seen.add(url)
                    title = p.get("title")
                    yield {"title": title if isinstance(title, str) else json.dumps(title, ensure_ascii=False)[:200],
                           "url": url, "format": fmt, "licence": p.get("license_id") or p.get("license_title"),
                           "page": f"{base.split('/api')[0]}/dataset/{p.get('name', '')}"}
                    break                                     # one file per dataset


def candidates_socrata(host: Host, portal: str):
    seen = set()
    for q in QUERIES:
        try:
            r = host.get(f"https://{portal}/api/catalog/v1", params={"q": q, "only": "datasets", "limit": 25, "domains": portal,
                                                                          "search_context": portal})
            rows = r.json().get("results", [])
        except Exception as e:
            print(f"[{portal}] search {q!r} failed: {type(e).__name__}", flush=True)
            continue
        for row in rows:
            res = row.get("resource", {})
            uid = res.get("id")
            if uid and uid not in seen and res.get("type") == "dataset":
                seen.add(uid)
                yield {"title": res.get("name"), "url": f"https://{portal}/resource/{uid}.csv?$limit=5000", "format": "CSV",
                       "licence": row.get("metadata", {}).get("license"), "page": row.get("link")}


def download(host: Host, cand: dict, dest: Path) -> dict | None:
    r = host.get(cand["url"], stream=True, allow_redirects=True)
    if not r.ok:
        return {"error": f"HTTP {r.status_code}"}
    data, size = [], 0
    for chunk in r.iter_content(65536):
        size += len(chunk)
        if size > MAX_BYTES:
            return {"error": f"larger than {MAX_BYTES // 2**20} MB"}
        data.append(chunk)
    raw = b"".join(data)
    dest.write_bytes(raw)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "content_type": r.headers.get("content-type", "")}


def crawl(portal: str, n: int, lock: threading.Lock, manifest: Path):
    if portal in EXCLUDED:
        print(f"[{portal}] excluded (held-out city)")
        return
    base = CKAN.get(portal)
    host = Host(base or f"https://{portal}/")
    folder = OUT / portal
    folder.mkdir(parents=True, exist_ok=True)
    cands = candidates_socrata(host, portal) if portal in SOCRATA else candidates_ckan(host, base, portal)
    got = 0
    for cand in cands:
        if got >= n:
            break
        ext = FORMATS[cand["format"]]
        name = f"{got:03d}_{hashlib.md5(cand['url'].encode()).hexdigest()[:8]}.{ext}"
        try:
            info = download(host, cand, folder / name)
        except Exception as e:
            info = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        rec = {"portal": portal, "file": f"{portal}/{name}", **cand, **(info or {}), "fetched": time.strftime("%Y-%m-%d")}
        with lock, manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if "error" not in rec:
            got += 1
        if got % 10 == 0 and "error" not in rec:
            print(f"[{portal}] {got}/{n}", flush=True)
    print(f"[{portal}] done: {got} files", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portals", nargs="*", default=sorted(CKAN) + sorted(SOCRATA))
    ap.add_argument("--per-portal", type=int, default=120)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, lock = OUT / "manifest.jsonl", threading.Lock()
    threads = [threading.Thread(target=crawl, args=(p, a.per_portal, lock, manifest)) for p in a.portals]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
