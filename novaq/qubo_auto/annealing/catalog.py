"""Local catalogue index of open-data portals with semantic (embedding) retrieval.

Keyword search on the portals only finds a dataset when the agent guesses the publisher's wording ("LSOA population
weighted centroids", "里界圖", "Planungsraum"); a wrong guess and the data are never seen (2026-10-12 runs). As in
OGD4All (arXiv 2602.00012), the portals' metadata (title, publisher, description, field list, files) is harvested
once, embedded with a multilingual model, and searched by meaning. The portal APIs stay as a second route.

    python -m qubo_auto.annealing.catalog build data.gov.tw data.gov.uk datenregister.berlin.de
    python -m qubo_auto.annealing.catalog search "YouBike stations Taipei real-time"
"""
from __future__ import annotations
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results" / "data_cache" / "catalog"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
READABLE = {"CSV", "JSON", "GEOJSON", "XLSX", "XLS", "SHP", "ZIP", "WFS"}
CKAN = {"data.gov.uk": "https://data.gov.uk/api/action",
        "datenregister.berlin.de": "https://datenregister.berlin.de/api/3/action",
        "govdata.de": "https://ckan.govdata.de/api/3/action",
        "open.canada.ca": "https://open.canada.ca/data/api/action",
        "data.gov.au": "https://data.gov.au/data/api/3/action",
        "opendata.swiss": "https://ckan.opendata.swiss/api/3/action",
        "data.go.jp": "https://www.data.go.jp/data/api/3/action",
        "data.gov.ie": "https://data.gov.ie/api/3/action",
        "data.overheid.nl": "https://data.overheid.nl/data/api/3/action"}
COUNTRY = {"data.gov.tw": "tw", "data.gov.uk": "gb", "datenregister.berlin.de": "de", "govdata.de": "de",
           "open.canada.ca": "ca", "data.gov.au": "au", "opendata.swiss": "ch", "data.go.jp": "jp",
           "data.gov.ie": "ie", "data.overheid.nl": "nl", "data.gov.sg": "sg"}
PAGES = {"datenregister.berlin.de": "https://daten.berlin.de/datensaetze/", "govdata.de": "https://www.govdata.de/suche/daten/",
         "open.canada.ca": "https://open.canada.ca/data/en/dataset/", "data.gov.au": "https://data.gov.au/data/dataset/",
         "opendata.swiss": "https://opendata.swiss/en/dataset/", "data.go.jp": "https://www.data.go.jp/data/dataset/",
         "data.overheid.nl": "https://data.overheid.nl/dataset/"}


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "QuantumQUBOAgent research (open-data catalogue index)"
    return s


def _txt(v) -> str:
    """Multilingual portals (opendata.swiss) give {"de": ..., "fr": ..., "en": ...}: join the languages present."""
    if isinstance(v, dict):
        return " / ".join(dict.fromkeys(str(x) for x in v.values() if x))
    return "" if v is None else str(v)


def _fmt(v) -> str:
    v = str(v or "").upper().strip().lstrip(".")
    return {"JSON-LD": "JSON", "GEO+JSON": "GEOJSON", "ESRI SHAPEFILE": "SHP", "SHAPEFILE": "SHP"}.get(v, v)


# ---------------------------------------------------------------- harvesting
def harvest_govtw() -> list[dict]:
    raw = _session().get("https://data.gov.tw/datasets/export/csv", timeout=300).content.decode("utf-8-sig")
    csv.field_size_limit(10 ** 9)
    out = []
    for r in csv.DictReader(io.StringIO(raw)):
        fmts = [_fmt(f) for f in r["檔案格式"].split(";")]
        urls = [u.strip() for u in r["資料下載網址"].split(";")]
        files = [{"format": f, "url": u} for f, u in zip(fmts, urls) if u.startswith("http")]
        out.append({"id": f"gov-{r['資料集識別碼']}", "title": r["資料集名稱"], "publisher": r["提供機關"],
                    "description": r["資料集描述"][:600], "fields": r["主要欄位說明"][:400],
                    "updated": r["詮釋資料更新時間"][:10], "files": files[:6],
                    "page": f"https://data.gov.tw/dataset/{r['資料集識別碼']}"})
    return out


def harvest_ckan(portal: str, rows: int = 1000, pause: float = 0.5) -> list[dict]:
    base, s, out, start = CKAN[portal], _session(), [], 0
    while True:
        for attempt in range(4):
            try:
                res = s.get(f"{base}/package_search", params={"rows": rows, "start": start}, timeout=120).json()["result"]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            break
        for p in res["results"]:
            title = _txt(p.get("title") or p.get("name", ""))
            notes = _txt(p.get("notes") or p.get("description"))
            files = [{"format": _fmt(r.get("format")), "url": r.get("url", ""), "name": _txt(r.get("name"))[:80]}
                     for r in p.get("resources", []) if r.get("url")]
            out.append({"id": f"cat-{portal}-{p.get('name', p.get('id'))}"[:120], "title": title,
                        "publisher": _txt((p.get("organization") or {}).get("title"))[:80],
                        "description": " ".join(str(notes).split())[:600],
                        "fields": ", ".join(_txt(t.get("display_name") or t.get("name")) for t in p.get("tags", []))[:300],
                        "updated": (p.get("metadata_modified") or "")[:10], "files": files[:8],
                        "page": PAGES.get(portal, f"https://{portal}/dataset/") + str(p.get("name", ""))})
        start += len(res["results"])        # portals may cap a page below `rows` (data.go.jp returns 10)
        if start % 1000 < len(res["results"]) or start >= res["count"]:
            print(f"  {portal}: {min(start, res['count'])}/{res['count']}", flush=True)
        if start >= res["count"] or not res["results"]:
            return out
        time.sleep(pause if len(res["results"]) >= 100 else 0.1)
    return out


def harvest_datagovsg(pause: float = 0.2) -> list[dict]:
    """data.gov.sg (Singapore): its search API ignores the query, so the whole list (title, agency, format) is indexed
    here; files are fetched through the poll-download API (a Singapore request fell back to OpenStreetMap twice
    because the official Hawker Centres layer was never found, 2026-10-22)."""
    s, out, page, pages = _session(), [], 1, 1
    while page <= pages:
        for attempt in range(4):
            try:
                d = s.get("https://api-production.data.gov.sg/v2/public/api/datasets", params={"page": page},
                          timeout=60).json()["data"]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            break
        pages = d.get("pages", pages)
        for x in d["datasets"]:
            fmt = _fmt(x.get("format"))
            out.append({"id": f"cat-data.gov.sg-{x['datasetId']}", "title": x.get("name", ""),
                        "publisher": x.get("managedByAgencyName", "")[:80], "description": "", "fields": "",
                        "updated": (x.get("lastUpdatedAt") or "")[:10],
                        "files": [{"format": fmt, "url": f"https://api-open.data.gov.sg/v1/public/api/datasets/"
                                                          f"{x['datasetId']}/poll-download"}],
                        "page": f"https://data.gov.sg/datasets/{x['datasetId']}/view"})
        if page % 100 == 0:
            print(f"  data.gov.sg: page {page}/{pages}", flush=True)
        page += 1
        time.sleep(pause)
    return out


def _text(d: dict) -> str:
    return f"{d['title']} | {d['publisher']} | {d['description'][:300]} | {d['fields'][:200]}"


def build(portal: str, dest: Path | None = None) -> int:
    from fastembed import TextEmbedding
    docs = (harvest_govtw() if portal == "data.gov.tw" else harvest_datagovsg() if portal == "data.gov.sg"
            else harvest_ckan(portal))
    dest = dest or DIR
    dest.mkdir(parents=True, exist_ok=True)
    emb = np.array(list(TextEmbedding(MODEL).embed([_text(d) for d in docs], batch_size=256)), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    (dest / f"{portal}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    np.save(dest / f"{portal}.npy", emb.astype(np.float16))
    return len(docs)


# ---------------------------------------------------------------- search
_LOADED: dict[str, tuple[list[dict], np.ndarray]] = {}
_EMBEDDER = None


def portals_for(area: str) -> list[str] | None:
    """The catalogues of the country the area lies in (None: unknown, search all). Other countries' catalogues only
    add look-alike datasets (a Canadian fire-station list for an English request)."""
    if not area:
        return None
    try:
        from .opendata import _get
        hit = _get("https://nominatim.openstreetmap.org/search",
                   params={"q": area, "format": "json", "limit": 1, "addressdetails": 1}).json()
        cc = hit[0]["address"]["country_code"]
    except Exception:
        return None
    mine = [p for p in available() if COUNTRY.get(p) == cc]
    return mine or None


def available() -> list[str]:
    import os
    if os.environ.get("QUBO_NO_CATALOG"):              # unit tests and the "no catalogue" ablation
        return []
    return sorted(p.stem for p in DIR.glob("*.npy")) if DIR.exists() else []


def _load(portal: str):
    """Documents, dense vectors, and two character n-gram TF-IDF indexes (titles; publisher + fields + description).
    Titles carry the publisher's own wording; lexical character n-grams on titles found all six Taiwan targets in
    the top 20 while the dense model alone confused place names (catalog_eval, 2026-10-12)."""
    if portal not in _LOADED:
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [json.loads(l) for l in (DIR / f"{portal}.jsonl").read_text(encoding="utf-8").splitlines()]
        vt = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, lowercase=True)
        vf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2, sublinear_tf=True, max_features=300000)
        T = vt.fit_transform([d["title"] for d in docs])
        F = vf.fit_transform([f"{d['publisher']} {d['fields'][:300]} {d['description'][:300]}" for d in docs])
        _LOADED[portal] = (docs, np.load(DIR / f"{portal}.npy").astype(np.float32), (vt, T, vf, F))
    return _LOADED[portal]


def _compact(q: str) -> str:
    """Drop the spaces between CJK characters (titles are written without them)."""
    import re
    return re.sub(r"(?<=[\u3000-\u9fff])\s+(?=[\u3000-\u9fff])", "", q)


def _embed(texts: list[str]) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding(MODEL)
    e = np.array(list(_EMBEDDER.embed(texts)), dtype=np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def search(queries: list[str], portals: list[str] | None = None, k: int = 5) -> list[dict]:
    """Datasets closest in meaning to each query, as acquisition candidates (one per readable file, at most 3 files
    per dataset). Empty when no catalogue has been built."""
    portals = [p for p in (portals or available()) if p in available()]
    # "English description / 本地語言" needs: search each language separately (a mixed query dilutes the lexical match)
    queries = list(dict.fromkeys(p.strip() for q in queries for p in str(q).split(" / ") if p.strip()))
    if not portals or not queries:
        return []
    qv = _embed(queries)
    hits: dict[str, tuple[float, dict, str]] = {}
    for portal in portals:
        docs, emb, (vt, T, vf, F) = _load(portal)
        dense = qv @ emb.T
        for qi, q in enumerate(queries):
            qc = _compact(q)
            lex = (T @ vt.transform([qc]).T).toarray().ravel() + 0.3 * (F @ vf.transform([qc]).T).toarray().ravel()
            # reciprocal-rank fusion: lexical first, dense as a tie-breaker for wording the titles do not share
            top = np.argpartition(-lex, 200)[:200] if len(lex) > 200 else np.arange(len(lex))
            topd = np.argpartition(-dense[qi], 200)[:200] if len(lex) > 200 else np.arange(len(lex))
            pool = np.union1d(top, topd)
            rl = {i: r for r, i in enumerate(pool[np.argsort(-lex[pool])])}
            rd = {i: r for r, i in enumerate(pool[np.argsort(-dense[qi][pool])])}
            score = {i: 1 / (60 + rl[i]) + 0.5 / (60 + rd[i]) for i in pool}
            for i in sorted(score, key=score.get, reverse=True)[:k]:
                d = docs[i]
                if d["id"] not in hits or hits[d["id"]][0] < score[i]:
                    hits[d["id"]] = (float(score[i]), d, portal)
    out = []
    for score, d, portal in sorted(hits.values(), key=lambda h: -h[0]):
        files = [f for f in d["files"] if f["format"] in READABLE or "service=wfs" in f["url"].lower()][:3]
        wfs = [f for f in files if f["format"] == "WFS" or "service=wfs" in f["url"].lower()]
        if wfs:
            # a WFS link is a service, not a file: expand it into one GeoJSON download per layer (Berlin's
            # playgrounds, blocks and planning areas are published only this way)
            from .platforms import _wfs_candidates
            out += [{**c, "id": f"{d['id']}-{c['id']}", "similarity": round(score, 3)} for c in _wfs_candidates(
                wfs[0]["url"], d["title"], f"catalogue {portal}", f"{d['publisher']}｜{d['description'][:200]}", d["page"])]
            files = [f for f in files if f not in wfs]
        for j, f in enumerate(files):
            out.append({"id": f"{d['id']}" + (f"-{j}" if j else ""), "source": f"catalogue {portal}",
                        "title": d["title"] + (f" — {f.get('name') or f['format']}" if len(files) > 1 else ""),
                        "detail": f"{d['publisher']}｜更新 {d['updated']}｜{d['description'][:200]}｜欄位 {d['fields'][:150]}",
                        "format": "WFS" if "service=wfs" in f["url"].lower() else f["format"], "url": f["url"],
                        "page": d["page"], "similarity": round(score, 3)})
    return out


if __name__ == "__main__":
    if sys.argv[1] in ("build", "stage"):           # stage: build into staging/ so a running batch is unaffected
        for p in sys.argv[2:]:
            t = time.time()
            try:
                n = build(p, DIR / "staging" if sys.argv[1] == "stage" else None)
                print(f"{p}: {n} datasets in {time.time() - t:.0f} s", flush=True)
            except Exception as e:                       # one failing portal must not stop the others
                print(f"{p}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        for c in search([" ".join(sys.argv[2:])], k=8)[:12]:
            print(c["similarity"], c["source"], "|", c["title"][:80], "|", c["format"])
