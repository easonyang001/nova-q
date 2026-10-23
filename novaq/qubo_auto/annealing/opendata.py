"""Open-data connectors: find and download real datasets for the data step.

- data.gov.tw (Taiwan's government open-data platform): search + dataset resources. The search uses the endpoint
  the platform's own website calls (POST /api/front/dataset/list); it is not a documented public API and may
  change, so failures are reported, never raised.
- OpenStreetMap via Overpass: points of interest (existing facilities etc.) inside a bounding box. The public
  Overpass servers are often overloaded (HTTP 504), so several mirrors are tried; treat it as best effort.

Downloads go into the session's data folder; the user approves datasets before they are used.
"""
from __future__ import annotations
import json
import re
import urllib.parse
import time
from pathlib import Path
from typing import Any

from .websearch import _session
from .schema import public_view

GOVTW = "https://data.gov.tw"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
USER_AGENT = "QuantumQUBOAgent/0.1 (academic research prototype)"
MAX_DOWNLOAD_MB = 300


def _get(url: str, **kw):
    import socket
    import urllib3.util.connection as conn
    old, conn.allowed_gai_family = conn.allowed_gai_family, (lambda: socket.AF_INET)   # no IPv6 route here
    try:
        s = _session()
        s.headers["User-Agent"] = USER_AGENT
        return s.request(kw.pop("method", "GET"), url, timeout=kw.pop("timeout", 30), **kw)
    finally:
        conn.allowed_gai_family = old


def search_govtw(query: str, limit: int = 10) -> list[dict]:
    """[{nid, title, agency, description, area, updated, url}]; [] on failure."""
    body = {"bool": [{"fulltext": {"value": query}}], "filter": [], "page_num": 1, "page_limit": limit,
            "tids": [], "sort": "_score_desc"}
    try:
        r = _get(f"{GOVTW}/api/front/dataset/list", method="POST", json=body)
        hits = r.json()["payload"]["search_result"]
    except Exception:
        return []
    return [{"nid": h.get("nid"), "title": h.get("title") or h.get("content", "")[:60], "agency": h.get("agency_name", ""),
             "description": (h.get("content") or "")[:300], "area": h.get("coverage_spatial", ""),
             "updated": (h.get("changed") or {}).get("date", "")[:10], "url": f"{GOVTW}/dataset/{h.get('nid')}"}
            for h in hits]


def resources_govtw(nid: int) -> list[dict]:
    """Downloadable files of a dataset: [{description, format, url}] (preferring CSV / JSON / XLSX)."""
    try:
        res = _get(f"{GOVTW}/api/v2/rest/dataset/{int(nid)}").json()["result"]
    except Exception:
        return []
    out = [{"description": d.get("resourceDescription") or "", "format": (d.get("resourceFormat") or "").upper(),
            "url": d.get("resourceDownloadUrl") or d.get("resourceAccessUrl") or ""}
           for d in res.get("distribution", [])]
    rank = {"CSV": 0, "JSON": 1, "GEOJSON": 1, "XLSX": 2, "XLS": 2}
    return sorted([o for o in out if o["url"]], key=lambda o: rank.get(o["format"], 9))


def download(url: str, dest: Path, fmt: str = "", name: str = "") -> Path:
    """Fetch one file (size-capped) into `dest`; the extension follows the declared format or content type."""
    if "api-open.data.gov.sg" in url and url.rstrip("/").endswith("poll-download"):
        # data.gov.sg hands out a short-lived signed file URL per request
        url = _get(url, timeout=60).json()["data"]["url"]
    r = _get(url, stream=True, timeout=120)
    r.raise_for_status()
    import tempfile
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.NamedTemporaryFile(dir=dest, prefix="_partial_", delete=False) as tmp:   # streamed, not in memory
        for chunk in r.iter_content(1 << 20):
            total += len(chunk)
            if total > MAX_DOWNLOAD_MB << 20:
                tmp.close(); Path(tmp.name).unlink(missing_ok=True)
                raise ValueError(f"file larger than {MAX_DOWNLOAD_MB} MB: {url}")
            tmp.write(chunk)
    raw = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    ctype = r.headers.get("content-type", "").lower()
    head = raw[:600].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<head" in head[:300]:
        # moved or retired files often redirect to the publisher's home page (Berlin LOR zip, 2026-10-12)
        raise ValueError(f"the link returned a web page, not a data file: {url}")
    ext = {"CSV": ".csv", "JSON": ".json", "GEOJSON": ".geojson", "XLSX": ".xlsx", "XLS": ".xls",
           "SHP": ".zip", "ZIP": ".zip"}.get(fmt.upper()) or (".zip" if raw[:2] == b"PK" and "sheet" not in ctype else None) or (
        ".json" if "json" in ctype else ".xlsx" if "spreadsheet" in ctype else ".csv")
    dest.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w一-鿿-]+", "_", name or Path(url).stem)[:60] or "dataset"
    path = dest / f"{stem}{ext}"
    if path.exists() and path.read_bytes() != raw:
        import hashlib
        path = dest / f"{stem}_{hashlib.sha256(raw).hexdigest()[:10]}{ext}"
    path.write_bytes(raw)
    return path


def osm_points(tags: dict[str, str], bbox: tuple[float, float, float, float], dest: Path, name: str,
               limit: int = 5000) -> Path:
    """OSM nodes/ways with all `tags` inside bbox (south, west, north, east) -> CSV (lat, lon, name, tags)."""
    sel = "".join(f'["{k}"="{v}"]' for k, v in tags.items())
    s, w, n, e = bbox
    query = f"[out:json][timeout:60];(node{sel}({s},{w},{n},{e});way{sel}({s},{w},{n},{e}););out center {limit};"
    last = ""
    for attempt, url in enumerate(OVERPASS * 2):
        try:
            r = _get(url, method="POST", data={"data": query}, timeout=90, headers={"Accept": "application/json"})
            if r.status_code == 200:
                rows = []
                for el in r.json().get("elements", []):
                    lat = el.get("lat", (el.get("center") or {}).get("lat"))
                    lon = el.get("lon", (el.get("center") or {}).get("lon"))
                    t = el.get("tags", {})
                    rows.append({"lat": lat, "lon": lon, "name": t.get("name", ""), "osm_id": el.get("id"),
                                 **{k: v for k, v in t.items() if k != "name"}})
                if not rows:
                    # an empty file used a download slot and then failed to load (Berlin playgrounds, 2026-10-12)
                    raise ValueError(f"OpenStreetMap has no {tags} inside {bbox}; check the area or the tags")
                import pandas as pd
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{name}.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                return path
            last = f"HTTP {r.status_code} from {url}"
        except ValueError as ex:
            if str(ex).startswith("OpenStreetMap has no"):
                raise
            last = f"{type(ex).__name__} from {url}"           # e.g. a JSON error from an overloaded server: retry
        except Exception as ex:
            last = f"{type(ex).__name__} from {url}"
        time.sleep(3 + 2 * attempt)
    raise RuntimeError(f"OpenStreetMap (Overpass) unavailable right now: {last}")


def geocode_bbox(area: str) -> tuple[float, float, float, float] | None:
    """(south, west, north, east) of a place name via OSM Nominatim; None when unknown."""
    try:
        d = _get("https://nominatim.openstreetmap.org/search", params={"q": area, "format": "json", "limit": 1}).json()
        s_, n, w, e = map(float, d[0]["boundingbox"])
        return s_, w, n, e
    except Exception:
        return None


_SCOUT = (Path(__file__).parent.parent / "prompts" / "annealing" / "scout.txt").read_text(encoding="utf-8")


_DOMAIN = re.compile(r"(?<![@\w.-])((?:[a-z0-9-]+\.)+(?:gov|org|net|com|edu|swiss|gv|go)(?:\.[a-z]{2})?|"
                     r"(?:[a-z0-9-]+\.)+(?:sg|fr|de|uk|au|ch|jp|tw|nl|ie|ca|us))\b", re.I)


def requested_domains(spec_text: str) -> list[str]:
    """Portals or publishers the user named by their web address ('data.gov.sg', 'opendata.paris.fr'), with or
    without http://. A source named only in words still reaches the ranking through the specification."""
    hosts = [m.group(1).lower() for m in _DOMAIN.finditer(spec_text)]
    hosts += [urllib.parse.urlparse(u).hostname or "" for u in re.findall(r'https?://[^\s<>"\'\\]+', spec_text)]
    return sorted({h[4:] if h.startswith("www.") else h for h in hosts if h and "." in h})


def from_requested(candidate: dict, domains: list[str]) -> bool:
    text = " ".join(str(candidate.get(k, "")) for k in ("url", "page", "source")).lower()
    return any(d in text for d in domains)


def scout(spec_text: str, schema: dict, client, model: str) -> dict:
    """The local LLM proposes searches; code runs them. Returns {"plan", "candidates"} where each candidate is
    {"id", "source", "title", "detail", "format", "url" | "osm"} for the user to choose from."""
    from .platforms import CATALOGUE
    raw = client.call(model=model, prompt=_SCOUT.replace("{spec}", spec_text[:3000]).replace(
        "{schema}", json.dumps(public_view(schema), ensure_ascii=False, indent=1)).replace("{platforms}", CATALOGUE),
        step_name="scout", json_mode=True)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {"govtw_queries": [], "osm": [], "area": "", "explanation": "scout answer was not valid JSON"}
    candidates, seen = [], set()
    pages = re.findall(r'https?://[^\s<>"\'\\]+', spec_text)
    requested_pages = set(pages)
    from .websearch import lookup
    for query in (plan.get("web_queries") or [])[:2]:
        finding = lookup(str(query), region="wt-wt")
        pages.extend(item["url"] for item in finding.get("results", []))
    for page in list(dict.fromkeys(pages))[:4]:
        try:
            for candidate in resources_from_page(page):
                candidate["requested_source"] = page in requested_pages
                candidates.append(candidate)
        except Exception:
            continue
    for q in (plan.get("govtw_queries") or [])[:4]:
        for h in search_govtw(str(q), limit=5):
            if h["nid"] in seen:
                continue
            seen.add(h["nid"])
            res = resources_govtw(h["nid"])
            if not res:
                continue
            # a dataset can hold several files (e.g. one per year): list them with their descriptions so the
            # ranking can take the period the specification asks for (the first file was the OLDEST year, 2026-10-11)
            for k, r in enumerate(res[:3]):
                label = f"{h['title']}" + (f" — {r['description'][:40]}" if len(res) > 1 and r["description"] else "")
                candidates.append({"id": f"gov-{h['nid']}" + (f"-{k}" if k else ""), "source": "政府資料開放平台",
                                   "title": label,
                                   "detail": f"{h['agency']}｜{h['area']}｜更新 {h['updated']}｜{h['description'][:80]}",
                                   "format": r["format"], "url": r["url"], "page": h["url"]})
    # semantic search in the local catalogue index (no need to guess the publisher's wording); see catalog.py
    try:
        from . import catalog
        needs = [str(n) for n in (plan.get("needs") or [])[:4]]
        needs += [str(q) for q in (plan.get("govtw_queries") or [])[:2]]
        needs += [str(i.get("query", "")) for i in (plan.get("platform_queries") or [])[:2] if isinstance(i, dict)]
        # 8 per need: with 5, look-alike titles (population tables "in LOR-Planungsräumen") pushed the planning-area
        # geometry itself out of the list (Berlin, 2026-10-13)
        candidates += catalog.search(needs, catalog.portals_for(plan.get("area", "")), k=8)[:40]
    except Exception:
        pass
    # the portals the user named: search them first - their own catalogue entry if we index it, else the web
    # restricted to that site (a named source that was never searched lost to OpenStreetMap, Singapore 2026-10-22)
    domains = requested_domains(spec_text)
    if domains:
        try:
            from . import catalog
            mine = [p for p in catalog.available() if any(d.endswith(p) or p.endswith(d) for d in domains)]
            needs = [str(n) for n in (plan.get("needs") or [])[:4]]
            if mine and needs:
                candidates += catalog.search(needs, mine, k=8)[:24]
        except Exception:
            pass
        for d in domains[:2]:
            for need in [str(n).split(" / ")[0] for n in (plan.get("needs") or [])[:2]]:
                finding = lookup(f"site:{d} {need}"[:200], region="wt-wt")
                for item in finding.get("results", [])[:3]:
                    try:
                        candidates += resources_from_page(item["url"])
                    except Exception:
                        continue
    for c in candidates:
        if domains and from_requested(c, domains):
            c["requested_source"] = True
    from .platforms import search as platform_search
    for item in (plan.get("platform_queries") or [])[:5]:
        if isinstance(item, dict):
            candidates += platform_search(item.get("platform", ""), item.get("portal", ""), item.get("query", ""))
    bbox = geocode_bbox(plan["area"]) if plan.get("area") else None
    for o in (plan.get("osm") or [])[:3]:
        tags = {str(k): str(v) for k, v in (o.get("tags") or {}).items()}
        if tags and bbox:
            label = ",".join(f"{k}={v}" for k, v in tags.items())
            candidates.append({"id": f"osm-{label}", "source": "OpenStreetMap", "title": f"{plan['area']} 的 {label}",
                               "detail": f"用途：{o.get('purpose', '')}（OSM 在台灣的覆蓋不完整，數量可能偏少）",
                               "format": "CSV", "osm": {"tags": tags, "bbox": bbox}})
    unique = {}
    for candidate in candidates:
        key = candidate.get("url") or candidate["id"]
        if key not in unique or candidate.get("requested_source"):
            unique[key] = candidate
    return {"plan": plan, "candidates": list(unique.values())}


def resources_from_page(page):
    """Discover actual download links from a fetched publisher page, never guessed URLs."""
    import hashlib
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    response = _get(page, timeout=20)
    response.raise_for_status()
    suffix = Path(urlparse(response.url).path).suffix.lower().lstrip(".").upper()
    if suffix in READABLE and "html" not in response.headers.get("content-type", "").lower():
        return [{"id": "web-" + hashlib.sha256(response.url.encode()).hexdigest()[:12],
                 "source": urlparse(response.url).hostname, "title": Path(urlparse(response.url).path).stem,
                 "detail": "Explicit downloadable source supplied in specification", "format": suffix,
                 "url": response.url, "page": response.url}]
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1") or soup.find("title")
    title = title.get_text(" ", strip=True) if title else page
    description = soup.find("meta", attrs={"name": "description"})
    description = description.get("content", "") if description else soup.get_text(" ", strip=True)[:1600]
    license_link = next((a["href"] for a in soup.find_all("a", href=True)
                         if "creativecommons.org/licenses/by/4.0" in a["href"]), "")
    candidates, seen = [], set()
    for link in soup.find_all("a", href=True):
        url = urljoin(response.url, link["href"])
        suffix = Path(urlparse(url).path).suffix.lower().lstrip(".").upper()
        if suffix not in READABLE or url in seen:
            continue
        seen.add(url)
        candidates.append({"id": "web-" + hashlib.sha256(url.encode()).hexdigest()[:12],
            "source": urlparse(response.url).hostname, "title": title,
            "detail": link.get_text(" ", strip=True) + " | " + description, "format": suffix, "url": url, "page": response.url,
            "fetched_page_sha256": hashlib.sha256(response.content).hexdigest(),
            "license_from_page": "CC BY 4.0" if license_link else "",
            "license_evidence": response.url if license_link else ""})
        if len(candidates) >= 6:
            break
    return candidates


def fetch_candidate(c: dict, dest: Path) -> Path:
    if "osm" in c:
        return osm_points(c["osm"]["tags"], tuple(c["osm"]["bbox"]), dest, re.sub(r"\W+", "_", c["id"])[:60])
    return download(c["url"], dest, c.get("format", ""), c["title"])


_RANK = """You screen candidate datasets for an optimisation model. Use only the titles and descriptions below.

PROBLEM: {spec}

THE MODEL NEEDS (DATA_SCHEMA): {schema}

CANDIDATES:
{candidates}

Datasets are COMBINED later (joined by area or id), so a candidate is relevant if it feeds AT LEAST ONE key at the
right level (e.g. village vs district) for this problem's area, even if other keys must come from other datasets.
Boundary maps (SHP / GeoJSON / KML polygons, 界圖) give the coordinates (centroids) of their areas.
When several files of one dataset differ by period (e.g. one per year), only the one matching the requested period
(usually the most recent) is relevant.
If the specification names a source or publisher (e.g. "official YouBike open data", "ONS census"), only candidates
from that publisher are relevant for the data it names; crowd-sourced data (OpenStreetMap) is relevant only when no
official candidate exists for that data (it can differ from the official list: an OSM run lost 2 of 38 stations).

A candidate feeds a set only if its rows ARE that set's things: metro exits are not bike stations, a reference point
(e.g. the station the radius is measured from) is not the list of places to visit. Things that are merely nearby,
related or similar are not relevant for that set.

When the request covers ALL entities of a kind ("every station", "all villages"), only a complete list of them is
relevant for that set (a master list, register or full status feed), not a subset or statistics about some of them
(e.g. "complementary stations", "popular stations", origin-destination counts).
Where the detail lists the dataset's FIELDS (欄位), the candidate feeds a key only if those fields contain what the
key needs (coordinates or an address for positions, a count column for a population, a station id for a station
list); a file that lacks them would only end in a DATA GAP after download.
Where a PREVIEW is given, judge by the actual columns: prefer raw record tables (one row per place / area / item,
with ids, names, coordinates or measures) over reports, summaries, notes, documentation or web pages.

For EVERY candidate return whether it can feed a DATA_SCHEMA key for this problem's area. Return JSON only:
{"ranking": [{"id": "<id>", "relevant": true|false, "feeds": "<DATA_SCHEMA key or empty>",
              "reason": "<one short sentence, in Traditional Chinese>"}, ...]}"""
READABLE = {"CSV", "JSON", "GEOJSON", "XLSX", "XLS", "SHP", "ZIP"}


def preview(c: dict, max_bytes: int = 32768, timeout: float = 10) -> str:
    """Column names and the first record of a candidate, from its first bytes (geo services: 3 records). Judging by
    title alone took report spreadsheets and explanatory tables for data (London, Berlin, 2026-10-11)."""
    import csv as _csv
    import io
    url = c.get("url") or ""
    if not url or "osm" in c:
        return ""
    if "FeatureServer" in url or "MapServer" in url:
        url += "&resultRecordCount=3"
    elif "request=GetFeature" in url:
        url += "&count=3"
    elif "/exports/" in url:
        url += ("&" if "?" in url else "?") + "limit=3"
    try:
        r = _get(url, stream=True, timeout=timeout)
        raw = b""
        for chunk in r.iter_content(8192):
            raw += chunk
            if len(raw) >= max_bytes:
                break
        r.close()
    except Exception as e:
        return f"(preview failed: {type(e).__name__})"
    if raw[:2] == b"PK":
        return "(zip / spreadsheet: no preview)"
    from .datakit import _decode
    text = _decode(raw)
    head = text.lstrip()[:300].lower()
    if head.startswith(("<!doctype html", "<html")) or "<head>" in head:
        return "PREVIEW: a web page, not a data file"
    if head[:1] in "[{":
        cols = re.findall(r'"([^"]{1,40})"\s*:', text[:6000])
        return "PREVIEW: JSON keys " + ", ".join(dict.fromkeys(cols).keys())[:400]
    lines = text.splitlines()[:6]
    try:
        sep = _csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|").delimiter
    except _csv.Error:
        sep = ","
    rows = list(_csv.reader(io.StringIO("\n".join(lines)), delimiter=sep))
    return ("PREVIEW: " + " || ".join(" | ".join(x[:12] for x in row[:15]) for row in rows[:3]))[:500] if rows else ""


def rank_candidates(spec_text: str, schema: dict, candidates: list[dict], client, model: str) -> list[dict]:
    """Relevant candidates first, each with the key it would feed and a reason; unreadable formats flagged."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [c for c in candidates[:16] if "preview" not in c]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for c, text in zip(todo, pool.map(preview, todo)):
            c["preview"] = text
    lines = "\n".join(f"- id={c['id']} | {c['title']} | {c['detail'][:600]} | format {c['format']} | page={c.get('page', '')} | requested_source={c.get('requested_source', False)}"
                      + (f" | {c['preview']}" if c.get("preview") else "") for c in candidates)
    raw = client.call(model=model, prompt=_RANK.replace("{spec}", spec_text[:2000]).replace(
        "{schema}", json.dumps(public_view(schema), ensure_ascii=False)).replace("{candidates}", lines),
        step_name="scout_rank", json_mode=True)
    try:
        verdict = {r["id"]: r for r in json.loads(raw).get("ranking", []) if isinstance(r, dict) and "id" in r}
    except (json.JSONDecodeError, AttributeError):
        verdict = {}
    for c in candidates:
        v = verdict.get(c["id"], {})
        c["relevant"] = bool(v.get("relevant", False))
        c["feeds"], c["reason"] = str(v.get("feeds", "")), str(v.get("reason", "（未評估）"))
        c["readable"] = c.get("format", "").upper() in READABLE
    latest_per_dataset(candidates)
    official_over_crowd(candidates)
    return sorted(candidates, key=lambda c: (not c.get("requested_source", False), not c["relevant"], not c["readable"]))


def _feeds(c: dict) -> set[str]:
    return {x.strip().lower() for x in re.split(r"[,;/、，]", str(c.get("feeds", ""))) if x.strip()}


def official_over_crowd(candidates: list[dict]) -> None:
    """Crowd-sourced points (OpenStreetMap) are dropped when an official relevant candidate feeds the same data.
    Mixing the two duplicated and lost places (AED, YouBike, 2026-10-12); OSM stays when it supplies something the
    official sources do not."""
    official = [c for c in candidates if c.get("relevant") and "osm" not in c and c.get("source") != "OpenStreetMap"]
    covered = set().union(*(_feeds(c) for c in official)) if official else set()
    for c in candidates:
        if c.get("relevant") and ("osm" in c or c.get("source") == "OpenStreetMap"):
            mine = _feeds(c)
            if official and (not mine or mine & covered):
                c["relevant"] = False
                c["reason"] = "已有官方來源提供同樣的資料；不混用群眾編輯的 OpenStreetMap 點位"



def period_of(text: str) -> float | None:
    """The latest year named in a file label: ROC years ('112年') are converted, '迄今' / 'to date' counts as later
    than its year, AD years (19xx/20xx) are taken as they are. None if no year is named."""
    years = [int(y) + 1911 for y in re.findall(r"(?<!\d)(\d{2,3})\s*年", text) if 50 <= int(y) <= 200]
    years += [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)]
    if not years:
        return None
    return max(years) + (0.5 if re.search(r"迄今|至今|to date|present|current", text, re.I) else 0)


def latest_per_dataset(candidates: list[dict]) -> None:
    """Several relevant files of ONE dataset that differ by period (one per year): keep the latest, mark the rest.
    Code, not the LLM, decides this: the ranking kept all yearly files and downloads went to the oldest first
    (2026-10-11)."""
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        if c.get("relevant") and c.get("page"):
            groups.setdefault(c["page"], []).append(c)
    for files in groups.values():
        dated = [(period_of(c.get("title", "") + " " + c.get("detail", "")), c) for c in files]
        dated = [(p, c) for p, c in dated if p is not None]
        if len(dated) < 2 or len({p for p, _ in dated}) < 2:
            continue
        newest = max(p for p, _ in dated)
        for p, c in dated:
            if p < newest:
                c["relevant"] = False
                c["reason"] = "較舊的年度檔；同一資料集已選最新一期"

