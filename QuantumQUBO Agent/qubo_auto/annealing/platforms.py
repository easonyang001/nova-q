"""Open-data PLATFORM connectors (docs/data_landscape.md §1): one connector per platform type, not per site.

Each search returns candidates in the Scout's format {"id", "source", "title", "detail", "format", "url", "page"}.
Only real download URLs given by the platform APIs are returned; nothing is guessed.

  ckan          package_search on any CKAN portal (UK, Germany, Canada, Australia ...)
  socrata       the Socrata discovery API (US cities: New York, Chicago, San Francisco ...)
  opendatasoft  Explore API v2.1 (Paris, the cross-portal hub data.opendatasoft.com ...)
  arcgis        ArcGIS Hub datasets (feature layers exported as GeoJSON)
  europa        data.europa.eu, the EU-wide catalogue
  nomis         UK official statistics (census tables by small area)
  WFS services found in CKAN or data.europa.eu are expanded into GeoJSON (WGS84) downloads per feature type.
"""
from __future__ import annotations
import re
from urllib.parse import quote

CKAN = {"data.gov.uk": "https://data.gov.uk/api/3", "govdata.de": "https://ckan.govdata.de/api/3",
        "open.canada.ca": "https://open.canada.ca/data/api/3", "data.gov.au": "https://data.gov.au/data/api/3"}
OPENDATASOFT = ("opendata.paris.fr", "data.opendatasoft.com")
READABLE = {"CSV", "JSON", "GEOJSON", "XLSX", "XLS", "ZIP", "SHP"}
CATALOGUE = """- ckan: data.gov.uk (UK), govdata.de (Germany, all Länder incl. Berlin), open.canada.ca, data.gov.au
- socrata: US city portals (portal = a domain such as data.cityofnewyork.us, data.cityofchicago.org, or empty for all)
- opendatasoft: opendata.paris.fr (Paris), data.opendatasoft.com (many cities and agencies worldwide)
- arcgis: ArcGIS Hub (geographic layers from agencies worldwide; portal empty)
- europa: data.europa.eu (all EU national portals; portal empty)
- nomis: UK official statistics (census 2021 tables, population estimates) for small areas (LSOA, MSOA, output
  areas, wards, local authorities); query in plain words such as "usual residents" (portal empty)"""


def _get(url: str, **kw):
    from .opendata import _get as get
    return get(url, **kw)


def _fmt(value: str) -> str:
    v = (value or "").upper().split("/")[-1].replace("APPLICATION", "").strip(". ")
    return {"VND.MS-EXCEL": "XLS", "ZIPPED SHAPEFILE": "ZIP", "SHAPEFILE": "ZIP", "ESRI SHAPEFILE": "ZIP"}.get(v, v)


def _clean(text, n=300) -> str:
    return re.sub(r"<[^>]+>|\s+", " ", str(text or "")).strip()[:n]


def wfs_layers(url: str, limit: int = 4) -> list[dict]:
    """A WFS service -> one GeoJSON (WGS84) download per feature type, from its GetCapabilities. Any WFS server."""
    base = url.split("?")[0]
    text = _get(f"{base}?service=WFS&request=GetCapabilities").text
    names = []
    for block in re.findall(r"<(?:wfs:)?FeatureType[\s>].*?</(?:wfs:)?FeatureType>", text, re.S):
        name = re.search(r"<(?:wfs:)?Name>([^<]+)<", block)
        title = re.search(r"<(?:wfs:)?Title>([^<]*)<", block)
        if name:
            names.append((name.group(1).strip(), title.group(1).strip() if title else ""))
    json_ok = "json" in text.lower()
    out = []
    for name, title in names[:limit]:
        fmt = "application/json" if json_ok else "GML3"
        out.append({"layer": name, "title": title or name, "format": "GEOJSON" if json_ok else "GML",
                    "url": f"{base}?service=WFS&version=2.0.0&request=GetFeature&typeNames={quote(name)}"
                           f"&outputFormat={quote(fmt)}&srsName=EPSG:4326"})
    return [o for o in out if o["format"] == "GEOJSON"]


def _wfs_candidates(link: str, title: str, source: str, detail: str, page: str) -> list[dict]:
    try:
        return [{"id": f"wfs-{l['layer'][:40]}", "source": source, "title": f"{title} — {l['title']} (WFS layer)",
                 "detail": detail, "format": "GEOJSON", "url": l["url"], "page": page} for l in wfs_layers(link)]
    except Exception:
        return []


def ckan(portal: str, query: str, limit: int = 6) -> list[dict]:
    base = CKAN.get(portal)
    if not base:
        return []
    res = _get(f"{base}/action/package_search?q={quote(query)}&rows={limit}").json()["result"]["results"]
    out = []
    for pkg in res:
        title = pkg.get("title") or pkg.get("name", "")
        if isinstance(title, dict):
            title = next(iter(title.values()), "")
        files = [r for r in pkg.get("resources", []) if _fmt(r.get("format")) in READABLE and r.get("url")]
        if not files:                                   # geodata published only as a WFS service
            for r in pkg.get("resources", []):
                if "WFS" in _fmt(r.get("format")) or "service=wfs" in str(r.get("url", "")).lower():
                    out += _wfs_candidates(r["url"], title, f"CKAN {portal}", _clean(pkg.get("notes"), 200),
                                           f"https://{portal}/dataset/{pkg.get('name', '')}")
                    break
        for r in files[:3]:
            out.append({"id": f"ckan-{r.get('id', r['url'])[:40]}", "source": f"CKAN {portal}",
                        "title": f"{title} — {_clean(r.get('name') or r.get('description'), 80)}",
                        "detail": f"{_clean(pkg.get('organization', {}).get('title') if pkg.get('organization') else '', 60)}｜"
                                  f"{_clean(pkg.get('notes'), 200)}",
                        "format": _fmt(r.get("format")), "url": r["url"],
                        "page": f"https://{portal}/dataset/{pkg.get('name', '')}"})
    return out


def socrata(portal: str, query: str, limit: int = 6) -> list[dict]:
    url = f"https://api.us.socrata.com/api/catalog/v1?q={quote(query)}&limit={limit}&only=dataset"
    if portal:
        url += f"&domains={quote(portal)}"
    out = []
    for item in _get(url).json().get("results", []):
        r, domain = item.get("resource", {}), item.get("metadata", {}).get("domain", "")
        if not r.get("id") or not domain:
            continue
        out.append({"id": f"socrata-{r['id']}", "source": f"Socrata {domain}", "title": r.get("name", ""),
                    "detail": f"updated {str(r.get('updatedAt', ''))[:10]}｜{_clean(r.get('description'), 240)}",
                    "format": "CSV", "url": f"https://{domain}/api/views/{r['id']}/rows.csv?accessType=DOWNLOAD",
                    "page": item.get("permalink", "")})
    return out


def opendatasoft(portal: str, query: str, limit: int = 6) -> list[dict]:
    domain = portal if portal in OPENDATASOFT else OPENDATASOFT[1]
    url = (f"https://{domain}/api/explore/v2.1/catalog/datasets?where=search({quote(repr(query).replace(chr(39), chr(34)))})"
           f"&limit={limit}")
    out = []
    for d in _get(url).json().get("results", []):
        meta = (d.get("metas") or {}).get("default", {})
        geo = any(f.get("type") in ("geo_point_2d", "geo_shape") for f in d.get("fields", []))
        kind = "geojson" if geo else "csv"
        out.append({"id": f"ods-{d['dataset_id']}", "source": f"Opendatasoft {domain}", "title": meta.get("title", d["dataset_id"]),
                    "detail": f"{_clean(meta.get('publisher'), 60)}｜records {meta.get('records_count')}｜"
                              f"{_clean(meta.get('description'), 220)}",
                    "format": kind.upper(),
                    "url": f"https://{domain}/api/explore/v2.1/catalog/datasets/{d['dataset_id']}/exports/{kind}",
                    "page": f"https://{domain}/explore/dataset/{d['dataset_id']}/"})
    return out


def arcgis(portal: str, query: str, limit: int = 6) -> list[dict]:
    url = f"https://hub.arcgis.com/api/v3/datasets?q={quote(query)}&page%5Bsize%5D={limit}"
    out = []
    for d in _get(url).json().get("data", []):
        a = d.get("attributes", {})
        layer = a.get("url") or ""
        if not re.search(r"/(Feature|Map)Server/\d+$", layer):
            continue
        out.append({"id": f"arcgis-{d['id'][:40]}", "source": "ArcGIS Hub", "title": a.get("name", ""),
                    "detail": f"{_clean(a.get('source') or a.get('orgName'), 60)}｜records {a.get('recordCount')}｜"
                              f"{_clean(a.get('searchDescription') or a.get('description'), 200)}",
                    "format": "GEOJSON", "url": f"{layer}/query?where=1%3D1&outFields=*&outSR=4326&f=geojson",
                    "page": f"https://hub.arcgis.com/datasets/{d['id']}"})
    return out


def europa(portal: str, query: str, limit: int = 6) -> list[dict]:
    url = f"https://data.europa.eu/api/hub/search/search?q={quote(query)}&limit={limit}&filter=dataset"
    out = []
    for d in _get(url).json().get("result", {}).get("results", []):
        title = d.get("title") or {}
        title = title.get("en") or next(iter(title.values()), "") if isinstance(title, dict) else str(title)
        desc = d.get("description") or {}
        desc = desc.get("en") or next(iter(desc.values()), "") if isinstance(desc, dict) else str(desc)
        page = f"https://data.europa.eu/data/datasets/{d.get('id', '')}"
        detail = f"{(d.get('country') or {}).get('label', '')}｜{_clean(desc, 220)}"
        dists = d.get("distributions", [])[:8]

        def link_of(dist):
            for key in ("download_url", "access_url"):
                v = dist.get(key)
                v = v[0] if isinstance(v, list) and v else v
                if v:
                    return v
            return None

        def fmt_of(dist):
            f = dist.get("format")
            return _fmt(f.get("id") if isinstance(f, dict) else f)

        files = [(fmt_of(x), link_of(x)) for x in dists if fmt_of(x) in READABLE and link_of(x)]
        if files:
            fmt, link = files[0]
            out.append({"id": f"eu-{link[-40:]}", "source": "data.europa.eu", "title": f"{title} — {fmt}",
                        "detail": detail, "format": fmt, "url": link, "page": page})
        else:
            wfs = [link_of(x) for x in dists if "WFS" in fmt_of(x) and link_of(x)]
            if wfs:
                out += _wfs_candidates(wfs[0], title, "data.europa.eu (WFS)", detail, page)
    return out


NOMIS = "https://www.nomisweb.co.uk/api/v01/dataset"
# small-area geography types first: a model over areas needs them, and the data stage filters to the district
NOMIS_TYPES = ("TYPE151", "TYPE152", "TYPE150", "TYPE154", "TYPE153")


def nomis(portal: str, query: str, limit: int = 6) -> list[dict]:
    """UK official statistics (census, population estimates, labour market) from Nomis: one CSV candidate per dataset
    and small-area geography type (whole England and Wales; the data stage filters to the area). The census tables
    were missing from data.gov.uk's catalogue (London case, 2026-10-12)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2][:4]
    found = {}
    for w in words or [query]:
        res = _get(f"{NOMIS}/def.sdmx.json?search=*{quote(w)}*", timeout=60).json()
        for k in ((res.get("structure") or {}).get("keyfamilies") or {}).get("keyfamily", []) or []:
            name = k["name"]["value"]
            score = sum(x.lower() in name.lower() for x in words) + (1 if k["id"].startswith("NM_2021") else 0)
            found[k["id"]] = (max(score, found.get(k["id"], (0, ""))[0]), name)
    out = []
    for ds, (score, name) in sorted(found.items(), key=lambda kv: -kv[1][0])[:max(1, limit // 2)]:
        try:
            codes = _get(f"{NOMIS}/{ds}/geography/TYPE.def.sdmx.json", timeout=60).json()
            types = {c["value"]: c["description"]["value"] for c in codes["structure"]["codelists"]["codelist"][0]["code"]}
        except Exception:
            continue
        for t in [t for t in NOMIS_TYPES if t in types][:2]:
            out.append({"id": f"nomis-{ds}-{t}", "source": "Nomis (ONS official statistics)",
                        "title": f"{name} — {types[t]}", "detail": f"Nomis {ds}; whole England and Wales at {types[t]}",
                        "format": "CSV", "url": f"{NOMIS}/{ds}.data.csv?date=latest&geography={t}&measures=20100",
                        "page": f"https://www.nomisweb.co.uk/datasets/{ds.lower()}"})
    return out


PLATFORMS = {"ckan": ckan, "socrata": socrata, "opendatasoft": opendatasoft, "arcgis": arcgis, "europa": europa, "nomis": nomis}


def search(platform: str, portal: str, query: str, limit: int = 6) -> list[dict]:
    """Candidates from one platform; [] on any failure (a failed portal must not stop the other searches)."""
    fn = PLATFORMS.get(str(platform).lower())
    if fn is None:
        return []
    try:
        return fn(str(portal or ""), str(query), limit)
    except Exception:
        return []
