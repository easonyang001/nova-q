"""Geography for real-world data (docs/data_landscape.md §3.5): coordinate systems, swapped lat/lon, geocoding,
distance and travel-time matrices. Network results are cached under results/data_cache so a rerun is
reproducible and does not hammer free services (Nominatim: 1 request/s; OSRM demo: light use only).
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[3] / "results" / "data_cache"


# ---------------------------------------------------------------- coordinate systems
def detect_crs(x: pd.Series, y: pd.Series, prj: str = "") -> str:
    """EPSG code of a coordinate pair: from a .prj text when given, else from the value ranges.
    TWD97 TM2 (EPSG:3826) and TWD67 TM2 (EPSG:3828) overlap in range (about 800 m apart); without a .prj the
    newer TWD97 is assumed and the log should say so."""
    p = prj.upper()
    if "1967" in p or "TWD67" in p or "3828" in p:
        return "EPSG:3828"
    if "1997" in p or "TWD97" in p or "3826" in p or "TAIWAN" in p:
        return "EPSG:3826"
    m = re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]\s*$', prj.strip())
    if m:
        return f"EPSG:{m.group(1)}"
    if prj.strip():
        # "EPSG:27700" or a .prj WKT without an authority line: pyproj reads both. prj="EPSG:27700" used to fall through
        # to the value ranges and was taken for web mercator (UK points near (3.8, 3.7), 2026-10-12)
        try:
            from pyproj import CRS
            code = CRS.from_user_input(prj.strip()).to_epsg(min_confidence=70)
            if code:
                return f"EPSG:{code}"
        except Exception:
            pass
    xs, ys = pd.to_numeric(x, errors="coerce").dropna(), pd.to_numeric(y, errors="coerce").dropna()
    if xs.empty:
        return "unknown"
    xm, ym = xs.median(), ys.median()
    if -180 <= xm <= 180 and -90 <= ym <= 90:
        return "EPSG:4326"
    if 140_000 <= xm <= 360_000 and 2_400_000 <= ym <= 2_800_000:
        return "EPSG:3826"
    # metres outside Taiwan: national grids (British National Grid, UTM, Lambert ...) and web mercator share the same
    # ranges, so guessing gives points thousands of km away; ask for the system instead
    return "unknown"


def to_wgs84(df: pd.DataFrame, x: str, y: str, crs: str | None = None, prj: str = "", log=None) -> pd.DataFrame:
    """Add 'lon' / 'lat' (WGS84) converted from columns x / y in any EPSG system (via pyproj)."""
    from pyproj import Transformer
    crs = crs or detect_crs(df[x], df[y], prj)
    out = df.copy()
    if crs == "EPSG:4326":
        out["lon"], out["lat"] = pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce")
    elif crs == "unknown":
        raise ValueError(f"cannot tell the coordinate system of {x}/{y} from the numbers alone; take it from the "
                         f"source's .prj file, metadata or column names and pass crs='EPSG:....' (e.g. British National "
                         f"Grid EPSG:27700, ETRS89 / UTM 32N EPSG:25832, 33N EPSG:25833, web mercator EPSG:3857)")
    else:
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(pd.to_numeric(df[x], errors="coerce").to_numpy(), pd.to_numeric(df[y], errors="coerce").to_numpy())
        out["lon"], out["lat"] = lon, lat
    if log is not None:
        log.step(f"座標轉換 {crs} → WGS84", len(df), len(out),
                 "無 .prj，依數值範圍判斷為 TWD97" if crs == "EPSG:3826" and not prj else "")
    return out


def fix_swapped(df: pd.DataFrame, lat: str = "lat", lon: str = "lon",
                bbox: tuple[float, float, float, float] | None = None, log=None) -> pd.DataFrame:
    """Swap lat/lon where they are reversed: |lat| > 90, or (with bbox south,west,north,east) only the swapped
    pair falls inside the expected area."""
    out = df.copy()
    la, lo = pd.to_numeric(out[lat], errors="coerce"), pd.to_numeric(out[lon], errors="coerce")
    swap = la.abs() > 90
    if bbox:
        s, w, n, e = bbox
        inside = la.between(s, n) & lo.between(w, e)
        swapped_inside = lo.between(s, n) & la.between(w, e)
        swap |= ~inside & swapped_inside
    out.loc[swap, lat], out.loc[swap, lon] = lo[swap], la[swap]
    if log is not None and swap.any():
        log.step("修正經緯度對調", len(df), len(df), f"{int(swap.sum())} 筆")
    return out


# ---------------------------------------------------------------- distances
def haversine_matrix(lat: list[float], lon: list[float]) -> np.ndarray:
    """Straight-line (great-circle) distances in metres."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dla, dlo = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_to(lat, lon, lat0: float, lon0: float) -> np.ndarray:
    """Great-circle distance (m) from every point to ONE reference point (lat0, lon0)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    la0, lo0 = math.radians(float(lat0)), math.radians(float(lon0))
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la) * math.cos(la0) * np.sin((lo - lo0) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def within_radius(df: pd.DataFrame, lat: str, lon: str, center: tuple[float, float], radius_m: float,
                  log=None) -> pd.DataFrame:
    """Rows whose (lat, lon) lie within radius_m metres (great-circle) of center=(lat0, lon0); adds 'dist_m'.
    Rows without valid coordinates are dropped and counted in the log."""
    out = df.copy()
    la, lo = pd.to_numeric(out[lat], errors="coerce"), pd.to_numeric(out[lon], errors="coerce")
    valid = la.notna() & lo.notna()
    out = out[valid].copy()
    out["dist_m"] = distance_to(la[valid], lo[valid], *center)
    kept = out[out["dist_m"] <= radius_m]
    if log is not None:
        log.step(f"只保留距離 ({center[0]}, {center[1]}) {radius_m:g} 公尺內的資料", len(df), len(kept),
                 f"{int((~valid).sum())} 列沒有座標" if (~valid).any() else "")
        if len(kept) == 0 and len(out):
            log.flags.append(f"within_radius kept NOTHING: the nearest row is {out['dist_m'].min():.0f} m from "
                             f"{center}; check lat/lon columns (swapped? projected?) and the centre")
    return kept.reset_index(drop=True)


def _cached(kind: str, key: str, fetch):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fetch()
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def road_matrix(lat: list[float], lon: list[float], profile: str = "driving", log=None) -> dict:
    """Road distance (m) and travel time (s) matrices from OSRM, in blocks of at most 50 points per request,
    cached. Falls back to straight-line x 1.3 (and 30 km/h) where OSRM gives no route; the log counts them."""
    from ..opendata import _get
    n = len(lat)
    dist, dur = np.full((n, n), np.nan), np.full((n, n), np.nan)
    pts = [f"{lo:.6f},{la:.6f}" for la, lo in zip(lat, lon)]
    block = 50
    for i0 in range(0, n, block):
        for j0 in range(0, n, block):
            src, dst = list(range(i0, min(n, i0 + block))), list(range(j0, min(n, j0 + block)))
            idx = src + [j for j in dst if j not in src]
            coords = ";".join(pts[k] for k in idx)
            q = (f"https://router.project-osrm.org/table/v1/{profile}/{coords}?annotations=distance,duration"
                 f"&sources={';'.join(str(idx.index(k)) for k in src)}&destinations={';'.join(str(idx.index(k)) for k in dst)}")
            def fetch(q=q):
                for attempt in range(4):
                    r = _get(q, timeout=60)
                    if r.status_code == 200:
                        return r.json()
                    time.sleep(2 + 3 * attempt)
                return {"code": f"HTTP {r.status_code}"}
            res = _cached("osrm", q, fetch)
            if res.get("code") == "Ok":
                dist[np.ix_(src, dst)] = np.array(res["distances"], dtype=float)
                dur[np.ix_(src, dst)] = np.array(res["durations"], dtype=float)
            if i0 or j0:
                time.sleep(1)                                 # be light on the public demo server
    straight = haversine_matrix(lat, lon)
    missing = np.isnan(dist)
    dist[missing] = straight[missing] * 1.3
    dur[np.isnan(dur)] = dist[np.isnan(dur)] / (30 / 3.6)
    if log is not None:
        log.step(f"道路距離與時間矩陣（OSRM {profile}）", n, n,
                 f"{int(missing.sum())} 組無路線，以直線×1.3 估計" if missing.any() else "全部取自 OSRM")
    return {"distance_m": dist, "duration_s": dur, "estimated": missing}


# ---------------------------------------------------------------- geocoding
def _relaxations(address: str) -> list[tuple[str, str]]:
    """(query, precision) from exact to coarse. Nominatim knows few Taiwanese house numbers, so a full address
    often fails while its road or district succeeds; each fallback is labelled, never passed off as exact."""
    import unicodedata
    a = unicodedata.normalize("NFKC", address).strip()
    out = [(a, "address")]
    road = re.sub(r"\d+(之\d+)?號.*$", "", a)                        # drop house number and below
    road = re.sub(r"\d+[巷弄].*$", "", road)
    road = re.sub(r"[一二三四五六七八九十]+段$", lambda m: m.group(0), road)
    if road and road != a:
        out.append((road, "road"))
    m = re.match(r"^(.*?[市縣])?(.*?[區鄉鎮市])", a)
    if m and m.group(0) not in (a, road):
        out.append((m.group(0), "district"))
    return out


GEOCODE_MAX = 120                                         # about two minutes at Nominatim's 1 request / second


def geocode(addresses: list[str], country: str | None = None, log=None, relax: bool = True) -> pd.DataFrame:
    """Address / place name -> lat, lon via OSM Nominatim (1 request per second, cached). When the full address is
    not found, retries with the road, then the district, and labels the result's precision (address / road /
    district). Rows that still fail keep NaN and are counted; they are never filled with guesses."""
    from ..opendata import _get
    names = [str(a).strip() for a in addresses]
    # statistical areas ("Camden 001A", "E01000001") have no address: geocoding returns some street of a similar
    # name, and 256 of them took longer than the whole data stage (London, 2026-10-12)
    codes = sum(bool(re.fullmatch(r"[A-Za-z]{0,3}\d{5,}|.+\s\d{3}[A-Z]", n)) for n in names)
    if names and codes > len(names) / 2:
        raise ValueError("geocode: these look like statistical-area codes or labels, not addresses; their location "
                         "comes from the publisher's centroid or boundary file. Report it as a DATA GAP if missing")
    if len(names) > GEOCODE_MAX:
        raise ValueError(f"geocode: {len(names)} addresses would take about {len(names) * 1.1 / 60:.0f} minutes "
                         f"(1 request per second); use coordinates from the source, or report a DATA GAP")
    rows, fetched = [], 0

    def lookup(q):
        nonlocal fetched
        params = {"q": q, "format": "json", "limit": 1}
        if country:
            params["countrycodes"] = country

        def fetch():
            nonlocal fetched
            fetched += 1
            time.sleep(1.1)                               # Nominatim usage policy: max 1 request / second
            try:
                return _get("https://nominatim.openstreetmap.org/search", params=params).json()
            except Exception:
                return []
        return _cached("geocode", json.dumps(params, sort_keys=True, ensure_ascii=False), fetch)

    for a in addresses:
        q = str(a).strip()
        found = None
        for query, precision in (_relaxations(q) if relax else [(q, "address")]):
            hit = lookup(query)
            if hit:
                found = (hit[0], precision)
                break
        rows.append({"query": q, "lat": float(found[0]["lat"]) if found else math.nan,
                     "lon": float(found[0]["lon"]) if found else math.nan,
                     "precision": found[1] if found else "not found",
                     "matched": found[0].get("display_name", "") if found else ""})
    out = pd.DataFrame(rows)
    if log is not None:
        counts = out["precision"].value_counts().to_dict()
        log.step("地址轉座標（Nominatim，找不到時放寬到路名、行政區）", len(addresses), int(out["lat"].notna().sum()),
                 "精確度：" + "、".join(f"{k} {v}" for k, v in counts.items()) + f"；實際查詢 {fetched} 次，其餘來自快取")
    return out


# ---------------------------------------------------------------- areas
def _rings(geojson: dict) -> list[list[np.ndarray]]:
    """Polygons as [outer ring, holes...] arrays of (lon, lat)."""
    t, c = geojson.get("type"), geojson.get("coordinates") or []
    polys = [c] if t == "Polygon" else c if t == "MultiPolygon" else []
    return [[np.asarray(r, float)[:, :2] for r in p] for p in polys]


def _inside_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over points."""
    inside = np.zeros(len(lon), bool)
    x0, y0 = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    for a, b, c, d in zip(x0, y0, x1, y1):
        cross = (b > lat) != (d > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = a + (lat - b) * (c - a) / (d - b)
        inside ^= cross & (lon < xs)
    return inside


def points_in_polygon(lat, lon, geojson: dict) -> np.ndarray:
    """True where (lat, lon) lies inside a GeoJSON Polygon / MultiPolygon (holes respected)."""
    la, lo = np.asarray(lat, float), np.asarray(lon, float)
    out = np.zeros(len(la), bool)
    for outer, *holes in _rings(geojson):
        hit = _inside_ring(lo, la, outer)
        for h in holes:
            hit &= ~_inside_ring(lo, la, h)
        out |= hit
    return out


def area_polygon(name: str, country: str | None = None) -> dict:
    """Boundary of a named administrative area (e.g. '大安區, 臺北市', 'Camden, London') from OSM Nominatim, cached.
    When the name first matches a place POINT (Islington is also a neighbourhood inside the London Borough of
    Islington, 2026-10-13), the point's own address names the unit around it: that unit's boundary is used; then a
    search inside the city's box. The search order is the same for every country."""
    from ..opendata import _get

    def search(params):
        params = {"format": "json", "polygon_geojson": 1, **params}
        if country:
            params["countrycodes"] = country

        def fetch():
            time.sleep(1.1)
            try:
                return _get("https://nominatim.openstreetmap.org/search", params=params).json()
            except Exception:
                return []
        return _cached("area", json.dumps(params, sort_keys=True, ensure_ascii=False), fetch)

    def polygon(hits):
        return next((h["geojson"] for h in hits if (h.get("geojson") or {}).get("type") in ("Polygon", "MultiPolygon")), None)

    hits = search({"q": name, "limit": 5})
    found = polygon(hits)
    if found:
        return found
    base = name.split(",")[0].strip()
    for h in hits:                                   # the administrative unit around a same-named point
        for part in [p.strip() for p in str(h.get("display_name", "")).split(",")]:
            if base.lower() in part.lower() and part.lower() != base.lower():
                found = polygon(search({"q": part, "limit": 3}))
                if found:
                    return found
    city = name.split(",", 1)[1].strip() if "," in name else ""
    if city:                                         # an administrative boundary inside the city's box
        c = search({"q": city, "limit": 1})
        if c and c[0].get("boundingbox"):
            s_, n_, w_, e_ = c[0]["boundingbox"]
            found = polygon([h for h in search({"q": base, "limit": 10, "viewbox": f"{w_},{n_},{e_},{s_}", "bounded": 1})
                             if h.get("class") == "boundary"])
            if found:
                return found
    raise ValueError(f"in_area: no boundary found for {name!r}. Write the area as '<area>, <city>' or its official "
                     f"name; or filter by a name / code column of the table instead (ids often carry the area name)")


def in_area(df: pd.DataFrame, lat: str, lon: str, areas: str | list[str], country: str | None = None,
            log=None, code_column: str | None = None) -> pd.DataFrame:
    """Keep the rows whose coordinates lie inside the named area(s), using the areas' real boundaries. Use it to
    filter by area when the table has coordinates, or to check a filter on area CODES: a code list written from
    memory kept the wrong districts twice (Taipei AED, 2026-10-12)."""
    areas = [areas] if isinstance(areas, str) else list(areas)
    la, lo = pd.to_numeric(df[lat], errors="coerce").to_numpy(), pd.to_numeric(df[lon], errors="coerce").to_numpy()
    keep = np.zeros(len(df), bool)
    counts, codes = {}, {}
    for a in areas:
        hit = points_in_polygon(la, lo, area_polygon(a, country))
        counts[a] = int(hit.sum())
        keep |= hit
        if code_column and hit.any():
            # the publisher's own area code decides membership; the boundary only says WHICH code is this area
            # (OSM boundaries put points on border roads in the neighbouring district, Taipei AED, 2026-10-12)
            m = df.loc[hit, code_column].mode().iloc[0]
            codes[a] = m.item() if hasattr(m, "item") else m
    if code_column:
        keep = df[code_column].isin(list(codes.values())).to_numpy()
    out = df[keep].reset_index(drop=True)
    if log is not None:
        log.step(f"依實際邊界保留區域 {areas}", len(df), len(out),
                 f"各區點數 {counts}" + (f"；對應的區域代碼 {codes}（以發布者代碼篩選）" if codes else ""))
        empty = [a for a, n in counts.items() if n == 0]
        if empty:
            log.flags.append(f"in_area: no point lies inside {empty}; check the area names or the coordinates")
    return out


def merge_same_place(df: pd.DataFrame, name: str, lat: str, lon: str, max_m: float | None = 150,
                     log=None) -> pd.DataFrame:
    """One row per real place: rows with the same normalised name whose coordinates lie within max_m are merged
    (the first is kept). Two records of one metro station 20 m apart survived a coordinate-only dedupe (Taipei AED,
    2026-10-12); same-named places far apart (two '中山里' in different districts) stay separate.
    max_m=None merges by name alone, at the mean coordinates, for requests that say one name is one place (Basel AED:
    two records of one faculty 200 m apart stayed separate, 2026-10-24)."""
    from .text import normalize_text
    out = df.reset_index(drop=True)
    key = out[name].map(normalize_text)
    la, lo = pd.to_numeric(out[lat], errors="coerce").to_numpy(), pd.to_numeric(out[lon], errors="coerce").to_numpy()
    if max_m is None:
        first = ~key.duplicated()
        res = out[first].reset_index(drop=True)
        res[lat] = key[first].map(pd.Series(la).groupby(key).mean()).to_numpy()
        res[lon] = key[first].map(pd.Series(lo).groupby(key).mean()).to_numpy()
        if log is not None:
            log.step("合併同名的同一地點（只看名稱，座標取平均）", len(out), len(res), f"合併 {len(out) - len(res)} 筆")
        return res
    drop = np.zeros(len(out), bool)
    for _, idx in out.groupby(key).groups.items():
        idx = list(idx)
        for a_pos, a in enumerate(idx):
            if drop[a]:
                continue
            for b in idx[a_pos + 1:]:
                if not drop[b] and haversine_matrix([la[a], la[b]], [lo[a], lo[b]])[0, 1] <= max_m:
                    drop[b] = True
    res = out[~drop].reset_index(drop=True)
    if log is not None:
        log.step(f"合併同名且相距 {max_m:g} m 內的同一地點", len(out), len(res), f"合併 {int(drop.sum())} 筆")
    return res


def drop_stray_points(df: pd.DataFrame, lat: str, lon: str, max_km: float | None = None, log=None) -> pd.DataFrame:
    """Drop the few rows whose coordinates lie far from all the others (typos, (0,0), a point in another country),
    and say which. Without max_km the limit is 10 x the 75th-percentile distance from the median point (at least
    20 km). Only a small share may be dropped: more than 5 % means the data or the area is wrong, so it raises."""
    la, lo = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
    ok = la.notna() & lo.notna()
    c_la, c_lo = float(la[ok].median()), float(lo[ok].median())
    d = pd.Series(np.nan, index=df.index)
    d[ok] = [haversine_matrix([c_la, a], [c_lo, b])[0, 1] / 1000 for a, b in zip(la[ok], lo[ok])]
    limit = max_km or max(20.0, 10 * float(d[ok].quantile(0.75)))
    stray = (~ok) | (d > limit)
    if stray.mean() > 0.05:
        raise ValueError(f"drop_stray_points: {int(stray.sum())} of {len(df)} rows lie over {limit:.0f} km from the "
                         f"others; that is not a few typos: check the coordinate columns, their system, or the area")
    out = df[~stray].reset_index(drop=True)
    if log is not None:
        examples = df.loc[stray, [lat, lon]].head(3).values.tolist()
        log.step(f"移除離群座標（距中心 > {limit:.0f} km）", len(df), len(out), f"移除 {int(stray.sum())} 筆，例：{examples}")
    return out
