"""Ground truth for the end-to-end acceptance cases, computed from the official raw files only.

Independent of the system: plain pandas / pyshp / pyproj, no qubo_auto import. The system never reads this
directory or its output. Run once before a batch; the batch records the truth file's SHA256.

    python -I acceptance/truth.py results/acceptance_truth/raw_20261011 results/acceptance_truth/truth_20261011.json
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MRT_DAAN = (25.0330, 121.5436)                       # 捷運大安站, as stated in the YouBike request
FLINDERS_ST = (-37.8183, 144.9671)                   # Flinders Street Station, as stated in the Melbourne request
TAIPEI_BBOX = (24.96, 121.45, 25.22, 121.67)          # lat_min, lon_min, lat_max, lon_max


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def polygon_centroid(shape):
    """Area-weighted centroid of all rings of a polygon (shoelace; holes have opposite orientation)."""
    pts, parts = np.asarray(shape.points, float), list(shape.parts) + [len(shape.points)]
    area = cx = cy = 0.0
    for a, b in zip(parts[:-1], parts[1:]):
        x, y = pts[a:b, 0], pts[a:b, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area += cross.sum() / 2
        cx += ((x[:-1] + x[1:]) * cross).sum() / 6
        cy += ((y[:-1] + y[1:]) * cross).sum() / 6
    return cx / area, cy / area


def chargers(raw: Path, only_districts: list[str] | None = None, k: int = 5) -> dict:
    import shapefile
    from pyproj import Transformer
    pop = pd.read_csv(raw / "pop_112_now.csv", encoding="cp950", usecols=range(6))
    year = pop["年份"].max()
    month = pop.loc[pop["年份"] == year, "月份"].max()
    latest = pop[(pop["年份"] == year) & (pop["月份"] == month) & (pop["性別"] == "計")].copy()
    latest["code"] = latest["區域代碼"].astype(str)
    districts = dict(zip(latest.loc[latest["code"].str.len() == 8, "code"], latest.loc[latest["code"].str.len() == 8, "區域別"]))
    villages = latest[latest["code"].str.len() == 11].copy()
    villages["district"] = villages["code"].str[:8].map(districts)
    city_total = int(latest.loc[latest["code"] == "63000000", "總計"].iloc[0])
    z = zipfile.ZipFile(raw / "village_boundaries_taipei.zip")
    stem = next(n[:-4] for n in z.namelist() if n.endswith(".shp"))
    reader = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), dbf=io.BytesIO(z.read(stem + ".dbf")),
                              shx=io.BytesIO(z.read(stem + ".shx")), encoding="utf-8")
    to_wgs = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    geo = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        x, y = polygon_centroid(shp)
        lon, lat = to_wgs.transform(x, y)
        geo[(rec["TNAME"], rec["VNAME"])] = (lat, lon, str(rec["CPTVID"]))
    rows, unmatched = [], []
    for _, r in villages.iterrows():
        key = (r["district"], r["區域別"])
        if key not in geo:                                   # variant characters (e.g. 糖廍里): match by district + similar name
            same = [k for k in geo if k[0] == key[0] and k[1][0] == key[1][0] and k[1][-1] == key[1][-1]
                    and len(k[1]) == len(key[1]) and not any((key[0], v) == k for v in villages["區域別"])]
            if len(same) == 1:
                key = same[0]
            else:
                unmatched.append(r["區域別"])
                continue
        lat, lon, cptvid = geo[key]
        rows.append({"district": r["district"], "village": r["區域別"], "code": r["code"], "cptvid": cptvid,
                     "pop": int(r["總計"]), "lat": lat, "lon": lon})
    if unmatched or len(rows) != len(geo):
        raise SystemExit(f"truth join incomplete: unmatched {unmatched}, {len(rows)} rows vs {len(geo)} polygons")
    if only_districts:
        rows = [r for r in rows if r["district"] in only_districts]
    lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows])
    w = np.array([r["pop"] for r in rows], float)
    cover = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) <= 1000     # [site, village]
    chosen, covered = [], np.zeros(len(rows), bool)
    for _ in range(k):                                       # greedy max coverage: a feasible reference, not an optimum
        gain = (cover & ~covered[None, :]) @ w
        best = int(np.argmax(gain))
        chosen.append(best)
        covered |= cover[best]
    return {"period": f"{year}/{month}", "villages": rows, "n_villages": len(rows), "city_total": city_total,
            "village_total": int(w.sum()), "k": k, "radius_m": 1000, "districts": only_districts,
            "greedy": {"sites": [rows[i]["code"] for i in chosen], "covered": int(w[covered].sum())}}


def youbike(raw: Path, radius: float = 500) -> dict:
    return youbike_from_file(raw / "youbike_immediate.json", radius)


def youbike_from_file(path: Path, radius: float = 500) -> dict:
    """Live data: the judge recomputes the truth from the very file a run downloaded when it is the official feed,
    so both see the same moment (stations go in and out of service)."""
    yb = pd.DataFrame(json.loads(Path(path).read_text(encoding="utf-8")))
    d = haversine(yb["latitude"].astype(float), yb["longitude"].astype(float), *MRT_DAAN)
    yb["dist_m"] = d
    inside = yb[d <= radius - 10]
    border = yb[(d > radius - 10) & (d <= radius + 10)]   # 10 m either side of the stated radius: either answer is fine
    def rows(df):
        return [{"sno": str(r.sno), "sna": r.sna, "lat": float(r.latitude), "lon": float(r.longitude),
                 "dist_m": round(float(r.dist_m), 1)} for r in df.itertuples()]
    return {"centre": MRT_DAAN, "radius_m": radius, "stations": rows(inside), "optional": rows(border),
            "snapshot": str(yb["srcUpdateTime"].max())}


def aed(raw: Path, codes: tuple = (63000030,), groups: int = 3) -> dict:
    a = pd.read_csv(raw / "aed_taipei.csv", encoding="utf-8-sig")
    s = a[a["區域代碼"].isin(codes) & ~a["場所分類"].isin(["學校、大型集會場所", "其他"])].copy()
    s["name"] = s["場所名稱"].astype(str).str.strip()
    places = (s.groupby("name").agg(lat=("緯度", "mean"), lon=("經度", "mean"), n_aed=("name", "size"),
                                    category=("場所分類", "first")).reset_index())
    return {"rows": len(s), "places": places.to_dict("records"), "n_places": len(places), "groups": groups,
            "group_size": len(places) // groups}


def main(raw: str, out: str):
    raw = Path(raw)
    truth = {"ev_chargers": chargers(raw), "youbike_tour": youbike(raw), "aed_groups": aed(raw),
             "ev_chargers_s": chargers(raw, ["大安區"], 3), "youbike_tour_l": youbike(raw, 800),
             "aed_groups_l": aed(raw, (63000010, 63000020, 63000030, 63000040), 4),
             "taipei_bbox": TAIPEI_BBOX}
    Path(out).write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    c, y, g = truth["ev_chargers"], truth["youbike_tour"], truth["aed_groups"]
    print(f"chargers: {c['n_villages']} villages, total {c['village_total']} (city {c['city_total']}), "
          f"greedy covers {c['greedy']['covered']}")
    print(f"youbike: {len(y['stations'])} stations within 490 m, {len(y['optional'])} on the 500 m border")
    print(f"aed: {g['rows']} rows -> {g['n_places']} places, {g['groups']} groups of {g['group_size']}")




# ---------------------------------------------------------------- international cases

def _pmedian_reference(w, d, p, exact_limit_s=300):
    """Greedy + vertex interchange (Teitz-Bart); plus the exact MILP optimum when HiGHS proves it in time."""
    n = len(w)
    cost = lambda S: float((w * d[:, S].min(axis=1)).sum())
    S = []
    for _ in range(p):
        S.append(min((j for j in range(n) if j not in S), key=lambda j: cost(S + [j])))
    improved = True
    while improved:
        improved = False
        for a in list(S):
            for b in range(n):
                if b in S:
                    continue
                T = [b if s == a else s for s in S]
                if cost(T) < cost(S) - 1e-9:
                    S, improved = T, True
                    break
            if improved:
                break
    out = {"heuristic_sites": S, "heuristic_cost": cost(S), "exact_cost": None}
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
        nv = n + n * n                                   # x[j] open, y[i,j] i served by j
        c = np.concatenate([np.zeros(n), (w[:, None] * d).ravel()])
        A = lil_matrix((1 + n + n * n, nv))
        lo, hi = [p], [p]
        A[0, :n] = 1
        for i in range(n):
            A[1 + i, n + i * n: n + (i + 1) * n] = 1
            lo.append(1); hi.append(1)
        r = 1 + n
        for i in range(n):
            for j in range(n):
                A[r, n + i * n + j] = 1; A[r, j] = -1; lo.append(-np.inf); hi.append(0); r += 1
        res = milp(c, integrality=np.ones(nv), bounds=Bounds(0, 1), constraints=LinearConstraint(A.tocsr(), lo, hi),
                   options={"time_limit": exact_limit_s})
        if res.status == 0:
            x = np.round(res.x[:n]).astype(int)
            out.update(exact_cost=float(res.fun), exact_sites=[int(j) for j in np.flatnonzero(x)])
    except Exception as e:
        out["exact_error"] = str(e)[:200]
    return out


def london(raw: Path, level: str, boroughs: list[str], p: int) -> dict:
    """p-median over ONS 2021 small areas: Census 2021 usual residents (Nomis TS001) and the ONS population-weighted
    centroids (British National Grid, converted to WGS84). Straight-line distances."""
    from pyproj import Transformer
    if level == "MSOA":
        pop = pd.read_csv(raw / "nomis_ts001_msoa.csv")
        cent = pd.read_csv(raw / "msoa2021_pwc.csv", encoding="utf-8-sig").rename(columns={"MSOA21CD": "code", "X": "x", "Y": "y"})
    else:
        pop = pd.concat([pd.read_csv(raw / "nomis_part_0.csv"), pd.read_csv(raw / "nomis_part_25000.csv")])
        cent = pd.read_csv(raw / "lsoa2021_pwc.csv", encoding="utf-8-sig").rename(columns={"LSOA21CD": "code"})
    pop = pop.rename(columns={"GEOGRAPHY_CODE": "code", "GEOGRAPHY_NAME": "name", "OBS_VALUE": "pop"})
    pop["borough"] = pop["name"].str.rsplit(" ", n=1).str[0]
    pop = pop[pop["borough"].isin(boroughs)].merge(cent[["code", "x", "y"]], on="code", how="left")
    if pop["x"].isna().any():
        raise SystemExit(f"centroids missing for {pop.loc[pop.x.isna(), 'code'].tolist()[:5]}")
    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(pop["x"].to_numpy(), pop["y"].to_numpy())
    pop["lat"], pop["lon"] = lat, lon
    rows = [{"code": r.code, "name": r.name, "pop": int(r.pop), "lat": float(r.lat), "lon": float(r.lon),
             "x": float(r.x), "y": float(r.y)} for r in pop.itertuples()]
    w = np.array([r["pop"] for r in rows], float)
    la, lo = np.array([r["lat"] for r in rows]), np.array([r["lon"] for r in rows])
    d = haversine(la[:, None], lo[:, None], la[None, :], lo[None, :])
    ref = _pmedian_reference(w, d, p, exact_limit_s=600 if len(rows) <= 60 else 300)
    return {"level": level, "boroughs": boroughs, "p": p, "areas": rows, "n": len(rows), "pop_total": int(w.sum()),
            "reference": ref, "bbox": (51.28, -0.52, 51.70, 0.34)}


def _rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def _centroid(geom):
    """Area-weighted centroid (lon, lat) of a (multi)polygon; small areas, planar formula is adequate."""
    A = cx = cy = 0.0
    for poly in _rings(geom):
        for k, ring in enumerate(poly):
            pts = np.asarray(ring, float)
            x, y = pts[:, 0], pts[:, 1]
            cross = x[:-1] * y[1:] - x[1:] * y[:-1]
            sign = 1 if k == 0 else -1 if cross.sum() * 1 > 0 else 1      # holes subtract
            a = cross.sum() / 2
            A += a; cx += ((x[:-1] + x[1:]) * cross).sum() / 6; cy += ((y[:-1] + y[1:]) * cross).sum() / 6
    if A == 0:
        pts = np.asarray(_rings(geom)[0][0], float)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return cx / A, cy / A


def _inside(lon, lat, geom) -> bool:
    """Ray casting over all rings (even-odd), so holes are handled."""
    inside = False
    for poly in _rings(geom):
        for ring in poly:
            pts = np.asarray(ring, float)
            x, y = pts[:, 0], pts[:, 1]
            x2, y2 = np.roll(x, -1), np.roll(y, -1)
            cond = ((y > lat) != (y2 > lat)) & (lon < (x2 - x) * (lat - y) / np.where(y2 - y == 0, 1e-12, y2 - y) + x)
            inside ^= bool(cond.sum() % 2)
    return inside


def berlin(raw: Path, districts: list[str], k: int, radius: float = 500.0) -> dict:
    """New playgrounds at LOR planning-area centroids of the given districts; maximise residents (Umweltatlas 2019
    blocks) within `radius` of a new playground who are NOT within `radius` of an existing playground (block counted
    once). Blocks are assigned to planning areas by point in polygon."""
    plr = json.loads((raw / "berlin_lor2021_plr.geojson").read_text())["features"]
    blocks = json.loads((raw / "berlin_einwohnerdichte_2019.geojson").read_text())["features"]
    plays = json.loads((raw / "berlin_spielplaetze.geojson").read_text())["features"]
    areas = [f for f in plr if f["properties"]["bez"].split(" - ", 1)[-1] in districts]
    cand = []
    for f in areas:
        lon, lat = _centroid(f["geometry"])
        cand.append({"plr_id": f["properties"]["plr_id"], "name": f["properties"]["plr_name"], "lat": lat, "lon": lon,
                     "district": f["properties"]["bez"]})
    boxes = []
    for f in areas:
        pts = np.concatenate([np.asarray(r, float) for poly in _rings(f["geometry"]) for r in poly])
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    demand = []
    for b in blocks:
        ew = b["properties"].get("ew2019") or 0
        if ew <= 0 or not b.get("geometry"):
            continue
        lon, lat = _centroid(b["geometry"])
        for f, (x0, y0, x1, y1) in zip(areas, boxes):
            if x0 <= lon <= x1 and y0 <= lat <= y1 and _inside(lon, lat, f["geometry"]):
                demand.append({"block": b["properties"]["schluessel"], "ew": int(ew), "lat": lat, "lon": lon,
                               "plr_id": f["properties"]["plr_id"]})
                break
    pl = np.array([_centroid(f["geometry"]) for f in plays if f.get("geometry")])          # (lon, lat)
    dla, dlo = np.array([d["lat"] for d in demand]), np.array([d["lon"] for d in demand])
    served = (haversine(dla[:, None], dlo[:, None], pl[None, :, 1], pl[None, :, 0]) <= radius).any(axis=1)
    open_demand = [d for d, s in zip(demand, served) if not s]
    w = np.array([d["ew"] for d in open_demand], float)
    ola, olo = np.array([d["lat"] for d in open_demand]), np.array([d["lon"] for d in open_demand])
    cla, clo = np.array([c["lat"] for c in cand]), np.array([c["lon"] for c in cand])
    cover = haversine(cla[:, None], clo[:, None], ola[None, :], olo[None, :]) <= radius      # [candidate, block]
    chosen, covered = [], np.zeros(len(open_demand), bool)
    for _ in range(k):
        gain = (cover & ~covered[None, :]) @ w
        best = int(np.argmax(gain)); chosen.append(best); covered |= cover[best]
    ref = {"greedy_sites": chosen, "greedy_value": float(w[covered].sum()), "exact_value": None}
    try:                                                       # exact maximum coverage: x[c], z[b] <= sum x
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
        nc, nb = len(cand), len(open_demand)
        useful = np.flatnonzero(cover.any(axis=0))
        A = lil_matrix((1 + len(useful), nc + len(useful)))
        lo, hi = [k], [k]
        A[0, :nc] = 1
        for r, b in enumerate(useful):
            A[1 + r, nc + r] = 1
            for c in np.flatnonzero(cover[:, b]):
                A[1 + r, c] = -1
            lo.append(-np.inf); hi.append(0)
        obj = np.concatenate([np.zeros(nc), -w[useful]])
        res = milp(obj, integrality=np.ones(nc + len(useful)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(A.tocsr(), lo, hi), options={"time_limit": 600})
        if res.status == 0:
            ref.update(exact_value=float(-res.fun), exact_sites=[int(c) for c in np.flatnonzero(np.round(res.x[:nc]))])
    except Exception as e:
        ref["exact_error"] = str(e)[:200]
    return {"districts": districts, "k": k, "radius_m": radius, "candidates": cand, "n_candidates": len(cand),
            "blocks": demand, "n_blocks": len(demand), "residents": int(sum(d["ew"] for d in demand)),
            "open_blocks": [d["block"] for d in open_demand], "open_residents": int(w.sum()),
            "playgrounds_used": int(len(pl)), "reference": ref, "bbox": (52.33, 13.08, 52.68, 13.77)}


def intl(raw: str, out: str):
    raw = Path(raw)
    truth = {"london_ambulance_s": london(raw, "MSOA", ["Camden"], 3),
             "london_ambulance_l": london(raw, "LSOA", ["Camden", "Islington"], 8),
             # 250 m: at 500 m only 311 of 290,380 residents of Friedrichshain-Kreuzberg are not near a playground
             "berlin_playgrounds_s": berlin(raw, ["Friedrichshain-Kreuzberg"], 3, radius=250.0),
             "berlin_playgrounds_l": berlin(raw, ["Mitte", "Pankow", "Friedrichshain-Kreuzberg"], 8, radius=250.0)}
    Path(out).write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, t in truth.items():
        r = t["reference"]
        if k.startswith("london"):
            print(f"{k}: {t['n']} {t['level']}s, population {t['pop_total']}, p={t['p']}, heuristic "
                  f"{r['heuristic_cost']:.0f}, exact {r['exact_cost']}")
        else:
            print(f"{k}: {t['n_candidates']} candidates, {t['n_blocks']} blocks / {t['residents']} residents, "
                  f"{t['open_residents']} not near a playground, k={t['k']}, greedy {r['greedy_value']:.0f}, "
                  f"exact {r['exact_value']}")


def melbourne_bike_from_file(path: Path, radius: float = 1000) -> dict:
    """City of Melbourne bike-share dock locations (data.gov.au): every dock within `radius` of Flinders Street Station.
    Same structure as the YouBike truth so the Tour judge can be reused; 10 m either side of the radius is optional."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    elif path.suffix.lower() == ".geojson":
        g = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame([{**f["properties"], "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                           for f in g["features"]])
    else:
        df = pd.read_csv(path)
    d = haversine(df["lat"].astype(float), df["lon"].astype(float), *FLINDERS_ST)
    df["dist_m"] = d
    inside, border = df[d <= radius - 10], df[(d > radius - 10) & (d <= radius + 10)]
    def rows(x):
        return [{"sno": str(r.station_id), "sna": str(r.name), "lat": float(r.lat), "lon": float(r.lon),
                 "dist_m": round(float(r.dist_m), 1)} for r in x.itertuples()]
    return {"centre": FLINDERS_ST, "radius_m": radius, "stations": rows(inside), "optional": rows(border),
            "snapshot": "static dataset",
            "bbox": [-38.10, 144.55, -37.55, 145.30]}         # Greater Melbourne: [lat_min, lon_min, lat_max, lon_max]


def melbourne(raw: str, out: str):
    t = {"melbourne_bike_tour": melbourne_bike_from_file(Path(raw) / "bike_share_dock_locations.csv", 1000)}
    Path(out).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"melbourne_bike_tour: {len(t['melbourne_bike_tour']['stations'])} docks within 1 km of Flinders St")


def lv95_to_wgs84(e: float, n: float) -> tuple[float, float]:
    """swisstopo approximate formula (LV95 -> WGS84), accurate to about 1 m."""
    y, x = (e - 2600000) / 1e6, (n - 1200000) / 1e6
    lon = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x * x - 0.0436 * y ** 3
    lat = 16.9023892 + 3.238272 * x - 0.270978 * y * y - 0.002528 * x * x - 0.0447 * y * y * x - 0.0140 * x ** 3
    return lat * 100 / 36, lon * 100 / 36


def basel_aed(raw: str, out: str, plz: int = 4051, groups: int = 5):
    """Basel-Stadt AED locations (opendata.swiss, LV95 points): places in one postcode, grouped by place name."""
    a = pd.read_csv(Path(raw) / "defibrillatoren.csv")
    a = a[a["PLZ"] == plz].copy()
    xy = a["geometry"].str.extract(r"POINT \(([\d.]+) ([\d.]+)\)").astype(float)
    ll = [lv95_to_wgs84(e, n) for e, n in zip(xy[0], xy[1])]
    a["lat"], a["lon"] = [p[0] for p in ll], [p[1] for p in ll]
    a["name"] = a["Standort"].astype(str).str.strip()
    places = a.groupby("name").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_aed=("name", "size")).reset_index()
    t = {"basel_aed_groups": {"rows": len(a), "places": places.to_dict("records"), "n_places": len(places),
                              "groups": groups, "group_size": len(places) // groups,
                              "bbox": [47.51, 7.55, 47.60, 7.70]}}
    Path(out).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"basel_aed_groups: {len(a)} AEDs, {len(places)} places in PLZ {plz}, {groups} groups")


def nyc_pb(raw: str, out: str, district: str = "3", year: int = 2017, budget: int = 1_000_000):
    """NYC participatory budgeting (NYC Open Data wwhr-5ven): every project on one district's ballot; the exact
    best set under the budget by full enumeration, and the city's own rule (fund by votes until the money runs out)."""
    import itertools as it
    d = pd.read_csv(Path(raw) / "pb_projects.csv")
    g = d[(d["Vote Year"] == year) & (d["Council District"].astype(str) == str(district))]
    rows = [{"number": str(int(r["Project Number"])), "title": str(r["Title"]).strip(), "category": r["Category"],
             "cost": float(r["Cost"]), "votes": float(r["Votes"]), "winner": int(r["Winner"]),
             "lat": float(r["Latitude"]), "lon": float(r["Longitude"])} for _, r in g.iterrows()]
    n = len(rows)
    best = (-1.0, ())
    for k in range(n + 1):
        for c in it.combinations(range(n), k):
            if sum(rows[i]["cost"] for i in c) <= budget:
                v = sum(rows[i]["votes"] for i in c)
                if v > best[0]:
                    best = (v, c)
    greedy, spent = [], 0.0
    for i in sorted(range(n), key=lambda i: -rows[i]["votes"]):
        if spent + rows[i]["cost"] <= budget:
            greedy.append(i); spent += rows[i]["cost"]
    t = {"nyc_pb_budget": {"projects": rows, "n_projects": n, "budget": budget, "district": str(district), "year": year,
                           "optimum": {"votes": best[0], "projects": list(best[1]),
                                       "cost": sum(rows[i]["cost"] for i in best[1])},
                           "city_rule": {"votes": sum(rows[i]["votes"] for i in greedy), "projects": greedy,
                                         "cost": spent},
                           "actual_winners": [i for i, r in enumerate(rows) if r["winner"]],
                           "bbox": [40.68, -74.03, 40.80, -73.93]}}
    Path(out).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"nyc_pb_budget: {n} projects, optimum {best[0]:.0f} votes ({len(best[1])} projects, "
          f"${t['nyc_pb_budget']['optimum']['cost']:,.0f}); city rule {t['nyc_pb_budget']['city_rule']['votes']:.0f} votes")


def _hav(lat, lon, lat0, lon0):
    return haversine(np.asarray(lat, float), np.asarray(lon, float), lat0, lon0)


def test_set(raw: str, out: str):
    """Test set (built after the 2026-10-22 code freeze): Singapore hawker groups, Paris Velib tour, Cambridge PB."""
    import itertools as it
    raw = Path(raw)
    t = {}
    # 1. Singapore hawker centres within 2.5 km of Toa Payoh MRT, 3 equal groups (Groups judge)
    g = json.loads((raw / "hawker_centres.geojson").read_text(encoding="utf-8"))
    places = []
    for f in g["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        if float(_hav(lat, lon, 1.3327, 103.8474)) <= 2500:
            places.append({"name": f["properties"]["NAME"], "lat": lat, "lon": lon,
                           "postal": str(f["properties"]["ADDRESSPOSTALCODE"])})
    t["sg_hawker_groups"] = {"places": places, "n_places": len(places), "groups": 3, "group_size": len(places) // 3,
                             "bbox": [1.15, 103.6, 1.48, 104.1]}
    # 2. Paris Velib stations within 600 m of Gare de Lyon (Tour judge; 10 m either side of the radius optional)
    v = pd.read_csv(raw / "velib_stations.csv", sep=";", encoding="utf-8-sig")
    xy = v["coordonnees_geo"].str.split(",", expand=True).astype(float)
    v["lat"], v["lon"] = xy[0], xy[1]
    v["dist_m"] = _hav(v["lat"], v["lon"], 48.8443, 2.3744)
    rows = lambda x: [{"sno": str(r.stationcode), "sna": str(r.name), "lat": float(r.lat), "lon": float(r.lon),
                       "dist_m": round(float(r.dist_m), 1)} for r in x.itertuples()]
    t["paris_velib_tour"] = {"centre": [48.8443, 2.3744], "radius_m": 600, "stations": rows(v[v["dist_m"] <= 590]),
                             "optional": rows(v[(v["dist_m"] > 590) & (v["dist_m"] <= 610)]),
                             "snapshot": "opendata.paris.fr export 2026-10-22", "bbox": [48.80, 2.22, 48.92, 2.47]}
    # 3. Cambridge (MA) participatory budgeting, cycle 7 (FY22), budget $525,000 (Budget judge)
    d = pd.read_csv(raw / "cambridge_pb.csv")
    d = d[d["PB Cycle"] == 7]
    proj = [{"number": "", "title": str(r["Project"]).strip(), "cost": float(r["Project Cost"]),
             "votes": float(r["Total Votes"]), "winner": int(str(r["Winning project"]).lower() == "yes")}
            for _, r in d.iterrows()]
    budget = 525_000
    best = (-1.0, ())
    for k in range(len(proj) + 1):
        for c in it.combinations(range(len(proj)), k):
            if sum(proj[i]["cost"] for i in c) <= budget:
                val = sum(proj[i]["votes"] for i in c)
                if val > best[0]:
                    best = (val, c)
    greedy, spent = [], 0.0
    for i in sorted(range(len(proj)), key=lambda i: -proj[i]["votes"]):
        if spent + proj[i]["cost"] <= budget:
            greedy.append(i); spent += proj[i]["cost"]
    t["cambridge_pb_budget"] = {"projects": proj, "n_projects": len(proj), "budget": budget,
                                "optimum": {"votes": best[0], "projects": list(best[1]),
                                            "cost": sum(proj[i]["cost"] for i in best[1])},
                                "city_rule": {"votes": sum(proj[i]["votes"] for i in greedy), "projects": greedy,
                                              "cost": spent},
                                "actual_winners": [i for i, r in enumerate(proj) if r["winner"]],
                                "bbox": [42.35, -71.17, 42.41, -71.06]}
    Path(out).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"sg_hawker_groups: {len(places)} places -> 3 groups; paris_velib_tour: {len(t['paris_velib_tour']['stations'])} "
          f"stations (+{len(t['paris_velib_tour']['optional'])} optional); cambridge_pb_budget: {len(proj)} projects, "
          f"optimum {best[0]:.0f} votes / ${t['cambridge_pb_budget']['optimum']['cost']:,.0f}, city rule "
          f"{t['cambridge_pb_budget']['city_rule']['votes']:.0f}")


def test_set2(raw: str, out: str):
    """Second held-out set (built after the 2026-10-24 freeze): Dublin bike tour and Somerville budget (Basel AED groups
    uses truth_basel_20261021.json)."""
    import itertools as it
    raw = Path(raw)
    t = {}
    g = json.loads((raw / "dublinbikes_stations.geojson").read_text(encoding="utf-8"))
    rows = []
    for f in g["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        d = float(_hav(lat, lon, 53.3509, -6.2500))
        rows.append({"sno": str(f["properties"].get("station_id")), "sna": str(f["properties"].get("name")),
                     "lat": lat, "lon": lon, "dist_m": round(d, 1)})
    t["dublin_bike_tour"] = {"centre": [53.3509, -6.2500], "radius_m": 600,
                             "stations": [r for r in rows if r["dist_m"] <= 590],
                             "optional": [r for r in rows if 590 < r["dist_m"] <= 610],
                             "snapshot": "data.smartdublin.ie stations.geojson 2026-10-24", "bbox": [53.2, -6.5, 53.5, -6.0]}
    d = pd.read_csv(raw / "somerville_pb.csv")
    d = d[d["Year"] == 2023]
    proj = [{"number": "", "title": str(r["Project"]).strip(), "cost": float(r["Cost"]), "votes": float(r["Total.Score"]),
             "winner": 0} for _, r in d.iterrows()]            # "votes" holds the ballot's Total.Score
    budget = 1_000_000
    best = (-1.0, ())
    for k in range(len(proj) + 1):
        for c in it.combinations(range(len(proj)), k):
            if sum(proj[i]["cost"] for i in c) <= budget:
                v = sum(proj[i]["votes"] for i in c)
                if v > best[0]:
                    best = (v, c)
    greedy, spent = [], 0.0
    for i in sorted(range(len(proj)), key=lambda i: -proj[i]["votes"]):
        if spent + proj[i]["cost"] <= budget:
            greedy.append(i); spent += proj[i]["cost"]
    t["somerville_pb_budget"] = {"projects": proj, "n_projects": len(proj), "budget": budget,
                                 "optimum": {"votes": best[0], "projects": list(best[1]),
                                             "cost": sum(proj[i]["cost"] for i in best[1])},
                                 "city_rule": {"votes": sum(proj[i]["votes"] for i in greedy), "projects": greedy,
                                               "cost": spent},
                                 "bbox": [42.37, -71.14, 42.42, -71.07]}
    Path(out).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dublin_bike_tour: {len(t['dublin_bike_tour']['stations'])} stations (+{len(t['dublin_bike_tour']['optional'])} "
          f"optional); somerville_pb_budget: {len(proj)} projects, optimum {best[0]:.0f} points / "
          f"${t['somerville_pb_budget']['optimum']['cost']:,.0f}, score-order rule {t['somerville_pb_budget']['city_rule']['votes']:.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "intl":
        intl(*sys.argv[2:4])
    elif sys.argv[1] == "basel":
        basel_aed(*sys.argv[2:4])
    elif sys.argv[1] == "test_set2":
        test_set2(*sys.argv[2:4])
    elif sys.argv[1] == "test_set":
        test_set(*sys.argv[2:4])
    elif sys.argv[1] == "nyc_pb":
        nyc_pb(*sys.argv[2:4])
    elif sys.argv[1] == "melbourne":
        melbourne(*sys.argv[2:4])
    else:
        try:                                     # held-out sets kept out of the public repository (acceptance/local)
            from local import truth_extra
            getattr(truth_extra, sys.argv[1])(*sys.argv[2:4])
        except (ImportError, AttributeError):
            main(*sys.argv[1:3])
