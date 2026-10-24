"""Independent acceptance judge: data, model and delivery verdicts from the ORIGINAL request and the raw files.

Fixed before the batch; the system never sees it. It reads what the system produced (DATA_SCHEMA, the instance,
build_ir's output with variable_keys, the delivered solution) and compares it with truth.json from truth.py.

- data:     the decision entities map one-to-one to real entities, coordinates are real, measured values match
            entity by entity, nothing is lost or duplicated, required scale kept.
- model:    on random decisions that satisfy the original request, the model's value (helper variables optimised)
            ranks them like the independently computed metric; decisions that break the request are infeasible.
- delivery: the delivered solution satisfies the original request, and its independently computed quality is
            within a tolerance of a simple reference computed from the raw data.
"""
from __future__ import annotations
import itertools
import math
import random

import numpy as np

QUALITY_TOLERANCE = 0.10        # delivered quality may trail the raw-data reference by at most 10 %
RANK_THRESHOLD = 0.9            # Spearman between model value and the true metric on random valid decisions
SAMPLES = 30


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _sets(ds):
    return ds.get("sets", {}) if isinstance(ds, dict) else {}


def _params(ds):
    return ds.get("params", {}) if isinstance(ds, dict) else {}


# ---------------------------------------------------------------- mapping system ids to real entities

def _norm(text) -> str:
    """Identifier comparison that ignores separators and case ('松山區_莊敬里' == '松山區莊敬里', 'E02000193' == 'e02000193')."""
    import re as _re
    import unicodedata
    t = unicodedata.normalize("NFKC", str(text)).replace("台", "臺").lower()
    return _re.sub(r"[\s_\-–/·.,:;|()\[\]]+", "", t)


def map_entities(ds, data, set_name, truth_rows, id_fields, tolerance_m):
    """{system id (str): truth row index}. Identity first (ids or text params equal to a truth id / name), then
    coordinates (params with unit deg_lat/deg_lon indexed by the set) for whatever is left. Coordinates alone can
    pick a neighbour when two centroid methods differ on a large irregular area (seen on mountain villages)."""
    ids = [str(v) for v in data.get(set_name, [])]
    params = _params(ds)
    mapping, how = {}, []
    lookup = {}
    for j, r in enumerate(truth_rows):
        for f in id_fields:
            key = f(r) if callable(f) else str(r.get(f, ""))
            if key:
                lookup.setdefault(_norm(key), j)
    texts = [ids] + [[str(v) for v in data[p]] for p, d in params.items()
                     if d.get("index") == [set_name] and d.get("type") == "text" and isinstance(data.get(p), list)]
    for column in texts:
        for i, t in zip(ids, column):
            if i not in mapping and _norm(t) in lookup:
                mapping[i] = lookup[_norm(t)]
    if mapping:
        how.append("id / name match")
    lat_p = next((p for p, d in params.items() if d.get("unit") == "deg_lat" and d.get("index") == [set_name]), None)
    lon_p = next((p for p, d in params.items() if d.get("unit") == "deg_lon" and d.get("index") == [set_name]), None)
    pairs = None
    if not (lat_p and lon_p):
        for key, v in data.items():                     # lenient: a list of [lat, lon] pairs, one per id
            if (isinstance(v, list) and len(v) == len(ids) and v and all(isinstance(q, list) and len(q) == 2 for q in v)
                    and all(isinstance(a, (int, float)) and -90 <= a <= 90 and -180 <= b <= 180 for a, b in v)):
                pairs = [(a, b) for a, b in v]
                break
    if len(mapping) < len(ids) and ((lat_p and lon_p) or pairs):
        tlat = np.array([r["lat"] for r in truth_rows]); tlon = np.array([r["lon"] for r in truth_rows])
        coords = pairs if pairs else list(zip(data[lat_p], data[lon_p]))
        before = len(mapping)
        for i, (la, lo) in zip(ids, coords):
            if i in mapping:
                continue
            try:
                d = haversine(float(la), float(lo), tlat, tlon)
            except (TypeError, ValueError):
                continue
            j = int(np.argmin(d))
            if d[j] <= tolerance_m:
                mapping[i] = j
        if len(mapping) > before:
            how.append(f"coordinates within {tolerance_m} m")
    return mapping, " + ".join(how)


# ---------------------------------------------------------------- the model, read from build_ir's output

def parse_ir(raw):
    names = [str(v) for v in raw["variable_names"]]
    by_name = {v: i for i, v in enumerate(names)}

    def idx(k):
        if isinstance(k, int):
            return k
        k = str(k)
        return int(k) if k.lstrip("-").isdigit() else by_name[k]

    lin = {idx(k): float(v) for k, v in raw.get("linear", {}).items()}
    quad = {}
    for i, j, v in raw.get("quadratic", []):
        i, j = sorted((idx(i), idx(j)))
        quad[(i, j)] = quad.get((i, j), 0.0) + float(v)
    cons = []
    for c in raw.get("constraints", []):
        if c["kind"] == "linear":
            cons.append(("linear", {idx(k): float(v) for k, v in c["coeffs"].items()}, c["sense"], float(c["rhs"])))
        else:
            cons.append((c["kind"], [(idx(a), idx(b)) for a, b in c["pairs"]], None, None))
    return {"n": len(names), "sense": raw["sense"], "lin": lin, "quad": quad, "const": float(raw.get("constant", 0)),
            "cons": cons, "keys": raw.get("variable_keys") or [{}] * len(names)}


def model_value(m, fixed: dict[int, int], time_limit=20.0):
    """Best model objective with the decision variables fixed and helper variables optimised (MILP).
    Returns (value or None if infeasible, note)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    free = [i for i in range(m["n"]) if i not in fixed]
    pos = {v: k for k, v in enumerate(free)}
    const = m["const"] + sum(m["lin"].get(i, 0) * b for i, b in fixed.items())
    c = np.zeros(len(free))
    for i, v in m["lin"].items():
        if i in pos:
            c[pos[i]] += v
    for (i, j), v in m["quad"].items():
        fi, fj = i in fixed, j in fixed
        if fi and fj:
            const += v * fixed[i] * fixed[j]
        elif fi:
            c[pos[j]] += v * fixed[i]
        elif fj:
            c[pos[i]] += v * fixed[j]
        else:
            return None, "helper variables interact quadratically: not checkable by MILP"
    rows, lo, hi = [], [], []
    for kind, a, sense, rhs in m["cons"]:
        if kind == "linear":
            row, r = np.zeros(len(free)), rhs - sum(v * fixed[i] for i, v in a.items() if i in fixed)
            for i, v in a.items():
                if i in pos:
                    row[pos[i]] += v
            entries = [(row, r, sense)]
        else:
            entries = []
            for i, j in a:
                row = np.zeros(len(free))
                if kind == "exclusion":            # x_i + x_j <= 1
                    r = 1 - fixed.get(i, 0) - fixed.get(j, 0)
                    if i in pos: row[pos[i]] += 1
                    if j in pos: row[pos[j]] += 1
                else:                              # x_i <= x_j
                    r = 0 - fixed.get(i, 0) + fixed.get(j, 0)
                    if i in pos: row[pos[i]] += 1
                    if j in pos: row[pos[j]] -= 1
                entries.append((row, r, "le"))
        for row, r, s in entries:
            if not row.any():
                if (s == "le" and r < -1e-9) or (s == "ge" and r > 1e-9) or (s == "eq" and abs(r) > 1e-9):
                    return None, "infeasible"
                continue
            rows.append(row); lo.append(r if s in ("ge", "eq") else -np.inf); hi.append(r if s in ("le", "eq") else np.inf)
    sign = 1 if m["sense"] == "minimize" else -1
    if not free:
        return const, "no helper variables"
    res = milp(sign * c, integrality=np.ones(len(free)), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(np.array(rows), lo, hi)] if rows else None,
               options={"time_limit": time_limit})
    if res.status == 2 or res.x is None:
        return None, "infeasible" if res.status == 2 else f"milp status {res.status}"
    return const + float(c @ np.round(res.x)), "helper variables optimised"


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- per-case knowledge (from the original request)

class Case:
    """Subclasses define: entity rows from truth, how decisions look, the true metric and the reference."""
    better = "max"

    def __init__(self, truth):
        self.truth = truth


class Chargers(Case):
    better = "max"

    def __init__(self, truth, case_id="ev_chargers"):
        super().__init__(truth)
        self.name = case_id
        t = truth[case_id]
        self.rows, self.k = t["villages"], t["k"]
        lat = np.array([r["lat"] for r in self.rows]); lon = np.array([r["lon"] for r in self.rows])
        self.cover = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) <= t["radius_m"]
        self.w = np.array([r["pop"] for r in self.rows], float)
        self.reference = t["greedy"]["covered"]
        # both official codes: the population table's and the boundary map's (CPTVID)
        # ids may also carry the city in front ('臺北市南港區舊莊里', smoke17): without it the mountain villages fell
        # back to coordinate matching and their polygon centroids lie > 500 m from the official village point
        self.id_fields = ["code", "cptvid", "village", lambda r: r["district"] + r["village"],
                          lambda r: "臺北市" + r["district"] + r["village"]]
        self.tolerance_m = 500

    def metric(self, entities):            # entities: list of truth indices chosen as sites
        covered = self.cover[list(entities)].any(axis=0) if entities else np.zeros(len(self.rows), bool)
        return float(self.w[covered].sum())

    def random_valid(self, rng, ids):
        return [{"sites": rng.sample(ids, self.k)}]

    def invalid(self, rng, ids):
        return [rng.sample(ids, self.k + 1), rng.sample(ids, self.k - 1)]


class Tour(Case):
    better = "min"

    def __init__(self, truth, case_id="youbike_tour"):
        super().__init__(truth)
        self.name = case_id
        t = truth[case_id]
        self.rows = t["stations"] + t["optional"]
        self.required = set(range(len(t["stations"])))
        self.id_fields = ["sno", "sna", lambda r: r["sna"].replace("YouBike2.0_", "")]
        self.tolerance_m = 30

    def metric(self, order):                # order: truth indices in visiting sequence
        lat = np.array([self.rows[i]["lat"] for i in order]); lon = np.array([self.rows[i]["lon"] for i in order])
        return float(haversine(lat, lon, np.roll(lat, -1), np.roll(lon, -1)).sum())

    def reference_value(self, entities):
        """Nearest neighbour + 2-opt on the same stations: a simple reference, not an optimum."""
        pts = list(entities)
        lat = np.array([self.rows[i]["lat"] for i in pts]); lon = np.array([self.rows[i]["lon"] for i in pts])
        d = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        tour, left = [0], set(range(1, len(pts)))
        while left:
            nxt = min(left, key=lambda j: d[tour[-1], j]); tour.append(nxt); left.remove(nxt)
        improved = True
        while improved:
            improved = False
            for a, b in itertools.combinations(range(1, len(tour)), 2):
                i, j = tour[a - 1], tour[a]; k, l = tour[b], tour[(b + 1) % len(tour)]
                if d[i, k] + d[j, l] < d[i, j] + d[k, l] - 1e-9:
                    tour[a:b + 1] = reversed(tour[a:b + 1]); improved = True
        return float(sum(d[tour[t], tour[(t + 1) % len(tour)]] for t in range(len(tour))))


class Groups(Case):
    better = "min"

    def __init__(self, truth, case_id="aed_groups"):
        super().__init__(truth)
        self.name = case_id
        t = truth[case_id]
        self.rows, self.groups, self.size = t["places"], t["groups"], t["group_size"]
        lat = np.array([r["lat"] for r in self.rows]); lon = np.array([r["lon"] for r in self.rows])
        self.d = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        self.id_fields = ["name"]
        self.tolerance_m = 60

    def metric(self, assignment):           # {truth index: group label}
        total = 0.0
        for g in set(assignment.values()):
            members = [i for i, h in assignment.items() if h == g]
            total += sum(self.d[i, j] for i, j in itertools.combinations(members, 2))
        return float(total)

    def reference_value(self):
        """Greedy balanced grouping + pairwise swaps: a simple reference, not an optimum."""
        n = len(self.rows)
        order = sorted(range(n), key=lambda i: (self.rows[i]["lat"], self.rows[i]["lon"]))
        assign = {i: k // self.size for k, i in enumerate(order)}
        best = self.metric(assign)
        improved = True
        while improved:
            improved = False
            for i, j in itertools.combinations(range(n), 2):
                if assign[i] != assign[j]:
                    assign[i], assign[j] = assign[j], assign[i]
                    v = self.metric(assign)
                    if v < best - 1e-9:
                        best, improved = v, True
                    else:
                        assign[i], assign[j] = assign[j], assign[i]
        return best


class PMedian(Case):
    """Ambulance sites: choose p areas; every area is served by its nearest open site (population-weighted)."""
    better = "min"

    def __init__(self, truth, case_id):
        super().__init__(truth)
        t = truth[case_id]
        self.name, self.rows, self.k = case_id, t["areas"], t["p"]
        lat = np.array([r["lat"] for r in self.rows]); lon = np.array([r["lon"] for r in self.rows])
        self.d = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        self.w = np.array([r["pop"] for r in self.rows], float)
        ref = t["reference"]
        self.reference = ref["exact_cost"] if ref.get("exact_cost") is not None else ref["heuristic_cost"]
        self.reference_kind = "exact optimum" if ref.get("exact_cost") is not None else "greedy + interchange"
        self.id_fields = ["code", "name"]
        self.tolerance_m = 300

    def metric(self, entities):
        return float((self.w * self.d[:, list(entities)].min(axis=1)).sum()) if entities else float("inf")


class NewCoverage(Case):
    """New playgrounds: k candidate sites; residents within the radius of a new site who were NOT within the radius
    of an existing playground, each block counted once (blocks assigned by the truth)."""
    better = "max"

    def __init__(self, truth, case_id):
        super().__init__(truth)
        t = truth[case_id]
        self.name, self.rows, self.k = case_id, t["candidates"], t["k"]
        open_set = set(t["open_blocks"])
        blocks = [b for b in t["blocks"] if b["block"] in open_set]
        cla = np.array([c["lat"] for c in self.rows]); clo = np.array([c["lon"] for c in self.rows])
        bla = np.array([b["lat"] for b in blocks]); blo = np.array([b["lon"] for b in blocks])
        self.cover = haversine(cla[:, None], clo[:, None], bla[None, :], blo[None, :]) <= t["radius_m"]
        self.w = np.array([b["ew"] for b in blocks], float)
        ref = t["reference"]
        self.reference = ref["exact_value"] if ref.get("exact_value") is not None else ref["greedy_value"]
        self.totals = (t["residents"], t["open_residents"])
        self.id_fields = ["plr_id", "name"]
        self.tolerance_m = 200

    def metric(self, entities):
        covered = self.cover[list(entities)].any(axis=0) if entities else np.zeros(len(self.w), bool)
        return float(self.w[covered].sum())


class Budget(Case):
    """Participatory budgeting: fund a set of ballot projects whose total cost stays within the budget, most votes."""
    better = "max"

    def __init__(self, truth, case_id):
        super().__init__(truth)
        t = truth[case_id]
        self.name, self.rows, self.budget = case_id, t["projects"], t["budget"]
        self.cost = np.array([r["cost"] for r in self.rows], float)
        self.votes = np.array([r["votes"] for r in self.rows], float)
        self.reference = t["optimum"]["votes"]
        self.id_fields = ["number", "title"]
        self.tolerance_m = 100

    def fits(self, entities):
        return float(self.cost[list(entities)].sum()) <= self.budget

    def metric(self, entities):
        return float(self.votes[list(entities)].sum())

    def random_valid(self, rng):
        """A random fundable set: projects in random order, each added if it still fits, stopping at a random size
        (sizes vary, so the true metric varies)."""
        order = rng.sample(range(len(self.rows)), len(self.rows))
        stop = rng.randint(1, len(self.rows))
        chosen = []
        for i in order:
            if len(chosen) >= stop:
                break
            if self.fits(chosen + [i]):
                chosen.append(i)
        return chosen

    def over_budget(self, rng):
        order = rng.sample(range(len(self.rows)), len(self.rows))
        chosen = []
        for i in order:
            chosen.append(i)
            if not self.fits(chosen):
                return chosen
        return chosen


CASES = {"ev_chargers": Chargers, "youbike_tour": Tour, "aed_groups": Groups,
         "ev_chargers_s": Chargers, "youbike_tour_l": Tour, "aed_groups_l": Groups,
         "london_ambulance_s": PMedian, "london_ambulance_l": PMedian,
         "berlin_playgrounds_s": NewCoverage, "berlin_playgrounds_l": NewCoverage,
         "melbourne_bike_tour": Tour, "basel_aed_groups": Groups, "nyc_pb_budget": Budget,
         # test set (2026-10-22, after the code freeze): existing judges, new cities and data
         "sg_hawker_groups": Groups, "paris_velib_tour": Tour, "cambridge_pb_budget": Budget,
         # second held-out set (2026-10-24)
         "dublin_bike_tour": Tour, "somerville_pb_budget": Budget}
TOUR_PREFIXES = ("youbike_tour", "melbourne_bike_tour", "paris_velib_tour", "dublin_bike_tour")
GROUP_PREFIXES = ("aed_groups", "basel_aed_groups", "sg_hawker_groups")
BUDGET = ("nyc_pb_budget", "cambridge_pb_budget", "somerville_pb_budget")
SELECT_K = ("ev_chargers", "ev_chargers_s", "london_ambulance_s", "london_ambulance_l", "berlin_playgrounds_s",
            "berlin_playgrounds_l")


def make_case(case_id, truth):
    return CASES[case_id](truth, case_id)


# ---------------------------------------------------------------- verdicts

def decision_structure(m, ds):
    """Sets used by decision variables: (entity-like sets, other sets)."""
    used = [tuple(sorted(k)) for k in m["keys"] if k]
    return sorted(set(used))


def judge(case_id: str, truth: dict, ds: dict, data: dict, raw_ir: dict | None, solution: list | None,
          requirements: dict) -> dict:
    """Never raises: an unexpected output is reported as a failed or unverifiable gate, not a crash."""
    partial: dict = {}
    try:
        return _judge(case_id, truth, ds, data, raw_ir, solution, requirements, partial)
    except Exception as e:
        import traceback
        out = partial.get("out") or {"data": {"ok": False, "checks": {}}, "model": {"ok": None, "checks": {}},
                                     "delivery": {"ok": False, "checks": {}}}
        out["judge_error"] = f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-600:]}"
        for gate in ("model", "delivery"):
            if out[gate].get("ok") is False and not out[gate]["checks"]:
                out[gate]["ok"] = None
        return out


def _judge(case_id: str, truth: dict, ds: dict, data: dict, raw_ir: dict | None, solution: list | None,
           requirements: dict, partial: dict) -> dict:
    case = make_case(case_id, truth)
    out = {"data": {"ok": False, "checks": {}}, "model": {"ok": False, "checks": {}},
           "delivery": {"ok": False, "checks": {}}}
    partial["out"] = out
    if not isinstance(ds, dict) or "sets" not in ds:
        out["data"]["checks"]["schema"] = "DATA_SCHEMA is not in the sets/params format"
        return out
    m = parse_ir(raw_ir) if raw_ir else None
    shapes = decision_structure(m, ds) if m else []
    out["model"]["checks"]["decision_sets"] = shapes
    entity_set = requirements.get("entity_set_hint")
    decision_sets = {s for shape in shapes for s in shape}
    # sets the decisions are about come first, so a tie (sites and villages are the same places) picks them
    def id_list(v):
        return (isinstance(v, list) and v and all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in v)
                and len({str(x) for x in v}) == len(v))
    declared = [s for s in _sets(ds) if id_list(data.get(s))]
    extra = [k for k, v in data.items() if k not in declared and id_list(v)]     # lenient: undeclared id lists
    candidates = sorted(declared + extra, key=lambda s: (s not in decision_sets, s not in declared))
    best = None
    for s in candidates:                    # the set whose ids map best to the real entities
        mapping, how = map_entities(ds, data, s, case.rows, case.id_fields, case.tolerance_m)
        if best is None or len(mapping) > len(best[1]):
            best = (s, mapping, how)
    entity_set, mapping, how = best if best else (entity_set, {}, "")
    ids = [str(v) for v in data.get(entity_set, [])] if entity_set else []
    d = out["data"]["checks"]
    d["entity_set"], d["mapping"] = entity_set, how
    d["entity_set_declared"] = entity_set in _sets(ds)
    d["instance_entities"] = len(ids)
    d["mapped"] = len(mapping)
    targets = list(mapping.values())
    d["duplicates"] = sorted({case.rows[j].get("village") or case.rows[j].get("sna") or case.rows[j].get("name")
                              for j in targets if targets.count(j) > 1})[:20]
    required = set(range(len(case.rows))) if not hasattr(case, "required") else case.required
    missing = sorted(required - set(targets))
    d["missing_real_entities"] = len(missing)
    d["missing_examples"] = [case.rows[j].get("village") or case.rows[j].get("sna") or case.rows[j].get("name")
                             for j in missing[:10]]
    d["unmapped_ids"] = [i for i in ids if i not in mapping][:10]
    lat_p = [p for p, q in _params(ds).items() if q.get("unit") == "deg_lat"]
    bbox = (truth.get(case_id) or {}).get("bbox") or truth.get("taipei_bbox")
    outside = 0
    for p in lat_p:
        lon_p = next((q for q, r in _params(ds).items() if r.get("unit") == "deg_lon" and r.get("index") == _params(ds)[p].get("index")), None)
        if lon_p and isinstance(data.get(p), list):
            outside += sum(not (bbox[0] <= float(a) <= bbox[2] and bbox[1] <= float(b) <= bbox[3])
                           for a, b in zip(data[p], data[lon_p]) if isinstance(a, (int, float)) and isinstance(b, (int, float)))
    d["coordinates_outside_area"] = outside
    # identity matched by name/id: the coordinates must still belong to that entity (a copied or invented position
    # is wrong even when the name is right); 3x the mapping tolerance absorbs different centroid methods
    lat_e = next((p for p, q in _params(ds).items() if q.get("unit") == "deg_lat" and q.get("index") == [entity_set]), None)
    lon_e = next((p for p, q in _params(ds).items() if q.get("unit") == "deg_lon" and q.get("index") == [entity_set]), None)
    wrong, same = [], 0
    if lat_e and lon_e and isinstance(data.get(lat_e), list):
        pts = list(zip(ids, data[lat_e], data[lon_e]))
        seen = {}
        for i, la, lo in pts:
            if not isinstance(la, (int, float)) or not isinstance(lo, (int, float)):
                continue
            key = (round(la, 6), round(lo, 6))
            same += key in seen
            seen.setdefault(key, i)
            if i in mapping:
                r = case.rows[mapping[i]]
                if float(haversine(la, lo, r["lat"], r["lon"])) > 3 * case.tolerance_m:
                    wrong.append(i)
    d["coordinates_far_from_own_entity"] = wrong[:10]
    d["identical_coordinates"] = same
    data_ok = (len(ids) > 0 and len(mapping) == len(ids) and not d["duplicates"] and not missing and outside == 0
               and not wrong and same == 0)
    if case_id in SELECT_K and not case_id.startswith("berlin"):
        # measured values entity by entity: some numeric param indexed by a mapped set equals the real population
        found = None
        for p, q in _params(ds).items():
            if q.get("index") == [entity_set] and q.get("type") in ("int", "number") and isinstance(data.get(p), list):
                vals = data[p]
                exact = sum(1 for i, v in zip(ids, vals) if i in mapping and isinstance(v, (int, float))
                            and abs(v - case.rows[mapping[i]]["pop"]) < 0.5)
                if found is None or exact > found[1]:
                    found = (p, exact, float(np.nansum([v for v in vals if isinstance(v, (int, float))])))
        if found:
            d["population_param"], d["population_exact_matches"], d["population_total"] = found
        d["population_truth_total"] = (truth[case_id]["village_total"] if case_id.startswith("ev_chargers")
                                       else truth[case_id]["pop_total"])
        data_ok = data_ok and bool(found) and found[1] == len(case.rows) and abs(found[2] - d["population_truth_total"]) < 1
    if case_id.startswith("berlin"):
        # demand side: some numeric parameter must hold the residents, all of them or those not yet served
        sums = {p: float(np.nansum([v for v in data[p] if isinstance(v, (int, float))]))
                for p, q in _params(ds).items() if q.get("type") in ("int", "number") and isinstance(data.get(p), list)
                and data[p] and not isinstance(data[p][0], list)}
        hit = [p for p, v in sums.items() if any(abs(v - t) <= 0.01 * t for t in case.totals)]
        d["resident_totals"] = {p: round(v) for p, v in sums.items()}
        d["resident_truth"] = {"all": case.totals[0], "not_near_a_playground": case.totals[1]}
        data_ok = data_ok and bool(hit)
    if case_id in BUDGET:
        # measured values entity by entity: votes exactly, cost up to one unit factor (dollars, thousands, ...)
        def per_entity(p):
            return {i: v for i, v in zip(ids, data[p]) if i in mapping and isinstance(v, (int, float))}
        nums = [p for p, q in _params(ds).items() if q.get("index") == [entity_set] and isinstance(data.get(p), list)]
        votes_p = next((p for p in nums if len(per_entity(p)) == len(case.rows) and all(
            abs(v - case.votes[mapping[i]]) < 0.5 for i, v in per_entity(p).items())), None)
        cost_p = None
        for p in nums:
            vals = per_entity(p)
            ratios = {round(v / case.cost[mapping[i]], 9) for i, v in vals.items() if case.cost[mapping[i]]}
            if len(vals) == len(case.rows) and len(ratios) == 1 and next(iter(ratios)) > 0:
                cost_p = p
                break
        budget_ok = any(isinstance(v, (int, float)) and cost_p and abs(v - case.budget * next(iter(
            {round(x / case.cost[mapping[i]], 9) for i, x in per_entity(cost_p).items()}))) < 0.5
            for v in data.values())
        d["votes_param"], d["cost_param"], d["budget_value_found"] = votes_p, cost_p, bool(budget_ok)
        data_ok = data_ok and bool(votes_p) and bool(cost_p) and bool(budget_ok)
    if case_id.startswith(GROUP_PREFIXES):
        groups = [s for s in candidates if s != entity_set and isinstance(data.get(s), list) and len(data[s]) == case.groups]
        d["group_set"] = groups[0] if groups else None
    out["data"]["ok"] = bool(data_ok)
    if m is None:
        return out
    # ---------------- model: random valid decisions rank like the true metric; invalid ones are infeasible
    mc = out["model"]["checks"]
    rng = random.Random(0)
    inv = {v: k for k, v in mapping.items()}
    key_index = {tuple(sorted((s, str(v)) for s, v in k.items())): i for i, k in enumerate(m["keys"]) if k}
    decision_vars = [i for i, k in enumerate(m["keys"]) if k]

    def fix(keys):
        idx = []
        for k in keys:
            t = tuple(sorted((s, str(v)) for s, v in k.items()))
            if t not in key_index:
                return None
            idx.append(key_index[t])
        chosen = set(idx)
        return {i: int(i in chosen) for i in decision_vars}

    samples, invalid = [], []
    mapped_ids = [i for i in ids if i in mapping]
    key_ids = {}
    for k in m["keys"]:
        for sname, v in (k or {}).items():
            key_ids.setdefault(sname, []).append(str(v))
    key_ids = {k: list(dict.fromkeys(v)) for k, v in key_ids.items()}
    # the decision set that names the real entities (its ids are the instance ids), and the other one
    ent_name = max(key_ids, key=lambda n: len(set(key_ids[n]) & set(mapped_ids)), default=None)
    if ent_name is not None and not set(key_ids[ent_name]) & set(mapped_ids):
        ent_name = None
    others = [n for n in key_ids if n != ent_name]
    E = ent_name
    partition_like = case_id.startswith(TOUR_PREFIXES + GROUP_PREFIXES)   # every entity must be in every solution
    if E and partition_like and len(mapped_ids) != len(key_ids.get(E, [])):
        mc["structure"] = (f"{len(key_ids.get(E, []))} decision entities but only {len(mapped_ids)} map to real ones: "
                           f"model unverifiable because the data is wrong")
        out["model"]["ok"] = None
    elif E and case_id in SELECT_K and shapes == [(E,)]:
        # half random, half clustered (a site and its nearest neighbours): overlapping service areas are where
        # double counting shows; random picks of distant sites hide it (seen in the Berlin self-test)
        tla = np.array([case.rows[mapping[i]]["lat"] for i in mapped_ids])
        tlo = np.array([case.rows[mapping[i]]["lon"] for i in mapped_ids])
        for n_sample in range(SAMPLES):
            if n_sample % 2 == 0:
                pick = rng.sample(mapped_ids, case.k)
            else:
                c0 = rng.randrange(len(mapped_ids))
                near = np.argsort(haversine(tla[c0], tlo[c0], tla, tlo))[:case.k]
                pick = [mapped_ids[j] for j in near]
            samples.append(([{E: i} for i in pick], case.metric([mapping[i] for i in pick])))
        invalid = [[{E: i} for i in rng.sample(mapped_ids, n)] for n in (case.k + 1, case.k - 1)]
    elif E and case_id in BUDGET and shapes == [(E,)]:
        back = {j: i for i, j in mapping.items()}
        for _ in range(SAMPLES):
            pick = case.random_valid(rng)
            samples.append(([{E: back[j]} for j in pick], case.metric(pick)))
        invalid = [[{E: back[j]} for j in case.over_budget(rng)] for _ in range(2)]
    elif E and case_id.startswith(TOUR_PREFIXES) and len(shapes) == 1 and len(shapes[0]) == 2 and len(others) == 1:
        pos_set = others[0]
        steps = key_ids[pos_set]
        for _ in range(SAMPLES):
            order = rng.sample(mapped_ids, len(mapped_ids))
            samples.append(([{E: i, pos_set: p} for i, p in zip(order, steps)],
                            case.metric([mapping[i] for i in order])))
        order = rng.sample(mapped_ids, len(mapped_ids)); order[1] = order[0]          # a station twice, one missing
        invalid = [[{E: i, pos_set: p} for i, p in zip(order, steps)]]
    elif E and case_id.startswith(GROUP_PREFIXES) and len(mapped_ids) != case.groups * case.size:
        mc["structure"] = (f"the instance has {len(mapped_ids)} mapped places instead of {case.groups * case.size}: "
                           f"model unverifiable because the data is wrong")
        out["model"]["ok"] = None
    elif E and case_id.startswith(GROUP_PREFIXES) and len(shapes) == 1 and len(others) == 1 and len(key_ids[others[0]]) == case.groups:
        gname, labels = others[0], key_ids[others[0]]
        d["group_set"] = d.get("group_set") or gname
        for _ in range(SAMPLES):
            perm = rng.sample(mapped_ids, len(mapped_ids))
            assign = {i: labels[k // case.size] for k, i in enumerate(perm)}
            samples.append(([{E: i, gname: g} for i, g in assign.items()],
                            case.metric({mapping[i]: g for i, g in assign.items()})))
        perm = rng.sample(mapped_ids, len(mapped_ids))
        sizes = [case.size + 1, case.size - 1] + [case.size] * (case.groups - 2)
        assign, k = {}, 0
        for g, n in zip(labels, sizes):
            for i in perm[k:k + n]:
                assign[i] = g
            k += n
        invalid = [[{E: i, gname: g} for i, g in assign.items()]]
    else:
        mc["structure"] = "decision variables not recognised for this request (see decision_sets); model unverifiable"
        out["model"]["ok"] = None                     # unverifiable, reported apart from "wrong"
    if samples:
        values, truths, notes = [], [], set()
        for keys, t in samples:
            variants = [keys]
            if case_id.startswith(TOUR_PREFIXES):
                # the same closed tour, rotated or reversed: a model may fix the first stop to break symmetry
                pos = others[0]
                order = [k[E] for k in keys]
                steps = [k[pos] for k in keys]
                n_ = len(order)
                variants = [[{E: o, pos: s_} for o, s_ in zip(seq[r:] + seq[:r], steps)]
                            for seq in (order, order[::-1]) for r in range(n_)]
            fixed, v, note = None, None, ""
            for variant in variants:
                f_ = fix(variant)
                if f_ is None:
                    continue
                v, note = model_value(m, f_)
                fixed = f_
                if v is not None:
                    break
            if fixed is None:
                notes.add("a valid decision has no matching decision variable")
                continue
            notes.add(note)
            if v is None:
                values.append(None); truths.append(t)
                continue
            values.append(v); truths.append(t)
        feasible = [i for i, v in enumerate(values) if v is not None]
        mc["valid_decisions_feasible"] = f"{len(feasible)}/{len(samples)}"
        if feasible:
            mv = np.array([values[i] for i in feasible]); tv = np.array([truths[i] for i in feasible])
            # both oriented "higher is better"
            mv = mv if m["sense"] == "maximize" else -mv
            tv = tv if case.better == "max" else -tv
            mc["rank_agreement"] = round(spearman(mv, tv), 3)
        rejected = []
        for keys in invalid:
            fixed = fix(keys)
            v, _ = model_value(m, fixed) if fixed is not None else (None, "")
            rejected.append(v is None)
        mc["invalid_decisions_rejected"] = f"{sum(rejected)}/{len(rejected)}"
        mc["notes"] = sorted(notes)
        out["model"]["ok"] = (len(feasible) == len(samples) and mc.get("rank_agreement", 0) >= RANK_THRESHOLD
                              and all(rejected))
    # ---------------- delivery
    dc = out["delivery"]["checks"]
    if solution is None:
        dc["solution"] = "none delivered"
        return out
    chosen = [k for k, b in zip(m["keys"], solution) if b and k]
    dc["decisions"] = len(chosen)
    try:
        if E is None:
            raise ValueError("decision variables do not name the real entities")
        if case_id in SELECT_K:
            sites = [str(k.get(E)) for k in chosen]
            ok_req = len(sites) == case.k and len(set(sites)) == case.k and all(s in mapping for s in sites)
            value = case.metric([mapping[s] for s in sites]) if ok_req else None
            ref = case.reference
        elif case_id in BUDGET:
            picked = [str(k.get(E)) for k in chosen]
            ok_req = len(set(picked)) == len(picked) and all(s in mapping for s in picked) and \
                case.fits([mapping[s] for s in picked])
            dc["true_cost"] = float(case.cost[[mapping[s] for s in picked if s in mapping]].sum())
            value = case.metric([mapping[s] for s in picked]) if ok_req else None
            ref = case.reference
        elif case_id.startswith(TOUR_PREFIXES):
            pos_set = others[0]
            order_pos = {str(p): n for n, p in enumerate(data[pos_set] if isinstance(data.get(pos_set), list)
                                                          else key_ids[pos_set])}
            seq = sorted(chosen, key=lambda k: order_pos.get(str(k.get(pos_set)), 1e9))
            stations = [str(k.get(E)) for k in seq]
            visited = {mapping[s] for s in stations if s in mapping}
            ok_req = (len(stations) == len(ids) and len(set(stations)) == len(ids)
                      and len({str(k.get(pos_set)) for k in seq}) == len(seq) and all(s in mapping for s in stations)
                      and case.required <= visited)                 # every required real station is on the tour
            dc["required_stations_missing"] = len(case.required - visited)
            value = case.metric([mapping[s] for s in stations]) if ok_req else None
            ref = case.reference_value([mapping[s] for s in mapped_ids])
        else:
            assign = {str(k.get(E)): str(k.get(others[0])) for k in chosen}
            sizes = {g: list(assign.values()).count(g) for g in set(assign.values())}
            ok_req = (len(chosen) == len(ids) and len(assign) == len(ids) and len(sizes) == case.groups
                      and all(n == case.size for n in sizes.values())
                      and {mapping[i] for i in assign if i in mapping} == set(range(len(case.rows))))  # every real place
            value = case.metric({mapping[i]: g for i, g in assign.items()}) if ok_req else None
            ref = case.reference_value()
    except Exception as e:
        dc["error"] = f"{type(e).__name__}: {e}"
        return out
    dc["meets_original_requirements"] = bool(ok_req)
    dc["true_quality"], dc["reference_quality"] = value, ref
    if value is not None:
        gap = (ref - value) / ref if case.better == "max" else (value - ref) / ref
        dc["gap_vs_reference"] = round(gap, 4)
        out["delivery"]["ok"] = bool(ok_req and gap <= QUALITY_TOLERANCE)
    return out
