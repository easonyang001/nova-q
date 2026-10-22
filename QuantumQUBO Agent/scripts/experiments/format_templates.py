"""Model-specific input formats for the 100-dataset sweep: each dataset is prepared for an optimisation model of its
problem type, not into a generic list. Same DATA_SCHEMA conventions as the real cases (sets / params, derived
parameters computed with a named prep tool, user settings, reducible sets shrunk only with prep.reduce_points).

    template_for(kind, problem_type) -> (name, model source, spec text builder, sanity checker)
"""
from __future__ import annotations

from typing import Any

import numpy as np

_HEADER = "import numpy as np\n"


def _model(schema: str, example: str, decision_set: str) -> str:
    """A module whose model is trivial (choose members of the decision set); only the data format matters here."""
    return _HEADER + f"""
DATA_SCHEMA = {schema}
REQUIREMENT_MAP = {{}}
MODEL_CHOICE = "format template: the data format is what is tested"


def make_example_data(size, seed):
{example}


def build_ir(data):
    ids = list(data["{decision_set}"])
    n = len(ids)
    return {{"name": "template", "sense": "maximize", "variable_names": [f"x{{i}}" for i in range(n)],
            "variable_keys": [{{"{decision_set}": i}} for i in ids], "linear": {{str(i): 1.0 for i in range(n)}},
            "quadratic": [], "constant": 0, "constraints": []}}


def describe_solution(data, x):
    return {{"chosen": [i for i, b in zip(data["{decision_set}"], x) if b]}}
"""


def _ex(lines: str) -> str:
    return "\n".join("    " + l for l in lines.strip().splitlines())


PT = '{"meaning": "%s", "index": ["%s"], "type": "number", "source": "public", "unit": "%s"}'
DIST = ('{"meaning": "straight-line distance in metres", "index": ["%s", "%s"], "type": "number", "source": "public", '
        '"unit": "m", "derived": {"tool": "haversine_matrix", "from": ["raw:lat", "raw:lon"], "how": "pairwise great-circle distance"}}')

TEMPLATES: dict[str, dict[str, Any]] = {
    "sites_cover": {
        "schema": '{"sets": {"sites": {"meaning": "one real place that can host a facility", "source": "public", '
                  '"reducible": {"method": "reduce_points", "weight": null}}}, "params": {'
                  '"lat": ' + PT % ("latitude WGS84", "sites", "deg_lat") + ', "lon": ' + PT % ("longitude WGS84", "sites", "deg_lon") +
                  ', "dist": ' + DIST % ("sites", "sites") +
                  ', "k": {"meaning": "facilities to open", "index": [], "type": "int", "source": "user", "value": 5}'
                  ', "radius": {"meaning": "coverage radius", "index": [], "type": "number", "source": "user", "unit": "m", "value": 1000}}}',
        "example": _ex('ids = [f"S{i}" for i in range(size)]\nlat = [25.0 + i / 200 for i in range(size)]\nlon = [121.5 + i / 200 for i in range(size)]\n'
                       'd = [[abs(i - j) * 600.0 for j in range(size)] for i in range(size)]\n'
                       'return {"sites": ids, "lat": lat, "lon": lon, "dist": d, "k": 2, "radius": 1000}'),
        "set": "sites", "max": 150,
        "spec": "Choose 5 of the {thing} in {place} as facility sites so that as many other {thing} as possible lie "
                "within 1000 m of a chosen site. Candidates: every {thing} in {place}, one per real place; if there are "
                "more than 150, reduce them to 150 representatives with prep.reduce_points. Coordinates WGS84.",
    },
    "stops_tour": {
        "schema": '{"sets": {"stops": {"meaning": "one stop to visit", "source": "public", '
                  '"reducible": {"method": "reduce_points", "weight": null}}}, "params": {'
                  '"lat": ' + PT % ("latitude WGS84", "stops", "deg_lat") + ', "lon": ' + PT % ("longitude WGS84", "stops", "deg_lon") +
                  ', "dist": ' + DIST % ("stops", "stops") + '}}',
        "example": _ex('ids = [f"T{i}" for i in range(size)]\nlat = [25.0 + i / 300 for i in range(size)]\nlon = [121.5] * size\n'
                       'd = [[abs(i - j) * 370.0 for j in range(size)] for i in range(size)]\n'
                       'return {"stops": ids, "lat": lat, "lon": lon, "dist": d}'),
        "set": "stops", "max": 30,
        "spec": "Plan one closed inspection tour through {thing} in {place}, each visited once, shortest straight-line "
                "length. Stops: the {thing} of {place}, one per real place, reduced to 30 representatives with "
                "prep.reduce_points if there are more. Coordinates WGS84.",
    },
    "items_groups": {
        "schema": '{"sets": {"items": {"meaning": "one real place to assign", "source": "public", '
                  '"reducible": {"method": "reduce_points", "weight": null}}, '
                  '"groups": {"meaning": "the teams to fill, defined by the user", "source": "user"}}, "params": {'
                  '"lat": ' + PT % ("latitude WGS84", "items", "deg_lat") + ', "lon": ' + PT % ("longitude WGS84", "items", "deg_lon") +
                  ', "dist": ' + DIST % ("items", "items") + '}}',
        "example": _ex('ids = [f"I{i}" for i in range(size)]\nlat = [25.0 + i / 200 for i in range(size)]\nlon = [121.5] * size\n'
                       'd = [[abs(i - j) * 550.0 for j in range(size)] for i in range(size)]\n'
                       'return {"items": ids, "groups": ["G1", "G2"], "lat": lat, "lon": lon, "dist": d}'),
        "set": "items", "max": 60,
        "spec": "Split the {thing} of {place} into 4 inspection teams G1..G4 of equal size, each team's places close "
                "together. Items: the {thing}, one per real place, reduced to 60 representatives with "
                "prep.reduce_points if there are more. The teams are given by the user: groups = [G1, G2, G3, G4].",
    },
    "nodes_edges": {
        "schema": '{"sets": {"nodes": {"meaning": "one network node", "source": "public", '
                  '"reducible": {"method": "reduce_points", "weight": null}}}, "params": {'
                  '"lat": ' + PT % ("latitude WGS84", "nodes", "deg_lat") + ', "lon": ' + PT % ("longitude WGS84", "nodes", "deg_lon") +
                  ', "edge_m": {"meaning": "length of each possible link: only pairs at most 1000 m apart", '
                  '"index": ["nodes", "nodes"], "type": "number", "source": "public", "unit": "m", "sparse": true, '
                  '"derived": {"tool": "haversine_matrix", "from": ["raw:lat", "raw:lon"], "how": "pairs within 1000 m"}}}}',
        "example": _ex('ids = [f"N{i}" for i in range(size)]\nlat = [25.0 + i / 300 for i in range(size)]\nlon = [121.5] * size\n'
                       'rows = [[ids[i], ids[i + 1], 370.0] for i in range(size - 1)]\n'
                       'return {"nodes": ids, "lat": lat, "lon": lon, "edge_m": rows}'),
        "set": "nodes", "max": 80,
        "spec": "Build a network over the {thing} of {place}: nodes are the {thing}, one per real place, reduced to 80 "
                "representatives with prep.reduce_points if there are more; a link is possible between two nodes at "
                "most 1000 m apart, with its straight-line length.",
    },
    "areas_weight": {
        "schema": '{"sets": {"areas": {"meaning": "an area at the lowest administrative level in the table", "source": "public"}}, '
                  '"params": {"weight": {"meaning": "the measure for the latest period, all categories combined", '
                  '"index": ["areas"], "type": "number", "source": "public", "min": 0}, '
                  '"k": {"meaning": "areas to choose", "index": [], "type": "int", "source": "user", "value": 5}}}',
        "example": _ex('ids = [f"A{i}" for i in range(size)]\nreturn {"areas": ids, "weight": [100.0 + i for i in range(size)], "k": 2}'),
        "set": "areas", "max": 5000,
        "spec": "Choose the 5 areas of {place} with the most {thing}. Areas: the lowest administrative level in the "
                "data; {thing} for the LATEST period, all categories (sex, age, ...) combined; no total rows.",
    },
    "slots_demand": {
        "schema": '{"sets": {"slots": {"meaning": "consecutive time periods, oldest first", "source": "public"}}, '
                  '"params": {"demand": {"meaning": "the measure in each period, all categories combined", '
                  '"index": ["slots"], "type": "number", "source": "public", "min": 0}, '
                  '"staff": {"meaning": "staff available per period", "index": [], "type": "int", "source": "user", "value": 10}}}',
        "example": _ex('ids = [f"2025-{i + 1:02d}" for i in range(size)]\nreturn {"slots": ids, "demand": [50.0 + i for i in range(size)], "staff": 10}'),
        "set": "slots", "max": 60,
        "spec": "Schedule staff over the most recent 12 periods (months, weeks or days, as the data has them) of {thing} "
                "in {place}: slots are those periods in time order with their total {thing}, all categories combined.",
    },
    "items_cost": {
        "schema": '{"sets": {"items": {"meaning": "one budget item (agency, project or programme)", "source": "public"}}, '
                  '"params": {"cost": {"meaning": "its amount for the latest year", "index": ["items"], "type": "number", '
                  '"source": "public", "min": 0}, "budget": {"meaning": "total budget", "index": [], "type": "number", '
                  '"source": "user", "value": 1000000}}}',
        "example": _ex('ids = [f"B{i}" for i in range(size)]\nreturn {"items": ids, "cost": [1000.0 * (i + 1) for i in range(size)], "budget": 5000}'),
        "set": "items", "max": 2000,
        "spec": "Select {thing} items of {place} within a budget: items are the individual entries (agencies, projects or "
                "programmes, the most detailed level, no totals) with their amount for the LATEST year.",
    },
    "assets_returns": {
        "schema": '{"sets": {"assets": {"meaning": "one asset or price series", "source": "public"}, '
                  '"periods": {"meaning": "consecutive periods, oldest first", "source": "public"}}, '
                  '"params": {"price": {"meaning": "price or index level of each asset in each period", '
                  '"index": ["assets", "periods"], "type": "number", "source": "public", "min": 0}}}',
        "example": _ex('a = [f"F{i}" for i in range(size)]\np = [f"2025-{m:02d}" for m in range(1, 7)]\n'
                       'return {"assets": a, "periods": p, "price": [[100.0 + i + m for m in range(6)] for i in range(size)]}'),
        "set": "assets", "max": 200,
        "spec": "Build a portfolio from the {thing} of {place}: assets are the individual series, periods the most recent "
                "24 periods they ALL share (time order); price[asset][period] without gaps.",
    },
    "features_obs": {
        "schema": '{"sets": {"features": {"meaning": "one measured variable", "source": "public"}, '
                  '"observations": {"meaning": "one complete record (time and place)", "source": "public"}}, '
                  '"params": {"value": {"meaning": "the value of each feature in each observation", '
                  '"index": ["observations", "features"], "type": "number", "source": "public"}}}',
        "example": _ex('f = ["pm25", "no2", "o3"]\no = [f"r{i}" for i in range(size)]\n'
                       'return {"features": f, "observations": o, "value": [[float(i + j) for j in range(3)] for i in range(size)]}'),
        "set": "features", "max": 100,
        "spec": "Select features from the {thing} of {place}: features are the measured variables (at least 3), "
                "observations the most recent 200 records in which all of them are present; value[observation][feature].",
    },
}

TYPE_TO_TEMPLATE = {
    "points": {"location": "sites_cover", "set_cover": "sites_cover", "routing": "stops_tour",
               "assignment": "items_groups", "clustering": "items_groups", "graph": "nodes_edges",
               "network_design": "nodes_edges"},
    "stats": {"scheduling": "slots_demand", "resource_allocation": "items_cost", "knapsack": "items_cost",
              "portfolio": "assets_returns", "feature_selection": "features_obs"},
}


def template_for(kind: str, problem_type: str) -> str:
    return TYPE_TO_TEMPLATE.get(kind, {}).get(problem_type) or ("sites_cover" if kind == "points" else "areas_weight")


def module_source(name: str) -> str:
    t = TEMPLATES[name]
    return _model(t["schema"].replace("null", "None").replace("true", "True"), t["example"], t["set"])


def spec_text(name: str, thing: str, place: str) -> str:
    return TEMPLATES[name]["spec"].format(thing=thing, place=place)


def sanity(name: str, data: dict, bbox: tuple | None) -> dict:
    """Independent, format-specific checks (never the system's own verdicts)."""
    t = TEMPLATES[name]
    s = t["set"]
    ids = data.get(s) or []
    c: dict[str, Any] = {"n": len(ids), "nonempty": len(ids) > 0, "within_max": len(ids) <= t["max"],
                         "unique_ids": len(set(map(str, ids))) == len(ids)}
    if "lat" in data and bbox:
        inside = sum(1 for a, o in zip(data["lat"], data["lon"]) if isinstance(a, (int, float)) and
                     isinstance(o, (int, float)) and bbox[0] <= a <= bbox[2] and bbox[1] <= o <= bbox[3])
        c["inside_country"] = f"{inside}/{len(ids)}"
        c["coords_ok"] = inside == len(ids)
    for key in ("dist",):
        if key in data:
            m = np.array(data[key], float)
            c["dist_square"] = m.shape == (len(ids), len(ids))
            c["dist_ok"] = bool(c["dist_square"] and np.allclose(np.diag(m), 0) and (m >= 0).all() and np.allclose(m, m.T, atol=1))
    if "edge_m" in data:
        rows = data["edge_m"]
        c["edges"] = len(rows)
        c["edges_ok"] = all(len(r) == 3 and 0 < float(r[2]) <= 1100 for r in rows) and len(rows) > 0
    if "weight" in data or "demand" in data or "cost" in data:
        v = data.get("weight") or data.get("demand") or data.get("cost") or []
        c["numeric_ok"] = len(v) == len(ids) and all(isinstance(x, (int, float)) and x >= 0 for x in v)
        names = [str(x).strip() for x in ids]
        c["no_total_rows"] = not any(n.lower().startswith(("total", "gesamt", "insgesamt", "總計", "合計")) for n in names)
    if name == "slots_demand":
        c["slots_ordered"] = list(map(str, ids)) == sorted(map(str, ids))
    if name == "assets_returns":
        p = np.array(data.get("price") or [[]], dtype=object)
        c["matrix_ok"] = (len(data.get("price", [])) == len(ids) and all(len(r) == len(data.get("periods", [])) for r in data["price"])
                          and all(isinstance(x, (int, float)) for r in data["price"] for x in r))
    if name == "features_obs":
        vals = data.get("value", [])
        c["features_ge_3"] = len(ids) >= 3
        c["matrix_ok"] = (len(vals) == len(data.get("observations", [])) and all(len(r) == len(ids) for r in vals)
                          and all(isinstance(x, (int, float)) for r in vals for x in r))
    checks = [v for k, v in c.items() if isinstance(v, bool)]
    c["ok"] = all(checks)
    return c
