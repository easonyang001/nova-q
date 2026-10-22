"""DATA_SCHEMA as index sets + parameters: declaration checks, data checks, Modeler and Data Adapter integration."""
import pandas as pd

from qubo_auto.annealing import schema
from qubo_auto.annealing.contracts import check_data
from qubo_auto.annealing.modeler import load_module, validate

DS = {
    "sets": {"sites": {"meaning": "candidate site", "source": "public",
                       "reducible": {"method": "reduce_points", "weight": "pop"}},
             "hours": {"meaning": "hour of the day", "source": "user"}},
    "params": {
        "pop": {"meaning": "people near the site", "index": ["sites"], "type": "int", "source": "public",
                "unit": "person", "min": 0},
        "lat": {"meaning": "latitude", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lat"},
        "demand": {"meaning": "charging demand", "index": ["sites", "hours"], "type": "number", "source": "public",
                   "unit": "kWh", "max_missing_fraction": 0.25},
        "road": {"meaning": "road links", "index": ["sites", "sites"], "type": "number", "source": "public",
                 "unit": "m", "sparse": True},
        "k": {"meaning": "sites to open", "index": [], "type": "int", "source": "user", "value": 2},
    },
}
DATA = {"sites": ["A", "B", "C"], "hours": [8, 9], "pop": [10, 20, 30], "lat": [25.0, 25.1, 25.2],
        "demand": [[1.0, 2.0], [3.0, None], [5.0, 6.0]], "road": [["A", "B", 120.0], ["B", "C", 80.5]], "k": 2}


def test_declaration_errors_name_the_problem():
    assert schema.declaration_errors(DS) == []
    assert "sets" in schema.declaration_errors({"pop": "people per site"})[0]          # legacy flat text schema
    bad = {"sets": {"sites": {"meaning": "site", "source": "web"}},
           "params": {"d": {"meaning": "distance", "index": ["sites", "zones"], "type": "float", "source": "public",
                            "unit": "km", "value": 3, "derived": {"tool": "magic_distance", "from": ["xy"]}}}}
    text = "\n".join(schema.declaration_errors(bad))
    for fragment in ("source must be one of", "'zones' is not a declared set", "type must be one of",
                     "unit 'km' is not canonical; use m", "'value' is only for user settings",
                     "'magic_distance' is not a prep function", "derived from 'xy'"):
        assert fragment in text, fragment


def test_data_shapes_ids_sparse_rows_ranges_and_settings():
    assert schema.data_errors(DS, DATA) == []
    cases = [
        ({"demand": [[1.0, 2.0], [3.0, 4.0]]}, "length 2 at top level but set 'sites' has 3 ids"),
        ({"sites": ["A", "A", "C"]}, "ids repeat"),
        ({"road": [["A", "Z", 5.0]]}, "id 'Z' not in set 'sites'"),
        ({"lat": [25.0, 2770000.0, 25.2]}, "outside [-90, 90]"),
        ({"demand": [[None, None], [3.0, None], [5.0, 6.0]]}, "3 of 6 values missing"),
        ({"k": 3}, "differs from the confirmed setting 2"),
        ({"pop": [10, -1, 30]}, "below the declared minimum 0"),
        ({"extra": 1}, "not declared in DATA_SCHEMA"),
    ]
    for change, fragment in cases:
        errors = schema.data_errors(DS, {**DATA, **change})
        assert any(fragment in e for e in errors), (fragment, errors)
    # small synthetic examples may scale confirmed settings
    assert schema.data_errors(DS, {**DATA, "k": 1}, check_values=False) == []


def test_contract_is_derived_from_the_schema():
    contract = schema.contract(DS)
    assert contract["k"] == {"source_kind": "user", "value": 2}
    assert contract["sites"]["source_kind"] == "public" and contract["lat"]["unit"] == "deg_lat"
    errors, _ = check_data(DATA, contract)
    assert errors == []
    assert "sites" in schema.describe(DS) and "SPARSE" in schema.describe(DS) and "= 2" in schema.describe(DS)


MODEL = '''
import numpy as np
DATA_SCHEMA = {
  "sets": {"sites": {"meaning": "candidate site", "source": "public",
                     "reducible": {"method": "reduce_points", "weight": "pop"}}},
  "params": {
    "name": {"meaning": "site name", "index": ["sites"], "type": "text", "source": "public"},
    "pop": {"meaning": "people near the site", "index": ["sites"], "type": "int", "source": "public", "min": 0},
    "lat": {"meaning": "latitude", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lat"},
    "lon": {"meaning": "longitude", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lon"},
    "dist": {"meaning": "distance between sites", "index": ["sites", "sites"], "type": "number", "source": "public",
             "unit": "m", "derived": {"tool": "haversine_matrix", "from": ["lat", "lon"]}},
    "k": {"meaning": "sites to open", "index": [], "type": "int", "source": "user", "value": 2}%EXTRA%}}
REQUIREMENT_MAP = {}

def make_example_data(size, seed):
    rng = np.random.default_rng(seed)
    lat = (25 + rng.random(size) * 0.1).tolist()
    lon = (121.5 + rng.random(size) * 0.1).tolist()
    dist = [[abs(a - b) * 1e5 + abs(c - e) * 1e5 for b, e in zip(lat, lon)] for a, c in zip(lat, lon)]
    return {"sites": [f"S{i}" for i in range(size)], "name": [f"site {i}" for i in range(size)],
            "pop": [int(v) for v in rng.integers(1, 100, size)], "lat": lat, "lon": lon, "dist": dist, "k": 2}

def build_ir(data):
    n, d, p = len(data["sites"]), np.asarray(data["dist"]), data["pop"]
    return {"name": "spread", "sense": "maximize", "variable_names": [f"open_{s}" for s in data["sites"]],
            "variable_keys": [{"sites": s} for s in data["sites"]],
            "linear": {str(i): float(p[i]) for i in range(n)},
            "quadratic": [[i, j, float(d[i][j]) / 1000] for i in range(n) for j in range(i + 1, n)], "constant": 0,
            "constraints": [{"name": "open_k", "kind": "linear", "coeffs": {str(i): 1 for i in range(n)},
                             "sense": "eq", "rhs": data["k"]}]}

def describe_solution(data, x):
    return {"open": [(data["name"][i], data["lat"][i], data["lon"][i]) for i, v in enumerate(x) if v]}
'''


def test_structured_model_validates_and_carries_its_contract():
    ns = load_module(MODEL.replace("%EXTRA%", ""))
    assert ns["DATA_CONTRACT"]["k"]["value"] == 2                     # derived on load: one declaration
    errors, report = validate(ns, strict_schema=True)
    assert errors == [] and report["n_variables"] == 6


def test_unused_parameters_and_legacy_schemas_are_rejected_in_strict_mode():
    extra = ',\n    "price": {"meaning": "unused price", "index": ["sites"], "type": "number", "source": "public"}'
    ns = load_module(MODEL.replace("%EXTRA%", extra).replace('"k": 2}', '"k": 2, "price": [1.0] * size}'))
    errors, report = validate(ns, strict_schema=True)
    assert errors == [] and any("price" in w for w in report["warnings"])      # reported, not blocking
    legacy = load_module(MODEL.replace("%EXTRA%", "").replace("DATA_SCHEMA = {", "DATA_SCHEMA_OLD = {", 1)
                         .replace("REQUIREMENT_MAP = {}", 'REQUIREMENT_MAP = {}\nDATA_SCHEMA = {"pop": "people"}'))
    assert "sets" in validate(legacy, strict_schema=True)[0][0]
    assert not any("sets" in e for e in validate(legacy)[0])            # old saved models still load


def test_example_data_must_follow_the_schema():
    ns = load_module(MODEL.replace("%EXTRA%", "").replace('"lat": lat, "lon"', '"lat": lat[:-1], "lon"'))
    errors, _ = validate(ns, strict_schema=True)
    assert any("make_example_data does not follow DATA_SCHEMA" in e and "'lat'" in e for e in errors), errors


def test_adapter_must_compute_derived_parameters_with_the_declared_tool():
    from qubo_auto.annealing.adapter import build_adapter
    ns = load_module(MODEL.replace("%EXTRA%", ""))
    tables = {"sites": pd.DataFrame({"name": ["A", "B", "C"], "pop": [5, 7, 9],
                                     "lat": [25.03, 25.04, 25.05], "lon": [121.50, 121.52, 121.55]})}
    body = ("    t = tables['sites']\n"
            "    out = {'sites': t['name'].tolist(), 'name': t['name'].tolist(), 'pop': t['pop'].tolist(),\n"
            "           'lat': t['lat'].tolist(), 'lon': t['lon'].tolist(), 'k': 2}\n")
    meta = ("ADAPTER_METADATA = {k: {'source_urls': ['https://example.org/sites.csv']} for k in "
            "['sites', 'name', 'pop', 'lat', 'lon', 'dist']}\nADAPTER_METADATA['lat']['unit'] = 'deg_lat'\n"
            "ADAPTER_METADATA['lon']['unit'] = 'deg_lon'\nADAPTER_METADATA['dist']['unit'] = 'm'\n"
            "ADAPTER_METADATA['k'] = {}\n")
    by_hand = meta + "def load_data(tables, log):\n" + body + (
        "    out['dist'] = [[abs(a - b) * 111000 for b in out['lat']] for a in out['lat']]\n    return out\n")
    with_tool = meta + "def load_data(tables, log):\n" + body + (
        "    out['dist'] = prep.haversine_matrix(out['lat'], out['lon']).tolist()\n    return out\n")
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append(prompt)
            return f"```python\n{by_hand if len(prompts) == 1 else with_tool}```"

    code, data, errors, report = build_adapter(ns, tables, "open 2 sites", Fake(), "m",
                                               sources=[{"title": "sites", "url": "https://example.org/sites.csv"}])
    assert errors == [] and data["sites"] == ["A", "B", "C"] and len(data["dist"]) == 3
    assert "derived by prep.haversine_matrix" in prompts[0]
    assert "REQUIRED TOOL NOT USED: DATA_SCHEMA declares 'dist'" in prompts[1]


def test_data_search_only_sees_public_items():
    view = schema.public_view(DS)
    assert set(view["sets"]) == {"sites", "hours"} and "k" not in view["params"] and "pop" in view["params"]  # hours indexes public demand
    assert schema.public_view({"pop": "people"}) == {"pop": "people"}


def test_common_model_mistakes_get_specific_feedback():
    nested = {"sets": {"sites": ["A", "B"]}, "params": {"pop": [1, 2]}}
    assert "must be FLAT" in schema.data_errors(DS, nested)[0]
    combined = {"sets": {"s": {"meaning": "site", "source": "public", "reducible": None}},
                "params": {"coords": {"meaning": "lat/lon", "index": ["s"], "type": "number", "source": "public",
                                      "unit": "deg_lat, deg_lon", "min": None}}}
    errors = schema.declaration_errors(combined)
    assert len(errors) == 1 and "two params" in errors[0]                    # None fields are simply "not given"


def test_variable_keys_let_anyone_read_a_solution_as_data_ids():
    data = {"sites": ["A", "B"], "hours": [8]}
    raw = {"variable_names": ["a", "b", "slack"], "variable_keys": [{"sites": "A"}, {"sites": "B"}, {}]}
    assert schema.variable_key_errors(DS, data, raw) == []
    assert schema.decisions(raw, [0, 1, 1]) == [{"sites": "B"}]
    for keys, fragment in [(None, "must return variable_keys"), ([{"sites": "Z"}, {}, {}], "not in set 'sites'"),
                           ([{"zones": "A"}, {}, {}], "not a declared set"), ([{}, {}, {}], "all empty"),
                           ([{"sites": "A"}, {"sites": "A"}, {}], "same variable_keys")]:
        bad = {**raw, "variable_keys": keys}
        assert fragment in schema.variable_key_errors(DS, data, bad)[0], fragment
    fixed_start = {"variable_names": ["b", "slack"], "variable_keys": [{"sites": "B"}, {}]}   # site A left out
    assert "no decision variable: ['A']" in schema.variable_key_errors(DS, data, fixed_start)[0]


def test_constant_measured_values_are_rejected_on_real_data_only():
    ds = {"sets": {"areas": {"meaning": "area", "source": "public"}},
          "params": {"pop": {"meaning": "people", "index": ["areas"], "type": "int", "source": "public"}}}
    data = {"areas": [f"A{i}" for i in range(6)], "pop": [1] * 6}
    assert any("never constant" in e for e in schema.data_errors(ds, data))
    assert schema.data_errors(ds, data, check_values=False) == []          # synthetic examples may be uniform


def test_plan_mode_runs_the_agents_plan_through_the_tools(monkeypatch):
    import json as _json
    from qubo_auto.annealing import adapter
    monkeypatch.setattr(adapter, "ADAPTER_MODE", "plan")
    ns = load_module(MODEL.replace("%EXTRA%", ""))
    tables = {"sites": pd.DataFrame({"name": ["A", "B", "C", "X"], "kind": ["shop", "shop", "shop", "other"],
                                     "pop": [5, 7, 9, 1], "lat": [25.03, 25.04, 25.05, 25.06],
                                     "lon": [121.50, 121.52, 121.55, 121.56]})}
    meta = {k: {"source_urls": ["https://example.org/sites.csv"]} for k in ["sites", "name", "pop", "lat", "lon", "dist"]}
    meta["lat"]["unit"], meta["lon"]["unit"], meta["dist"]["unit"], meta["k"] = "deg_lat", "deg_lon", "m", {}
    plan = {"table": "sites", "steps": [{"op": "exclude", "column": "kind", "values": ["other"]}],
            "output": {"sets": {"sites": "name"}, "params": {"name": "name", "pop": "pop", "lat": "lat", "lon": "lon"},
                       "settings": {"k": 2}, "derived": {"dist": {"tool": "haversine_matrix", "lat": "lat", "lon": "lon"}}},
            "notes": "shops only", "metadata": meta}
    steps = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            steps.append(step_name)
            return _json.dumps(plan)

    code, data, errors, report = adapter.build_adapter(ns, tables, "open 2 sites", Fake(), "m",
                                                       sources=[{"title": "sites", "url": "https://example.org/sites.csv"}])
    assert errors == [] and data["sites"] == ["A", "B", "C"] and len(data["dist"]) == 3
    assert steps == ["adapter_plan"] and report["adapter_mode"] == "plan"
