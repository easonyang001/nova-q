"""Failures seen in test set 2 (2026-10-24), each fixed generically."""
import json

import pandas as pd

from qubo_auto.annealing import clarify, prep
from qubo_auto.annealing.adapter import PORTAL_ROW_ID
from qubo_auto.annealing.contracts import check_requirements
from qubo_auto.annealing.modeler import load_module, validate

GROUPS = '''import numpy as np
MODEL_CHOICE = "one-hot assignment of sites to 5 teams; same-team pairs cost their distance"
DATA_SCHEMA = {"sets": {"sites": {"meaning": "places", "source": "private"},
                        "teams": {"meaning": "teams", "source": "model", "size_of": 5}},
               "params": {"dist": {"meaning": "distance", "index": ["sites", "sites"], "type": "number",
                                   "source": "private", "unit": "m"}}}
REQUIREMENT_MAP = {"single_assignment": ["single_assignment_*"]}
def make_example_data(size, seed):
    size = (size // 5) * 5                     # whole teams only: 4 sites -> none
    sites = [f"S{i}" for i in range(size)]
    return {"sites": sites, "dist": [[float(abs(i - j)) for j in range(size)] for i in range(size)]}
def build_ir(data):
    sites, teams = data["sites"], [str(g) for g in range(5)]
    idx = {(s, g): k for k, (s, g) in enumerate((s, g) for s in sites for g in teams)}
    quad = [[idx[a, g], idx[b, g], data["dist"][i][j] + 1.0] for g in teams for i, a in enumerate(sites)
            for j, b in enumerate(sites) if i < j]
    cons = [{"name": f"single_assignment_{s}", "kind": "linear", "coeffs": {idx[s, g]: 1 for g in teams},
             "sense": "eq", "rhs": 1} for s in sites]
    return {"name": "groups", "variable_names": [f"x_{s}_{g}" for s, g in idx], "sense": "minimize",
            "variable_keys": [{"sites": s, "teams": g} for s, g in idx], "linear": {}, "quadratic": quad,
            "constraints": cons}
def describe_solution(data, x):
    return {"n": sum(x)}
'''


def test_merge_same_place_by_name_only_uses_mean_coordinates():
    df = pd.DataFrame({"name": ["Fakultät", "Fakultät", "Museum"], "lat": [47.550, 47.552, 47.56],
                       "lon": [7.580, 7.580, 7.59]})
    assert len(prep.merge_same_place(df, "name", "lat", "lon")) == 3          # 220 m apart: kept by default
    out = prep.merge_same_place(df, "name", "lat", "lon", max_m=None)
    assert len(out) == 2 and abs(out.loc[out["name"] == "Fakultät", "lat"].item() - 47.551) < 1e-9


def test_portal_row_ids_are_recognised():
    assert PORTAL_ROW_ID.match("row-3j4m~khmq.utc9") and PORTAL_ROW_ID.match("row-xr6z-p99n~csiy")
    assert not PORTAL_ROW_ID.match("109") and not PORTAL_ROW_ID.match("Park A")


def test_requirement_mapping_is_checked_on_the_full_example_not_an_empty_shrunk_one():
    ns = load_module(GROUPS)
    errors, report = validate(ns)
    assert report["_full_data"]["sites"], "the full-size example has sites"
    assert report["_data"]["sites"], "shrinking never ends on an empty instance"
    spec = {"constraints": [{"name": "single_assignment", "strength": "hard"}]}
    assert check_requirements(spec, ns, ns["build_ir"](report["_full_data"])) == []


def test_draft_contradicting_the_user_is_redrafted(monkeypatch):
    drafts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            if step_name == "clarify_check":
                start = json.loads(prompt[prompt.index("DRAFT:") + 6:prompt.index("Return JSON")])
                bad = any("Connolly" in a for a in start["assumptions"])
                return json.dumps({"contradictions": [{"item": "start at Connolly", "user_said": "any start"}]
                                   if bad else []})
            drafts.append("YOUR DRAFT CONTRADICTS" in prompt)
            assumption = "tour can start at any station" if drafts[-1] else "tour starts at Connolly station"
            return json.dumps({"draft": {"problem_summary": "tour", "assumptions": [assumption]},
                               "questions": [], "coverage": {}, "ready_to_confirm": False, "web_queries": []})

    s = clarify.new_session("tour of stations near Connolly", "sa")
    s.rounds.append({"questions": [], "answers": "any station can be the start"})
    s = clarify.step(s, Fake(), "m", web=False)
    assert drafts == [False, True] and s.draft.assumptions == ["tour can start at any station"]


def test_data_check_compares_sets_read_from_tables_whatever_their_source_label():
    from qubo_auto.annealing.datacheck import compare
    ds = {"sets": {"sites": {"meaning": "places", "source": "private"},
                   "teams": {"meaning": "teams", "source": "model", "size_of": 5}}, "params": {}}
    data = {"sites": [str(i) for i in range(38)], "teams": [str(i) for i in range(5)]}
    assert compare({"counts": {"sites": 35}}, data, ds)              # open data labelled "private": still checked
    assert not compare({"counts": {"teams": 4}}, data, ds)           # model sets are not the checker's business


def test_repeated_names_are_questioned_but_codes_with_numbers_are_not():
    from qubo_auto.annealing.adapter import repeated_names
    ds = {"sets": {"sites": {"meaning": "places", "source": "private"}},
          "params": {"site_name": {"meaning": "name", "index": ["sites"], "type": "text", "source": "private"}}}
    assert repeated_names(ds, {"sites": ["Fakultät", "Fakultät_1", "Museum"]}) == ["Fakultät"]
    assert repeated_names(ds, {"sites": ["160", "161"], "site_name": ["Fakultät", "Fakultät"]}) == ["Fakultät"]
    assert repeated_names(ds, {"sites": ["S_1", "S_2", "S_3"]}) == []


def test_geojson_inside_a_zip_is_read(tmp_path):
    import io
    import zipfile
    from qubo_auto.annealing.datakit import load_tables
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"Standort": "Museum"}, "geometry": {"type": "Point", "coordinates": [7.59, 47.56]}}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("defibrillatoren.geojson", json.dumps(fc))
    path = tmp_path / "Defibrillatoren_geojson.geojson"
    path.write_bytes(buf.getvalue())
    (name, df), = load_tables([path]).items()
    assert name.endswith("#defibrillatoren") and df["Standort"].tolist() == ["Museum"]
