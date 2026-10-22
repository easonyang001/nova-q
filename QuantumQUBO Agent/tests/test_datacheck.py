"""Data-level requirement check by double computation (counts and totals, never ids)."""
import pandas as pd

from qubo_auto.annealing import datacheck
from qubo_auto.annealing.adapter import build_adapter
from qubo_auto.annealing.modeler import load_module

DS = {"sets": {"places": {"meaning": "AED place", "source": "public"},
               "teams": {"meaning": "team", "source": "user"},
               "areas": {"meaning": "area", "source": "public", "reducible": {"method": "reduce_points", "weight": "pop"}}},
      "params": {"pop": {"meaning": "people", "index": ["areas"], "type": "int", "source": "public"}}}


def test_compare_counts_and_totals_and_skips_reducible_counts():
    data = {"places": ["S0", "S1", "S2"], "teams": [0, 1], "areas": ["a"], "pop": [100]}
    issues = datacheck.compare({"counts": {"places": 2, "areas": 5, "teams": 9}, "totals": {"pop": 100}}, data, DS)
    assert len(issues) == 1 and "'places' has 3 entities" in issues[0]          # areas reducible, teams not public
    assert "sums to 100" in datacheck.compare({"totals": {"pop": 150}}, data, DS)[0]
    assert datacheck.compare({"counts": {"places": 3}, "totals": {"pop": 100.2}}, data, DS) == []


MODEL = '''
DATA_SCHEMA = {"sets": {"places": {"meaning": "AED place", "source": "public"}},
  "params": {"lat": {"meaning": "lat", "index": ["places"], "type": "number", "source": "public", "unit": "deg_lat"}}}
REQUIREMENT_MAP = {}
def make_example_data(size, seed):
    return {"places": [f"P{i}" for i in range(size)], "lat": [25.0 + i / 100 for i in range(size)]}
def build_ir(data):
    n = len(data["places"])
    return {"name": "m", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "variable_keys": [{"places": p} for p in data["places"]], "linear": {str(i): 1 for i in range(n)},
            "quadratic": [], "constant": 0, "constraints": []}
def describe_solution(data, x):
    return {"chosen": [p for p, b in zip(data["places"], x) if b], "lat": data["lat"]}
'''


def test_adapter_gets_the_disagreement_as_feedback_and_repairs():
    aed = pd.DataFrame({"name": ["Hotel A", "Hotel A", "School B", "Mall C"], "category": ["hotel", "hotel", "school", "shop"],
                        "lat": [25.01, 25.01, 25.02, 25.03]})
    meta = "ADAPTER_METADATA = {'places': {}, 'lat': {'unit': 'deg_lat'}}\n"
    wrong = meta + ("def load_data(tables, log):\n    t = tables['aed']\n"
             "    return {'places': [f'S{i}' for i in range(len(t))], 'lat': t['lat'].tolist()}\n")
    right = meta + ("def load_data(tables, log):\n    t = tables['aed']\n    t = t[t['category'] != 'school']\n"
             "    t = prep.dedupe(t, ['name'], log=log)\n"
             "    return {'places': t['name'].tolist(), 'lat': t['lat'].tolist()}\n")
    checker = ("def expected(tables):\n    t = tables['aed']\n"
               "    return {'counts': {'places': int(t[t['category'] != 'school']['name'].nunique())}, 'totals': {}}\n")
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append((step_name, prompt))
            if step_name == "data_checker":
                return f"```python\n{checker}```"
            n = sum(1 for s, _ in prompts if s == "adapter")
            return f"```python\n{wrong if n == 1 else right}```"

    code, data, errors, report = build_adapter(load_module(MODEL), {"aed": aed}, "AED places, no schools, one per place",
                                               Fake(), "m", data_check=True)
    adapter_prompts = [p for s, p in prompts if s == "adapter"]
    assert errors == [] and data["places"] == ["Hotel A", "Mall C"]
    assert "DATA CHECK: set 'places' has 4 entities" in adapter_prompts[1] and "expects 2" in adapter_prompts[1]
    assert report["data_check"]["issues"] == []
    assert "Hotel A" not in adapter_prompts[0].split("THE USER'S TABLES")[0]            # checker sees no adapter code
    checker_prompt = next(p for s, p in prompts if s == "data_checker")
    assert "load_data" not in checker_prompt


def test_infeasible_with_open_data_check_disagreement_goes_back_to_the_data(tmp_path, monkeypatch):
    from qubo_auto.annealing import repair as R
    calls = []
    monkeypatch.setattr(R, "prepare_real_data", lambda code, *a, **kw: (calls.append(("data", kw.get("notes", ""))) or
                                                                       ({"places": ["P0"]}, [], {})))
    monkeypatch.setattr(R.modeler, "build_model", lambda *a, **kw: calls.append(("model", "")) or ("c", {}, [], {}))
    fix = R.make_repair("{}", "[]", "spec", None, "m", "", tmp_path, {})
    bundle = {"code": "c", "data": {}, "metadata": {"report": {"data_check": {"issues": ["set 'places' has 320 entities"]}}}}
    out = fix({"stage": "diagnose", "evidence": ["infeasible"]}, bundle)
    assert [c[0] for c in calls] == ["data"] and "320 entities" in calls[0][1] and out["status"] == "repaired"
    calls.clear()
    fix({"stage": "diagnose", "evidence": ["infeasible"]}, {"code": "c", "data": {}, "metadata": {}})
    assert [c[0] for c in calls] == ["model", "data"]                     # no data evidence: the model first


def test_checker_must_use_the_declared_names():
    import pandas as pd
    ds = {"sets": {"sites": {"meaning": "station", "source": "public"}},
          "params": {"lat": {"meaning": "", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lat"}}}
    answers = ["def expected(tables):\n    return {'counts': {'stations': 18}, 'totals': {}}\n",
               "def expected(tables):\n    return {'counts': {'sites': 18}, 'totals': {}}\n"]
    prompts = []

    class Fake:
        def call(self, model, prompt, **kw):
            prompts.append(prompt)
            return f"```python\n{answers[len(prompts) - 1]}```"
    code, expected, errors = datacheck.build("spec", ds, {"t": pd.DataFrame({"a": [1]})}, Fake(), "m")
    assert expected == {"counts": {"sites": 18}, "totals": {}} and "unknown names ['stations']" in prompts[1]
    assert "set 'sites' has 19 entities" in datacheck.compare(expected, {"sites": list(range(19))}, ds)[0]


def test_checker_prompt_carries_the_detected_situations():
    import pandas as pd
    ds = {"sets": {"areas": {"meaning": "village", "source": "public"}},
          "params": {"pop": {"meaning": "people", "index": ["areas"], "type": "int", "source": "public"}}}
    seen = []

    class Fake:
        def call(self, model, prompt, **kw):
            seen.append(prompt)
            return "```python\ndef expected(tables):\n    return {'counts': {}, 'totals': {}}\n```"
    datacheck.build("spec", ds, {"t": pd.DataFrame({"a": [1]})}, Fake(), "m",
                    situations="- table 't', column '性別': rows split by category -> must call prep.keep_category")
    assert "rows split by category" in seen[0] and "must not be summed together" in seen[0]
