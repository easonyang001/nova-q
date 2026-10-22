import json

import pandas as pd

from qubo_auto.annealing import dataagent


def feed():
    return pd.DataFrame({"sno": [f"S{i}" for i in range(30)], "sna": [f"站{i}" for i in range(30)],
                         "mday": [f"2026-10-12 10:{i:02d}:00" for i in range(30)],
                         "lat": [25.03 + i * 1e-4 for i in range(30)], "lon": [121.54] * 30})


def test_read_only_tools_show_what_a_column_really_is():
    t = {"feed": feed(), "other": pd.DataFrame({"code": ["Ｓ0 ", "S1", "X"]})}
    values = json.loads(dataagent.explore(t, {"tool": "values", "table": "feed", "column": "mday"}))
    assert values["distinct_per_row"] == 1.0                       # an update time per row, not stacked periods
    keys = json.loads(dataagent.explore(t, {"tool": "keys", "left": ["feed", "sno"], "right": ["other", "code"]}))
    assert keys["matching_distinct"] == 2                           # full-width Ｓ0 and S1 match after normalising
    assert "no table" in dataagent.explore(t, {"tool": "rows", "table": "missing"})
    rows = json.loads(dataagent.explore(t, {"tool": "rows", "table": "feed", "column": "sna", "contains": "站2"}))
    assert rows["matching"] == 1 + 10                               # 站2 and 站20..站29


def test_agent_looks_then_plans_and_learns_from_a_failed_dry_run():
    answers = iter([
        {"table": "feed", "steps": [], "output": {"sets": {"stations": "sno"}}},          # a bare plan, too early
        {"explore": {"tool": "values", "table": "feed", "column": "mday"}},
        {"explore": {"tool": "rows", "table": "feed"}},
        {"plan": {"table": "feed", "steps": [{"op": "keep", "column": "mday", "values": ["2027"]}],
                  "output": {"sets": {"stations": "sno"}}}},
        {"plan": {"table": "feed", "steps": [], "output": {"sets": {"stations": "sno"}, "params": {"lat": "lat"}}},
         "notes": "all stations; mday is an update time", "metadata": {}},
    ])
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append(prompt)
            return json.dumps(next(answers))

    history = []
    raw = dataagent.plan("BASE", {"feed": feed()}, Fake(), "m", history)
    final = json.loads(raw)
    assert final["output"]["sets"]["stations"] == "sno" and final["notes"].startswith("all stations")
    assert "first LOOK at the data" in prompts[1]                   # a plan before looking is sent back
    assert "distinct_per_row" in prompts[2]                          # the tool result is shown on the next turn
    assert "FAILED" in prompts[4] and "values there" in prompts[4]   # so is the failed dry run
    assert len(history) == 5


def test_dry_run_checks_the_output_format_and_refuses_custom_steps():
    ds = {"sets": {"stations": {"meaning": "station", "source": "public"}},
          "params": {"lat": {"meaning": "lat", "index": ["stations"], "type": "number", "source": "public", "unit": "deg_lat"}}}
    short = {"plan": {"table": "feed", "steps": [], "output": {"sets": {"stations": "sno"}}}}
    assert "missing ['lat']" in dataagent.dry_run(short, {"feed": feed()}, ds)
    full = {"plan": {"table": "feed", "steps": [], "output": {"sets": {"stations": "sno"}, "params": {"lat": "lat"}}}}
    assert dataagent.dry_run(full, {"feed": feed()}, ds) is None
    custom = {"plan": {"table": "feed", "steps": [{"op": "custom", "expr": "df"}], "output": {}}}
    assert "custom steps are not accepted" in dataagent.dry_run(custom, {"feed": feed()}, ds)
