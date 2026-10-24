import pandas as pd
import pytest

from qubo_auto.annealing import dataplan


def village_table():
    return pd.DataFrame({"年": [115] * 6, "區域別": ["總計", "大安區", "龍坡里", "龍泉里", "萬華區", "中山里"],
                         "人口": ["1,000", "60", "30", "30", "40", "40"]})


def test_plan_runs_tested_tools_step_by_step_and_shows_each_result():
    plan = {"table": "pop", "steps": [
        {"op": "attach_parent", "column": "區域別", "new": "區"},
        {"op": "keep_level", "column": "區域別", "levels": "里"},
        {"op": "to_number", "columns": ["人口"]}],
        "output": {"sets": {"villages": "區域別"}, "params": {"pop": "人口", "district": "區"}, "settings": {"k": 2}}}
    data, seen, log = dataplan.execute(plan, {"pop": village_table()})
    assert data == {"villages": ["龍坡里", "龍泉里", "中山里"], "pop": [30, 30, 40],
                    "district": ["大安區", "大安區", "萬華區"], "k": 2}
    assert len(seen) == 4 and "rows 6 -> 3" in seen[2]                           # every step is observed


def test_plan_errors_name_the_step_and_what_exists():
    bad = {"table": "pop", "steps": [{"op": "keep", "column": "區域別", "values": ["信義區"]}], "output": {}}
    with pytest.raises(ValueError, match=r"step 0 \(keep\).*values there"):
        dataplan.execute(bad, {"pop": village_table()})
    with pytest.raises(ValueError, match="no column"):
        dataplan.execute({"table": "pop", "steps": [{"op": "exclude", "column": "類別", "values": ["x"]}]},
                         {"pop": village_table()})
    with pytest.raises(KeyError, match="is not one of"):
        dataplan.execute({"table": "人口", "steps": []}, {"pop": village_table()})


def test_wrong_step_order_is_reported_not_silently_wrong():
    order = {"table": "pop", "steps": [{"op": "keep_level", "column": "區域別", "levels": "里"},
                                       {"op": "attach_parent", "column": "區域別", "new": "區"}], "output": {}}
    with pytest.raises(ValueError, match="BEFORE keep_level"):
        dataplan.execute(order, {"pop": village_table()})


def test_repeated_ids_and_custom_steps():
    t = pd.DataFrame({"name": ["A", "A", "B"], "lat": [25.0, 25.5, 25.1], "lon": [121.5, 121.5, 121.6]})
    with pytest.raises(ValueError, match="ids repeat"):
        dataplan.assemble(t, {"sets": {"sites": "name"}})
    plan = {"table": "t", "steps": [{"op": "custom", "expr": "df[df.lat < 25.4]", "why": "drop the far copy"}],
            "output": {"sets": {"sites": "name"}, "params": {"lat": "lat", "lon": "lon"},
                       "derived": {"d": {"tool": "haversine_matrix", "lat": "lat", "lon": "lon"}}}}
    data, _, log = dataplan.execute(plan, {"t": t})
    assert data["sites"] == ["A", "B"] and len(data["d"]) == 2 and any("REVIEW" in f for f in log.flags)


def test_plan_becomes_an_ordinary_adapter_module_and_failures_show_the_steps():
    answer = {"table": "pop", "steps": [{"op": "keep_level", "column": "區域別", "levels": "里"}],
              "output": {"sets": {"villages": "區域別"}}, "notes": "villages only", "metadata": {}}
    ns = {}
    exec(dataplan.plan_to_code(answer), ns)
    from qubo_auto.annealing import prep
    assert ns["load_data"]({"pop": village_table()}, prep.PrepLog())["villages"] == ["龍坡里", "龍泉里", "中山里"]
    gap = {}
    exec(dataplan.plan_to_code({"gap": "no station list"}), gap)
    with pytest.raises(ValueError, match="DATA GAP: no station list"):
        gap["load_data"]({}, None)
    with pytest.raises(ValueError, match="RESULTS OF THE STEPS THAT RAN"):
        dataplan.execute({"table": "pop", "steps": [{"op": "keep_level", "column": "區域別", "levels": "里"},
                                                    {"op": "keep", "column": "區域別", "values": ["信義里"]}]},
                         {"pop": village_table()})


def test_each_table_is_prepared_before_the_join():
    pop = pd.DataFrame({"區域別": ["總計", "大安區", "龍坡里", "龍泉里"], "人口": [999, 60, 30, 30]})
    geo = pd.DataFrame({"TNAME": ["大安區", "大安區"], "VNAME": ["龍坡里", "龍泉里"], "lat": [25.02, 25.03],
                        "lon": [121.53, 121.54]})
    plan = {"prepare": {"pop": [{"op": "attach_parent", "column": "區域別", "new": "區"},
                                {"op": "keep_level", "column": "區域別", "levels": "里"}]},
            "table": "geo",
            "steps": [{"op": "join", "table": "pop", "left_on": ["TNAME", "VNAME"], "right_on": ["區", "區域別"]}],
            "output": {"sets": {"v": "VNAME"}, "params": {"pop": "人口", "lat": "lat"}}}
    data, seen, _ = dataplan.execute(plan, {"pop": pop, "geo": geo})
    assert data["v"] == ["龍坡里", "龍泉里"] and data["pop"] == [30, 30]
    assert seen[0].startswith("prepare pop") and any(x.startswith("main step 0 join") for x in seen)


def test_a_value_under_params_is_taken_as_a_setting():
    data = dataplan.assemble(village_table(), {"sets": {"v": "區域別"}, "params": {"k": 3, "pop": "人口"}})
    assert data["k"] == 3 and len(data["pop"]) == 6


def test_reduce_points_step_shrinks_a_reducible_set():
    t = pd.DataFrame({"name": [f"P{i}" for i in range(40)], "lat": [25.0 + i * 1e-3 for i in range(40)],
                      "lon": [121.5] * 40})
    plan = {"table": "t", "steps": [{"op": "reduce_points", "k": 10, "lat": "lat", "lon": "lon"}],
            "output": {"sets": {"sites": "name"}}}
    data, _, _ = dataplan.execute(plan, {"t": t})
    assert len(data["sites"]) == 10


def test_distance_to_one_point_as_a_step_and_as_a_derived_param():
    # a model may declare "distance of each station to the reference point"; a plan must be able to produce it
    t = {"t": pd.DataFrame({"id": ["a", "b"], "lat": [-37.818, -37.82], "lon": [144.967, 144.97]})}
    c = [-37.8183, 144.9671]
    plan = {"table": "t", "steps": [{"op": "distance_to", "lat": "lat", "lon": "lon", "center": c, "new": "d"}],
            "output": {"sets": {"s": "id"}, "params": {"lat": "lat", "lon": "lon", "dist": "d"},
                       "derived": {"dist2": {"tool": "distance_to", "lat": "lat", "lon": "lon", "center": c}}}}
    data, _, _ = dataplan.execute(plan, t)
    assert data["dist"] == data["dist2"] and data["dist"][0] < 50 < data["dist"][1] < 400


def test_model_defined_sets_are_sized_from_the_data_and_never_copy_an_entity_column():
    t = {"t": pd.DataFrame({"id": ["10", "25", "4"], "lat": [0.0, 0.1, 0.2], "lon": [0.0, 0.1, 0.2]})}
    out = {"sets": {"sites": "id"}, "settings": {"positions": {"index_of": "sites"}}, "params": {"lat": "lat"}}
    data, _, _ = dataplan.execute({"table": "t", "output": out}, t)
    assert data["positions"] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="index_of"):         # positions filled with the station ids (2026-10-21)
        dataplan.execute({"table": "t", "output": {"sets": {"sites": "id", "positions": "id"}}}, t)
    from qubo_auto.annealing import dataagent
    ds = {"sets": {"sites": {"source": "public"}, "positions": {"source": "mock"}}, "params": {}}
    msg = dataagent.dry_run({"plan": {"table": "t", "output": {"sets": {"sites": "id", "positions": "lat"}}}}, t, ds)
    assert msg and "defined by the model" in msg


def test_area_code_rule_applies_only_when_the_code_column_is_used():
    # NYC budgeting (2026-10-22): the plan kept the council district the user named and never touched 'Borough
    # Code', yet was forced into in_area on it
    from qubo_auto.annealing import adapter
    from qubo_auto.annealing.prep.situations import Situation
    sit = Situation("t", "Borough Code", "areas identified only by codes", ("in_area",), "4 codes")
    rec = adapter._Recorder()
    untouched = 'PLAN = {"steps": [{"op": "keep", "column": "Council District", "values": ["3"]}]}'
    assert adapter.enforcement_errors([sit], rec, {"t"}, untouched) == []
    filtered = 'PLAN = {"steps": [{"op": "keep", "column": "Borough Code", "values": ["1"]}]}'
    assert adapter.enforcement_errors([sit], rec, {"t"}, filtered)


def test_named_portals_are_recognised_and_substitutes_are_detected():
    from qubo_auto.annealing.opendata import requested_domains, from_requested
    assert requested_domains("請用新加坡政府開放資料（data.gov.sg）的 Hawker Centres 資料。") == ["data.gov.sg"]
    assert requested_domains("Vélib' 站點（opendata.paris.fr）") == ["opendata.paris.fr"]
    assert requested_domains("請用 NYC Open Data 的資料，聯絡 a@b.com") == []
    assert from_requested({"url": "https://api-open.data.gov.sg/v1/public/api/datasets/d_1/poll-download"}, ["data.gov.sg"])
    assert not from_requested({"source": "OpenStreetMap", "url": ""}, ["data.gov.sg"])


def test_dataset_metadata_is_not_taken_for_data():
    import json as _json
    from qubo_auto.annealing.opendata import is_metadata_json, METADATA_URL
    cols = [{"id": i, "name": n, "dataTypeName": "text", "fieldName": n.lower()} for i, n in enumerate(["A", "B", "C", "D"])]
    rows = [{"vote_year": "2017", "council_district": "3", "votes": "1405"}] * 4
    assert is_metadata_json(_json.dumps(cols).encode())
    assert not is_metadata_json(_json.dumps(rows).encode())
    assert METADATA_URL.search("/api/views/wwhr-5ven/columns.json") and not METADATA_URL.search("/api/views/wwhr-5ven/rows.json")
