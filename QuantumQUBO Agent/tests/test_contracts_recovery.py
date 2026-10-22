import copy
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from qubo_auto.annealing.contracts import check_data, check_requirements, changed_semantics
from qubo_auto.annealing.artifacts import ArtifactStore
from qubo_auto.annealing.adapter import check_output
from qubo_auto.annealing import acquisition, workflow
from qubo_auto.annealing.baselines import make_policy, DefaultPolicy
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.datakit import load_tables
from qubo_auto.annealing.progress import load_progress
from qubo_auto.annealing.provenance import source_record
from test_workflow import CODE, DATA, CFG, ir


def test_wrong_units_period_and_unknown_metadata_are_distinct():
    contract = {"demand": {"unit": "kg", "period": "daily", "min": 0, "source_kind": "public"}}
    errors, review = check_data({"demand": [2]}, contract, {"demand": {"unit": "g", "period": "monthly", "source_urls": ["fabricated"]}}, [{"url": "actual"}])
    # wrong unit and period are errors; a cited URL missing from the download record goes to human review
    assert len(errors) == 2 and any("來源網址" in r for r in review)
    errors, review = check_data({"demand": [2]}, contract)
    assert errors == [] and len(review) == 3


def test_nan_missing_and_out_of_range_data_rejected():
    for values in [[float("nan")], [None], [-1], [21]]:
        errors, _ = check_data({"v": values}, {"v": {"min": 0, "max": 20}})
        assert errors


def test_confirmed_user_setting_cannot_be_changed_in_data():
    errors, _ = check_data({"capacity": 200}, {"capacity": {"source_kind": "user", "value": 20}})
    assert errors and "confirmed user setting" in errors[0]


def test_data_repair_cannot_relax_a_user_bound(tmp_path):
    code = 'DATA_CONTRACT = {"values": {}, "capacity": {"source_kind": "user"}}\n' + CODE
    store = ArtifactStore(tmp_path)
    original = store.commit(code, {**DATA, "capacity": -1}, "user setting")
    result = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG,
        repair=lambda *a: {"status": "repaired", "code": code, "data": DATA}, log=lambda _: None)
    assert store.current()["version"] == original
    assert "使用者設定" in result["delivery"]["recovery_message"]


def test_requested_publisher_archive_is_discovered_and_preferred(monkeypatch):
    from qubo_auto.annealing import opendata, websearch
    page = "https://publisher.test/wine"
    html = '<h1>Wine Quality</h1><meta name="description" content="Red and white wine physicochemical records"><a href="/files/wine.zip">Download</a><a href="https://creativecommons.org/licenses/by/4.0/">License</a>'
    response = SimpleNamespace(text=html, content=html.encode(), url=page, headers={"content-type": "text/html"}, raise_for_status=lambda: None)
    monkeypatch.setattr(opendata, "_get", lambda *a, **kw: response)
    monkeypatch.setattr(websearch, "lookup", lambda *a, **kw: {"results": [{"url": page}]})
    class Client:
        def call(self, **kw):
            if kw["step_name"] == "scout":
                return json.dumps({"web_queries": ["wine"], "govtw_queries": [], "osm": []})
            return json.dumps({"ranking": []})  # title-only model refuses every source
    found = opendata.scout(json.dumps({"summary": "source " + page + "\nnext"}), {}, Client(), "m")
    assert len(found["candidates"]) == 1
    candidate = opendata.rank_candidates("spec", {}, found["candidates"], Client(), "m")[0]
    assert candidate["requested_source"] and candidate["readable"]
    assert candidate["url"] == "https://publisher.test/files/wine.zip"
    assert "physicochemical" in candidate["detail"]
    assert candidate["license_from_page"] == "CC BY 4.0"


def test_missing_metadata_is_flagged_for_review_not_retried():
    import pandas as pd
    from qubo_auto.annealing import adapter, modeler
    ns = modeler.load_module('DATA_CONTRACT = {"values": {}, "capacity": {}}\n' + CODE)
    body = 'def load_data(tables, log):\n    return {"values": tables["actual"]["values"].tolist(), "capacity": 1}\n'
    class Client:
        calls = 0
        def call(self, **kw):
            self.calls += 1
            return body if self.calls == 1 else 'ADAPTER_METADATA = {"values": {}, "capacity": {}}\n' + body
    client = Client()
    _, data, errors, report = adapter.build_adapter(ns, {"actual": pd.DataFrame({"values": [3, 1]})}, "spec", client, "m")
    # paperwork does not cost an attempt: the data are kept and the missing declarations go to the human review
    assert client.calls == 1 and not errors and data == DATA
    assert any("沒有聲明" in r and "capacity" in r for r in report["review_required"])


def test_requirement_family_mapping_scales_with_instance():
    spec = {"constraints": [{"name": "assign", "strength": "hard"}]}
    for size in (6, 24):
        assert not check_requirements(spec, {"REQUIREMENT_MAP": {"assign": ["one_group_*"]}},
                                      {"constraints": [{"name": f"one_group_{i}"} for i in range(size)]})
    assert check_requirements(spec, {"REQUIREMENT_MAP": {"assign": ["missing_*"]}}, {"constraints": [{"name": "one_group_0"}]})
    assert check_requirements(spec, {"REQUIREMENT_MAP": {"assign": ["*"]}}, {"constraints": [{"name": "one_group_0"}]})


def test_gml_archive_preserves_isolated_nodes_and_ignores_readme(tmp_path):
    import networkx as nx
    graph = nx.Graph([(1, 2)])
    graph.add_node(3)
    file = tmp_path / "network.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("graph.gml", "\n".join(nx.generate_gml(graph)))
        archive.writestr("readme.txt", "Not a table\nfield,readme,text\n")
    tables = load_tables([file])
    assert len(tables) == 1
    table = next(iter(tables.values()))
    assert len(table) == 3 and table.filter(like="adj_").to_numpy().sum() == 2


def test_explicit_sampling_retains_ids_and_records_seed():
    import pandas as pd
    from qubo_auto.annealing import prep
    frame = pd.DataFrame({"x": range(10)})
    log = prep.PrepLog()
    sampled = prep.sample_rows(frame, 4, seed=0, log=log)
    assert sampled.equals(frame.sample(n=4, random_state=0))
    assert "seed=0" in log.markdown()


def test_partial_requirement_mapping_is_rejected():
    spec = {"constraints": [{"name": "budget", "strength": "hard"}, {"name": "capacity", "strength": "hard"}]}
    raw = {"constraints": [{"name": "budget_ir"}]}
    assert check_requirements(spec, {"REQUIREMENT_MAP": {"budget": ["budget_ir"]}}, raw)
    assert check_requirements(spec, {"REQUIREMENT_MAP": {"budget": ["budget_ir"], "capacity": ["missing"]}}, raw)
    raw["constraints"].append({"name": "capacity_ir"})
    assert not check_requirements(spec, {"REQUIREMENT_MAP": {"budget": ["budget_ir"], "capacity": ["capacity_ir"]}}, raw)


def test_same_name_bound_change_detected_and_not_committed(tmp_path):
    before = ir()
    after = before.model_copy(deep=True)
    after.constraints[0].rhs = 200
    assert changed_semantics(before, after)
    # An unknown feasibility outcome routes to diagnosis; repair keeps the same
    # name but changes the capacity formula on the original data.
    store = ArtifactStore(tmp_path)
    first = store.commit(CODE, DATA, "original")
    modified = CODE.replace('"rhs": data["capacity"]', '"rhs": 200')
    from unittest.mock import patch
    inspection = {"feasibility": "infeasible", "issues": ["conflict"], "baseline_solution": None,
                  "baseline_objective": None, "proven_optimal": False}
    with patch.object(workflow, "inspect_problem", side_effect=[inspection, {**inspection, "feasibility": "feasible"}]):
        result = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG,
            repair=lambda *a: {"status": "repaired", "code": modified, "data": DATA}, log=lambda _: None)
    assert store.current()["version"] == first
    assert "重新確認" in result["delivery"]["recovery_message"]


def test_public_data_with_unresolved_contract_needs_confirmation(tmp_path):
    store = ArtifactStore(tmp_path)
    store.commit(CODE, DATA, "real", {"source": "real"})
    result = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG, log=lambda _: None)
    assert result["total_solver_calls"] == 0
    assert result["diagnoses"][0]["stage"] == "data"
    assert load_progress(tmp_path)["status"] == "needs_input"


def test_real_adapter_cache_reuses_exact_inputs_and_invalidates_changed_file(tmp_path, monkeypatch):
    file = tmp_path / "actual.csv"
    file.write_text("values\n3\n1\n")
    calls = []
    def adapter(*a, **kw):
        calls.append(True)
        return "adapter", DATA, [], {}
    monkeypatch.setattr(acquisition, "build_adapter", adapter)
    for _ in range(2):
        acquisition.prepare_real_data(CODE, "spec", None, "m", tmp_path, log=lambda _: None)
    assert len(calls) == 1
    file.write_text("values\n4\n1\n")
    acquisition.prepare_real_data(CODE, "spec", None, "m", tmp_path, log=lambda _: None)
    assert len(calls) == 2


def test_source_licence_is_unknown_unless_observed_in_publisher_page(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("a\n1\n")
    record = source_record({"source": "政府資料開放平台", "title": "x"}, file)
    assert record["license"] == "待確認" and len(record["sha256"]) == 64


def test_semicolon_and_zip_data_are_readable(tmp_path):
    file = tmp_path / "wine.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("wine-red.csv", '"fixed acidity";"quality"\n7.4;5\n')
    table = load_tables([file])["wine#wine-red"]
    assert list(table.columns) == ["fixed acidity", "quality"]
    assert table.iloc[0]["quality"] == 5


def test_workflow_resume_keeps_solver_budget_and_saved_result(tmp_path):
    store = ArtifactStore(tmp_path)
    store.commit(CODE, DATA, "test")
    settings = copy.deepcopy(CFG)
    settings["loop"].update(max_solver_calls=1, gap_target=-1, patience=10)
    first = workflow.run_workflow(store, get_backend("sa"), lambda: make_policy("penalty_cem"), settings, log=lambda _: None)
    checkpoint = next((tmp_path / "checkpoints").glob("*.json"))
    settings["loop"]["max_solver_calls"] = 3
    resumed = workflow.run_workflow(store, get_backend("sa"), lambda: make_policy("penalty_cem"), settings,
        log=lambda _: None, resume_checkpoint=checkpoint)
    assert first["total_solver_calls"] == 1
    assert resumed["total_solver_calls"] == 3
    assert len(resumed["loop"]["rounds"]) == 3
    assert load_progress(tmp_path)["status"] == "complete"


def test_imported_prep_remains_instrumented():
    import pandas as pd
    import sys
    from qubo_auto.annealing.adapter import _execute_adapter
    previous = sys.modules.get('prep')
    code = 'import prep\ndef load_data(tables, log):\n    sampled = prep.sample_rows(tables["actual"], 2, seed=0, log=log)\n    return {"ids": sampled.index.tolist()}\n'
    data, errors, execution = _execute_adapter(code, {'actual': pd.DataFrame({'x': range(4)})}, [], [])
    assert not errors and 'sample_rows' in execution['tools_used']
    assert len(data['ids']) == 2 and sys.modules.get('prep') is previous


def test_download_candidates_with_same_title_do_not_overwrite(tmp_path, monkeypatch):
    from qubo_auto.annealing import opendata
    def response(url, **kw):
        raw = b'first,1\n' if url.endswith('first') else b'second,2\n'
        return SimpleNamespace(headers={'content-type': 'text/csv'}, raise_for_status=lambda: None,
                               iter_content=lambda size: [raw])
    monkeypatch.setattr(opendata, '_get', response)
    first = opendata.download('https://publisher/first', tmp_path, 'CSV', 'same title')
    second = opendata.download('https://publisher/second', tmp_path, 'CSV', 'same title')
    assert first != second and first.read_bytes() == b'first,1\n' and second.read_bytes() == b'second,2\n'


def test_download_rejects_a_web_page_served_for_a_data_link(tmp_path, monkeypatch):
    from qubo_auto.annealing import opendata
    page = b'<!doctype html>\n<html lang="de"><head><title>Senatsverwaltung</title></head></html>'
    monkeypatch.setattr(opendata, '_get', lambda url, **kw: SimpleNamespace(
        headers={'content-type': 'text/html'}, raise_for_status=lambda: None, iter_content=lambda size: [page]))
    with pytest.raises(ValueError, match="web page"):
        opendata.download('https://publisher/LOR_SHP.zip', tmp_path, 'ZIP', 'LOR')
    assert not list(tmp_path.glob('LOR*'))

def test_requirement_guaranteed_by_the_variable_design_can_be_mapped_as_implied():
    spec = {"constraints": [{"name": "single_tour", "strength": "hard"}, {"name": "visit_once", "strength": "hard"}]}
    raw = {"constraints": [{"name": "visit_once_0"}]}
    ok = {"REQUIREMENT_MAP": {"single_tour": ["implied: position encoding gives one closed tour"], "visit_once": ["visit_once_*"]}}
    assert check_requirements(spec, ok, raw) == []
    lazy = {"REQUIREMENT_MAP": {"single_tour": ["implied:"], "visit_once": ["visit_once_*"]}}
    assert check_requirements(spec, lazy, raw)                                   # an empty justification is not enough
    assert "implied:" in check_requirements(spec, {"REQUIREMENT_MAP": {}}, raw)[0]


def test_requirement_map_with_translated_keys_says_which_names_to_copy():
    spec = {"constraints": [{"name": "選址數量限制", "strength": "hard"}]}
    errs = check_requirements(spec, {"REQUIREMENT_MAP": {"site_count": ["k"]}}, {"constraints": [{"name": "k"}]})
    assert "copied EXACTLY" in errs[0] and "選址數量限制" in errs[0] and "site_count" in errs[0]


def test_only_crowd_sourced_points_trigger_one_more_search_for_the_official_source(tmp_path, monkeypatch):
    from qubo_auto.annealing import opendata
    rounds, fetched, specs = [], [], []
    osm = {"id": "osm-x", "source": "OpenStreetMap", "osm": {"tags": {}}, "title": "OSM points", "relevant": True,
           "readable": True, "feeds": "values"}
    official = {"id": "gov-1", "source": "政府資料開放平台", "url": "https://example.org/official.csv",
                "title": "Official list", "relevant": True, "readable": True, "feeds": "values"}

    def scout(spec, schema, client, model):
        specs.append(spec)
        rounds.append(len(rounds))
        return {"plan": {"govtw_queries": [f"q{len(rounds)}"]}, "candidates": []}

    def rank(spec, schema, cands, client, model):
        return [dict(osm)] if len(rounds) == 1 else [dict(official), dict(osm)]

    def fetch(c, dest):
        fetched.append(c["id"])
        path = dest / f"{c['id']}.csv"
        path.write_text("values\n3\n1\n")
        return path
    monkeypatch.setattr(opendata, "scout", scout)
    monkeypatch.setattr(opendata, "rank_candidates", rank)
    monkeypatch.setattr(opendata, "fetch_candidate", fetch)
    monkeypatch.setattr(acquisition, "build_adapter", lambda *a, **kw: ("adapter", DATA, [], {}))
    acquisition.prepare_real_data(CODE, "spec", None, "m", tmp_path / "data", log=lambda _: None)
    assert fetched and fetched[0] == "gov-1"                      # the second search found the official source
    assert "OFFICIAL SOURCE NOT FOUND YET" in specs[1]
