from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from qubo_auto.annealing.artifacts import ArtifactStore
from qubo_auto.annealing.assessment import evaluate_delivery, inspect_problem
from qubo_auto.annealing.encodings import LinearConstraint, ProblemIR
from qubo_auto.annealing import modeler, workflow
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import DefaultPolicy
from qubo_auto.annealing.loop import LoopResult
from qubo_auto.annealing.runtime import StageTimeout, generated_module


CODE = '''DATA_SCHEMA = {"values": "list", "capacity": "number"}
def make_example_data(size, seed):
    return {"values": [3, 1], "capacity": 1}
def build_ir(data):
    return {"name": "selection", "variable_names": ["a", "b"], "sense": "maximize",
            "linear": {0: data["values"][0], 1: data["values"][1]},
            "constraints": [{"name": "capacity", "kind": "linear", "coeffs": {0: 1, 1: 1},
                             "sense": "le", "rhs": data["capacity"]}]}
def describe_solution(data, x):
    return {"selected": [i for i, b in enumerate(x) if b]}
'''
DATA = {"values": [3, 1], "capacity": 1}
CFG = {"loop": {"max_solver_calls": 2, "patience": 3},
       "param_caps": {"sa": {"num_reads": 5, "num_sweeps": 100}},
       "workflow": {"max_repairs": 1, "preflight_time_s": 1}}


def ir():
    return ProblemIR(name="selection", instance_id="test", n_variables=2, sense="maximize",
                     linear={0: 3, 1: 1}, constraints=[LinearConstraint(name="capacity", coeffs={0: 1, 1: 1}, sense="le", rhs=1)])


def test_preflight_reports_impossible_capacity_without_solver(monkeypatch):
    problem = ir()
    problem.constraints[0].sense = "ge"
    problem.constraints[0].rhs = 3
    monkeypatch.setattr("qubo_auto.annealing.assessment.run_task", lambda *a, **kw: pytest.fail("solver should not run"))
    check = inspect_problem(problem)
    assert check["feasibility"] == "infeasible"
    assert "capacity" in check["issues"][0] and "penalty" in check["issues"][0]


def test_preflight_catches_joint_conflict():
    problem = ir()
    problem.constraints = [LinearConstraint(name="zero", coeffs={0: 1}, sense="eq", rhs=0),
                           LinearConstraint(name="one", coeffs={0: 1}, sense="eq", rhs=1)]
    assert inspect_problem(problem)["feasibility"] == "infeasible"


def test_time_limit_is_unknown_not_infeasible(monkeypatch):
    problem = ir()
    problem.constraints[0].sense = "eq"
    monkeypatch.setattr("qubo_auto.annealing.assessment.run_task", lambda *a, **kw: {"status": 1, "solution": None, "message": "timeout"})
    check = inspect_problem(problem)
    assert check["feasibility"] == "unknown" and not check["proven_optimal"]


def test_baseline_checks_original_constraints_and_sense():
    problem = ir()
    check = inspect_problem(problem)
    assert check["baseline_objective"] == 3
    assert check["baseline_solution"] == [1, 0]
    bad = evaluate_delivery(problem, SimpleNamespace(best_solution=[1, 1]), check)
    assert bad["status"] == "no_solution" and bad["evidence"] == ["capacity"]
    worse = evaluate_delivery(problem, SimpleNamespace(best_solution=[0, 1]), check)
    assert worse["status"] == "limited" and worse["quality_gap"] > 0
    assert evaluate_delivery(problem, SimpleNamespace(best_solution=[1, 0]), check)["status"] == "usable"


def test_version_restore_keeps_model_and_data_together(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.commit(CODE, DATA, "first")
    second = store.commit(CODE + "\n# new model", {**DATA, "capacity": 2}, "second")
    assert first != second and len(store.versions()) == 2
    store.restore(first)
    assert store.current()["data"] == DATA and store.current()["code"] == CODE
    with pytest.raises(ValueError):
        store.restore("../model.py")


def test_generated_code_hang_is_stopped_from_ui_thread():
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generated_module, "while True: pass", 1)
        with pytest.raises(StageTimeout):
            job.result(timeout=5)
    ns = modeler.load_module(CODE)
    assert ns["build_ir"](DATA)["sense"] == "maximize"  # a fresh task still works


def test_outer_workflow_delivers_normal_case(tmp_path):
    store = ArtifactStore(tmp_path)
    store.commit(CODE, DATA, "test")
    output = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG, log=lambda _: None)
    assert output["delivery"]["status"] in ("usable", "limited")
    assert output["solution"] is not None and output["total_solver_calls"] <= 2
    assert (tmp_path / "workflow_state.json").exists()
    assert all(c.satisfied(output["solution"]) for c in ir().constraints)


def test_invalid_model_routes_to_repair_and_keeps_versions(tmp_path):
    store = ArtifactStore(tmp_path)
    broken = CODE.replace('return {"name": "selection"', 'raise ValueError("wrong column")\n    return {"name": "selection"')
    old = store.commit(broken, DATA, "broken")
    routed = []
    def repair(issue, bundle):
        routed.append(issue["stage"])
        return {"status": "repaired", "code": CODE, "data": DATA, "message": "fixed column"}
    output = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG, repair=repair, log=lambda _: None)
    assert routed == ["model"]
    assert output["version"] != old and store.read(old)["code"] == broken
    assert output["total_solver_calls"] <= 2 and output["solution"] is not None


def test_impossible_request_does_not_burn_solver_calls(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    impossible = CODE.replace('"sense": "le"', '"sense": "ge"')
    store.commit(impossible, {**DATA, "capacity": 3}, "impossible")
    monkeypatch.setattr(workflow, "run_loop", lambda *a, **kw: pytest.fail("must not anneal an impossible request"))
    output = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG,
        repair=lambda *a: {"status": "needs_input", "message": "需要增加容量或確認需求"}, log=lambda _: None)
    assert output["delivery"]["status"] == "no_solution" and output["total_solver_calls"] == 0
    assert "確認需求" in output["delivery"]["recovery_message"]


def test_worse_annealer_result_delivers_attributed_baseline(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    store.commit(CODE, DATA, "test")
    result = LoopResult(problem="selection", instance_id="test", backend="sa", simulated=True,
        policy="default", rounds=[], best_round=None, best_objective=1, best_feasibility_rate=1,
        best_solution=[0, 1], reference_objective=3, best_gap=2/3, stop_reason="budget", solver_calls=2)
    monkeypatch.setattr(workflow, "run_loop", lambda *a, **kw: result)
    output = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, CFG, log=lambda _: None)
    assert output["solution"] == [1, 0]
    assert output["delivery"]["solution_source"] == "baseline"
    assert output["delivery"]["status"] == "limited" and output["delivery"]["objective"] == 3


def test_acquisition_switches_failed_source_then_prepares(tmp_path, monkeypatch):
    from qubo_auto.annealing import acquisition as A
    candidates = [{"id": "bad", "title": "bad", "url": "bad", "relevant": True, "readable": True},
                  {"id": "good", "title": "good", "url": "good", "source": "test", "relevant": True, "readable": True}]
    monkeypatch.setattr(A.opendata, "scout", lambda *a: {"candidates": candidates})
    monkeypatch.setattr(A.opendata, "rank_candidates", lambda *a: candidates)
    called = []
    def fetch(c, directory):
        called.append(c["id"])
        if c["id"] == "bad":
            raise TimeoutError("download timeout")
        p = directory / "good.csv"
        p.write_text("values\n3\n1\n")
        return p
    monkeypatch.setattr(A.opendata, "fetch_candidate", fetch)
    monkeypatch.setattr(A, "build_adapter", lambda *a, **kw: ("code", DATA, [], {"notes": "", "prep_log": "used real data"}))
    data, errors, report = A.prepare_real_data(CODE, "spec", None, "m", tmp_path, log=lambda _: None)
    assert called == ["bad", "good"] and errors == [] and data == DATA
    assert report["sources"][0]["url"] == "good"
