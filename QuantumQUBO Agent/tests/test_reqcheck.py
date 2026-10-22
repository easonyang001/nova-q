"""Requirement-level validation: an independent Checker against the model on random valid / invalid decisions."""
from qubo_auto.annealing.reqcheck import Validator, fixed_value, parse

DATA = {"sites": ["A", "B", "C", "D", "E", "F"], "value": [5, 1, 4, 2, 6, 3]}
CHECKER = '''
OBJECTIVE_SENSE = "maximize"
def evaluate(data, decisions):
    ids = [d["sites"] for d in decisions]
    v = []
    if len(ids) != 2: v.append(f"opened {len(ids)} sites, required exactly 2")
    if len(set(ids)) != len(ids): v.append("a site is opened twice")
    val = {s: x for s, x in zip(data["sites"], data["value"])}
    return {"feasible": not v, "violations": v, "objective": float(sum(val[i] for i in set(ids)))}
def sample_decisions(data, rng):
    return [{"sites": s} for s in rng.sample(data["sites"], 2)]
'''


def model(sense="maximize", sign=1, rhs=2, op="eq"):
    n = len(DATA["sites"])
    return {"variable_names": [f"x{i}" for i in range(n)], "variable_keys": [{"sites": s} for s in DATA["sites"]],
            "sense": sense, "linear": {str(i): sign * v for i, v in enumerate(DATA["value"])}, "quadratic": [],
            "constant": 0, "constraints": [{"name": "k", "kind": "linear", "coeffs": {str(i): 1 for i in range(n)},
                                            "sense": op, "rhs": rhs}]}


def test_fixed_value_respects_constraints():
    m = parse(model())
    assert fixed_value(m, [{"sites": "A"}, {"sites": "E"}])[0] == 11
    assert fixed_value(m, [{"sites": "A"}])[0] is None                         # breaks "exactly 2"
    assert "no model variable" in fixed_value(m, [{"sites": "Z"}])[1]


def test_correct_model_agrees_with_the_checker():
    v = Validator(CHECKER, [("sites",)], samples=12)
    assert v.preflight(DATA, model()) == [] and v.last["rank_agreement"] == 1.0
    assert v.last["invalid_rejected"] == v.last["invalid"] > 0
    assert v.solution(DATA, model(), [1, 0, 0, 0, 1, 0]) == []
    assert "opened 3 sites" in v.solution(DATA, model(), [1, 1, 0, 0, 1, 0])[0]


def test_relaxed_bound_and_flipped_objective_are_reported():
    v = Validator(CHECKER, [("sites",)], samples=12)
    issues = v.preflight(DATA, model(rhs=5, op="le"))                         # "at most 5" for "exactly 2"
    assert any("accepts a solution that breaks" in i for i in issues)
    issues = v.preflight(DATA, model(sign=-1))                                # maximising the negative value
    assert any("ranks valid solutions differently" in i for i in issues)
    assert Validator(CHECKER, [("stations",)]).preflight(DATA, model()) == []  # other decision sets: not comparable


MODEL_CODE = '''DATA_SCHEMA = {"sets": {"sites": {"meaning": "site", "source": "public"}},
  "params": {"value": {"meaning": "value", "index": ["sites"], "type": "number", "source": "public"}}}
REQUIREMENT_MAP = {}
def make_example_data(size, seed):
    return {"sites": ["A", "B", "C", "D", "E", "F"], "value": [5, 1, 4, 2, 6, 3]}
def build_ir(data):
    n = len(data["sites"])
    return {"name": "pick", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "variable_keys": [{"sites": s} for s in data["sites"]],
            "linear": {str(i): v for i, v in enumerate(data["value"])}, "quadratic": [], "constant": 0,
            "constraints": [{"name": "k", "kind": "linear", "coeffs": {str(i): 1 for i in range(n)}, "sense": "%S", "rhs": %R}]}
def describe_solution(data, x):
    return {"open": [s for s, b in zip(data["sites"], x) if b]}
'''


def test_workflow_repairs_a_model_the_checker_disagrees_with(tmp_path):
    from qubo_auto.annealing import workflow
    from qubo_auto.annealing.artifacts import ArtifactStore
    from qubo_auto.annealing.backends import get_backend
    from qubo_auto.annealing.baselines import DefaultPolicy
    loose, right = MODEL_CODE.replace("%S", "le").replace("%R", "5"), MODEL_CODE.replace("%S", "eq").replace("%R", "2")
    store = ArtifactStore(tmp_path)
    store.commit(loose, DATA, "loose", {"source": "synthetic"})
    asked = []

    def repair(issue, bundle):
        asked.append(issue)
        return {"status": "repaired", "code": right, "data": bundle["data"], "message": "fixed"}

    cfg = {"loop": {"max_solver_calls": 2, "patience": 3}, "param_caps": {"sa": {"num_reads": 20, "num_sweeps": 200}},
           "workflow": {"max_repairs": 1, "preflight_time_s": 1}}
    out = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, cfg, repair=repair,
                                validator=Validator(CHECKER, [("sites",)], samples=10), log=lambda m: None)
    assert asked and asked[0]["stage"] == "model" and "accepts a solution that breaks" in asked[0]["evidence"][0]
    assert sum(out["solution"]) == 2 and out["delivery"]["formula_changes"]          # repaired, change disclosed
    assert "requirement_warnings" not in out["delivery"]


def test_without_repair_the_disagreement_is_delivered_as_a_warning(tmp_path):
    from qubo_auto.annealing import workflow
    from qubo_auto.annealing.artifacts import ArtifactStore
    from qubo_auto.annealing.backends import get_backend
    from qubo_auto.annealing.baselines import DefaultPolicy
    store = ArtifactStore(tmp_path)
    store.commit(MODEL_CODE.replace("%S", "le").replace("%R", "5"), DATA, "loose", {"source": "synthetic"})
    cfg = {"loop": {"max_solver_calls": 1, "patience": 3}, "param_caps": {"sa": {"num_reads": 20, "num_sweeps": 200}},
           "workflow": {"max_repairs": 0, "preflight_time_s": 1}}
    out = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, cfg,
                                validator=Validator(CHECKER, [("sites",)], samples=10), log=lambda m: None)
    assert out["solution"] is not None and out["delivery"]["status"] == "limited"
    assert any("breaks" in w for w in out["delivery"]["requirement_warnings"])


def test_decisions_the_model_leaves_out_are_not_comparable_not_an_error():
    raw = model()
    raw["variable_keys"][0] = {}                         # the model fixed site A out (e.g. symmetry breaking)
    v = Validator(CHECKER, [("sites",)], samples=12)
    issues = v.preflight(DATA, raw)
    assert not any("rejects" in i for i in issues) and v.last["not_comparable"] > 0


def test_checker_code_can_use_the_toolkit_for_distances():
    from qubo_auto.annealing.reqcheck import _evaluate
    code = ("OBJECTIVE_SENSE = 'maximize'\n"
            "def evaluate(data, decisions):\n"
            "    d = prep.haversine_matrix(data['lat'], data['lon'])\n"
            "    return {'feasible': True, 'violations': [], 'objective': float(d[0][1])}\n"
            "def sample_decisions(data, rng):\n    return []\n")
    got = _evaluate(code, {"lat": [25.0330, 25.0350], "lon": [121.5436, 121.5436]}, [])
    assert abs(got["objective"] - 222.4) < 1


def test_a_late_checker_objection_does_not_discard_the_annealer_solution(tmp_path):
    from qubo_auto.annealing import workflow
    from qubo_auto.annealing.artifacts import ArtifactStore
    from qubo_auto.annealing.backends import get_backend
    from qubo_auto.annealing.baselines import DefaultPolicy

    class Objecting:                            # agrees before solving, objects to every solution (a wrong Checker)
        last = {}
        def preflight(self, data, raw):
            return []
        def solution(self, data, raw, x):
            return ["the delivered solution breaks the confirmed requirements: start must be site A"]

    store = ArtifactStore(tmp_path)
    store.commit(MODEL_CODE.replace("%S", "eq").replace("%R", "2"), DATA, "right", {"source": "synthetic"})
    asked = []
    cfg = {"loop": {"max_solver_calls": 1, "patience": 3}, "param_caps": {"sa": {"num_reads": 20, "num_sweeps": 200}},
           "workflow": {"max_repairs": 2, "preflight_time_s": 1}}
    out = workflow.run_workflow(store, get_backend("sa"), DefaultPolicy, cfg, validator=Objecting(), log=lambda m: None,
                                repair=lambda issue, bundle: asked.append(issue) or {"status": "needs_input"})
    assert not asked                                         # no calls left to re-solve: no repair is attempted
    assert out["solution"] is not None and sum(out["solution"]) == 2
    assert out["delivery"].get("solution_source") in ("annealer", "baseline")
    assert any("start must be" in w for w in out["delivery"]["requirement_warnings"])


def test_checker_code_may_import_the_toolkit_by_name():
    from qubo_auto.annealing.reqcheck import _evaluate
    code = ("import prep\nimport numpy as np\nOBJECTIVE_SENSE = 'maximize'\n"
            "def evaluate(data, decisions):\n    return {'feasible': True, 'violations': [],"
            " 'objective': float(prep.distance_to([25.0350], [121.5436], 25.0330, 121.5436)[0])}\n"
            "def sample_decisions(data, rng):\n    return []\n")
    assert abs(_evaluate(code, {}, [])["objective"] - 222.4) < 1
