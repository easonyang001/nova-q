"""Annealing loop tests (P0 infrastructure, P2 engine, P3 RAG) — local only, no QPU / LLM calls."""
import itertools
import json

import pytest

from qubo_auto.annealing.backends import SimulatedAnnealingBackend, SolverRunResult
from qubo_auto.annealing.examples import max_cut, select_k_with_conflicts
from qubo_auto.annealing.feedback import extract_feedback
from qubo_auto.annealing.guardrail import GuardrailError, ParamSpec, check_params
from qubo_auto.annealing.problem import qubo_energy


def _argmin(qubo, offset, n):
    xs = [list(x) for x in itertools.product((0, 1), repeat=n)]
    energies = [qubo_energy(qubo, offset, x) for x in xs]
    best = min(energies)
    return [x for x, e in zip(xs, energies) if abs(e - best) < 1e-9]


@pytest.mark.parametrize("factory", [select_k_with_conflicts, max_cut])
def test_qubo_argmin_is_feasible_optimum(factory):
    p = factory()
    qubo, offset = p.to_qubo()
    for x in _argmin(qubo, offset, p.n_variables):
        assert p.is_feasible(x)
        assert p.objective(x) == p.reference_objective


def test_penalty_qubos_are_zero_iff_satisfied():
    p = select_k_with_conflicts()
    for x in itertools.product((0, 1), repeat=p.n_variables):
        x = list(x)
        for name, (pq, off) in p.penalty_qubos.items():
            value = qubo_energy(pq, off, x)
            assert value >= 0
            assert (value == 0) == p.constraints[name](x)


def test_penalty_weights_leave_objective_untouched():
    p = select_k_with_conflicts()
    q1, _ = p.to_qubo()
    q2, _ = p.to_qubo({"select_k": 50.0})
    assert q1 != q2
    with pytest.raises(ValueError):
        p.to_qubo({"objective": 2.0})


def test_guardrail_rejects_unknown_type_and_range():
    specs = {"num_reads": ParamSpec(type="int", min=1, max=10, default=5),
             "t": ParamSpec(type="float", min=0.5, max=2.0)}
    assert check_params({}, specs) == {"num_reads": 5}
    assert check_params({"t": 1}, specs) == {"num_reads": 5, "t": 1.0}
    with pytest.raises(GuardrailError) as e:
        check_params({"num_reads": 11, "t": "fast", "chain": 1}, specs)
    assert len(e.value.errors) == 3


def test_feedback_counts_occurrences_and_violations():
    p = select_k_with_conflicts()
    best = [0, 0, 0, 0, 0, 0, 0, 0]
    for x in itertools.product((0, 1), repeat=p.n_variables):
        if p.is_feasible(list(x)) and p.objective(list(x)) == p.reference_objective:
            best = list(x)
    too_many = [1] * 8
    result = SolverRunResult(backend="test", params={}, samples=[best, too_many], num_occurrences=[3, 1],
                             energies=[0.0, 1.0], wall_time_s=0.0)
    fb = extract_feedback(p, result)
    assert fb.n_samples == 4
    assert fb.feasibility_rate == 0.75
    assert fb.constraint_violation_rates == {"select_k": 0.25, "conflicts": 0.25}
    assert fb.gap == 0.0


def test_sa_backend_end_to_end():
    p = max_cut()
    qubo, offset = p.to_qubo()
    result = SimulatedAnnealingBackend().run(qubo, offset, p.n_variables, {"num_reads": 50, "seed": 1})
    assert sum(result.num_occurrences) == 50
    assert all(len(x) == p.n_variables for x in result.samples)
    fb = extract_feedback(p, result)
    assert fb.feasibility_rate == 1.0
    assert fb.best_feasible_objective == p.reference_objective


# ---------------------------------------------------------------- loop / guardrails / RAG (no LLM)
import yaml
from pathlib import Path

from qubo_auto.annealing.backends import DWaveQPUBackend
from qubo_auto.annealing.baselines import Policy, RuleBasedPolicy
from qubo_auto.annealing.examples import knapsack_random
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import Proposal, RunConfig, build_context, run_loop, validate_config
from qubo_auto.annealing.rag import KnowledgeBase
from qubo_auto.annealing.tools import penalty_lower_bound, verify_small

ROOT = Path(__file__).parent.parent


def _cfg(tmp_path, **loop):
    cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text())
    cfg["loop"].update(loop)
    cfg["logging"]["loop_log"] = str(tmp_path / "loop.jsonl")
    return cfg


def test_verify_small_strict_and_weak_penalty():
    p = knapsack_random(n=6, seed=1)
    assert verify_small(p)["passed"]
    assert not verify_small(p, {"capacity": 0.5})["passed"]


def test_validate_config_rejects_floor_unknown_and_caps(tmp_path):
    p = select_k_with_conflicts()
    ctx = build_context(p, SimulatedAnnealingBackend(), _cfg(tmp_path))
    floor = penalty_lower_bound(p)["select_k"]
    _, errs = validate_config(ctx, RunConfig(penalty_weights={"select_k": floor / 2, "bogus": 1.0}), [], False)
    assert any("must be > lower bound" in e for e in errs) and any("unknown penalty" in e for e in errs)
    _, errs = validate_config(ctx, RunConfig(params={"num_sweeps": 10**9}), [], False)
    assert any("outside" in e for e in errs)          # config cap (20000) applies, not only backend max


class _Crashing(Policy):
    name = "crashing"
    def initial(self, ctx):
        return Proposal(config=ctx.default_config, source=self.name)
    def next(self, ctx, history):
        raise RuntimeError("LLM went away")


def test_policy_failure_falls_back_to_rule(tmp_path):
    p = knapsack_random(n=10, seed=2)
    res = run_loop(p, SimulatedAnnealingBackend(), _Crashing(), _cfg(tmp_path, max_solver_calls=3, gap_target=-1),
                   root=tmp_path, verbose=False)
    assert res.solver_calls >= 2
    assert all(r.source.startswith("fallback") for r in res.rounds[1:])


def test_rule_policy_never_repeats_configs(tmp_path):
    p = knapsack_random(n=10, seed=3)
    res = run_loop(p, SimulatedAnnealingBackend(), RuleBasedPolicy(),
                   _cfg(tmp_path, max_solver_calls=6, gap_target=-1, patience=10), root=tmp_path, verbose=False)
    keys = [r.config.key() for r in res.rounds]
    assert len(keys) == len(set(keys))


def test_hardware_budget_refuses_before_any_call(tmp_path):
    budget = HardwareBudget(tmp_path / "ledger.jsonl", {"dwave_qpu": {"max_calls": 0, "max_solver_time_s": 0}})
    res = run_loop(select_k_with_conflicts(), DWaveQPUBackend(), RuleBasedPolicy(), _cfg(tmp_path),
                   budget=budget, root=tmp_path, verbose=False)
    assert res.stop_reason == "hardware_budget" and res.solver_calls == 0
    assert not (tmp_path / "ledger.jsonl").exists()


def test_rag_experience_isolation(tmp_path):
    exp = tmp_path / "exp.jsonl"
    rec = {"instance_id": "A", "problem_type": "knapsack", "backend": "sa", "change": {}, "before": {}, "after": {},
           "effective": True, "features": {"problem_type": "knapsack", "n_variables": 20, "density": 1.0,
                                           "coef_dynamic_range": 300.0}}
    exp.write_text("\n".join(json.dumps({**rec, "id": f"e{i}", "instance_id": iid}) for i, iid in enumerate("ABC")))
    kb = KnowledgeBase(["experience"], 5, ROOT / "knowledge", exp, exclude_instance_ids=["B"])
    summary = {"problem_type": "knapsack", "n_variables": 20, "density": 1.0, "coef_dynamic_range": 300.0}
    got = {e.id for e in kb.retrieve_experience(summary, "sa", instance_id="A")}
    assert got == {"e2"}                                # self (A) and test-split (B) never retrieved


def test_rag_docs_are_backend_scoped():
    kb = KnowledgeBase(["backend_docs"], 10, ROOT / "knowledge", Path("/nonexistent"))
    ids = {e.id for e in kb.retrieve_docs("backend_docs", {"sa", "num_reads", "insufficient_sampling"})}
    assert ids and all(not i.startswith("bd-dwave") for i in ids)


def test_graph_coloring_qubo_is_correct():
    from qubo_auto.annealing.examples import graph_coloring_random
    for seed in range(3):
        assert verify_small(graph_coloring_random(n=5, k=3, seed=seed))["passed"]


def test_milp_reference_matches_enumeration():
    import numpy as np
    from qubo_auto.annealing.examples import _best_k_subset, milp_reference
    rng = np.random.default_rng(0)
    for _ in range(5):
        vals = [int(v) for v in rng.integers(1, 21, size=14)]
        conf = [(i, j) for i in range(14) for j in range(i + 1, 14) if rng.random() < 0.2]
        a = np.zeros((len(conf), 14))
        for r, (i, j) in enumerate(conf):
            a[r, i] = a[r, j] = 1
        assert milp_reference(vals, a, np.ones(len(conf)), np.ones((1, 14)), [4], maximize=True) == _best_k_subset(vals, 4, conf)


@pytest.mark.skipif(not Path("/research/datasets/QOBLIB/07-independentset").exists(), reason="QOBLIB not available")
def test_qoblib_instances_load_and_match_references(tmp_path):
    from qubo_auto.annealing.qoblib import market_split, max_independent_set
    ms = market_split("ms_03_050_002")             # loader asserts the official solution satisfies A x = b
    assert ms.reference_objective == 0.0 and len(ms.penalty_qubos) == 3
    mis = max_independent_set("karate")             # loader asserts the official set is independent
    assert mis.reference_objective == 20.0
    ctx = build_context(mis, SimulatedAnnealingBackend(), _cfg(tmp_path))
    assert ctx.default_config.penalty_weights["independence"] > ctx.lower_bounds["independence"]


def test_formulation_patterns_are_not_backend_scoped():
    kb = KnowledgeBase(["formulation_patterns"], 10, ROOT / "knowledge", Path("/nonexistent"))
    ids = {e.id for e in kb.retrieve_docs("formulation_patterns", {"sa", "cardinality_equality"})}
    assert "fp-cardinality" in ids                     # tagged dwave_qpu, but still relevant on SA


# ---------------------------------------------------------------- P5 bridge (no LLM)
_ONE_HOT_QUBO = '''
import numpy as np
def build_qubo(instance):
    v, A = instance["values"], instance["A"]
    n = len(v); Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -v[i] - A
        for j in range(i + 1, n):
            Q[i, j] = 2 * A
    return Q, float(A)
'''
_GOOD = '''
import numpy as np
SENSE = "maximize"
def build_objective(instance):
    v = instance["values"]; n = len(v); Q = np.zeros((n, n))
    for i in range(n): Q[i, i] = -v[i]
    return Q, 0.0
def build_penalties(instance):
    n = len(instance["values"]); Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -1.0
        for j in range(i + 1, n): Q[i, j] = 2.0
    return {"pick_one": (Q, 1.0, float(instance["A"]))}
def decision_variables(instance):
    return list(range(len(instance["values"])))
def objective_value(instance, x):
    return float(sum(v * b for v, b in zip(instance["values"], x)))
def check_constraints(instance, x):
    return {"pick_one": sum(x) == 1}
'''


def _one_hot_case():
    from qubo_auto.verification.test_cases import TestCase
    inst = {"values": [3, 7, 5, 2], "A": 10}
    return TestCase(name="pick1", description="", n_variables=4, instance=inst,
                    optimal_value=-7.0, optimal_bitstrings=[(0, 1, 0, 0)], all_strings=None)


def test_bridge_accepts_correct_decomposition():
    from qubo_auto.agents.coder_agent import compile_build_qubo
    from qubo_auto.annealing.bridge import compile_module, to_problem_instance, verify_decomposition
    case = _one_hot_case()
    ns = compile_module(_GOOD)
    assert verify_decomposition(ns, compile_build_qubo(_ONE_HOT_QUBO), [case]) == []
    p = to_problem_instance(ns, case.instance, "pick_one", "test_pick1", reference=7.0)
    assert verify_small(p)["passed"]


@pytest.mark.parametrize("old,new,expected_error", [
    ("Q[i, i] = -v[i]\n", "Q[i, i] = -2 * v[i]\n", "differs from build_qubo"),          # objective part wrong
    ('{"pick_one": sum(x) == 1}', '{"pick_one": sum(x) == 2}', "violates"),               # wrong feasibility check
    ("return float(sum(", "return -float(sum(", "objective_value"),                        # wrong objective sign
])
def test_bridge_rejects_wrong_decompositions(old, new, expected_error):
    from qubo_auto.agents.coder_agent import compile_build_qubo
    from qubo_auto.annealing.bridge import compile_module, verify_decomposition
    assert old in _GOOD
    errors = verify_decomposition(compile_module(_GOOD.replace(old, new)), compile_build_qubo(_ONE_HOT_QUBO),
                                  [_one_hot_case()])
    assert any(expected_error in e for e in errors), errors


def test_bridge_catches_inequality_encoded_as_equality():
    """Front-end knapsack QUBO (sum w x - W)^2 without slack: passes tight test cases, wrong in general."""
    from qubo_auto.agents.coder_agent import compile_build_qubo
    from qubo_auto.annealing.bridge import compile_module, verify_decomposition
    from qubo_auto.verification.test_cases import TestCase
    bq = compile_build_qubo('''
import numpy as np
def build_qubo(instance):
    w, v, W, lam = instance["weights"], instance["values"], instance["W"], 1000.0
    n = len(w); Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -v[i] + lam * (w[i] ** 2 - 2 * W * w[i])
        for j in range(i + 1, n): Q[i, j] = 2 * lam * w[i] * w[j]
    return Q, lam * W ** 2
''')
    dec = compile_module('''
import numpy as np
SENSE = "maximize"
def build_objective(instance):
    v = instance["values"]; Q = np.diag([-float(a) for a in v]); return Q, 0.0
def build_penalties(instance):
    w, W = instance["weights"], instance["W"]; n = len(w); Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = w[i] ** 2 - 2 * W * w[i]
        for j in range(i + 1, n): Q[i, j] = 2 * w[i] * w[j]
    return {"capacity": (Q, float(W ** 2), 1000.0)}
def decision_variables(instance): return list(range(len(instance["weights"])))
def objective_value(instance, x): return float(sum(a * b for a, b in zip(instance["values"], x)))
def check_constraints(instance, x): return {"capacity": sum(a * b for a, b in zip(instance["weights"], x)) <= instance["W"]}
''')
    tight = TestCase(name="tight", description="", n_variables=3, instance={"weights": [3, 4, 5], "values": [30, 50, 60], "W": 8},
                     optimal_value=-90.0, optimal_bitstrings=[(1, 0, 1)], all_strings=None)
    errors = verify_decomposition(dec, bq, [tight])
    assert any("cannot reach 0" in e for e in errors), errors


# ---------------------------------------------------------------- encoding library (spec v0.2)
from qubo_auto.annealing.encodings import (Encoded, ExclusionConstraint, ImplicationConstraint, LinearConstraint,
                                           compile_problem, encode, verify_encoding)


@pytest.mark.parametrize("c,choice", [
    (LinearConstraint(name="eq", coeffs={0: 2, 1: 3, 2: 1}, sense="eq", rhs=3), "square"),
    (LinearConstraint(name="le", coeffs={0: 2, 1: 3, 2: 4}, sense="le", rhs=5), "binary_slack"),
    (LinearConstraint(name="le", coeffs={0: 2, 1: 3, 2: 4}, sense="le", rhs=5), "unary_slack"),
    (LinearConstraint(name="ge", coeffs={0: 1, 1: 1, 2: 1, 3: 1}, sense="ge", rhs=2), "binary_slack"),
    (LinearConstraint(name="mixed", coeffs={0: 3, 1: -2, 2: 1}, sense="le", rhs=1), "binary_slack"),
    (ExclusionConstraint(name="ex", pairs=[(0, 1), (1, 2)]), "pairwise"),
    (ImplicationConstraint(name="im", pairs=[(0, 1), (2, 1)]), "pairwise"),
])
def test_exact_encodings_pass_two_way_check(c, choice):
    enc = encode(c, choice, next_aux=4)
    r = verify_encoding(c, enc, n_decision=4)
    assert r["checked"] and r["passed"], r


def test_unbalanced_is_reported_approximate():
    c = LinearConstraint(name="le", coeffs={0: 2, 1: 3, 2: 4}, sense="le", rhs=5)
    r = verify_encoding(c, encode(c, "unbalanced", next_aux=3, l1=1.0, l2=1.0), n_decision=3)
    assert r["checked"] and r["exact"] is False


def test_inequality_encoded_as_equality_is_caught():
    c = LinearConstraint(name="cap", coeffs={0: 3, 1: 4, 2: 5}, sense="le", rhs=8)
    as_eq = encode(LinearConstraint(name="cap", coeffs=c.coeffs, sense="eq", rhs=8), "square", next_aux=3)
    r = verify_encoding(c, as_eq, n_decision=3)
    assert not r["passed"] and any("satisfied" in e for e in r["errors"])


def test_compiled_knapsack_matches_reference():
    from qubo_auto.annealing.examples import knapsack_ir
    ir = knapsack_ir(n=6, seed=1)
    p = compile_problem(ir, {"capacity": "binary_slack"})
    assert p.reference_objective == 81.0 and verify_small(p)["passed"]


# ---------------------------------------------------------------- loop with per-round encodings (spec v0.2)
def test_loop_validates_encodings_and_recompiles(tmp_path):
    from qubo_auto.annealing.examples import knapsack_ir
    ir = knapsack_ir(n=8, seed=0)
    ctx = build_context(None, SimulatedAnnealingBackend(), _cfg(tmp_path), ir=ir)
    assert ctx.default_config.encodings == {"capacity": {"encoding": "binary_slack", "params": {}}}
    bad = RunConfig(encodings={"capacity": {"encoding": "magic", "params": {}}})
    assert any("not available" in e for e in validate_config(ctx, bad, [], False)[1])
    bad = RunConfig(encodings={"capacity": {"encoding": "unbalanced", "params": {"l3": 1.0}}})
    assert any("unknown parameter" in e for e in validate_config(ctx, bad, [], False)[1])
    ok, errs = validate_config(ctx, RunConfig(encodings={"capacity": {"encoding": "unbalanced", "params": {"l1": 10, "l2": 0.25}}},
                                              penalty_weights={"capacity": 1.0}), [], False)
    assert not errs                                   # weight 1 < floor is allowed for approximate encodings
    assert ctx.problem_for(ok).n_variables == 8       # no slack bits
    assert ctx.problem_for(ctx.default_config).n_variables > 8


def test_loop_runs_ir_problem_with_fixed_encoding(tmp_path):
    from qubo_auto.annealing.examples import knapsack_ir
    res = run_loop(None, SimulatedAnnealingBackend(), RuleBasedPolicy(), _cfg(tmp_path, max_solver_calls=3, gap_target=-1),
                   root=tmp_path, verbose=False, ir=knapsack_ir(n=10, seed=2))
    assert res.solver_calls >= 1 and all(r.config.encodings["capacity"]["encoding"] == "binary_slack" for r in res.rounds)


# ---------------------------------------------------------------- Clarification Agent (no LLM)
class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
    def call(self, **_):
        return self.replies.pop(0)


def test_clarify_session_requires_answers_and_confirmation():
    import json as _json
    from qubo_auto.annealing.clarify import confirm, new_session, step
    draft = {"problem_summary": "pick sites", "constraints": [{"name": "budget", "description": "cost <= B",
             "strength": "hard", "type": "inequality"}]}
    ask = _json.dumps({"draft": draft, "questions": [{"id": "Q1", "question": "budget?", "blocking": True}],
                       "ready_to_confirm": False})
    from qubo_auto.annealing.clarify import CHECKLIST
    full = {k: {"status": "confirmed"} for k in CHECKLIST}
    partial = {**full, "equity": {"status": "unknown"}}
    done = _json.dumps({"draft": draft, "questions": [], "coverage": full, "ready_to_confirm": True})
    half = _json.dumps({"draft": draft, "questions": [], "coverage": partial, "ready_to_confirm": True})
    s = new_session("城市選址", "fujitsu_da")
    s = step(s, _FakeClient([ask]), "m")
    with pytest.raises(ValueError):
        confirm(s)                                    # blocking question still open
    s = step(s, _FakeClient([half]), "m", answers="Q1: 1000")
    assert not s.ready_to_confirm                     # one checklist item is still unknown
    with pytest.raises(ValueError):
        confirm(s)
    s = step(s, _FakeClient([done]), "m", answers="equity: none needed")
    assert s.rounds[0]["answers"] == "Q1: 1000" and s.ready_to_confirm
    assert confirm(s).confirmed
    with pytest.raises(ValueError):
        step(s, _FakeClient([done]), "m")             # confirmed sessions are frozen


def test_clarify_caps_questions_and_accepts_assumptions_on_confirm():
    import json as _json
    from qubo_auto.annealing.clarify import CHECKLIST, confirm, new_session, step
    cov = {k: {"status": "assumed", "note": "default"} for k in CHECKLIST}
    many = _json.dumps({"draft": {"problem_summary": "x"}, "coverage": cov, "ready_to_confirm": True,
                        "questions": [{"id": f"Q{i}", "question": "?", "blocking": False} for i in range(9)]})
    s = step(new_session("x", "sa"), _FakeClient([many]), "m")
    assert len(s.pending_questions) == 5 and s.ready_to_confirm
    s = confirm(s)
    assert all(v.status == "confirmed" and "accepted" in v.note for v in s.coverage.values())


# ---------------------------------------------------------------- Modeling Agent validation (no LLM)
_PARKS = '''
import numpy as np
DATA_SCHEMA = {"population": "people served by each site", "cost": "cost per site", "budget": "total budget",
               "too_close": "pairs of sites that may not both be built"}
def make_example_data(size, seed):
    rng = np.random.default_rng(seed)
    return {"population": [int(v) for v in rng.integers(100, 1000, size)], "cost": [int(v) for v in rng.integers(1, 5, size)],
            "budget": 6, "too_close": [[0, 1], [2, 3]]}
def build_ir(data):
    n = len(data["population"])
    return {"name": "parks", "sense": "maximize", "variable_names": [f"site_{i}" for i in range(n)],
            "linear": {str(i): p for i, p in enumerate(data["population"])}, "quadratic": [], "constant": 0,
            "constraints": [{"name": "budget", "kind": "linear", "coeffs": {str(i): c for i, c in enumerate(data["cost"])},
                             "sense": "le", "rhs": data["budget"]},
                            {"name": "spacing", "kind": "exclusion", "pairs": data["too_close"]}]}
def describe_solution(data, x):
    return {"sites": [i for i, b in enumerate(x) if b], "population": sum(p for p, b in zip(data["population"], x) if b)}
def visualize(data, x):
    return {"type": "map", "title": "parks", "points": [{"x": float(i), "y": float(p), "group": "selected" if b else "candidate"}
                                                       for i, (p, b) in enumerate(zip(data["population"], x))]}
'''


def test_modeler_validates_good_model_and_reference():
    from qubo_auto.annealing.modeler import load_module, reference, to_ir, validate
    ns = load_module(_PARKS)
    errors, report = validate(ns)
    assert errors == [] and report["n_variables"] == 6 and "small_example_solution" in report
    ir = to_ir(ns["build_ir"](ns["make_example_data"](30, 1)), "parks30")
    assert reference(ir) is not None                  # 30 variables: MILP path


@pytest.mark.parametrize("old,new,expected", [
    ('"budget": 6,', '"budget": 0,\n            "must": 1,', None),                                      # still feasible (empty set)
    ('"sense": "le", "rhs": data["budget"]}', '"sense": "ge", "rhs": 10**6}', "no feasible solution"),
    ('"pairs": data["too_close"]}', '"pairs": [[0, 99]]}', "out of range"),
])
def test_modeler_rejects_bad_models(old, new, expected):
    from qubo_auto.annealing.modeler import load_module, validate
    assert old in _PARKS
    errors, _ = validate(load_module(_PARKS.replace(old, new)))
    if expected is None:
        assert errors == []
    else:
        assert any(expected in e for e in errors), errors


def test_modeler_accepts_variable_names_as_keys():
    from qubo_auto.annealing.modeler import load_module, to_ir
    ns = load_module(_PARKS.replace('"linear": {str(i): p for i, p', '"linear": {f"site_{i}": p for i, p'))
    ir = to_ir(ns["build_ir"](ns["make_example_data"](6, 0)), "names")
    assert set(ir.linear) == set(range(6))
    with pytest.raises(ValueError, match="unknown variable reference"):
        to_ir({"sense": "maximize", "variable_names": ["a"], "linear": {"b": 1}}, "bad")


def test_test_runner_refuses_huge_brute_force():
    import numpy as np
    from qubo_auto.verification.test_cases import TestCase
    from qubo_auto.verification.test_runner import run_tests
    big = TestCase(name="big", description="", n_variables=40, instance={}, optimal_value=0.0,
                   optimal_bitstrings=[tuple([0] * 40)], all_strings=None)
    results, cls = run_tests(lambda inst: (np.zeros((40, 40)), 0.0), [big])
    # too large to enumerate is the test case's problem, not the code's: never sent back to the Coder
    assert cls == "unverifiable" and results[0].status == "unverifiable"
    assert "too many for brute-force" in results[0].error_message


def test_case_builder_rejects_cases_too_large_to_verify():
    from qubo_auto.agents.debugger_agent import _validate_case
    from qubo_auto.verification.test_runner import MAX_BRUTE_FORCE_VARS
    n = MAX_BRUTE_FORCE_VARS + 1
    case = {"n_variables": n, "instance": {}, "ground_truth": {"optimal_value": 0, "optimal_bitstrings": [[0] * n]}}
    assert any("smaller instance" in e for e in _validate_case(case, 0))
    ok = {"n_variables": 3, "instance": {}, "ground_truth": {"optimal_value": 0, "optimal_bitstrings": [[0, 1, 0]]}}
    assert _validate_case(ok, 0) == []


def test_unbalanced_reward_bound():
    """A loose budget with a large l1 pays the solver to select nothing; the bound must expose that."""
    from qubo_auto.annealing.encodings import LinearConstraint, unbalanced_max_reward
    budget = LinearConstraint(name="budget", coeffs={i: 1.0 for i in range(20)}, sense="le", rhs=30.0)
    # h = 30 - sum(x) lies in [10, 30]; l1=300, l2=1 peaks at h=150, clipped to 30 -> 300*30 - 900
    assert unbalanced_max_reward(budget, l1=300, l2=1) == 8100
    # l1=10, l2=1 peaks at h=5, below the feasible range: the best is h=10 -> 100 - 100 = 0
    assert unbalanced_max_reward(budget, l1=10, l2=1) == 0
    tight = LinearConstraint(name="cap", coeffs={0: 5.0, 1: 7.0}, sense="le", rhs=8.0)   # h in [0, 8] (x=(1,1) infeasible)
    assert abs(unbalanced_max_reward(tight, l1=4, l2=1) - 4.0) < 1e-9                    # peak at h=2: 8 - 4


# ---------------------------------------------------------------- problem-level verification
def _mis_case():
    """Maximum independent set on the path 0-1-2: the optimum is {0, 2}, size 2 (a maximisation)."""
    from qubo_auto.verification.reference import ReferenceCase
    return ReferenceCase(name="path3", description="", instance={"n": 3, "edges": [[0, 1], [1, 2]]},
                         optimal_value=2.0, reference_solution={"selected": [0, 2]})


def _mis_evaluate(instance, solution):
    sel = set(solution["selected"])
    if not sel <= set(range(instance["n"])):
        raise ValueError("index out of range")
    return all(not (i in sel and j in sel) for i, j in instance["edges"]), float(len(sel))


def _mis_qubo(penalty, order=(0, 1, 2), n_aux=0):
    import numpy as np
    def build(instance):
        pos = {v: k for k, v in enumerate(order)}
        Q = np.zeros((3 + n_aux, 3 + n_aux))
        for v in range(3):
            Q[pos[v], pos[v]] -= 1.0
        for i, j in instance["edges"]:
            a, b = sorted((pos[i], pos[j]))
            Q[a, b] += penalty
        return Q, 0.0
    def decode(instance, x):
        return {"selected": [v for v in range(3) if x[order.index(v)]]}
    return build, decode


def test_reference_tests_accept_any_layout_and_aux_variables():
    from qubo_auto.verification.reference import run_reference_tests
    for order, n_aux in [((0, 1, 2), 0), ((2, 0, 1), 0), ((0, 1, 2), 2)]:
        build, decode = _mis_qubo(2.0, order, n_aux)
        results, cls = run_reference_tests(build, decode, _mis_evaluate, [_mis_case()])
        assert cls == "success" and results[0].match, (order, n_aux, results[0].error_message)


def test_reference_tests_reject_infeasible_ground_states():
    from qubo_auto.verification.reference import run_reference_tests
    # penalty 0.5 is too weak: selecting all three vertices ties with the optimum (-3 + 1 = -2)
    build, decode = _mis_qubo(0.5)
    results, cls = run_reference_tests(build, decode, _mis_evaluate, [_mis_case()])
    assert cls == "formulation_error" and "INFEASIBLE" in results[0].error_message


def test_reference_tests_blame_code_for_broken_decode():
    from qubo_auto.verification.reference import run_reference_tests
    build, _ = _mis_qubo(2.0)
    results, cls = run_reference_tests(build, lambda inst, x: 1 / 0, _mis_evaluate, [_mis_case()])
    assert cls == "coding_error" and "decode raised" in results[0].error_message


def test_checker_is_trusted_only_when_it_reproduces_the_benchmark():
    from dataclasses import replace
    from qubo_auto.verification.reference import validate_checker
    case = _mis_case()
    assert validate_checker(_mis_evaluate, [case]) == ([True], [])
    trusted, notes = validate_checker(lambda inst, sol: (True, 99.0), [case])        # wrong objective
    assert trusted == [False] and "99" in notes[0]
    trusted, notes = validate_checker(_mis_evaluate, [replace(case, reference_solution=None)])
    assert trusted == [False] and "no reference solution" in notes[0]
    # a benchmark whose stated solution contradicts its stated value (as quadratic_assignment does) is not trusted
    trusted, _ = validate_checker(_mis_evaluate, [replace(case, reference_solution={"selected": [0]})])
    assert trusted == [False]


def test_cga_request_is_upper_triangular():
    from qubo_auto.annealing.backends import _cga_index, _upper_triangular
    q = _upper_triangular({(2, 0): 1.5, (0, 2): 0.5, (1, 1): -1.0, (3, 1): 0.0})
    assert q == {(0, 2): 2.0, (1, 1): -1.0}
    assert [_cga_index(k) for k in (3, "c[4]", "y[5]", "6", "z")] == [3, 4, 5, 6, None]


def test_clarifier_web_lookup_runs_once_and_is_stored(monkeypatch):
    import json
    from qubo_auto.annealing import clarify as C, websearch
    calls = {"llm": 0, "search": []}

    class FakeClient:
        def call(self, model, prompt, **kw):
            calls["llm"] += 1
            looked_up = "QUERY: 勞基法 輪班 休息" in prompt
            return json.dumps({
                "draft": {"problem_summary": "排班", "sources": ([{"claim": "輪班間隔 >= 11 小時",
                                                                     "url": "https://law.moj.gov.tw/x"}] if looked_up else [])},
                "questions": [], "coverage": {}, "ready_to_confirm": False,
                "web_queries": [] if looked_up else ["勞基法 輪班 休息"]})

    monkeypatch.setattr(websearch, "lookup", lambda q, **kw: calls["search"].append(q) or
                        {"query": q, "time": 0, "error": "", "results": [{"title": "t", "url": "https://law.moj.gov.tw/x",
                                                                          "excerpt": "更換班次時至少應有連續十一小時之休息時間"}]})
    s = C.step(C.new_session("護理師排班，要符合勞基法", "cga"), FakeClient(), "m", web=True)
    assert calls["search"] == ["勞基法 輪班 休息"] and calls["llm"] == 2
    assert s.web_findings[0]["query"] == "勞基法 輪班 休息" and s.draft.sources[0].url == "https://law.moj.gov.tw/x"
    assert "參考來源" in C.render(s)
    C.step(s, FakeClient(), "m", answers="ok", web=True)          # findings reused: no second search
    assert calls["search"] == ["勞基法 輪班 休息"]
    off = C.step(C.new_session("同上", "cga"), FakeClient(), "m", web=False)
    assert off.web_findings == []


def test_clarifier_sources_must_be_fetched_pages():
    from qubo_auto.annealing.clarify import DraftSpec, Source, _check_sources
    findings = [{"query": "q", "results": [{"url": "https://www.mohw.gov.tw/dl-47213-037e2918.html", "title": "", "excerpt": ""}]}]
    d = DraftSpec(problem_summary="x", sources=[Source(claim="a", url="https://www.mohw.gov.tw/dl-472913-037e2918.html"),
                                                Source(claim="b", url="https://example.com/made-up")])
    _check_sources(d, findings)
    assert d.sources[0].url == "https://www.mohw.gov.tw/dl-47213-037e2918.html"     # garbled URL corrected
    assert d.sources[1].claim.startswith("（未驗證")                                  # not a fetched page


def test_constraint_families_keep_prompts_small():
    from qubo_auto.annealing.families import compress_messages, expand, families, group_rates, group_values
    names = [f"daily_n{n}_d{d}" for n in range(12) for d in range(28)] + ["cover_d1"]
    fam = families(names)
    assert set(fam) == {"daily_n#_d#", "cover_d#"} and len(fam["daily_n#_d#"]) == 336
    assert families(["a1", "b2"]) == {"a1": ["a1"], "b2": ["b2"]}          # small problems: unchanged
    w = {n: 10.0 for n in names}
    assert group_values(w, fam) == {"daily_n#_d# (×336)": 10.0, "cover_d1": 10.0}    # singletons keep their name
    w["daily_n0_d0"] = 20.0
    assert group_values(w, fam)["daily_n#_d# (×336)"] == {"min": 10.0, "max": 20.0}
    r = group_rates({"daily_n0_d0": 0.5}, fam)["daily_n#_d# (×336)"]
    assert r["max"] == 0.5 and r["violated_members"] == 1
    assert len(expand({"daily_n#_d# (×336)": 7.0}, fam)) == 336 and expand({"cover_d1": 1.0}, fam) == {"cover_d1": 1.0}
    msgs = compress_messages([f"penalty 'daily_n{i}_d1'=5 must be > 6" for i in range(300)])
    assert len(msgs) == 1 and "299 more" in msgs[0]


def test_model_validation_survives_numpy_data_and_points_at_the_line():
    from qubo_auto.annealing.modeler import load_module, validate
    base = '''import numpy as np
DATA_SCHEMA = {"w": "weights"}
def make_example_data(size, seed):
    return {"w": np.ones(size)}
def build_ir(data):
    w = np.asarray(data["w"])
    n = len(w)
    return {"name": "t", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "linear": {f"x{i}": float(w[i]) for i in range(n)}, "quadratic": [], "constant": 0.0, "constraints": []}
def describe_solution(data, x):
    return {}
def visualize(data, x):
    return {"type": "schedule", "title": "t", "rows": ["x"], "cols": [str(i) for i in range(len(x))], "cells": [[str(v) for v in x]]}
'''
    assert validate(load_module(base))[0] == []                     # numpy data is fine (stored as JSON lists)
    bad = base.replace('w = np.asarray(data["w"])', 'w = data["w"]; _ = w.shape')
    errs = validate(load_module(bad))[0]
    assert "line 6" in errs[0] and "JSON round trip" in errs[0]       # the failing line is reported


def test_model_validation_rejects_an_empty_objective():
    from qubo_auto.annealing.modeler import load_module, validate
    code = '''import numpy as np
DATA_SCHEMA = {"n": "sites"}
def make_example_data(size, seed):
    return {"n": size}
def build_ir(data):
    n = data["n"]
    return {"name": "t", "sense": "minimize", "variable_names": [f"x{i}" for i in range(n)], "linear": {},
            "quadratic": [], "constant": 0.0,
            "constraints": [{"name": "k", "kind": "linear", "coeffs": {f"x{i}": 1 for i in range(n)}, "sense": "eq", "rhs": 2}]}
def describe_solution(data, x):
    return {}
def visualize(data, x):
    return {"type": "schedule", "title": "t", "rows": ["x"], "cols": [str(i) for i in range(len(x))], "cells": [[str(v) for v in x]]}
'''
    assert any("objective is empty" in e for e in validate(load_module(code))[0])


def test_detects_constraint_penalty_written_into_the_objective():
    from qubo_auto.annealing.modeler import penalty_in_objective, to_ir
    base = {"name": "t", "sense": "minimize", "variable_names": ["a", "b", "c"], "constant": 0.0,
            "constraints": [{"name": "pick2", "kind": "linear", "coeffs": {"a": 1, "b": 1, "c": 1}, "sense": "eq", "rhs": 2}]}
    lam = 50.0     # lambda*(a+b+c-2)^2 expanded: 2*lambda on every pair
    doubled = to_ir({**base, "linear": {"a": 1.0, "b": 2.0, "c": 3.0},
                     "quadratic": [["a", "b", 2 * lam], ["a", "c", 2 * lam], ["b", "c", 2 * lam]]}, "x")
    clean = to_ir({**base, "linear": {"a": 1.0, "b": 2.0, "c": 3.0}, "quadratic": [["a", "b", 4.0]]}, "x")
    assert penalty_in_objective(doubled) == ["pick2"] and penalty_in_objective(clean) == []


def test_sortable_model_is_sent_back_unless_declared_separable():
    from qubo_auto.annealing.modeler import load_module, validate
    code = '''import numpy as np
DATA_SCHEMA = {"s": "scores"}
def make_example_data(size, seed):
    return {"s": list(range(1, size + 1))}
def build_ir(data):
    n = len(data["s"])
    return {"name": "t", "sense": "maximize", "variable_names": [f"x{i}" for i in range(n)],
            "linear": {f"x{i}": float(data["s"][i]) for i in range(n)}, "quadratic": [], "constant": 0.0,
            "constraints": [{"name": "k", "kind": "linear", "coeffs": {f"x{i}": 1 for i in range(n)}, "sense": "eq", "rhs": 2}]}
def describe_solution(data, x):
    return {}
def visualize(data, x):
    return {"type": "schedule", "title": "t", "rows": ["x"], "cols": [str(i) for i in range(len(x))], "cells": [[str(v) for v in x]]}
'''
    errs, _ = validate(load_module(code))
    assert any("solvable by sorting" in e for e in errs)                           # agent must re-model ...
    errs, rep = validate(load_module(code + 'MODEL_CHOICE = "separable: scores are independent by definition"\n'))
    assert errs == [] and rep["warnings"] and "separable" in rep["model_choice"]    # ... unless it says why


def test_llm_client_retries_without_thinking_when_budget_is_exhausted(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from qubo_auto import llm_client as L
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "m")
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            thinking = kw["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            content, reason = ("", "length") if thinking else ("42", "stop")
            usage = NS(total_tokens=10, cost=0.0, model_dump=lambda: {"completion_tokens": 5})
            return NS(choices=[NS(message=NS(content=content), finish_reason=reason)], usage=usage)

    c = L.LLMClient(transcript_dir=tmp_path, run_id="t", use_cache=False)
    fake = NS(chat=NS(completions=FakeCompletions()), models=NS(list=lambda: NS(data=[NS(id="m", max_model_len=4096, model_extra={})])))
    c._clients["local"] = fake
    assert c.call("m", "question", step_name="s") == "42"
    assert [k["extra_body"]["chat_template_kwargs"]["enable_thinking"] for k in calls] == [True, False]
    names = sorted(p.name.split("_1")[0] for p in (tmp_path / "t").glob("*.json"))
    assert names == ["s_exhausted", "s_nothink"]                 # both attempts are on record
    import pytest
    with pytest.raises(L.PromptTooLong):
        c.call("m", "x" * 20000, step_name="big")                 # ~5.7k tokens > 4096: refused before sending


def test_figures_are_specified_by_the_model_and_drawn_by_code(tmp_path):
    from qubo_auto.annealing.modeler import check_visualize, figure
    from qubo_auto.annealing.render import check_spec
    data, x = {"p": [1, 2]}, [1, 0]
    good = {"visualize": lambda d, x: {"type": "map", "points": [{"x": 0, "y": 0, "group": "selected"}, {"x": 1, "y": 1}]}}
    assert check_visualize(good, data, x, rep := {}) == [] and rep["figure_type"] == "map"
    assert figure(good, data, x, tmp_path / "f.png").exists()
    assert "define visualize" in check_visualize({}, data, x, {})[0]                       # missing -> sent back
    bad = {"visualize": lambda d, x: {"type": "routes", "nodes": [{"x": 0, "y": 0}], "routes": [[0, 5]]}}
    assert "valid node indices" in check_visualize(bad, data, x, {})[0]
    assert check_spec({"type": "pie"})                                                       # unknown type rejected


def test_model_retries_repair_previous_code_and_figure_is_a_separate_step():
    from qubo_auto.annealing import modeler as M
    from test_schema import MODEL
    good = MODEL.replace("%EXTRA%", "")                          # a valid sets/params model without a figure function
    broken = good.replace("def build_ir(data):", "def build_ir(data):\n    raise ValueError('oops')")
    vis = ('def visualize(data, x):\n    return {"type": "map", "points": [{"x": float(i), "y": 0.0, '
           '"group": "selected" if b else "candidate"} for i, b in enumerate(x)]}\n')
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append((step_name, prompt))
            if step_name == "visualizer":
                return f"```python\n{vis}```"
            return f"```python\n{broken if len([p for p in prompts if p[0] == 'model']) == 1 else good}```"

    code, ns, errors, report = M.build_model("{}", "[]", Fake(), "m")
    assert errors == [] and "visualize" in ns and "def visualize" in code
    model_prompts = [p for s, p in prompts if s == "model"]
    assert "YOUR PREVIOUS MODULE" in model_prompts[1] and "raise ValueError('oops')" in model_prompts[1]
    assert "visualize" not in model_prompts[0].split("CONFIRMED SPECIFICATION")[0]   # model step no longer asked to draw


def test_datakit_reads_taiwan_style_files_and_profiles_them(tmp_path):
    import json
    import pandas as pd
    from qubo_auto.annealing.datakit import load_tables, profiles
    (tmp_path / "village.csv").write_bytes("里名,人口數,緯度,經度\n大安里,5123,25.03,121.54\n龍泉里,8800,25.02,121.53\n".encode("cp950"))
    (tmp_path / "stations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "A站"}, "geometry": {"type": "Point", "coordinates": [121.5, 25.0]}},
        {"type": "Feature", "properties": {"name": "B區"}, "geometry": {"type": "Polygon",
         "coordinates": [[[121.0, 25.0], [121.2, 25.0], [121.2, 25.2], [121.0, 25.2]]]}}]}), encoding="utf-8")
    pd.DataFrame({"x": [1, 2]}).to_excel(tmp_path / "t.xlsx", index=False)
    t = load_tables([tmp_path / "village.csv", tmp_path / "stations.geojson", tmp_path / "t.xlsx"])
    assert list(t["village"].columns) == ["里名", "人口數", "緯度", "經度"] and t["village"]["人口數"].sum() == 13923
    assert abs(t["stations"]["lon"][1] - 121.1) < 1e-9 and t["stations"]["name"][0] == "A站"     # polygon -> centroid
    text = profiles(t)
    assert "'人口數' (int64): min 5123" in text and "TABLE 't'" in text and len(text) < 12000


def test_adapter_checks_types_on_full_data_and_repairs(tmp_path):
    import pandas as pd
    from qubo_auto.annealing.adapter import build_adapter
    from qubo_auto.annealing.modeler import load_module
    model = load_module(_PARKS)                                    # needs population (list of int), cost, budget
    tables = {"parks": pd.DataFrame({"pop": [300, 500, 700, 900], "cost": [1, 2, 2, 3]})}
    wrong = "def load_data(tables):\n    t = tables['parks']\n    return {'population': int(t['pop'].sum()), 'cost': t['cost'].tolist(), 'budget': 4}\n"
    right = ("ADAPTER_NOTES = 'pop -> population'\ndef load_data(tables):\n    t = tables['parks']\n"
             "    return {'population': t['pop'].tolist(), 'cost': t['cost'].tolist(), 'budget': 4, 'too_close': [[0, 1]]}\n")
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append(prompt)
            return f"```python\n{wrong if len(prompts) == 1 else right}```"

    code, data, errors, report = build_adapter(model, tables, "parks", Fake(), "m")
    assert errors == [] and data["population"] == [300, 500, 700, 900] and report["n_variables"] == 4
    assert "'population' is a number but the model expects a list" in prompts[1]        # type check fed back
    assert "YOUR PREVIOUS CODE" in prompts[1] and report["notes"] == "pop -> population"
    assert "300" not in prompts[0].split("THE USER'S TABLES")[0]                           # data only via profiles


def test_hard_constraints_from_the_spec_must_appear_in_the_model():
    import json
    from qubo_auto.annealing.modeler import missing_hard_constraints
    spec = json.dumps({"constraints": [{"name": "capacity", "strength": "hard"}, {"name": "pref", "strength": "soft"}]})
    assert "capacity" in missing_hard_constraints(spec, {"constraints": []})[0]
    assert missing_hard_constraints(spec, {"constraints": ["capacity (linear/le)"]}) == []
    assert missing_hard_constraints(json.dumps({"constraints": []}), {"constraints": []}) == []


def test_graph_and_selection_figures(tmp_path):
    from qubo_auto.annealing.render import check_spec, render
    g = {"type": "graph", "nodes": [{"label": "a", "group": "A"}, {"label": "b", "group": "B"}, {"label": "c"}],
         "edges": [[0, 1], [1, 2, 3.0]], "highlight": [[0, 1]]}
    s = {"type": "selection", "items": [{"label": "x", "value": 3, "selected": True}, {"label": "y", "value": 1}]}
    assert check_spec(g) == [] and check_spec(s) == []
    assert render(g, tmp_path / "g.png").exists() and render(s, tmp_path / "s.png").exists()
    assert check_spec({"type": "graph", "nodes": [{}], "edges": [[0, 5]]})          # bad edge index
    assert check_spec({"type": "selection", "items": [{"label": "x"}]})              # value missing


def test_sortable_check_does_not_flag_overlapping_constraints():
    from qubo_auto.annealing.modeler import to_ir, trivially_separable
    names = [f"x_{i}_{j}" for i in range(3) for j in range(3)]
    rows = [{"name": f"row{i}", "kind": "linear", "coeffs": {f"x_{i}_{j}": 1 for j in range(3)}, "sense": "eq", "rhs": 1} for i in range(3)]
    cols = [{"name": f"col{j}", "kind": "linear", "coeffs": {f"x_{i}_{j}": 1 for i in range(3)}, "sense": "eq", "rhs": 1} for j in range(3)]
    base = {"name": "a", "sense": "minimize", "variable_names": names, "linear": {n: float(k) for k, n in enumerate(names)},
            "quadratic": [], "constant": 0.0}
    assert not trivially_separable(to_ir({**base, "constraints": rows + cols}, "x"))   # assignment: not sortable
    assert trivially_separable(to_ir({**base, "constraints": rows}, "x"))             # disjoint "choose 1" groups: sortable


# ---------------------------------------------------------------- preprocessing toolkit (real-data situations)
def test_prep_text_numbers_dates():
    import pandas as pd
    from qubo_auto.annealing import prep
    assert prep.normalize_text("　台北市 （大安區）") == "臺北市 (大安區)"
    nums = prep.to_number(pd.Series(["1,234", "NT$300", "12.5%", "(50)", "-", "…", "abc", "１２３"]))
    assert nums.tolist()[:4] == [1234.0, 300.0, 12.5, -50.0] and nums.isna().tolist()[4:7] == [True, True, True] and nums[7] == 123
    d = prep.roc_to_date(pd.Series(["113年7月31日", "113/07/31", "1130731", "2024-07-31", "-"]))
    assert all(str(x.date()) == "2024-07-31" for x in d[:4]) and pd.isna(d[4])


def test_prep_table_levels_totals_and_periods():
    import pandas as pd
    from qubo_auto.annealing import prep
    t = pd.DataFrame({"年": [114, 115, 115, 115, 115, 115], "區域別": ["總計", "總計", "大安區", "龍坡里", "龍泉里", "萬華區"],
                      "人口": [1, 100, 60, 30, 30, 40]})
    t = prep.latest_period(t, ["年"])
    t = prep.attach_parent(t, "區域別", new="區")
    v = prep.keep_level(t, "區域別", "里")
    assert v["區域別"].tolist() == ["龍坡里", "龍泉里"] and v["區"].tolist() == ["大安區", "大安區"]
    assert v["人口"].sum() == 60                       # the city total row ("總計", no suffix) is NOT a village
    feed = pd.DataFrame({"sno": range(30), "mday": [f"2026-10-12 10:{i:02d}:00" for i in range(30)]})
    log = prep.PrepLog()
    prep.latest_period(feed, "mday", log=log)                     # each row's own update time, not a period
    assert any("own update time" in f for f in log.flags)
    import pytest
    with pytest.raises(ValueError, match="BEFORE keep_level"):           # parents already filtered away: no 區
        prep.attach_parent(prep.keep_level(t, "區域別", "里"), "區域別", new="區")
    assert prep.drop_total_rows(pd.DataFrame({"name": ["A", "合計", "B"], "v": [1, 3, 2]}))["name"].tolist() == ["A", "B"]
    assert len(prep.find_records({"success": True, "result": {"records": [{"a": 1}, {"a": 2}]}})) == 2


def test_prep_header_row_detection():
    import pandas as pd
    from qubo_auto.annealing import prep
    raw = pd.DataFrame([["臺北市人口統計表", None, None], ["資料來源：民政局", None, None],
                        ["區", "里", "人口"], ["大安區", "龍坡里", 3000], ["大安區", "龍泉里", 4000]])
    row = prep.find_header_row(raw)
    assert row == 2 and list(prep.apply_header(raw, row).columns) == ["區", "里", "人口"]


def test_prep_geo_crs_and_swaps():
    import pandas as pd
    from qubo_auto.annealing import prep
    df = pd.DataFrame({"x": [302000.0], "y": [2770000.0]})
    assert prep.detect_crs(df["x"], df["y"]) == "EPSG:3826"
    w = prep.to_wgs84(df, "x", "y")
    assert abs(w["lon"][0] - 121.515) < 0.01 and abs(w["lat"][0] - 25.037) < 0.01        # Taipei Main Station area
    uk = pd.DataFrame({"x": [406612.4], "y": [424748.3]})                        # British National Grid metres
    assert prep.detect_crs(uk["x"], uk["y"], "EPSG:27700") == "EPSG:27700"        # a plain code given as prj
    assert abs(prep.to_wgs84(uk, "x", "y", prj="EPSG:27700")["lat"][0] - 53.72) < 0.01
    import pytest
    with pytest.raises(ValueError, match="EPSG:27700"):                           # no guessing web mercator
        prep.to_wgs84(uk, "x", "y")
    with pytest.raises(ValueError, match="statistical-area"):                     # no network call is made
        prep.geocode(["Camden 001A", "Camden 002B", "Islington 003C"])
    square = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
                                                 [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]]]}
    assert prep.points_in_polygon([1.5, 0.7, 3.0], [1.5, 0.7, 1.0], square).tolist() == [True, False, False]  # hole
    from qubo_auto.annealing.prep import situations
    coded = pd.DataFrame({"區域代碼": [63000010, 63000030, 63000030], "緯度": [25.05, 25.03, 25.02],
                          "經度": [121.55, 121.54, 121.53]})
    assert [x.tools for x in situations.detect({"aed": coded})] == [("in_area",)]
    assert situations.forbidden_patterns("df = df.drop_duplicates(subset=['緯度', '經度'])")      # one building, many venues
    assert not situations.forbidden_patterns("df = df.drop_duplicates(subset=['名稱', 'lat', 'lon'])")
    same = pd.DataFrame({"name": ["忠孝新生站", "忠孝新生站 ", "中山里", "中山里"],
                         "lat": [25.0420, 25.0421, 25.05, 25.10], "lon": [121.5330, 121.5331, 121.52, 121.55]})
    assert prep.merge_same_place(same, "name", "lat", "lon")["lat"].tolist() == [25.0420, 25.05, 25.10]
    pts = pd.DataFrame({"lat": [25.03 + i * 1e-3 for i in range(40)] + [3.8], "lon": [121.5] * 40 + [3.7]})
    assert len(prep.drop_stray_points(pts, "lat", "lon")) == 40                   # one stray point removed
    with pytest.raises(ValueError, match="not a few typos"):
        prep.drop_stray_points(pd.concat([pts] + [pts.tail(1)] * 5), "lat", "lon")
    s = prep.fix_swapped(pd.DataFrame({"lat": [121.5, 25.0], "lon": [25.0, 121.5]}), bbox=(21.8, 119.3, 25.4, 122.1))
    assert s["lat"].tolist() == [25.0, 25.0]
    d = prep.haversine_matrix([25.0, 25.0], [121.5, 121.51])
    assert 990 < d[0, 1] < 1020                          # 0.01 degree of longitude at 25N is about 1 km


def test_prep_join_variants_and_reduction():
    import pandas as pd
    from qubo_auto.annealing import prep
    left = pd.DataFrame({"區": ["萬華區", "大安區", "大安區"], "里": ["糖蔀里", "龍坡里", "不存在里"]})
    right = pd.DataFrame({"區": ["萬華區", "大安區"], "里": ["糖廍里", "龍坡里"], "人口": [3000, 5000]})
    log = prep.PrepLog()
    j = prep.match(left, right, ["區", "里"], ["區", "里"], log=log)
    assert j["人口"].tolist()[:2] == [3000, 5000] and pd.isna(j["人口"][2])
    assert "模糊對上 1" in log.steps[0]["note"] and "對不上 1" in log.steps[0]["note"]
    pts = pd.DataFrame({"lat": [25.0 + i * 0.001 for i in range(20)] + [25.1 + i * 0.001 for i in range(20)],
                        "lon": [121.5] * 40, "pop": [10] * 40})
    r = prep.reduce_points(pts, 2, weight="pop")
    assert len(r) == 2 and r["pop"].sum() == 400 and sorted(r["cluster_size"]) == [20, 20]


def test_adapter_must_use_required_tools_and_cannot_write_sources():
    import pandas as pd
    from qubo_auto.annealing.adapter import build_adapter
    from qubo_auto.annealing.modeler import load_module
    model = load_module(_PARKS)
    pop = pd.DataFrame({"區域別": ["總計", "大安區", "龍坡里", "龍泉里"] * 2, "性別": ["計"] * 4 + ["男"] * 4,
                        "人口": [600, 600, 300, 300, 290, 290, 140, 150]})
    unused = pd.DataFrame({"年份": [114, 115], "v": [1, 2]})          # stacked periods, but never read
    by_hand = ("def load_data(tables, log):\n    t = tables['pop']\n    t = t[(t['性別'] == '計') & t['區域別'].str.endswith('里')]\n"
               "    log.source('made up', 'x', 'y', 'CC BY 4.0')\n"
               "    return {'population': t['人口'].tolist(), 'cost': [1] * len(t), 'budget': 2, 'too_close': [[0, 1]]}\n")
    with_tools = ("def load_data(tables, log):\n    t = prep.keep_category(tables['pop'], '性別', '計', log=log)\n"
                  "    t = prep.keep_level(t, '區域別', '里', log=log)\n"
                  "    return {'population': t['人口'].tolist(), 'cost': [1] * len(t), 'budget': 2, 'too_close': [[0, 1]]}\n")
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append(prompt)
            return f"```python\n{by_hand if len(prompts) == 1 else with_tools}```"

    code, data, errors, report = build_adapter(model, {"pop": pop, "other": unused}, "parks", Fake(), "m",
                                               sources=[{"title": "人口統計", "url": "https://data.gov.tw/x", "license": "政府資料開放授權條款"}])
    assert errors == [] and data["population"] == [300, 300] and report["tools_used"] == ["keep_category", "keep_level"]
    feedback = prompts[1]
    assert "REQUIRED TOOL NOT USED" in feedback and "prep.keep_category" in feedback and "prep.keep_level" in feedback
    assert "FORBIDDEN: writing source records" in feedback
    assert "latest_period" not in feedback.split("PREVIOUS ATTEMPT FAILED")[1]     # the unused table imposes nothing
    assert "CC BY" not in report["prep_log"] and "政府資料開放授權條款" in report["prep_log"]   # sources come from the system


def test_join_on_non_unique_names_is_flagged():
    import pandas as pd
    from qubo_auto.annealing import prep
    villages = pd.DataFrame({"區": ["大安區", "中正區"], "里": ["中央里", "中央里"], "x": [1, 2]})
    pop = pd.DataFrame({"區": ["大安區", "中正區"], "里": ["中央里", "中央里"], "人口": [100, 200]})
    log = prep.PrepLog()
    j0 = prep.match(villages, pop, "里", "里", log=log)        # name only, but both carry 區: the tool adds it itself
    assert not log.flags and j0["人口"].tolist() == [100, 200] and "自動加上上層鍵" in log.steps[-1]["note"]
    log = prep.PrepLog()
    prep.match(villages[["里", "x"]], pop, "里", "里", log=log)   # one side has no parent level: flagged
    assert log.flags and "add the parent level" in log.flags[0]
    log2 = prep.PrepLog()
    j = prep.match(villages, pop, ["區", "里"], ["區", "里"], log=log2)   # with the district: unique
    assert not log2.flags and j["人口"].tolist() == [100, 200]


def test_join_codes_exactly_and_flag_poor_matches():
    import pandas as pd
    from qubo_auto.annealing import prep
    a = pd.DataFrame({"code": ["63000010002", "63000010003", "63000010004"]})
    b = pd.DataFrame({"code2": ["6300500020", "63000010003"], "v": [1, 2]})
    log = prep.PrepLog()
    j = prep.match(a, b, "code", "code2", log=log)
    assert j["v"].isna().tolist() == [True, False, True]            # no fuzzy "near" codes
    assert any("unmatched" in f for f in log.flags)                  # 2 of 3 unmatched -> must be fixed
    j2 = prep.match(a, b, left_keys="code", right_keys="code2")      # guessed keyword names are accepted
    assert len(j2) == 3


def test_within_radius_keeps_points_near_one_centre_and_flags_empty_results():
    import pandas as pd
    from qubo_auto.annealing import prep
    df = pd.DataFrame({"lat": [25.0330, 25.0350, 25.0600, None], "lon": [121.5436, 121.5436, 121.5436, 121.5]})
    log = prep.PrepLog()
    kept = prep.within_radius(df, "lat", "lon", (25.0330, 121.5436), 500, log=log)
    assert len(kept) == 2 and kept["dist_m"].max() < 300 and "1 列沒有座標" in log.steps[0]["note"]
    assert abs(prep.distance_to([25.0350], [121.5436], 25.0330, 121.5436)[0] - 222.4) < 1
    prep.within_radius(df, "lon", "lat", (25.0330, 121.5436), 500, log=log)          # swapped columns
    assert any("kept NOTHING" in f for f in log.flags)


def test_looping_answers_are_detected():
    from qubo_auto.annealing.modeler import looped
    assert looped("x = 1\n" + "    # Let's use the constraint: u[i] - u[j] <= N - 1\n" * 50)
    assert not looped("\n".join(f"    c{i} = data['value'][{i}] * 2  # line {i}" for i in range(200)))


def test_facility_names_ending_in_district_words_are_not_admin_levels():
    import pandas as pd
    from qubo_auto.annealing import prep
    from qubo_auto.annealing.prep import situations
    aed = pd.DataFrame({"場所名稱": ["國立臺灣師範大學校本部", "大安森林公園管理處", "安和里民活動中心", "大安區公所",
                                     "台北101購物中心", "遠東百貨", "捷運大安站", "仁愛醫院"]})
    assert not [s for s in situations.detect({"aed": aed}) if s.tools == ("keep_level",)]
    pop = pd.DataFrame({"區域別": ["總計", "松山區", "莊敬里", "東榮里", "信義區", "三犁里"]})
    assert [s for s in situations.detect({"pop": pop}) if s.tools == ("keep_level",)]
    log = prep.PrepLog()
    prep.keep_level(pd.concat([aed] * 3), "場所名稱", "里", log=log)
    assert any("does not look like administrative-unit names" in f for f in log.flags)


def test_only_schema_mismatches_send_data_failures_back_to_the_model():
    from qubo_auto.annealing.repair import schema_mismatch
    assert schema_mismatch(["missing DATA_SCHEMA keys ['lat']"])
    assert not schema_mismatch(["沒有可讀的真實資料；需要搜尋其他來源或上傳檔案"])
    assert not schema_mismatch(["RuntimeError: KeyError: \"Coordinate columns '緯度' missing\""])


def test_ambiguous_join_says_how_to_recover_a_missing_parent_level():
    import pandas as pd
    from qubo_auto.annealing import prep
    pop = pd.DataFrame({"里": ["中央里", "中央里", "東榮里"], "人口": [1, 2, 3]})
    geo = pd.DataFrame({"TNAME": ["松山區", "中正區", "松山區"], "VNAME": ["中央里", "中央里", "東榮里"]})
    log = prep.PrepLog()
    prep.match(pop, geo, "里", "VNAME", log=log)
    flag = next(f for f in log.flags if "ambiguous" in f)
    assert "right ['TNAME']" in flag and "prep.attach_parent" in flag and "BEFORE removing" in flag


def test_latest_yearly_file_of_a_dataset_is_kept():
    from qubo_auto.annealing.opendata import latest_per_dataset, period_of
    assert period_of("110年每月各里人口數") == 2021 and period_of("112年迄今-每月各里人口數") == 2023.5
    assert period_of("Population mid-2022 estimates") == 2022 and period_of("里界圖") is None
    page = "https://data.gov.tw/dataset/136896"
    cands = [{"title": f"人口 — {t}", "page": page, "relevant": True, "detail": ""}
             for t in ("110年每月各里人口數", "111年每月各里人口數", "112年迄今-每月各里人口數")]
    cands.append({"title": "臺北市里界圖", "page": "https://data.gov.tw/dataset/121163", "relevant": True, "detail": ""})
    latest_per_dataset(cands)
    assert [c["relevant"] for c in cands] == [False, False, True, True]


def test_join_adds_the_parent_level_itself_when_both_tables_have_one():
    import pandas as pd
    from qubo_auto.annealing import prep
    pop = pd.DataFrame({"區": ["松山區", "中正區", "松山區"], "里": ["中央里", "中央里", "東榮里"], "人口": [1, 2, 3]})
    geo = pd.DataFrame({"TNAME": ["中正區", "松山區", "松山區"], "VNAME": ["中央里", "中央里", "東榮里"], "x": [10, 20, 30]})
    log = prep.PrepLog()
    j = prep.match(pop, geo, "里", "VNAME", log=log)
    assert j["x"].tolist() == [20, 10, 30]                                   # each 中央里 found in its own district
    assert not any("ambiguous" in f for f in log.flags) and "自動加上上層鍵" in log.steps[-1]["note"]


def test_json_cells_with_lists_become_text(tmp_path):
    import json
    from qubo_auto.annealing.datakit import load_tables
    from qubo_auto.annealing.prep import situations
    f = tmp_path / "parking.json"
    f.write_text(json.dumps([{"name": "A", "entrances": [{"x": 1}, {"x": 2}], "lat": 25.0},
                             {"name": "B", "entrances": [], "lat": 25.1}]), encoding="utf-8")
    t = load_tables([f])["parking"]
    assert isinstance(t["entrances"][0], str) and json.loads(t["entrances"][0])[1]["x"] == 2
    situations.detect({"parking": t})                                       # no crash on former list cells


def test_nested_json_records_and_zero_coordinates(tmp_path):
    import json
    from qubo_auto.annealing.datakit import load_tables
    from qubo_auto.annealing import schema
    f = tmp_path / "parks.json"
    f.write_text(json.dumps({"data": {"UPDATETIME": "x", "park": [{"id": "A", "tw97x": "302864.7"}, {"id": "B", "tw97x": "303000.1"}]}}))
    t = load_tables([f])
    assert list(t) == ["parks.data.park"] and len(t["parks.data.park"]) == 2
    ds = {"sets": {"p": {"meaning": "lot", "source": "public"}},
          "params": {"lat": {"meaning": "", "index": ["p"], "type": "number", "source": "public", "unit": "deg_lat"},
                     "lon": {"meaning": "", "index": ["p"], "type": "number", "source": "public", "unit": "deg_lon"}}}
    errs = schema.data_errors(ds, {"p": ["A", "B"], "lat": [25.0, 0], "lon": [121.5, 0]})
    assert any("(0, 0)" in e and "prep.to_wgs84" in e for e in errs)


def test_projected_coordinates_stored_as_text_are_detected():
    import pandas as pd
    from qubo_auto.annealing.prep import situations
    df = pd.DataFrame({"name": ["A", "B", "C"], "tw97x": ["302864.78", "303000.10", "301500.55"],
                       "tw97y": ["2771988.95", "2770001.20", "2769500.10"]})
    assert any(s.tools == ("to_wgs84",) for s in situations.detect({"parking": df}))


def test_loader_handles_html_pages_broken_rows_and_title_lines(tmp_path):
    import pytest
    from qubo_auto.annealing.datakit import load_tables
    page = tmp_path / "points.csv"
    page.write_text("<!DOCTYPE html>\n<html><head><title>Export</title></head></html>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="web page"):
        load_tables([page])
    broken = tmp_path / "chargers.csv"
    # like a real Taichung file: the last row ends inside a quoted field
    broken.write_text('"id","name","lat"\n"1","A","24.1"\n"2","B","24.2"\n"3","C","24.3', encoding="utf-8")
    t = load_tables([broken])["chargers"]
    assert list(t["name"]) == ["A", "B", "C"] and t["lat"].dtype.kind == "f" and "tolerant" in t.attrs["load_warnings"][0]
    torn = tmp_path / "torn.csv"
    torn.write_text('id,name,lat\n1,A,24.1\n2,B,24.2\n3,C\n4,D,24.4\n"5","E","24.5', encoding="utf-8")
    t = load_tables([torn])["torn"]
    assert list(t["name"]) == ["A", "B", "D", "E"] and "1 malformed row(s) skipped" in t.attrs["load_warnings"][0]
    titled = tmp_path / "budget.csv"
    titled.write_text("City budget 2025\nSource: Treasury\n\nagency,amount\nParks,10\nRoads,20\nLibraries,5\n", encoding="utf-8")
    t = load_tables([titled])["budget"]
    assert list(t.columns) == ["agency", "amount"] and t["amount"].sum() == 35


def test_official_sources_win_over_crowd_sourced_points_for_the_same_data():
    from qubo_auto.annealing.opendata import official_over_crowd
    cands = [{"id": "gov-1", "source": "政府資料開放平台", "relevant": True, "feeds": "sites, lat, lon"},
             {"id": "osm-aed", "source": "OpenStreetMap", "osm": {}, "relevant": True, "feeds": "sites, lat, lon"},
             {"id": "osm-parks", "source": "OpenStreetMap", "osm": {}, "relevant": True, "feeds": "parks"}]
    official_over_crowd(cands)
    assert [c["relevant"] for c in cands] == [True, False, True]       # OSM kept only for what officials lack
    alone = [{"id": "osm-x", "source": "OpenStreetMap", "osm": {}, "relevant": True, "feeds": "sites"}]
    official_over_crowd(alone)
    assert alone[0]["relevant"]                                          # no official source: OSM is used


def test_latest_period_accepts_a_single_column_name():
    import pandas as pd
    from qubo_auto.annealing import prep
    df = pd.DataFrame({"年份": [113, 114, 114], "v": [1, 2, 3]})
    assert prep.latest_period(df, "年份")["v"].tolist() == [2, 3]


def test_data_scope_rules_lead_the_specification_for_the_data_agents():
    from qubo_auto.annealing.clarify import DraftSpec, spec_for_data
    d = DraftSpec(problem_summary="AED teams", data_scope=["only 大安區", "exclude 學校、大型集會場所 and 其他",
                                                          "one place per name even with several AEDs"])
    text = spec_for_data(d)
    assert text.index("DATA SCOPE") < text.index("{") and "exclude 學校" in text and "one place per name" in text


def test_slack_encoding_divides_by_the_gcd_and_stays_exact():
    # costs that are multiples of 1,000 (NYC budget): 20 slack bits became 10, feasibility unchanged
    from qubo_auto.annealing.encodings import LinearConstraint, enc_binary_slack, verify_encoding
    c = LinearConstraint(name="budget", coeffs={0: 200000, 1: 150000, 2: 575000, 3: 450000, 4: 60000},
                         sense="le", rhs=1000000)
    enc = enc_binary_slack(c, next_aux=5)
    assert len(enc.aux) == 8                                     # gcd 5,000: rhs 1,000,000 -> 200
    rep = verify_encoding(c, enc, n_decision=5)
    assert rep["checked"] and rep["passed"], rep
    odd = LinearConstraint(name="b", coeffs={0: 4, 1: 6, 2: 10}, sense="le", rhs=13)   # gcd 2, rhs -> 6
    rep = verify_encoding(odd, enc_binary_slack(odd, next_aux=3), n_decision=3)
    assert rep["passed"], rep


def test_reasoning_loop_detector():
    from qubo_auto.llm_client import looping
    block = "    *   `S0` at pos 0.\n    *   `S1` at pos 1.\n    *   `S2` at pos 2.\n    *   This is a closed loop.\n\n"
    assert looping("Plan: positions x[i,t].\n" + block * 6)
    assert not looping("Plan: positions x[i,t].\n" + block * 2 + "Now the distance terms between t and t+1.\n")


def test_reasoning_loop_detector_catches_long_passages():
    import random
    from qubo_auto.llm_client import looping
    random.seed(1)
    words = "station tour distance check position closed loop verify sum each once node edge order".split()
    long = " ".join(random.choice(words) + str(random.randint(0, 99)) for _ in range(250))    # ~2,000 characters
    assert looping("start " + long * 3) and not looping("start " + long * 2)
