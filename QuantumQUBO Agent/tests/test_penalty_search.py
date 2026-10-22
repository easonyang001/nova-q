"""Penalty integration contracts: compilation, original feasibility, budget and resume."""
import copy
import itertools
import json

import pytest

from qubo_auto.annealing.backends import SimulatedAnnealingBackend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.encodings import LinearConstraint, ProblemIR, compile_problem
from qubo_auto.annealing.loop import build_context, run_loop
from qubo_auto.annealing.parameterized import ParameterizedProblem
from qubo_auto.annealing.problem import qubo_energy


def ir():
    return ProblemIR(name="penalty_test", instance_id="penalty_test", n_variables=3,
        sense="maximize", linear={0: 5, 1: 3, 2: 2},
        constraints=[LinearConstraint(name="capacity", coeffs={0: 2, 1: 1, 2: 1}, sense="le", rhs=2)])


def cfg(calls=4):
    return {"loop": {"max_solver_calls": calls, "patience": 100, "gap_target": -1},
        "param_caps": {"sa": {"num_reads": 10, "num_sweeps": 50}},
        "penalty_search": {"compiler": "pyqubo", "grid_points": 3, "cem_population": 2}}


@pytest.mark.parametrize("encoding", ["binary_slack", "unary_slack", "unbalanced"])
def test_pyqubo_energy_and_variable_indices_match_native(encoding):
    native = compile_problem(ir(), {"capacity": encoding})
    compiled = ParameterizedProblem(native, {"capacity"} if encoding == "unbalanced" else ())
    model = compiled._model
    for weight in [0.5, 10, 51.25]:
        q, off = compiled.to_qubo({"capacity": weight})
        nq, noff = native.to_qubo({"capacity": weight})
        for bits in itertools.product([0, 1], repeat=native.n_variables):
            assert qubo_energy(q, off, bits) == pytest.approx(qubo_energy(nq, noff, bits))
        assert compiled._model is model  # placeholder updates do not compile again


def test_pyqubo_slack_diagnostics_do_not_replace_original_constraints():
    native = compile_problem(ir())
    compiled = ParameterizedProblem(native)
    x = [0] * native.n_variables  # feasible decision, incorrectly zero-filled slack
    assert compiled.is_feasible(x)
    diagnostic = compiled.penalty_diagnostics(x, {"capacity": 10})
    assert diagnostic["capacity"]["penalty_satisfied"] is False


def test_approximate_penalties_not_reported_as_broken_constraints():
    native = compile_problem(ir(), {"capacity": "unbalanced"})
    compiled = ParameterizedProblem(native, {"capacity"})
    assert compiled.penalty_diagnostics([0, 0, 0], {"capacity": 10}) == {}


@pytest.mark.parametrize("name", ["penalty_grid", "penalty_cem", "penalty_tpe", "penalty_sequential"])
def test_resume_matches_uninterrupted_search_and_accounts_calls(tmp_path, name):
    checkpoint = tmp_path / f"{name}.json"
    first = run_loop(None, SimulatedAnnealingBackend(), make_policy(name, seed=7), cfg(2),
        ir=ir(), seed=7, root=tmp_path / "partial", checkpoint=checkpoint, verbose=False)
    assert first.solver_calls == 2
    resumed = run_loop(None, SimulatedAnnealingBackend(), make_policy(name, seed=7), cfg(4),
        ir=ir(), seed=7, root=tmp_path / "resume", checkpoint=checkpoint, resume=True, verbose=False)
    whole = run_loop(None, SimulatedAnnealingBackend(), make_policy(name, seed=7), cfg(4),
        ir=ir(), seed=7, root=tmp_path / "whole", verbose=False)
    assert resumed.solver_calls == whole.solver_calls
    assert [r.config for r in resumed.rounds] == [r.config for r in whole.rounds]
    assert resumed.best_solution == whole.best_solution
    assert resumed.best_objective == whole.best_objective
    saved = json.loads(checkpoint.read_text())
    assert not saved["pending_call"]
    assert saved["feasible_penalty_weights"]
    assert "encoded_penalty_diagnostics" in resumed.rounds[0].feedback


def test_checkpoint_rejects_changed_model_and_indeterminate_call(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_cem"), cfg(1),
        ir=ir(), root=tmp_path, checkpoint=checkpoint, verbose=False)
    changed = ir()
    changed.linear[0] = 99
    with pytest.raises(ValueError, match="different problem"):
        run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_cem"), cfg(3),
            ir=changed, root=tmp_path, checkpoint=checkpoint, resume=True, verbose=False)
    saved = json.loads(checkpoint.read_text())
    saved["pending_call"] = True
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="unfinished solver call"):
        run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_cem"), cfg(3),
            ir=ir(), root=tmp_path, checkpoint=checkpoint, resume=True, verbose=False)


def test_feasible_stop_survives_resume_without_extra_call(tmp_path):
    settings = cfg(5)
    settings["loop"]["stop_after_feasible_count"] = 1
    checkpoint = tmp_path / "checkpoint.json"
    first = run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_sequential"), settings,
        ir=ir(), root=tmp_path, checkpoint=checkpoint, verbose=False)
    assert first.stop_reason == "feasible_count"
    settings = copy.deepcopy(settings)
    settings["loop"]["max_solver_calls"] = 10
    resumed = run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_sequential"), settings,
        ir=ir(), root=tmp_path, checkpoint=checkpoint, resume=True, verbose=False)
    assert resumed.solver_calls == first.solver_calls == 1


def test_lazy_grid_stops_at_enumeration_bound(tmp_path):
    settings = cfg(10)
    settings["penalty_search"]["max_grid_candidates"] = 2
    result = run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_grid"), settings,
        ir=ir(), root=tmp_path, verbose=False)
    assert result.solver_calls == 2
    assert result.stop_reason == "policy_stop:penalty_grid"


def test_compiler_cache_and_unconstrained_problem(tmp_path):
    problem_ir = ir()
    context = build_context(None, SimulatedAnnealingBackend(), cfg(), ir=problem_ir)
    assert context.problem_for(context.default_config) is context.problem
    problem_ir.constraints = []
    result = run_loop(None, SimulatedAnnealingBackend(), make_policy("penalty_tpe"), cfg(),
        ir=problem_ir, root=tmp_path, verbose=False)
    assert result.solver_calls == 1
    assert result.best_solution is not None
    assert result.stop_reason == "policy_stop:penalty_tpe"


def test_constant_only_pyqubo_keeps_unused_variables():
    problem_ir = ir()
    problem_ir.linear = {}
    problem_ir.constraints = []
    compiled = ParameterizedProblem(compile_problem(problem_ir))
    qubo, offset = compiled.to_qubo()
    assert qubo == {}
    assert compiled.n_variables == 3
    assert compiled.penalty_diagnostics([0, 0, 0], {}) == {}
