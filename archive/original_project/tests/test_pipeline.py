"""Unit and integration tests for the QUBO pipeline (no LLM calls)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from qubo_auto.schemas import (
    Constraint, Objective, QUBOFormulation, QUBOTerm,
    PenaltyTerm, StructuredSpec, Variable, VerificationResult,
)
from qubo_auto.verification.brute_force import brute_force_qubo, validate_Q, is_symmetric
from qubo_auto.agents.coder_agent import _extract_code, compile_build_qubo
from qubo_auto.benchmarks_oracle import (
    max_cut_generator, max_cut_oracle,
    number_partition_generator, number_partition_oracle,
    knapsack_generator, knapsack_oracle,
)


# ---------------------------------------------------------------------------
# brute_force tests
# ---------------------------------------------------------------------------

class TestBruteForceQubo:
    def test_simple_diagonal(self):
        Q = np.array([[-1.0, 0.0], [0.0, -1.0]])
        x, e = brute_force_qubo(Q, 0.0)
        assert e == pytest.approx(-2.0)
        assert x[0] == pytest.approx(1.0)
        assert x[1] == pytest.approx(1.0)

    def test_penalty_prevents_both_on(self):
        # x0 + x1 = 1 penalty: (x0+x1-1)^2 = x0+x1 - 2*x0*x1
        Q = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x, e = brute_force_qubo(Q, -1.0)
        assert e == pytest.approx(-1.0)

    def test_empty(self):
        Q = np.zeros((0, 0))
        x, e = brute_force_qubo(Q, 5.0)
        assert e == pytest.approx(5.0)

    def test_single_variable(self):
        Q = np.array([[-3.0]])
        x, e = brute_force_qubo(Q, 0.0)
        assert e == pytest.approx(-3.0)
        assert x[0] == pytest.approx(1.0)

    def test_n_too_large_raises(self):
        Q = np.zeros((21, 21))
        with pytest.raises(ValueError, match="too large"):
            brute_force_qubo(Q, 0.0)


class TestValidateQ:
    def test_symmetric(self):
        Q = np.eye(3)
        assert validate_Q(Q, 3) == []

    def test_wrong_shape(self):
        Q = np.eye(3)
        errors = validate_Q(Q, 4)
        assert any("shape" in e for e in errors)

    def test_asymmetric(self):
        Q = np.array([[1.0, 2.0], [0.0, 1.0]])
        errors = validate_Q(Q, 2)
        assert any("symmetric" in e for e in errors)


# ---------------------------------------------------------------------------
# Coder extraction tests
# ---------------------------------------------------------------------------

class TestCodeExtraction:
    def test_extract_from_fence(self):
        raw = "```python\ndef build_qubo(instance):\n    pass\n```"
        code = _extract_code(raw)
        assert "def build_qubo" in code

    def test_extract_raises_on_missing(self):
        with pytest.raises(ValueError):
            _extract_code("No code here at all.")

    def test_compile_simple_function(self):
        code = (
            "import numpy as np\n"
            "def build_qubo(instance):\n"
            "    n = instance['n']\n"
            "    Q = np.eye(n) * -1\n"
            "    return Q, 0.0\n"
        )
        fn = compile_build_qubo(code)
        Q, offset = fn({"n": 3})
        assert Q.shape == (3, 3)
        assert offset == 0.0

    def test_compile_raises_on_missing_function(self):
        with pytest.raises(ValueError, match="build_qubo"):
            compile_build_qubo("x = 1")


# ---------------------------------------------------------------------------
# Benchmark oracle sanity checks
# ---------------------------------------------------------------------------

class TestMaxCutOracle:
    def test_all_same_group_zero_cut(self):
        instance = {"n_nodes": 3, "edges": [(0, 1, 2), (1, 2, 3)]}
        x = np.zeros(3)
        obj, feasible = max_cut_oracle(x, instance)
        # oracle returns -best_cut, so obj should be <= 0
        assert feasible is True

    def test_optimal_cut(self):
        instance = {"n_nodes": 2, "edges": [(0, 1, 5)]}
        x = np.array([0.0, 1.0])
        obj, feasible = max_cut_oracle(x, instance)
        assert feasible is True
        assert obj == pytest.approx(-5.0)

    def test_generator_produces_valid_instance(self):
        inst = max_cut_generator(0, 6)
        assert "n_nodes" in inst
        assert "edges" in inst
        assert inst["n_nodes"] >= 3


class TestNumberPartitionOracle:
    def test_perfect_partition(self):
        instance = {"numbers": [1, 1, 2, 2]}
        x = np.array([0.0, 0.0, 1.0, 1.0])
        obj, feasible = number_partition_oracle(x, instance)
        assert feasible is True
        assert obj == pytest.approx(0.0)


class TestKnapsackOracle:
    def test_empty_selection(self):
        instance = {"n_items": 3, "weights": [2, 3, 4], "values": [3, 4, 5], "capacity": 5}
        x = np.zeros(3)
        obj, feasible = knapsack_oracle(x, instance)
        assert feasible is True

    def test_overweight_infeasible(self):
        instance = {"n_items": 2, "weights": [3, 4], "values": [5, 6], "capacity": 5}
        x = np.array([1.0, 1.0])
        obj, feasible = knapsack_oracle(x, instance)
        assert feasible is False


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_structured_spec_round_trip(self):
        spec = StructuredSpec(
            problem_name="test",
            variables=[Variable(name="x0", index=0)],
            objective=Objective(direction="minimize", expression="x0"),
            constraints=[Constraint(name="c1", type="eq", expression="x0", rhs="1")],
            instance_parameters=["n"],
        )
        raw = spec.model_dump_json()
        recovered = StructuredSpec.model_validate_json(raw)
        assert recovered.problem_name == "test"

    def test_qubo_formulation_round_trip(self):
        f = QUBOFormulation(
            n_variables=2,
            objective_terms=[QUBOTerm(coefficient=-1.0, var_indices=[0])],
            penalty_terms=[
                PenaltyTerm(
                    constraint_name="c1",
                    penalty_weight=5.0,
                    terms=[QUBOTerm(coefficient=1.0, var_indices=[0, 1])],
                )
            ],
            penalty_justification="weight > max_obj_coeff",
            offset=0.0,
        )
        raw = f.model_dump_json()
        recovered = QUBOFormulation.model_validate_json(raw)
        assert recovered.n_variables == 2
