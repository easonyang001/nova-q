"""Problem instances for the execution-feedback loop.

The QUBO is kept as separate parts so refinement can only touch what it is allowed to:
    H(x) = objective(x) + sum_c  weight_c * penalty_c(x)
`objective` is the minimisation form of the original objective and is never modified;
each `penalty_c` has unit weight, is integer-valued and >= 0, and its minimum over any auxiliary
(slack) variables is 0 exactly when constraint c holds. Feasibility is always judged on the
original constraint, never on the penalty value.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Callable, Literal

QUBO = dict[tuple[int, int], float]


def add_qubo(target: QUBO, source: QUBO, scale: float = 1.0) -> QUBO:
    """target += scale * source, with keys normalised to (min, max)."""
    for (i, j), c in source.items():
        key = (min(i, j), max(i, j))
        target[key] = target.get(key, 0.0) + scale * float(c)
    return target


def qubo_energy(qubo: QUBO, offset: float, x: list[int]) -> float:
    return offset + sum(c * x[i] * x[j] for (i, j), c in qubo.items())


@dataclass
class ProblemInstance:
    name: str
    n_variables: int
    sense: Literal["minimize", "maximize"]
    objective: Callable[[list[int]], float]              # original objective, in its own sense
    objective_qubo: QUBO                                 # minimisation form (negated if maximize)
    constraints: dict[str, Callable[[list[int]], bool]] = field(default_factory=dict)  # name -> satisfied?
    penalty_qubos: dict[str, tuple[QUBO, float]] = field(default_factory=dict)         # name -> (unit QUBO, offset)
    default_penalty_weights: dict[str, float] = field(default_factory=dict)
    objective_offset: float = 0.0
    reference_objective: float | None = None             # known optimum of the original problem
    constraint_types: dict[str, str] = field(default_factory=dict)  # e.g. "cardinality", "pairwise_exclusion"
    instance_id: str = ""                                # unique per generated instance (RAG train/test split)

    def to_qubo(self, penalty_weights: dict[str, float] | None = None) -> tuple[QUBO, float]:
        weights = {**self.default_penalty_weights, **(penalty_weights or {})}
        unknown = set(weights) - set(self.penalty_qubos)
        if unknown:
            raise ValueError(f"Unknown penalty names: {sorted(unknown)}")
        qubo = add_qubo({}, self.objective_qubo)
        offset = self.objective_offset
        for name, (pq, p_off) in self.penalty_qubos.items():
            add_qubo(qubo, pq, weights[name])
            offset += weights[name] * p_off
        return {k: v for k, v in qubo.items() if v != 0.0}, offset

    def is_feasible(self, x: list[int]) -> bool:
        return all(check(x) for check in self.constraints.values())

    def brute_force_reference(self, max_n: int = 20) -> float | None:
        """Best feasible objective by enumeration (small instances only)."""
        if self.n_variables > max_n:
            return None
        values = [self.objective(list(x)) for x in itertools.product((0, 1), repeat=self.n_variables)
                  if self.is_feasible(list(x))]
        # (with slack variables this enumerates them too; the objective ignores them)
        if not values:
            return None
        return max(values) if self.sense == "maximize" else min(values)
