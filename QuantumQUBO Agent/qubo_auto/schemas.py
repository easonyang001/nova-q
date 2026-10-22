from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Variable(BaseModel):
    name: str
    type: Literal["binary"] = "binary"
    index: int | str  # int for concrete vars; str pattern like "i" or "0..N-1" for families
    description: str = ""


class Objective(BaseModel):
    direction: Literal["minimize", "maximize"]
    expression: str
    notes: str = ""


class Constraint(BaseModel):
    name: str
    type: Literal["eq", "leq", "geq"]
    expression: str
    rhs: str
    notes: str = ""


class StructuredSpec(BaseModel):
    problem_name: str
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint]
    domain_notes: str = ""
    instance_parameters: list[str]


class QUBOTerm(BaseModel):
    coefficient: float | str  # float for fixed values; str for symbolic (e.g. "4*a_i")
    var_indices: list[int] = Field(..., min_length=1, max_length=2)


class PenaltyTerm(BaseModel):
    constraint_name: str
    penalty_weight: float | str  # str allowed for symbolic weights
    terms: list[QUBOTerm]


class QUBOFormulation(BaseModel):
    n_variables: int
    variable_mapping: dict[str, int] = Field(default_factory=dict)
    objective_terms: list[QUBOTerm]
    penalty_terms: list[PenaltyTerm]
    penalty_justification: str
    offset: float | str = 0.0


class VerificationResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class TestInstanceResult(BaseModel):
    instance_id: int
    status: Literal["success", "coding_error", "formulation_error", "unverifiable"]
    error_message: str = ""
    qubo_optimum: float | None = None
    problem_optimum: float | None = None
    match: bool = False


class RunResult(BaseModel):
    run_id: str
    benchmark: str
    seed: int
    status: Literal["success", "failed", "partial", "unverifiable"]
    n_iterations: int = 0
    wall_time_s: float = 0.0
    models_used: dict[str, str] = Field(default_factory=dict)
    tokens_used: int = 0
    cost_used: float = 0.0
    verification: dict[str, int] = Field(default_factory=dict)
    error_trace: str = ""
    failure_modes: list[str] = Field(default_factory=list)
