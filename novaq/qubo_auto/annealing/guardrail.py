"""Deterministic checks applied to every configuration before it reaches a backend."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class ParamSpec(BaseModel):
    type: Literal["int", "float"]
    min: float
    max: float
    default: float | int | None = None    # None = leave unset (backend decides)
    description: str = ""


class GuardrailError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def check_params(params: dict[str, Any], specs: dict[str, ParamSpec]) -> dict[str, Any]:
    """Return params with defaults filled in, or raise GuardrailError listing every problem.

    Rejects (never silently clips) unknown names, wrong types and out-of-range values, so the
    reason can be fed back to whoever proposed the configuration.
    """
    errors: list[str] = []
    for name in params:
        if name not in specs:
            errors.append(f"unknown parameter '{name}' (allowed: {sorted(specs)})")

    clean: dict[str, Any] = {}
    for name, spec in specs.items():
        value = params.get(name, spec.default)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"'{name}' must be {spec.type}, got {value!r}")
            continue
        if spec.type == "int":
            if float(value) != int(value):
                errors.append(f"'{name}' must be an integer, got {value!r}")
                continue
            value = int(value)
        else:
            value = float(value)
        if not spec.min <= value <= spec.max:
            errors.append(f"'{name}'={value} outside [{spec.min}, {spec.max}]")
            continue
        clean[name] = value

    if errors:
        raise GuardrailError(errors)
    return clean


def capped_specs(specs: dict[str, ParamSpec], caps: dict[str, float] | None) -> dict[str, ParamSpec]:
    """Apply config caps (tighter maxima) on top of a backend's own ranges."""
    out = dict(specs)
    for name, cap in (caps or {}).items():
        if name in out:
            spec = out[name]
            new_max = min(spec.max, float(cap))
            default = spec.default if spec.default is None or spec.default <= new_max else new_max
            out[name] = spec.model_copy(update={"max": new_max, "default": default})
    return out


class HardwareBudget:
    """Cumulative real-hardware usage, persisted in a JSONL ledger; checked before every real call."""

    def __init__(self, ledger: Path, limits: dict[str, dict[str, float]]):
        self.ledger = ledger
        self.limits = limits

    def usage(self, backend: str) -> dict[str, float]:
        calls, solver_s = 0, 0.0
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["backend"] == backend:
                    calls += 1
                    solver_s += rec.get("solver_time_s") or 0.0
        return {"calls": calls, "solver_time_s": solver_s}

    def check(self, backend: str) -> None:
        limit = self.limits.get(backend)
        if limit is None:
            raise GuardrailError([f"no hardware budget configured for '{backend}'"])
        used = self.usage(backend)
        errors = []
        if used["calls"] >= limit.get("max_calls", 0):
            errors.append(f"hardware budget: {backend} calls {used['calls']} >= max_calls {limit.get('max_calls', 0)}")
        if used["solver_time_s"] >= limit.get("max_solver_time_s", 0):
            errors.append(f"hardware budget: {backend} solver time {used['solver_time_s']:.3f}s "
                          f">= max_solver_time_s {limit.get('max_solver_time_s', 0)}")
        if errors:
            raise GuardrailError(errors)

    def record(self, backend: str, solver_time_s: float | None, note: str = "") -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": time.time(), "backend": backend,
                                "solver_time_s": solver_time_s, "note": note}) + "\n")
