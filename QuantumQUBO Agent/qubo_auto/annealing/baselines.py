"""Baseline policies (spec §10.3). They use the same engine, guardrails and budget as the agents."""
from __future__ import annotations
import math
from typing import Any

import numpy as np

from .loop import LoopContext, Proposal, RoundRecord, RunConfig, score


class Policy:
    name = "base"
    allow_repeats = False
    tokens_used = 0

    def initial(self, ctx: LoopContext) -> Proposal:
        raise NotImplementedError

    def next(self, ctx: LoopContext, history: list[RoundRecord]) -> Proposal:
        raise NotImplementedError

    def repair(self, ctx: LoopContext, history: list[RoundRecord], proposal: Proposal,
               errors: list[str]) -> Proposal | None:
        return None


class DefaultPolicy(Policy):
    """Baseline 1: backend defaults + formula penalty, never adjusted (re-runs use the budget)."""
    name = "default"
    allow_repeats = True

    def initial(self, ctx):
        return Proposal(config=ctx.default_config, source=self.name, rationale="defaults")

    def next(self, ctx, history):
        params = {k: v for k, v in history[0].config.params.items() if k != "seed"}
        return Proposal(config=history[0].config.model_copy(update={"params": params}),
                        source=self.name, rationale="unchanged rerun")


class RuleBasedPolicy(Policy):
    """Baseline 2 and the engine's fallback: violated constraint -> double its penalty;
    otherwise double the first effort parameter that is not yet at its cap."""
    name = "rule"

    def initial(self, ctx):
        return Proposal(config=ctx.default_config, source=self.name, rationale="defaults")

    def next(self, ctx, history):
        last = history[-1]
        if not last.feedback:                         # failed round: go back to the best config
            best = max(history, key=lambda r: score(r.feedback, ctx.problem.sense))
            return Proposal(config=best.config, source=self.name, rationale="revert after failed round")
        tried = {r.config.key() for r in history}
        viol = last.feedback["constraint_violation_rates"]
        if viol and max(viol.values()) > 0:
            name = max(viol, key=viol.get)
            weights = {**last.config.penalty_weights, name: last.config.penalty_weights[name] * 2}
            cand = last.config.model_copy(update={"penalty_weights": weights})
            if cand.key() not in tried:
                return Proposal(config=cand, source=self.name,
                                diagnosis={"labels": ["penalty_too_weak"], "evidence": f"{name} violated {viol[name]:.0%}"},
                                rationale=f"double penalty of most violated constraint '{name}'")
        gap = last.feedback["gap"]
        if gap is not None and gap <= ctx.cfg["loop"].get("gap_target", 0.0):
            return Proposal(config=last.config, stop=True, source=self.name,
                            diagnosis={"labels": ["converged"], "evidence": "gap target reached"})
        for p in ctx.effort_params:
            spec, cur = ctx.specs[p], last.config.params.get(p, ctx.specs[p].default)
            if cur is None or cur >= spec.max:
                continue
            new = min(spec.max, cur * 2)
            new = int(new) if spec.type == "int" else float(new)
            cand = last.config.model_copy(update={"params": {**last.config.params, p: new}})
            if cand.key() not in tried:
                return Proposal(config=cand, source=self.name,
                                diagnosis={"labels": ["insufficient_sampling"], "evidence": f"gap={gap}"},
                                rationale=f"double {p} ({cur} -> {new})")
        return Proposal(config=last.config, stop=True, source=self.name,
                        diagnosis={"labels": ["converged"], "evidence": "no rule applies; effort at caps"})


def _search_space(ctx: LoopContext) -> dict[str, tuple[float, float, bool, str]]:
    """name -> (low, high, log, kind). Penalties: floor..10x floor; effort params: default/4..cap."""
    space = {}
    for name, lb in ctx.lower_bounds.items():
        base = lb * 1.01 if lb > 0 else 0.1          # strictly above the floor
        space[f"penalty:{name}"] = (base, base * 10, True, "float")
    for p in ctx.effort_params:
        spec = ctx.specs[p]
        lo = max(spec.min, (spec.default or spec.min) / 4)
        space[f"param:{p}"] = (lo, spec.max, True, spec.type)
    return space


def _config_from(values: dict[str, float], ctx: LoopContext) -> RunConfig:
    weights = {k.split(":", 1)[1]: float(v) for k, v in values.items() if k.startswith("penalty:")}
    params = {}
    for k, v in values.items():
        if k.startswith("param:"):
            name = k.split(":", 1)[1]
            params[name] = int(round(v)) if ctx.specs[name].type == "int" else float(v)
    return RunConfig(penalty_weights={**ctx.default_config.penalty_weights, **weights}, params=params)


class RandomSearchPolicy(Policy):
    """Baseline 3: log-uniform random search over penalties and effort parameters."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _sample(self, ctx):
        values = {}
        for k, (lo, hi, log, kind) in _search_space(ctx).items():
            v = math.exp(self.rng.uniform(math.log(lo), math.log(hi))) if log else self.rng.uniform(lo, hi)
            values[k] = v
        return Proposal(config=_config_from(values, ctx), source=self.name, rationale="random sample")

    def initial(self, ctx):
        return self._sample(ctx)

    def next(self, ctx, history):
        return self._sample(ctx)

    def repair(self, ctx, history, proposal, errors):
        return self._sample(ctx)


class OptunaPolicy(Policy):
    """Baseline 4: Bayesian optimisation (Optuna TPE) over the same space. Needs `pip install optuna`."""
    name = "optuna"

    def __init__(self, seed: int = 0):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna = optuna
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.trial = None

    def _ask(self, ctx):
        self.trial = self.study.ask()
        values = {}
        for k, (lo, hi, log, kind) in _search_space(ctx).items():
            values[k] = (self.trial.suggest_int(k, int(math.ceil(lo)), int(hi), log=log) if kind == "int"
                         else self.trial.suggest_float(k, lo, hi, log=log))
        return Proposal(config=_config_from(values, ctx), source=self.name, rationale=f"TPE trial {self.trial.number}")

    def initial(self, ctx):
        return self._ask(ctx)

    def next(self, ctx, history):
        has_feasible, obj, feas = score(history[-1].feedback, ctx.problem.sense)
        value = obj if has_feasible else -1e12 + (0 if feas == -math.inf else feas)
        self.study.tell(self.trial, value)
        return self._ask(ctx)

    def repair(self, ctx, history, proposal, errors):
        self.study.tell(self.trial, state=self.optuna.trial.TrialState.FAIL)
        return self._ask(ctx)


def make_policy(name: str, seed: int = 0, **kwargs: Any) -> Policy:
    if name.startswith("penalty_"):
        from .penalty_search import GridPenaltyPolicy, CEMPenaltyPolicy, TPEPenaltyPolicy, SequentialPenaltyPolicy
        policies = {p.name: p for p in (GridPenaltyPolicy, CEMPenaltyPolicy, TPEPenaltyPolicy, SequentialPenaltyPolicy)}
        if name in policies:
            return policies[name](seed)
    if name == "default":
        return DefaultPolicy()
    if name == "rule":
        return RuleBasedPolicy()
    if name == "random":
        return RandomSearchPolicy(seed)
    if name == "optuna":
        return OptunaPolicy(seed)
    if name == "agent":
        from .agents import AgentPolicy
        return AgentPolicy(**kwargs)
    raise ValueError(f"unknown policy {name!r}")
