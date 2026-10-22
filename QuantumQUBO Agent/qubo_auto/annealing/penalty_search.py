"""History-based penalty policies; shared backends, budget and original-problem ranking.

Grid/CEM design follows QHyper's hyperoptimizer; sequential updates follow
Kaiwu's penalty-method controller. These are adapters of the ideas, not SDK imports.
TPE directly uses Optuna. All policies reconstruct from saved history on resume.
"""
import itertools
import math

import numpy as np

from .baselines import Policy
from .loop import Proposal, score


def bounds(ctx):
    settings = ctx.cfg.get("penalty_search", {})
    multiplier = float(settings.get("max_multiplier", 16))
    if not math.isfinite(multiplier) or multiplier <= 1:
        raise ValueError("max_multiplier must be finite and > 1")
    result = {}
    for name, default in ctx.default_config.penalty_weights.items():
        low = max(ctx.lower_bounds[name] * 1.01, 0.1)
        result[name] = (low, max(low, default) * multiplier)
    return result


class PenaltySearchPolicy(Policy):
    resumable = True

    def __init__(self, seed=0):
        self.seed = seed

    def initial(self, ctx):
        if not ctx.default_config.penalty_weights:
            return self.proposal(ctx, {}, "unconstrained problem: solve once with backend defaults")
        return self.next(ctx, [])

    def proposal(self, ctx, weights, rationale):
        return Proposal(config=ctx.default_config.model_copy(update={"penalty_weights": weights}),
                        source=self.name, rationale=rationale)

    def exhausted(self, ctx):
        return Proposal(config=ctx.default_config, source=self.name, stop=True, rationale="search exhausted")


class GridPenaltyPolicy(PenaltySearchPolicy):
    name = "penalty_grid"

    def next(self, ctx, history):
        space = bounds(ctx)
        if not space:
            return self.exhausted(ctx)
        points = int(ctx.cfg.get("penalty_search", {}).get("grid_points", 3))
        if points < 2:
            raise ValueError("grid_points must be >= 2")
        # Lazy product avoids QHyper-style full meshgrid memory growth.
        tried = {tuple(r.config.penalty_weights[k] for k in space) for r in history}
        grid = itertools.product(*(np.geomspace(lo, hi, points) for lo, hi in space.values()))
        # Bound even proposal enumeration, not just hardware calls.
        limit = int(ctx.cfg.get("penalty_search", {}).get("max_grid_candidates", 4096))
        if limit < 1:
            raise ValueError("max_grid_candidates must be positive")
        for values in itertools.islice(grid, limit):
            if values not in tried:
                return self.proposal(ctx, dict(zip(space, map(float, values))), "bounded penalty grid")
        return self.exhausted(ctx)


class CEMPenaltyPolicy(PenaltySearchPolicy):
    name = "penalty_cem"

    def next(self, ctx, history):
        space = bounds(ctx)
        if not space:
            return self.exhausted(ctx)
        settings = ctx.cfg.get("penalty_search", {})
        batch = int(settings.get("cem_population", 4))
        frac = float(settings.get("cem_elite_fraction", 0.5))
        if batch < 2 or not 0 < frac <= 1:
            raise ValueError("CEM requires population >= 2 and 0 < elite_fraction <= 1")
        lo, hi = np.log(np.array(list(space.values()), dtype=float)).T
        mean, deviation = (lo + hi) / 2, (hi - lo) / 3
        completed = len(history) // batch * batch
        # Reconstruct distribution epoch by epoch; no opaque optimizer pickle.
        for start in range(0, completed, batch):
            population = history[start:start + batch]
            valid = [r for r in population if r.feedback and r.source == self.name]
            elite = sorted(valid, key=lambda r: score(r.feedback, ctx.problem.sense), reverse=True)
            elite = elite[:max(2, math.ceil(len(valid) * frac))]
            if elite:
                vectors = np.array([[math.log(r.config.penalty_weights[k]) for k in space] for r in elite])
                mean = vectors.mean(axis=0)
                deviation = np.maximum(vectors.std(axis=0), (hi - lo) * 0.05)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, len(history)]))
        tried = {tuple(r.config.penalty_weights[k] for k in space) for r in history}
        for _ in range(64):
            # Bounded rejection and a variance floor prevent degenerate CEM hangs.
            vector = rng.normal(mean, deviation)
            if np.all((vector >= lo) & (vector <= hi)):
                values = tuple(map(float, np.exp(vector)))
                if values not in tried:
                    return self.proposal(ctx, dict(zip(space, values)), f"CEM epoch {len(history) // batch}")
        values = np.exp(rng.uniform(lo, hi))
        return self.proposal(ctx, dict(zip(space, map(float, values))), "CEM bounded uniform fallback")


class TPEPenaltyPolicy(PenaltySearchPolicy):
    name = "penalty_tpe"

    def next(self, ctx, history):
        import optuna
        space = bounds(ctx)
        if not space:
            return self.exhausted(ctx)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=self.seed + len(history), n_startup_trials=2))
        distributions = {k: optuna.distributions.FloatDistribution(lo, hi, log=True)
                         for k, (lo, hi) in space.items()}
        for record in history:
            params = {k: record.config.penalty_weights[k] for k in space}
            if record.source != self.name or any(not lo <= params[k] <= hi for k, (lo, hi) in space.items()):
                continue
            feedback = record.feedback
            if feedback and feedback["best_feasible_objective"] is not None:
                objective = feedback["best_feasible_objective"]
                utility = objective if ctx.problem.sense == "maximize" else -objective
                # Disjoint score ranges guarantee feasible > infeasible at any objective scale.
                scale = max(1.0, abs(ctx.problem.objective_offset) +
                            sum(abs(v) for v in ctx.problem.objective_qubo.values()))
                value = 2 + math.atan(utility / scale) / math.pi
            else:
                rates = (feedback or {}).get("constraint_violation_rates", {})
                value = -sum(rates.values()) / max(1, len(rates)) if feedback else -2.0
            study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=value))
        trial = study.ask()
        weights = {k: trial.suggest_float(k, lo, hi, log=True) for k, (lo, hi) in space.items()}
        return self.proposal(ctx, weights, "Optuna TPE penalty search; two startup trials")


class SequentialPenaltyPolicy(PenaltySearchPolicy):
    name = "penalty_sequential"
    allow_repeats = True

    def next(self, ctx, history):
        if not history:
            return self.proposal(ctx, ctx.default_config.penalty_weights, "initial penalty-method solve")
        space = bounds(ctx)
        weights = dict(history[-1].config.penalty_weights)
        rates = (history[-1].feedback or {}).get("constraint_violation_rates", {})
        for name, rate in rates.items():
            if rate > 0:
                weights[name] = min(space[name][1], weights[name] * 2)
        return self.proposal(ctx, weights, "increase violated penalties; bounded by search limits")
