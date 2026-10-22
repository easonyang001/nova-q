"""Closed-loop workflow engine (spec §8): Configure → Guardrail → Execute → Feedback → Diagnose → Refine.

The engine is deterministic. A *policy* (an agent team or a baseline) only proposes configurations;
the engine validates them, runs them, decides when to stop and keeps the best result. This is where
the §8.3 guardrails live, so every policy gets exactly the same protection and the same budget.
"""
from __future__ import annotations
import json
import hashlib
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .backends import Backend
from .feedback import Feedback, extract_feedback, log_run
from .guardrail import GuardrailError, HardwareBudget, ParamSpec, capped_specs, check_params
from .problem import ProblemInstance
from .tools import penalty_lower_bound, qubo_summary, scale_for_backend

# Parameters that trade solver effort for quality, per backend (used by baselines and fallback).
EFFORT_PARAMS = {"sa": ["num_sweeps", "num_reads"], "dwave_qpu": ["annealing_time", "num_reads"],
                 "fujitsu_da": ["time_limit_sec"], "cga": ["timeout"], "dwave_hybrid": ["time_limit"]}
SCALE_RANGE = (1e-3, 1e3)


class RunConfig(BaseModel):
    penalty_weights: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    scale: float = 1.0
    # constraint -> {"encoding": name, "params": {...}}; only used when the problem is given as an IR (spec v0.2)
    encodings: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def key(self) -> str:
        """Identity used for duplicate detection (the RNG seed does not make a config new)."""
        params = {k: v for k, v in self.params.items() if k != "seed"}
        return json.dumps({"p": self.penalty_weights, "q": params, "s": self.scale, "e": self.encodings},
                          sort_keys=True)


class Proposal(BaseModel):
    config: RunConfig
    rationale: str = ""
    diagnosis: dict[str, Any] | None = None     # diagnosis of the previous round that led to this proposal
    stop: bool = False
    source: str = ""
    retrieved: list[str] = Field(default_factory=list)   # RAG entry ids shown to the policy
    cited: list[str] = Field(default_factory=list)       # RAG entry ids the policy said it relied on


class RoundRecord(BaseModel):
    index: int
    config: RunConfig
    source: str
    rationale: str = ""
    diagnosis: dict[str, Any] | None = None
    retrieved: list[str] = Field(default_factory=list)
    cited: list[str] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)   # guardrail rejections before this config
    feedback: dict[str, Any] | None = None
    solution: list[int] | None = None
    error: str = ""
    improved: bool = False


class LoopResult(BaseModel):
    problem: str
    instance_id: str
    backend: str
    simulated: bool
    policy: str
    rounds: list[RoundRecord]
    best_round: int | None
    best_objective: float | None
    best_feasibility_rate: float | None
    best_solution: list[int] | None
    reference_objective: float | None
    best_gap: float | None
    stop_reason: str
    solver_calls: int
    llm_tokens: int = 0
    wall_time_s: float = 0.0


@dataclass
class LoopContext:
    """Everything a policy may look at. Built once per loop."""
    problem: ProblemInstance
    backend: Backend
    specs: dict[str, ParamSpec]                 # backend ranges with config caps applied
    summary: dict[str, Any]                     # qubo_summary features (also the RAG key)
    lower_bounds: dict[str, float]              # penalty floors (already multiplied by floor factor)
    default_config: RunConfig
    cfg: dict[str, Any]
    effort_params: list[str] = field(default_factory=list)
    rag: Any = None                             # rag.KnowledgeBase or None
    seed: int = 0
    ir: Any = None                              # encodings.ProblemIR when encodings may change (spec v0.2)

    def legal_actions(self) -> dict[str, Any]:
        """Machine-readable action space shown to agents (spec §9.2, stage-1 categories)."""
        actions = {
            "set_penalty": {name: {"must_exceed": lb, "note": "floor applies to exact encodings only"}
                            for name, lb in self.lower_bounds.items()},
            "set_param": {k: {"type": s.type, "min": s.min, "max": s.max, "default": s.default,
                              "description": s.description} for k, s in self.specs.items() if k != "seed"},
            "set_scale": {"min": SCALE_RANGE[0], "max": SCALE_RANGE[1]},
            "revert_to_best": {},
        }
        if self.ir is not None:
            actions["request_reencoding"] = {"description": "ask the Encoding Agent to change constraint encodings "
                                                            "(use for formulation-level problems such as precision_loss "
                                                            "or penalties that dominate the landscape)"}
        return actions

    def problem_for(self, config: "RunConfig") -> ProblemInstance:
        """The ProblemInstance for a config: recompiled from the IR when encodings are part of the config."""
        if self.ir is None:
            return self.problem
        from .encodings import compile_problem
        key = json.dumps(config.encodings, sort_keys=True)
        cache = self.__dict__.setdefault("_compiled", {})
        if key not in cache:
            from .parameterized import parameterize
            cache[key] = parameterize(compile_problem(self.ir, {k: v["encoding"] for k, v in config.encodings.items()},
                                         {k: v.get("params", {}) for k, v in config.encodings.items()}),
                                         self.cfg, config.encodings)
        return cache[key]


def summarize_feedback(fb: Feedback) -> dict[str, Any]:
    """Compact feedback for history, agents and the experience store."""
    def r(v):
        return None if v is None else round(v, 4)
    keep = ("solver", "chain_strength", "chain_break_fraction_mean", "samples_with_chain_breaks",
            "max_chain_length", "num_qubits_used", "result_status")
    return {
        "feasibility_rate": r(fb.feasibility_rate),
        "constraint_violation_rates": {k: r(v) for k, v in fb.constraint_violation_rates.items()},
        "best_feasible_objective": fb.best_feasible_objective,
        "mean_feasible_objective": r(fb.mean_feasible_objective),
        "reference_objective": fb.reference_objective,
        "gap": r(fb.gap),
        "n_samples": fb.n_samples,
        "n_unique": fb.n_unique,
        "diversity": r(fb.diversity),
        "best_energy_diagnostic_only": r(fb.best_energy),
        "solver_time_s": r(fb.solver_time_s),
        "wall_time_s": r(fb.wall_time_s),
        "backend_specific": {k: fb.backend_specific[k] for k in keep if k in fb.backend_specific},
    }


def score(feedback: dict[str, Any] | None, sense: str) -> tuple:
    """Original-problem ranking (§9.1): any feasible > none, then objective, then feasibility rate."""
    if not feedback:
        return (0, -math.inf, -math.inf)
    obj = feedback["best_feasible_objective"]
    if obj is None:
        return (0, -math.inf, feedback["feasibility_rate"])
    return (1, obj if sense == "maximize" else -obj, feedback["feasibility_rate"])


def build_context(problem: ProblemInstance | None, backend: Backend, cfg: dict[str, Any], rag=None,
                  seed: int = 0, ir=None, encodings: dict[str, dict[str, Any]] | None = None) -> LoopContext:
    default_encodings: dict[str, dict[str, Any]] = {}
    if ir is not None:
        from .encodings import compile_problem, options_for
        default_encodings = {c.name: {"encoding": next(iter(options_for(c))), "params": {}} for c in ir.constraints}
        default_encodings.update({k: {"encoding": v["encoding"], "params": dict(v.get("params") or {})}
                                  for k, v in (encodings or {}).items()})      # user-fixed starting encodings
        problem = compile_problem(ir, {k: v["encoding"] for k, v in default_encodings.items()},
                                  {k: v["params"] for k, v in default_encodings.items()})
    specs = capped_specs(backend.param_specs(), cfg.get("param_caps", {}).get(backend.name))
    from .parameterized import parameterize
    problem = parameterize(problem, cfg, default_encodings)
    factor = cfg["loop"].get("penalty_floor_factor", 1.0)
    floors = {k: v * factor for k, v in penalty_lower_bound(problem).items()}
    # defaults must sit strictly above the floor (a weight equal to it can tie feasible and infeasible states)
    weights = {k: v if v > floors.get(k, 0.0) else (floors[k] * 1.25 if floors[k] > 0 else 1.0)
               for k, v in problem.default_penalty_weights.items()}
    summary = qubo_summary(problem)
    if cfg.get("workflow_baseline"):
        summary["baseline"] = cfg["workflow_baseline"]
    if ir is not None:
        summary["encodings"] = {k: v["encoding"] for k, v in default_encodings.items()}
    context = LoopContext(problem=problem, backend=backend, specs=specs, summary=summary,
                       lower_bounds=floors,
                       default_config=RunConfig(penalty_weights=weights, encodings=default_encodings),
                       cfg=cfg, effort_params=EFFORT_PARAMS.get(backend.name, []), rag=rag, seed=seed, ir=ir)
    if ir is not None:
        context.__dict__["_compiled"] = {json.dumps(default_encodings, sort_keys=True): problem}
    return context


def validate_config(ctx: LoopContext, config: RunConfig, history: list[RoundRecord],
                    allow_repeats: bool) -> tuple[RunConfig | None, list[str]]:
    """§8.3 guardrails. Returns (clean config with defaults filled, []) or (None, errors)."""
    errors: list[str] = []
    weights = {**ctx.default_config.penalty_weights, **config.penalty_weights}
    encodings = {**ctx.default_config.encodings, **config.encodings}
    approximate: set[str] = set()
    if ctx.ir is not None:
        from .encodings import ENCODING_PARAMS, options_for
        by_name = {c.name: c for c in ctx.ir.constraints}
        for name, e in config.encodings.items():
            if name not in by_name:
                errors.append(f"unknown constraint '{name}' in encodings (allowed: {sorted(by_name)})")
                continue
            opts = options_for(by_name[name])
            if e.get("encoding") not in opts:
                errors.append(f"encoding '{e.get('encoding')}' not available for '{name}' (choose from {sorted(opts)})")
                continue
            spec = ENCODING_PARAMS.get(e["encoding"], {})
            for pname, pval in (e.get("params") or {}).items():
                if pname not in spec:
                    errors.append(f"unknown parameter '{pname}' for encoding '{e['encoding']}' (allowed: {sorted(spec)})")
                elif not isinstance(pval, (int, float)) or not spec[pname][0] <= pval <= spec[pname][1]:
                    errors.append(f"'{name}.{pname}'={pval} outside {list(spec[pname])}")
        approximate = {k for k, e in encodings.items() if e.get("encoding") == "unbalanced"}
        from .encodings import unbalanced_max_reward
        obj_range = sum(abs(v) for v in ctx.problem.objective_qubo.values())
        for name in sorted(approximate & set(by_name)):
            p = encodings[name].get("params") or {}
            reward = weights.get(name, 0.0) * unbalanced_max_reward(by_name[name], **{k: p[k] for k in ("l1", "l2") if k in p})
            if reward > obj_range:
                errors.append(f"unbalanced '{name}' (weight {weights.get(name, 0.0):.4g}, params {p}) rewards feasible "
                              f"points by up to {reward:.4g}, more than the whole objective range {obj_range:.4g}: the "
                              f"solver would be paid to stay far from the bound (e.g. select nothing). Lower l1 "
                              f"relative to l2, lower the weight, or use a slack encoding")
    for name, w in config.penalty_weights.items():
        if name not in ctx.problem.penalty_qubos:
            errors.append(f"unknown penalty '{name}' (allowed: {sorted(ctx.problem.penalty_qubos)})")
    for name, w in weights.items():
        if not math.isfinite(w):
            errors.append(f"penalty '{name}' must be finite")
            continue
        if name in approximate:                         # no optimum-preservation guarantee to protect: only w > 0
            if w <= 0:
                errors.append(f"penalty '{name}'={w} must be > 0")
            continue
        if name in ctx.lower_bounds and w <= ctx.lower_bounds[name]:
            errors.append(f"penalty '{name}'={w} must be > lower bound {ctx.lower_bounds[name]:.4g} "
                          f"(at or below it the problem's optimum may change)")
    if not SCALE_RANGE[0] <= config.scale <= SCALE_RANGE[1]:
        errors.append(f"scale {config.scale} outside {list(SCALE_RANGE)}")
    params = dict(config.params)
    if "seed" in ctx.specs and "seed" not in params:
        params["seed"] = ctx.seed + len(history)          # reproducible simulator runs
    try:
        params = check_params(params, ctx.specs)
    except GuardrailError as e:
        errors.extend(e.errors)
    if errors:
        return None, errors
    clean = RunConfig(penalty_weights={k: float(v) for k, v in weights.items()}, params=params, scale=config.scale,
                      encodings={k: {"encoding": v["encoding"], "params": dict(v.get("params") or {})}
                                 for k, v in encodings.items()})
    if not allow_repeats and any(r.config.key() == clean.key() for r in history):
        return None, ["duplicate configuration: this exact config was already run; propose a different one"]
    return clean, []


def run_loop(problem: ProblemInstance | None, backend: Backend, policy, cfg: dict[str, Any], *,
             simulate: bool = False, budget: HardwareBudget | None = None, rag=None, seed: int = 0,
             root: Path = Path("."), verbose: bool = True, ir=None, log=None,
             encodings: dict[str, dict[str, Any]] | None = None,
             checkpoint: Path | None = None, resume: bool = False) -> LoopResult:
    from .baselines import RuleBasedPolicy          # deterministic fallback (§2 principle 7)

    t0 = time.time()
    lc = cfg["loop"]
    if lc.get("max_solver_calls", 10) < 1:
        raise ValueError("max_solver_calls must be positive")
    target = lc.get("stop_after_feasible_count")
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 1):
        raise ValueError("stop_after_feasible_count must be a positive integer or null")
    ctx = build_context(problem, backend, cfg, rag=rag, seed=seed, ir=ir, encodings=encodings)
    problem = ctx.problem
    fallback = RuleBasedPolicy()
    real = not simulate and not backend.is_simulator
    history: list[RoundRecord] = []
    best: RoundRecord | None = None
    no_improve = consecutive_errors = calls = 0
    stop_reason = ""
    resumed_stop = ""
    # A resume must use the same IR, solver, policy, seed and settings. Increasing
    # only the total call budget is allowed. Native callable constraints cannot
    # be fingerprinted reliably, so resuming them is deliberately unsupported.
    identity_cfg = json.loads(json.dumps(cfg))
    identity_cfg.pop("logging", None)
    identity_cfg.get("loop", {}).pop("max_solver_calls", None)
    identity = {"ir": ir.model_dump(mode="json") if ir is not None else None,
                "objective": sorted((i, j, v) for (i, j), v in problem.objective_qubo.items()),
                "penalties": {k: [sorted((i, j, v) for (i, j), v in q.items()), off]
                              for k, (q, off) in problem.penalty_qubos.items()},
                "offset": problem.objective_offset, "n": problem.n_variables,
                "config": identity_cfg, "backend": backend.name, "real": real,
                "specs": {k: v.model_dump() for k, v in ctx.specs.items()},
                "seed": seed, "policy": policy.name, "encodings": encodings}
    identity["policy_seed"] = getattr(policy, "seed", seed)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        if checkpoint is None or ir is None or not getattr(policy, "resumable", False):
            raise ValueError("resume requires a checkpoint, an IR, and a history-based penalty policy")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("checkpoint belongs to a different problem, backend, policy or settings")
        if saved.get("pending_call"):
            raise ValueError("checkpoint has an unfinished solver call; inspect backend logs before retrying")
        history = [RoundRecord.model_validate(r) for r in saved["rounds"]]
        calls = saved["solver_calls"]
        no_improve, consecutive_errors = saved["no_improve"], saved["consecutive_errors"]
        resumed_stop = saved.get("stop_reason", "")
        best = max((r for r in history if r.feedback), key=lambda r: score(r.feedback, problem.sense), default=None)

    def save_checkpoint(pending=False, final_reason=""):
        if checkpoint is not None:
            from .artifacts import atomic_write
            atomic_write(checkpoint, json.dumps({"schema": 1, "fingerprint": fingerprint,
                "context": {"ir": identity["ir"], "config": cfg, "backend": backend.name,
                            "real": real, "seed": seed, "policy": policy.name,
                            "policy_seed": identity["policy_seed"], "encodings": encodings},
                "rounds": [r.model_dump() for r in history], "solver_calls": calls,
                "no_improve": no_improve, "consecutive_errors": consecutive_errors,
                "pending_call": pending, "stop_reason": final_reason,
                "feasible_penalty_weights": [r.config.penalty_weights for r in history
                     if r.feedback and r.feedback["best_feasible_objective"] is not None]}, allow_nan=False))

    def say(msg: str) -> None:
        if log is not None:                          # e.g. the UI streams progress
            log(msg)
        elif verbose:
            print(msg, flush=True)

    def ask(fn, *args) -> Proposal | None:
        try:
            return fn(*args)
        except Exception as e:                       # LLM / policy failure must not crash the loop
            say(f"  [policy error] {type(e).__name__}: {e}")
            return None

    proposal = (ask(policy.next, ctx, history) if history else ask(policy.initial, ctx)) or Proposal(config=ctx.default_config, source="fallback:default",
                                                    rationale="policy failed at cold start")
    while True:
        if resumed_stop and resumed_stop != "budget":
            stop_reason = resumed_stop
            break
        if calls >= lc.get("max_solver_calls", 10):
            stop_reason = "budget"
            break
        if proposal.stop:
            stop_reason = f"policy_stop:{proposal.source}"
            break
        # ---- guardrail with bounded retries, then deterministic fallback -------------------------
        rejected: list[dict[str, Any]] = []
        config, errors = validate_config(ctx, proposal.config, history, policy.allow_repeats)
        attempt = 0
        while config is None and attempt < lc.get("guardrail_retries", 2):
            rejected.append({"source": proposal.source, "config": proposal.config.model_dump(), "errors": errors})
            say(f"  [guardrail] rejected {proposal.source}: {errors}")
            repaired = ask(policy.repair, ctx, history, proposal, errors)
            if repaired is None:
                break
            proposal, attempt = repaired, attempt + 1
            config, errors = validate_config(ctx, proposal.config, history, policy.allow_repeats)
        if config is None:
            rejected.append({"source": proposal.source, "config": proposal.config.model_dump(), "errors": errors})
            fb_prop = fallback.next(ctx, history) if history else Proposal(config=ctx.default_config)
            config, errors = validate_config(ctx, fb_prop.config, history, False)
            if config is None:
                stop_reason = "no_valid_proposal"
                break
            proposal = fb_prop.model_copy(update={"source": "fallback:rule", "diagnosis": proposal.diagnosis})
            say(f"  [fallback] using rule-based proposal")

        # ---- execute -----------------------------------------------------------------------------
        rec = RoundRecord(index=len(history), config=config, source=proposal.source, rationale=proposal.rationale,
                          diagnosis=proposal.diagnosis, retrieved=proposal.retrieved, cited=proposal.cited,
                          rejected=rejected)
        try:
            if real:
                if budget is None:
                    raise GuardrailError(["real hardware call without a hardware budget"])
                budget.check(backend.name)
            prob = ctx.problem_for(config)
            qubo, offset = prob.to_qubo(config.penalty_weights)
            if config.scale != 1.0:
                qubo, offset, _ = scale_for_backend(qubo, offset, config.scale)
            calls += 1
            save_checkpoint(pending=True)
            result = backend.run(qubo, offset, prob.n_variables, config.params, simulate=simulate)
            if real:
                budget.record(backend.name, result.solver_time_s, note=problem.instance_id)
            fb = extract_feedback(prob, result)
            log_run(root / cfg.get("logging", {}).get("backend_log", "results/annealing_runs.jsonl"), result, fb,
                    extra={"problem": problem.name, "instance_id": problem.instance_id, "policy": policy.name,
                           "round": rec.index, "penalty_weights": config.penalty_weights, "scale": config.scale,
                           "encodings": config.encodings})
            rec.feedback, rec.solution = summarize_feedback(fb), fb.best_feasible_solution
            if hasattr(prob, "penalty_diagnostics") and result.samples:
                diagnostic_x = fb.best_feasible_solution or result.samples[min(range(len(result.energies)), key=result.energies.__getitem__)]
                rec.feedback["encoded_penalty_diagnostics"] = prob.penalty_diagnostics(diagnostic_x, config.penalty_weights)
            consecutive_errors = 0
        except GuardrailError as e:
            rec.error = "; ".join(e.errors)
            history.append(rec)
            save_checkpoint(final_reason="hardware_budget")
            stop_reason = "hardware_budget"
            say(f"  [budget] {rec.error}")
            break
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
            consecutive_errors += 1
            say(f"  [backend error] {rec.error}\n{traceback.format_exc(limit=2)}")

        if rec.feedback and (best is None or score(rec.feedback, problem.sense) > score(best.feedback, problem.sense)):
            rec.improved, best, no_improve = True, rec, 0
        else:
            no_improve += 1
        history.append(rec)
        save_checkpoint()
        if rec.feedback:
            f = rec.feedback
            say(f"  round {rec.index} [{rec.source}] feas={f['feasibility_rate']:.2f} "
                f"best={f['best_feasible_objective']} gap={f['gap']} "
                f"w={ {k: round(v, 2) for k, v in config.penalty_weights.items()} } "
                f"p={ {k: v for k, v in config.params.items() if k != 'seed'} }"
                f"{' e=' + str({k: (v['encoding'], v.get('params') or None) for k, v in config.encodings.items()}) if config.encodings else ''}"
                f"{'  *best' if rec.improved else ''}")

        # ---- stop conditions (§8.2) ----------------------------------------------------------------
        if consecutive_errors >= 2:
            stop_reason = "backend_error"
            break
        feasible_target = lc.get("stop_after_feasible_count")
        if feasible_target is not None and sum(bool(r.feedback and r.feedback["best_feasible_objective"] is not None)
                                               for r in history) >= feasible_target:
            stop_reason = "feasible_count"
            break
        if best and best.feedback["gap"] is not None and best.feedback["gap"] <= lc.get("gap_target", 0.0):
            stop_reason = "gap_target"
            break
        if no_improve >= lc.get("patience", 3):
            stop_reason = "no_improvement"
            break
        if calls >= lc.get("max_solver_calls", 10):
            stop_reason = "budget"
            break
        nxt = ask(policy.next, ctx, history)
        if nxt is None:
            nxt = fallback.next(ctx, history)
            nxt = nxt.model_copy(update={"source": "fallback:rule"})
        if nxt.stop:
            stop_reason = f"policy_stop:{nxt.source}"
            history[-1].diagnosis = history[-1].diagnosis or nxt.diagnosis
            break
        proposal = nxt

    save_checkpoint(final_reason=stop_reason)
    out = LoopResult(
        problem=problem.name, instance_id=problem.instance_id, backend=backend.name, simulated=not real,
        policy=policy.name, rounds=history, best_round=best.index if best else None,
        best_objective=best.feedback["best_feasible_objective"] if best else None,
        best_feasibility_rate=best.feedback["feasibility_rate"] if best else None,
        best_solution=best.solution if best else None, reference_objective=problem.reference_objective,
        best_gap=best.feedback["gap"] if best else None, stop_reason=stop_reason, solver_calls=calls,
        llm_tokens=getattr(policy, "tokens_used", 0), wall_time_s=time.time() - t0,
    )
    log_path = root / cfg.get("logging", {}).get("loop_log", "results/loop_runs.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(out.model_dump_json() + "\n")
    if rag is not None and cfg.get("rag", {}).get("write_experience", False):
        rag.write_experience(out, ctx.summary)
    return out
