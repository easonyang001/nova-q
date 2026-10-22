"""LLM agents for the feedback loop (spec §5.2, §5.3): Configurator, Diagnoser, Refiner, or one single agent.

Agents only decide. They see a compact, role-specific context and must answer with JSON matching a
schema. Validation failures are retried once with the error; persistent failures raise, and the
engine falls back to the rule policy. The engine's guardrails still check every proposal.
"""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..llm_client import LLMClient
from .baselines import Policy
from .families import compress_messages, expand, families, group_rates, group_values, label
from .loop import LoopContext, Proposal, RoundRecord, RunConfig, score

_PROMPTS = Path(__file__).parent.parent / "prompts" / "annealing"

DiagnosisLabel = Literal["penalty_too_weak", "penalty_too_strong", "precision_loss", "chain_break",
                         "insufficient_sampling", "schedule_mismatch", "formulation_suspect", "converged"]


class ConfiguratorOut(BaseModel):
    penalty_weights: dict[str, float] = Field(default_factory=dict)
    params: dict[str, float] = Field(default_factory=dict)
    scale: float = 1.0
    rationale: str = ""
    used_knowledge: list[str] = Field(default_factory=list, description="ids of retrieved entries relied on")


class DiagnoserOut(BaseModel):
    labels: list[DiagnosisLabel] = Field(..., min_length=1)
    evidence: str
    suggested_direction: str = ""
    used_knowledge: list[str] = Field(default_factory=list)


class Action(BaseModel):
    type: Literal["set_penalty", "set_param", "set_scale", "revert_to_best", "request_reencoding"]
    target: str | None = None
    value: float | None = None


class RefinerOut(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    rationale: str = ""
    stop: bool = False
    used_knowledge: list[str] = Field(default_factory=list)


class EncodingChoice(BaseModel):
    encoding: str
    params: dict[str, float] = Field(default_factory=dict)


class EncodingOut(BaseModel):
    choices: dict[str, EncodingChoice]                   # constraint name -> encoding
    penalty_weights: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    used_knowledge: list[str] = Field(default_factory=list)


class SingleOut(BaseModel):
    diagnosis: DiagnoserOut
    actions: list[Action] = Field(default_factory=list)
    rationale: str = ""
    stop: bool = False
    used_knowledge: list[str] = Field(default_factory=list)


def _j(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)


def _ids(cited: list[str]) -> list[str]:
    """Normalise cited knowledge ids ('[tk-x]' -> 'tk-x'), keep order, drop duplicates."""
    return list(dict.fromkeys(c.strip().strip("[]") for c in cited if c.strip()))


def _fam(ctx: LoopContext) -> dict[str, list[str]]:
    """Constraint families shown to agents (one per constraint on small problems; see families.py)."""
    return families(ctx.problem.penalty_qubos.keys())


def _cfg_view(c: RunConfig, fam: dict[str, list[str]] | None = None) -> dict[str, Any]:
    weights = {k: round(v, 4) for k, v in c.penalty_weights.items()}
    view = {"penalty_weights": group_values(weights, fam) if fam else weights,
            "params": {k: v for k, v in c.params.items() if k != "seed"}, "scale": c.scale}
    if c.encodings:
        if fam:
            view["encodings"] = group_values({k: e["encoding"] + (f" {e['params']}" if e.get("params") else "")
                                              for k, e in c.encodings.items()}, fam)
        else:
            view["encodings"] = c.encodings
    return view


def _metrics(fb: dict[str, Any] | None, fam: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
    if not fb:
        return None
    out = {k: fb[k] for k in ("feasibility_rate", "constraint_violation_rates", "best_feasible_objective",
                                "mean_feasible_objective", "gap", "diversity", "solver_time_s", "backend_specific")}
    if fam:
        out["constraint_violation_rates"] = group_rates(fb["constraint_violation_rates"], fam)
    return out


def _summary_view(ctx: LoopContext, fam: dict[str, list[str]]) -> dict[str, Any]:
    view = ctx.summary | {"constraints": group_values(ctx.summary["constraints"], fam),
                          "reference_objective": ctx.problem.reference_objective}
    if "encodings" in view:
        view["encodings"] = group_values(view["encodings"], fam)
    return view


def _actions_view(ctx: LoopContext, fam: dict[str, list[str]]) -> dict[str, Any]:
    acts = ctx.legal_actions()
    if any(len(m) > 1 for m in fam.values()):
        lbs = ctx.lower_bounds
        acts["set_penalty"] = {label(f, m): {
            "must_exceed": max((lbs.get(x, 0.0) for x in m), default=0.0),
            "note": "floor applies to exact encodings only"} for f, m in fam.items()}
        acts["families_note"] = ("names with '#' and (×n) are constraint FAMILIES (numbers replaced by #); "
                                 "a weight or encoding set for a family applies to all n members")
    return acts


def apply_actions(history: list[RoundRecord], sense: str, actions: list[Action],
                  fam: dict[str, list[str]] | None = None) -> RunConfig:
    """Deterministically turn agent actions into a config. A family target sets every member; unknown names
    pass through to the guardrail."""
    base = history[-1].config
    best = max(history, key=lambda r: score(r.feedback, sense)).config
    weights, params, scale = dict(base.penalty_weights), {k: v for k, v in base.params.items() if k != "seed"}, base.scale
    encodings = dict(base.encodings)
    for a in actions:
        if a.type == "revert_to_best":
            weights, params, scale = dict(best.penalty_weights), {k: v for k, v in best.params.items() if k != "seed"}, best.scale
            encodings = dict(best.encodings)
        elif a.type == "set_penalty" and a.target is not None and a.value is not None:
            weights.update(expand({a.target: a.value}, fam or {}))
        elif a.type == "set_param" and a.target is not None and a.value is not None:
            params[a.target] = a.value
        elif a.type == "set_scale" and a.value is not None:
            scale = a.value
    return RunConfig(penalty_weights=weights, params=params, scale=scale, encodings=encodings)


_KIND_TAGS = {("linear", "le"): {"inequality_with_binary_slack", "precision_loss"},
              ("linear", "ge"): {"inequality_with_binary_slack", "precision_loss"},
              ("linear", "eq"): {"linear_equality", "cardinality_equality"},
              ("exclusion", None): {"pairwise_exclusion"}, ("implication", None): {"pairwise_exclusion"}}


def _constraint_view(ctx: LoopContext) -> tuple[list[dict[str, Any]], set[str]]:
    """Facts the Encoding Agent needs per constraint (per family on large problems), plus RAG tags."""
    rows, tags = _constraint_rows(ctx)
    fam = _fam(ctx)
    if not any(len(m) > 1 for m in fam.values()):
        return rows, tags
    by_name = {r["name"]: r for r in rows}
    out = []
    for f, members in fam.items():
        rs = [by_name[m] for m in members if m in by_name]
        row = {"name": label(f, members), "type": rs[0]["type"],
               "encodings": rs[0]["encodings"],
               "weight_floor_for_exact_encodings": max((r["weight_floor_for_exact_encodings"] or 0.0) for r in rs)}
        for k in ("n_terms", "rhs", "n_pairs"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                row[k] = vals[0] if min(vals) == max(vals) else [min(vals), max(vals)]
        if any(r.get("slack_bits") for r in rs):
            row["slack_bits_max"] = {b: max((r.get("slack_bits") or {}).get(b, 0) for r in rs) for b in ("binary", "unary")}
        out.append(row)
    return out, tags


def _constraint_rows(ctx: LoopContext) -> tuple[list[dict[str, Any]], set[str]]:
    from .encodings import ENCODING_PARAMS, options_for
    rows, tags = [], set()
    for c in ctx.ir.constraints:
        key = (c.kind, getattr(c, "sense", None))
        tags |= _KIND_TAGS.get(key, set())
        row: dict[str, Any] = {"name": c.name, "type": f"{c.kind}" + (f" ({c.sense})" if hasattr(c, "sense") else ""),
                               "encodings": {e: ENCODING_PARAMS.get(e, {}) for e in options_for(c)},
                               "weight_floor_for_exact_encodings": ctx.lower_bounds.get(c.name)}
        if c.kind == "linear":
            vals = [abs(v) for v in c.coeffs.values()]
            lo = sum(v for v in c.coeffs.values() if v < 0)
            span = (c.rhs - lo) if c.sense != "ge" else (sum(v for v in c.coeffs.values() if v > 0) - c.rhs)
            row.update(n_terms=len(c.coeffs), coef_abs_range=[min(vals), max(vals)], rhs=c.rhs,
                       slack_bits={"binary": max(1, int(span).bit_length()) if span > 0 else 0,
                                   "unary": int(max(span, 0))} if c.sense != "eq" else None,
                       squared_penalty_coef_scale=max(vals) * max(max(vals), abs(c.rhs)))
        else:
            row.update(n_pairs=len(c.pairs))
        rows.append(row)
    return rows, tags


class AgentPolicy(Policy):
    def __init__(self, cfg: dict[str, Any], root: Path, rag=None, seed: int = 0, **_: Any):
        acfg = cfg["agents"]
        self.mode = acfg.get("mode", "multi")
        self.model = acfg.get("model") or os.environ.get("LLM_MODEL")
        if not self.model:
            raise RuntimeError("No model: set agents.model in config_loop.yaml or LLM_MODEL")
        self.temperature = acfg.get("temperature", 0.0)
        self.schema_retries = acfg.get("schema_retries", 1)
        self.history_rounds = acfg.get("history_rounds", 6)
        self.max_tokens = acfg.get("max_tokens_per_run", 400_000)
        self.root, self.rag = root, rag
        self.use_encoding = acfg.get("encoding_agent", True)   # spec v0.2; False = fixed default encodings
        self.name = f"agent-{self.mode}{'-rag' if rag else ''}{'' if self.use_encoding else '-fixedenc'}"
        self.client: LLMClient | None = None
        self.templates = {n: (_PROMPTS / f"{n}.txt").read_text(encoding="utf-8")
                          for n in ("configurator", "diagnoser", "refiner", "single", "encoding")}

    @property
    def tokens_used(self) -> int:
        return self.client.tokens_used if self.client else 0

    # ------------------------------------------------------------------ LLM plumbing
    def _call(self, template: str, schema: type[BaseModel], step: str, **fields: str) -> BaseModel:
        prompt = self.templates[template]
        fields["schema"] = json.dumps(schema.model_json_schema(), indent=1)
        for k, v in fields.items():
            prompt = prompt.replace("{" + k + "}", v)
        last_error = ""
        for attempt in range(self.schema_retries + 1):
            text = prompt if not last_error else (
                prompt + f"\n\nYOUR PREVIOUS ANSWER WAS INVALID: {last_error}\nReturn corrected JSON only.")
            raw = self.client.call(model=self.model, prompt=text, temperature=self.temperature,
                                   step_name=step, json_mode=True)
            try:
                return schema.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)[:800]
        raise ValueError(f"{step}: invalid output after {self.schema_retries + 1} attempts: {last_error}")

    def _knowledge(self, ctx: LoopContext, tags: set[str], stores: tuple[str, ...],
                   labels: set[str] | None = None) -> tuple[str, list[str]]:
        if self.rag is None:
            return "(RAG disabled)", []
        entries = []
        for store in stores:
            if store == "experience":
                entries += self.rag.retrieve_experience(ctx.summary, ctx.backend.name, ctx.problem.instance_id, labels)
            else:
                entries += self.rag.retrieve_docs(store, tags)
        if not entries:
            return "(nothing relevant retrieved)", []
        return "\n".join(e.render() for e in entries), [e.id for e in entries]

    def _tags(self, ctx: LoopContext, labels: set[str] | None = None) -> set[str]:
        tags = {ctx.backend.name, ctx.problem.name, *ctx.summary["constraints"].values(), *ctx.specs}
        if not ctx.problem.constraints:
            tags.add("unconstrained")
        return tags | (labels or set())

    def _common(self, ctx: LoopContext, history: list[RoundRecord]) -> dict[str, str]:
        fam = _fam(ctx)
        recent = history[-self.history_rounds:]
        best = max(history, key=lambda r: score(r.feedback, ctx.problem.sense))
        stale = 0
        for r in reversed(history):
            if r.improved:
                break
            stale += 1
        return {
            "status": _j(self._status(ctx, len(history), stale)),
            "problem": _j(_summary_view(ctx, fam)),
            "backend": ctx.backend.name,
            "current_config": _j(_cfg_view(history[-1].config, fam)),
            "latest": _j(_metrics(history[-1].feedback, fam) or {"error": history[-1].error}),
            "history": _j([{"round": r.index, "config": _cfg_view(r.config, fam), "metrics": _metrics(r.feedback, fam),
                            "diagnosis": (r.diagnosis or {}).get("labels")} for r in recent]),
            "best": _j({"round": best.index, "config": _cfg_view(best.config, fam), "metrics": _metrics(best.feedback, fam)}),
            "actions": _j(_actions_view(ctx, fam)),
            "tried": _j([_cfg_view(r.config, fam) for r in history]),
        }

    @staticmethod
    def _status(ctx: LoopContext, calls_used: int, stale: int) -> dict[str, Any]:
        lc = ctx.cfg["loop"]
        return {"gap_target": lc.get("gap_target", 0.0),
                "solver_calls_used": calls_used, "solver_call_budget": lc.get("max_solver_calls", 10),
                "rounds_without_improvement": stale, "patience": lc.get("patience", 3),
                "note": "The loop stops by itself at the budget, at the gap target, or after `patience` rounds "
                        "without improvement. Stop early only if nothing is worth the remaining calls."}

    # ------------------------------------------------------------------ Encoding Agent (spec v0.2)
    def _encode(self, ctx: LoopContext, history: list[RoundRecord], reason: str) -> tuple[RunConfig, Proposal]:
        """Ask the Encoding Agent for encodings (+ weights); returns (partial config, provenance)."""
        rows, tags = _constraint_view(ctx)
        fam = _fam(ctx)
        knowledge, ids = self._knowledge(ctx, tags | {ctx.backend.name}, ("formulation_patterns", "tuning", "backend_docs"))
        seen: dict[str, dict[str, Any]] = {}
        for r in history:
            k = json.dumps(r.config.encodings, sort_keys=True)
            m = _metrics(r.feedback) or {}
            if k not in seen or (m and score(r.feedback, ctx.problem.sense) > score(seen[k].get("_fb"), ctx.problem.sense)):
                view = _cfg_view(r.config, fam)
                seen[k] = {"encodings": view.get("encodings", {}), "penalty_weights": view["penalty_weights"],
                           "best_metrics": {x: m.get(x) for x in ("feasibility_rate", "best_feasible_objective", "gap")},
                           "_fb": r.feedback}
        hist = [{k: v for k, v in e.items() if k != "_fb"} for e in seen.values()]
        out: EncodingOut = self._call(
            "encoding", EncodingOut, "encoding",
            problem=_j({k: ctx.summary.get(k) for k in ("problem_type", "sense", "n_variables", "objective_coef_abs_max")}
                       | {"reference_objective": ctx.problem.reference_objective}),
            constraints=_j(rows), backend=ctx.backend.name,
            backend_notes=_j({k: v for k, v in ctx.backend.describe().items() if k in ("precision", "notes", "billing")}),
            status=_j(self._status(ctx, len(history), 0)), reason=reason, encoding_history=_j(hist),
            knowledge=knowledge)
        cfg = RunConfig(encodings=expand({k: {"encoding": c.encoding, "params": dict(c.params)}
                                          for k, c in out.choices.items()}, fam),
                        penalty_weights=expand(dict(out.penalty_weights), fam))
        return cfg, Proposal(config=cfg, rationale=out.rationale, source="encoding", retrieved=ids,
                             cited=_ids(out.used_knowledge))

    # ------------------------------------------------------------------ Policy interface
    def initial(self, ctx: LoopContext) -> Proposal:
        if self.client is None:
            self.client = LLMClient(transcript_dir=self.root / "transcripts",
                                    run_id=f"loop_{ctx.problem.instance_id}_{self.name}_{uuid.uuid4().hex[:6]}",
                                    max_tokens=self.max_tokens, use_cache=False)
        enc = enc_p = None
        if ctx.ir is not None and self.use_encoding:
            enc, enc_p = self._encode(ctx, [], "cold start: choose the initial encoding for every constraint")
        fam = _fam(ctx)
        knowledge, ids = self._knowledge(ctx, self._tags(ctx), ("backend_docs", "formulation_patterns", "tuning", "experience"))
        out: ConfiguratorOut = self._call(
            "configurator", ConfiguratorOut, "configurator",
            problem=_j(_summary_view(ctx, fam)),
            status=_j(self._status(ctx, 0, 0)), backend=ctx.backend.name, actions=_j(_actions_view(ctx, fam)),
            default_config=_j(_cfg_view(ctx.default_config if enc is None else
                                        ctx.default_config.model_copy(update={"encodings": enc.encodings}), fam)),
            knowledge=knowledge)
        cfg = RunConfig(penalty_weights={**ctx.default_config.penalty_weights, **expand(out.penalty_weights, fam)},
                        params=dict(out.params), scale=out.scale)
        if enc is not None:                               # Encoding Agent owns encodings and their weights
            cfg = cfg.model_copy(update={"encodings": enc.encodings,
                                         "penalty_weights": {**cfg.penalty_weights, **enc.penalty_weights}})
            return Proposal(config=cfg, rationale=f"[encoding] {enc_p.rationale}\n[configurator] {out.rationale}",
                            source="encoding+configurator", retrieved=list(dict.fromkeys(enc_p.retrieved + ids)),
                            cited=_ids(enc_p.cited + out.used_knowledge))
        return Proposal(config=cfg, rationale=out.rationale, source="configurator", retrieved=ids,
                        cited=_ids(out.used_knowledge))

    def next(self, ctx: LoopContext, history: list[RoundRecord]) -> Proposal:
        common = self._common(ctx, history)
        if self.mode == "single":
            knowledge, ids = self._knowledge(ctx, self._tags(ctx), ("backend_docs", "formulation_patterns", "tuning", "experience"))
            out: SingleOut = self._call("single", SingleOut, "single_agent", knowledge=knowledge, **common)
            diagnosis = out.diagnosis.model_dump()
            actions, rationale, stop, cited = out.actions, out.rationale, out.stop, out.used_knowledge + out.diagnosis.used_knowledge
            source = "single"
        else:
            k_diag, ids_d = self._knowledge(ctx, self._tags(ctx), ("tuning", "formulation_patterns", "backend_docs", "experience"))
            diag: DiagnoserOut = self._call("diagnoser", DiagnoserOut, "diagnoser", knowledge=k_diag, **common)
            labels = set(diag.labels)
            k_ref, ids_r = self._knowledge(ctx, self._tags(ctx, labels), ("backend_docs", "tuning", "experience"), labels)
            ref: RefinerOut = self._call("refiner", RefinerOut, "refiner", diagnosis=_j(diag.model_dump()),
                                         knowledge=k_ref, **common)
            diagnosis, actions, rationale, stop = diag.model_dump(), ref.actions, ref.rationale, ref.stop
            ids, cited, source = list(dict.fromkeys(ids_d + ids_r)), diag.used_knowledge + ref.used_knowledge, "diagnoser+refiner"
        if "converged" in diagnosis["labels"] and not actions:
            stop = True
        config = apply_actions(history, ctx.problem.sense, actions, _fam(ctx))
        if any(a.type == "request_reencoding" for a in actions) and ctx.ir is not None and self.use_encoding:
            reason = (f"re-encoding requested after round {history[-1].index}: diagnosis {diagnosis.get('labels')}; "
                      f"evidence: {diagnosis.get('evidence', '')[:600]}; refiner: {rationale[:400]}")
            enc, enc_p = self._encode(ctx, history, reason)
            config = config.model_copy(update={"encodings": {**config.encodings, **enc.encodings},
                                               "penalty_weights": {**config.penalty_weights, **enc.penalty_weights}})
            rationale = f"{rationale}\n[encoding] {enc_p.rationale}"
            ids, cited, source = list(dict.fromkeys(ids + enc_p.retrieved)), cited + enc_p.cited, source + "+encoding"
        return Proposal(config=config, rationale=rationale,
                        diagnosis=diagnosis, stop=stop, source=source, retrieved=ids, cited=_ids(cited))

    def repair(self, ctx: LoopContext, history: list[RoundRecord], proposal: Proposal,
               errors: list[str]) -> Proposal | None:
        """Guardrail rejected the proposal: ask the responsible agent again with the reasons."""
        note = ("\n\nGUARDRAIL REJECTED YOUR LAST PROPOSAL:\n- " + "\n- ".join(compress_messages(errors))
                + "\nFix these problems.")
        if not history:
            saved = self.templates["configurator"]
            self.templates["configurator"] = saved + note
            try:
                return self.initial(ctx)
            finally:
                self.templates["configurator"] = saved
        key = "single" if self.mode == "single" else "refiner"
        saved = self.templates[key]
        self.templates[key] = saved + note
        try:
            p = self.next(ctx, history)
            return p.model_copy(update={"source": p.source + ":repair"})
        finally:
            self.templates[key] = saved
