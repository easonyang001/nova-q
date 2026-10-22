"""RAG knowledge base (spec §7): four stores, feature-keyed retrieval, train/test isolation.

Stores:
  backend_docs, formulation_patterns, tuning   curated JSONL in knowledge/ (manual, versioned)
  experience                                   written automatically from finished loops

Retrieval keys are problem features and tags (backend, constraint types, diagnosis labels),
never the natural-language problem text (§7.2). Every retrieval returns entry ids so the loop log
records what each agent saw. A vector index can replace `_score_doc` later without changing callers.
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DOC_STORES = ("backend_docs", "formulation_patterns", "tuning")
BACKEND_TAGS = {"sa", "dwave_qpu", "fujitsu_da", "cga"}


@dataclass
class Entry:
    id: str
    store: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)
    source: str = ""

    def render(self, max_chars: int = 1500) -> str:
        # safety net: one entry must never be able to fill the LLM context (a per-constraint experience
        # record of a 3324-constraint problem was 157k characters)
        text = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " …(truncated)"
        return f"[{self.id}] {self.title}: {text}" + (f" (source: {self.source})" if self.source else "")


class KnowledgeBase:
    def __init__(self, stores: list[str], top_k: int, knowledge_dir: Path, experience_path: Path,
                 exclude_instance_ids: list[str] | None = None, experience_backends: list[str] | None = None):
        self.stores = list(stores)
        self.top_k = top_k
        self.experience_path = experience_path
        self.exclude = set(exclude_instance_ids or [])
        self.experience_backends = set(experience_backends) if experience_backends else None
        self.docs: dict[str, list[Entry]] = {}
        for store in DOC_STORES:
            path = knowledge_dir / f"{store}.jsonl"
            if store in self.stores and path.exists():
                self.docs[store] = [Entry(store=store, **json.loads(line))
                                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @classmethod
    def from_config(cls, rag_cfg: dict[str, Any], root: Path) -> "KnowledgeBase":
        return cls(stores=rag_cfg.get("stores", []), top_k=rag_cfg.get("top_k", 4),
                   knowledge_dir=root / rag_cfg.get("knowledge_dir", "knowledge"),
                   experience_path=root / rag_cfg.get("experience_path", "results/experience.jsonl"),
                   exclude_instance_ids=rag_cfg.get("exclude_instance_ids"),
                   experience_backends=rag_cfg.get("experience_backends"))

    # ---------------------------------------------------------------- curated stores
    def retrieve_docs(self, store: str, tags: set[str], k: int | None = None) -> list[Entry]:
        """Tag-overlap ranking. In backend-specific stores (backend docs, tuning), entries tied to a
        backend are only returned for that backend; formulation patterns are backend-independent."""
        if store not in self.docs:
            return []
        query_backends = tags & BACKEND_TAGS
        candidates = [e for e in self.docs[store]
                      if store == "formulation_patterns" or not (set(e.tags) & BACKEND_TAGS)
                      or set(e.tags) & query_backends]
        scored = [(len(tags & set(e.tags)), e) for e in candidates]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda t: (-t[0], t[1].id))
        return [e for _, e in scored[: k or self.top_k]]

    # ---------------------------------------------------------------- experience store
    def _experience(self) -> list[dict[str, Any]]:
        if "experience" not in self.stores or not self.experience_path.exists():
            return []
        return [json.loads(l) for l in self.experience_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def retrieve_experience(self, summary: dict[str, Any], backend: str, instance_id: str,
                            labels: set[str] | None = None, k: int | None = None) -> list[Entry]:
        """Nearest past rounds by problem features. Never returns the current or excluded instances."""
        out = []
        for rec in self._experience():
            if rec["instance_id"] == instance_id or rec["instance_id"] in self.exclude:
                continue
            if self.experience_backends is not None and rec["backend"] not in self.experience_backends:
                continue
            f = rec["features"]
            dist = (abs(math.log(f["n_variables"] + 1) - math.log(summary["n_variables"] + 1))
                    + abs(f["density"] - summary["density"])
                    + abs(math.log((f.get("coef_dynamic_range") or 1) + 1) - math.log((summary.get("coef_dynamic_range") or 1) + 1)))
            sim = (3.0 * (f["problem_type"] == summary["problem_type"]) + 2.0 * (rec["backend"] == backend)
                   + 1.0 * bool(labels and labels & set(rec.get("diagnosis_labels", []))) - dist)
            out.append((sim, rec))
        out.sort(key=lambda t: -t[0])
        entries = []
        for _, rec in out[: k or self.top_k]:
            entries.append(Entry(
                id=rec["id"], store="experience", title=f"{rec['problem_type']} n={rec['features']['n_variables']} on {rec['backend']}",
                text=(f"diagnosis={rec.get('diagnosis_labels')} change={rec['change']} "
                      f"before={rec['before']} after={rec['after']} effective={rec['effective']}"),
                source=f"run log {rec['instance_id']}"))
        return entries

    def write_experience(self, loop_result, summary: dict[str, Any]) -> int:
        """§7.1: one record per refinement step (config change -> did it help?)."""
        from .loop import score
        rounds = [r for r in loop_result.rounds if r.feedback]
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for prev, cur in zip(rounds, rounds[1:]):
            change = compress_change(_diff(prev.config.model_dump(), cur.config.model_dump()))
            if not change:
                continue
            rec = {
                "id": f"exp-{loop_result.instance_id}-{loop_result.policy}-{int(time.time())}-{cur.index}",
                "instance_id": loop_result.instance_id, "problem_type": summary["problem_type"],
                "backend": loop_result.backend, "simulated": loop_result.simulated, "policy": loop_result.policy,
                "features": {k: summary[k] for k in ("problem_type", "n_variables", "density", "coef_dynamic_range")},
                "diagnosis_labels": (cur.diagnosis or {}).get("labels", []),
                "change": change,
                "before": _brief(prev.feedback), "after": _brief(cur.feedback),
                "effective": score(cur.feedback, summary["sense"]) > score(prev.feedback, summary["sense"]),
            }
            lines.append(json.dumps(rec) + "\n")
        with self.experience_path.open("a", encoding="utf-8") as f:   # one write: safe with parallel runs
            f.write("".join(lines))
        return len(lines)


def _diff(a: dict, b: dict, prefix: str = "") -> dict[str, list]:
    out = {}
    for k in set(a) | set(b):
        if k == "seed":
            continue
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(_diff(va, vb, f"{prefix}{k}."))
        elif va != vb:
            out[f"{prefix}{k}"] = [va, vb]
    return out


def compress_change(change: dict[str, list]) -> dict[str, Any]:
    """Config changes grouped by constraint family on large problems: 'penalty_weights.coverage_d#_s早 (×28)'."""
    from .families import FAMILY_MIN, family_of
    if len(change) <= FAMILY_MIN:
        return change
    groups: dict[str, list] = {}
    for k, v in change.items():
        groups.setdefault(family_of(k), []).append(v)
    out: dict[str, Any] = {}
    for f, vals in groups.items():
        if len(vals) == 1:
            out[f] = vals[0]
        elif all(v == vals[0] for v in vals):
            out[f"{f} (×{len(vals)})"] = vals[0]
        else:
            out[f"{f} (×{len(vals)})"] = {"examples": vals[:2]}
    return out


def _brief(fb: dict[str, Any]) -> dict[str, Any]:
    from .families import families, group_rates
    out = {k: fb[k] for k in ("feasibility_rate", "best_feasible_objective", "gap", "constraint_violation_rates")}
    out["constraint_violation_rates"] = group_rates(out["constraint_violation_rates"],
                                                    families(out["constraint_violation_rates"]))
    return out
