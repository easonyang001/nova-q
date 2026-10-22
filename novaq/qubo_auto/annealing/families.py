"""Constraint families: what agents see when a problem has many constraints.

A scheduling model with 1680 variables had 3324 constraints but only 10 kinds ("daily_limit_n3_d17" ...). Listing
every constraint's weight, violation rate and encoding overflowed the LLM's 64k-token context. Agents therefore
see and act on families (the name with every number replaced by '#'); a family-level decision is expanded to all
members by code. Small problems keep one constraint per family, so their prompts are unchanged.
"""
from __future__ import annotations
import re
from typing import Any, Iterable

FAMILY_MIN = 30          # group only above this many constraints


def family_of(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def families(names: Iterable[str]) -> dict[str, list[str]]:
    names = list(names)
    if len(names) <= FAMILY_MIN:
        return {n: [n] for n in names}
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(family_of(n), []).append(n)
    return out


def label(f: str, members: list[str]) -> str:
    """How a family is shown: its single member's real name, or the pattern with a member count."""
    return members[0] if len(members) == 1 else f"{f} (×{len(members)})"


def _key(fam: dict[str, list[str]], f: str, members: list[str]) -> str:
    return label(f, members)


def group_values(values: dict[str, Any], fam: dict[str, list[str]]) -> dict[str, Any]:
    """{family: value} when all members agree, else {family: {"min", "max"}}; non-numeric values are counted."""
    out: dict[str, Any] = {}
    for f, members in fam.items():
        vals = [values[m] for m in members if m in values]
        if not vals:
            continue
        key = _key(fam, f, members)
        if all(v == vals[0] for v in vals):
            out[key] = round(vals[0], 4) if isinstance(vals[0], float) else vals[0]
        elif all(isinstance(v, (int, float)) for v in vals):
            out[key] = {"min": round(min(vals), 4), "max": round(max(vals), 4)}
        else:
            counts: dict[str, int] = {}
            for v in vals:
                counts[str(v)] = counts.get(str(v), 0) + 1
            out[key] = counts
    for k, v in values.items():                     # names outside the families (e.g. stale keys) stay visible
        if not any(k in m for m in fam.values()):
            out[k] = v
    return out


def group_rates(rates: dict[str, float], fam: dict[str, list[str]]) -> dict[str, Any]:
    """Violation rates per family: max, mean and how many members were violated at all."""
    out: dict[str, Any] = {}
    for f, members in fam.items():
        vals = [rates.get(m, 0.0) for m in members]
        if len(members) == 1:
            out[members[0]] = round(vals[0], 4)
        else:
            out[_key(fam, f, members)] = {"max": round(max(vals), 4), "mean": round(sum(vals) / len(vals), 4),
                                          "violated_members": sum(v > 0 for v in vals)}
    return out


def expand(values: dict[str, Any], fam: dict[str, list[str]]) -> dict[str, Any]:
    """Family-level (or single-constraint) decisions -> one entry per constraint. Accepts the "(×n)" form too."""
    out: dict[str, Any] = {}
    for k, v in values.items():
        f = re.sub(r"\s*\(×\d+\)$", "", k)
        for m in fam.get(f, [k]):
            out[m] = v
    return out


def compress_messages(messages: list[str], limit: int = 15) -> list[str]:
    """Guardrail errors repeat per constraint; keep one per family pattern with a count."""
    seen: dict[str, list[str]] = {}
    for m in messages:
        seen.setdefault(re.sub(r"\d+(\.\d+)?", "#", m), []).append(m)
    out = [(f"{ms[0]} (and {len(ms) - 1} more like it)" if len(ms) > 1 else ms[0]) for ms in seen.values()]
    return out[:limit] + ([f"... {len(out) - limit} more kinds of error"] if len(out) > limit else [])
