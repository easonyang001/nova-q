"""Explicit data and requirement contracts; declarations are not semantic proofs."""
from __future__ import annotations
import math
from fnmatch import fnmatchcase


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def check_data(data, contract=None, metadata=None, sources=None):
    errors, review = [], []
    for key, value in data.items():
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v)
               for v in leaves(value)):
            errors.append(f"{key}: NaN or infinity is not valid model data")
    if not contract:
        return errors, ["舊模型未提供 DATA_CONTRACT，單位／時間／地區的一致性尚未核對"]
    metadata = metadata or {}
    urls = {s.get("url") for s in sources or [] if s.get("url")}
    for key, rules in contract.items():
        if not isinstance(rules, dict):
            errors.append(f"DATA_CONTRACT[{key!r}] must be a dict")
            continue
        if key not in data:
            errors.append(f"{key}: required contract field is missing")
            continue
        if "value" in rules and data[key] != rules["value"]:
            errors.append(f"{key}: value differs from confirmed user setting {rules['value']!r}")
        info = metadata.get(key, {})
        if not isinstance(info, dict):
            errors.append(f"{key}: metadata must be a dict")
            continue
        for attr in ("unit", "period", "geography", "granularity"):
            expected = rules.get(attr)
            if expected:
                actual = info.get(attr)
                if actual is None:
                    review.append(f"{key}: 尚未聲明 {attr}，需求為 {expected}")
                elif actual != expected:
                    errors.append(f"{key}: {attr}={actual!r}, expected {expected!r}; do not silently substitute")
        values = list(leaves(data[key]))
        missing = sum(v is None or v == "" for v in values)
        if values and missing / len(values) > rules.get("max_missing_fraction", 0):
            errors.append(f"{key}: missing fraction {missing / len(values):.1%} exceeds contract")
        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if "min" in rules and v < rules["min"]:
                    errors.append(f"{key}: value {v} below minimum {rules['min']}")
                    break
                if "max" in rules and v > rules["max"]:
                    errors.append(f"{key}: value {v} above maximum {rules['max']}")
                    break
        if rules.get("source_kind") == "public":
            cited = info.get("source_urls", [])
            if isinstance(cited, str):
                cited = [cited]
            if not cited:
                review.append(f"{key}: 缺少實際來源對應，不能確認為公開真實資料")
            elif any(url not in urls for url in cited):
                # often the dataset page cited for its download link: a review item, not a data error (2026-10-12)
                review.append(f"{key}: 引用的來源網址不在下載紀錄中（可能是資料集頁面而非下載連結）")
        if info.get("assumptions"):
            review.append(f"{key}: {info['assumptions']}")
    return errors, review


def check_requirements(spec, ns, raw):
    """Check per-requirement coverage. It cannot verify a natural-language formula."""
    hard = [c for c in spec.get("constraints", []) if c.get("strength") == "hard"]
    mapping = ns.get("REQUIREMENT_MAP", {})
    names = {c["name"] for c in raw.get("constraints", [])}
    errors = []
    if not isinstance(mapping, dict):
        return ["REQUIREMENT_MAP must map confirmed hard requirement names to constraint names"]
    unknown_keys = [k for k in mapping if k not in {r["name"] for r in hard}]
    if unknown_keys and any(r["name"] not in mapping for r in hard):
        # keys translated or paraphrased (a Chinese requirement name mapped under an English key): say what to copy
        errors.append(f"REQUIREMENT_MAP keys must be the confirmed requirement names copied EXACTLY: "
                      f"{[r['name'] for r in hard]}; your keys {unknown_keys} are not among them")
    for requirement in hard:
        name = requirement["name"]
        targets = mapping.get(name)
        if not isinstance(targets, list) or not targets:
            errors.append(f"hard requirement {name!r} has no REQUIREMENT_MAP entry; map it to the IR constraint names "
                          f"that enforce it, or to [\"implied: <how the variables guarantee it>\"] when the variable "
                          f"design itself makes it true (e.g. position encoding gives a single closed tour; every QUBO "
                          f"variable is already binary). Add this exact key to REQUIREMENT_MAP: "
                          f"{name!r}: [<constraint names>] or {name!r}: [\"implied: <reason>\"]")
        else:
            # "implied: ..." = guaranteed by the variable design; still tested by the requirement Checker and the judge
            targets = [t for t in targets if not (isinstance(t, str) and t.strip().lower().startswith("implied:")
                                                  and len(t.strip()) > len("implied:") + 5)]
            missing = [target for target in targets if not isinstance(target, str) or target == "*" or
                       not any(fnmatchcase(actual, target) for actual in names)]
            if not missing:
                continue
            correction = f" Set REQUIREMENT_MAP[{name!r}] = [{name!r}]." if name in names else ""
            errors.append(f"hard requirement {name!r} maps to missing constraints {missing}; actual IR names: {sorted(names)[:30]}.{correction}")
    return errors


def changed_semantics(old_ir, new_ir):
    """Conservative repair gate: changed formulas need confirmation, even with identical names."""
    changes = []
    if old_ir.sense != new_ir.sense:
        changes.append("objective direction changed")
    if old_ir.n_variables != new_ir.n_variables:
        changes.append("decision variable layout changed")
    before = {c.name: c.model_dump() for c in old_ir.constraints}
    after = {c.name: c.model_dump() for c in new_ir.constraints}
    for name, constraint in before.items():
        if after.get(name) != constraint:
            changes.append(f"constraint {name!r} formula/coefficients/bound changed or was removed")
    if (old_ir.linear, old_ir.quadratic, old_ir.constant) != (new_ir.linear, new_ir.quadratic, new_ir.constant):
        changes.append("original objective coefficients changed")
    return changes
