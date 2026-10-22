"""Cross-stage repair by the LLM agents: the evidence of a failure goes back to the stage that can fix it.

- data stage:  the Data Adapter writes new preprocessing code with the evidence (same model, same sources);
- model stage: the Modeling Agent repairs its module with the evidence (repair, not rewrite), then the data are
               prepared again for the repaired model.

The callback only proposes. run_workflow accepts a repair only if it keeps the confirmed user settings, every
existing constraint, the objective direction and (on the same data) the formulation; otherwise the previous
version stays. A data failure before solving can also be sent back to the model (prepare_with_repair).
"""
from __future__ import annotations
import json
from pathlib import Path

from . import modeler
from .acquisition import prepare_real_data


def _evidence(items) -> str:
    return "\n".join(f"- {e}" for e in items)[:4000]


def make_repair(spec_json: str, qa_json: str, spec_text: str, client, model: str, backend: str, data_dir: Path,
                budget: dict, log=print):
    def repair(issue: dict, bundle: dict) -> dict:
        evidence = _evidence(issue.get("evidence", []))
        code = bundle["code"]
        stage = issue["stage"]
        check = ((bundle.get("metadata") or {}).get("report") or {}).get("data_check") or {}
        if stage == "diagnose" and check.get("issues"):
            # "no solution" cannot tell data from model; an open data-check disagreement points at the data first
            # (2026-10-11: 320 AED rows instead of 48 made "equal groups of three" infeasible)
            stage = "data"
            evidence += "\n" + _evidence(f"DATA CHECK: {m}" for m in check["issues"])
        if stage in ("model", "diagnose"):
            log(f"[修復] 回到建模：{evidence[:300]}")
            new_code, ns, errors, report = modeler.build_model(
                spec_json, qa_json, client, model, max_attempts=budget.get("model_attempts", 4), backend=backend,
                previous_code=code, patterns=budget.get("pattern_ids", False), repair_feedback="A check after data preparation / solving found:\n" + evidence)
            if errors:
                return {"status": "needs_input", "message": "模型修復未通過驗證：" + "; ".join(errors)[:800]}
            code = new_code
        else:
            log(f"[修復] 回到資料：{evidence[:300]}")
        data, errors, report = prepare_real_data(
            code, spec_text, client, model, data_dir,
            notes="PROBLEM FOUND AFTER THE PREVIOUS PREPARATION (fix it, keep the confirmed requirements):\n" + evidence,
            search_rounds=budget.get("data_search_rounds", 2), max_downloads=budget.get("max_downloads", 4), log=log,
            data_check=budget.get("data_check", False))
        if errors:
            return {"status": "needs_input", "message": "資料修復失敗：" + "; ".join(map(str, errors))[:800]}
        return {"status": "repaired", "code": code, "data": data, "message": f"{stage} 修復完成",
                "metadata": {"report": report}}
    return repair


SCHEMA_MARKERS = ("missing DATA_SCHEMA keys", "the model expects", "differs from the confirmed", "contract field",
                  "set '", "param '", "raised by the MODEL's build_ir", "REQUIRED TOOL NOT USED: DATA_SCHEMA")


def schema_mismatch(errors) -> bool:
    """Errors saying the data cannot match the model's declaration (as opposed to search or adapter-code failures)."""
    return any(m in str(e) for e in errors for m in SCHEMA_MARKERS)


def prepare_with_repair(code: str, spec_json: str, qa_json: str, spec_text: str, client, model: str, backend: str,
                        data_dir: Path, budget: dict, max_model_repairs: int = 1, log=print):
    """Data preparation; if it fails after its own retries, send the evidence back to the Modeling Agent (the
    model may ask for data the sources cannot give) and prepare again. Returns (code, data, errors, report, repairs)."""
    repairs = []
    for attempt in range(max_model_repairs + 1):
        data, errors, report = prepare_real_data(code, spec_text, client, model, data_dir,
                                                 search_rounds=budget.get("data_search_rounds", 2),
                                                 max_downloads=budget.get("max_downloads", 4), log=log,
            data_check=budget.get("data_check", False))
        if not errors or attempt == max_model_repairs or not schema_mismatch(errors):
            # only "the data cannot give what this model declares" goes back to the model; no data found or a bug
            # in the adapter's own code is not a model problem
            return code, data, errors, report, repairs
        evidence = _evidence(map(str, errors))
        log(f"[修復] 資料備妥失敗，回到建模：{evidence[:300]}")
        new_code, ns, model_errors, _ = modeler.build_model(
            spec_json, qa_json, client, model, max_attempts=budget.get("model_attempts", 4), backend=backend,
            previous_code=code, patterns=budget.get("pattern_ids", False), repair_feedback=("The real data could not be prepared for this model's DATA_SCHEMA. "
                                                 "Errors from the Data Adapter:\n" + evidence +
                                                 "\nKeep every confirmed requirement; change only what the data "
                                                 "cannot provide or what is declared wrongly."))
        repairs.append({"from": "data", "to": "model", "evidence": evidence[:1500], "ok": not model_errors})
        if model_errors:
            return code, data, errors, report, repairs
        code = new_code
    return code, data, errors, report, repairs
