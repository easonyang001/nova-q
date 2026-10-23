"""Outer recovery loop: model/data -> preflight -> solver loop -> delivery.

Repair callbacks return a validated replacement bundle, or a request for input.
Solver calls share one budget across repairs. Only results attached to the active
model/data version may be delivered; checkpoints allow restarting without retrieval.
"""
from __future__ import annotations
import copy
import uuid
import json
from pathlib import Path

from . import modeler
from .artifacts import ArtifactStore, atomic_write
from .assessment import evaluate_delivery, inspect_problem
from .loop import LoopResult, run_loop


def run_workflow(store: ArtifactStore, backend, policy_factory, cfg: dict, *, budget=None,
                 simulate=False, rag=None, encodings=None, repair=None, log=print,
                 resume_checkpoint: Path | None = None, validator=None) -> dict:
    limit = int(cfg["loop"].get("max_solver_calls", 10))
    if limit < 1:
        raise ValueError("solver-call budget must be positive")
    settings = cfg.get("workflow", {})
    max_repairs = int(settings.get("max_repairs", 2))
    calls, attempts, repairs, all_attempts = 0, 0, [], []
    state_path = store.root / "workflow_state.json"
    run_id = uuid.uuid4().hex[:12]
    saved_context = None
    diagnoses = []
    best_so_far = None          # the last version that produced a feasible annealer solution
    if resume_checkpoint is not None:
        saved_context = json.loads(resume_checkpoint.read_text())["context"]
        calls = saved_context["config"].get("workflow_prior_calls", 0)
        attempts = saved_context["config"].get("workflow_attempt", 0)
        run_id = saved_context["config"].get("workflow_run_id", run_id)
    while True:
        bundle = store.current()
        version = bundle["version"]
        metadata = bundle.get("metadata", {})
        ns = None
        req_issues, raw_ir = [], None
        try:
            ns = modeler.load_module(bundle["code"])
            from .contracts import check_data, check_requirements
            report = metadata.get("report", {})
            data_errors, reviews = check_data(bundle["data"], ns.get("DATA_CONTRACT"),
                                              report.get("data_metadata"), report.get("sources"))
            if data_errors or (metadata.get("source") == "real" and reviews and not metadata.get("data_review_approved")):
                raise ValueError("DATA_CONTRACT: " + "; ".join(data_errors or reviews))
            ir = modeler.to_ir(ns["build_ir"](bundle["data"]), f"ui_{store.root.name}_{version}")
            if metadata.get("confirmed_spec"):
                requirement_errors = check_requirements(metadata["confirmed_spec"], ns, ns["build_ir"](bundle["data"]))
                if requirement_errors:
                    raise ValueError("REQUIREMENT_MAP: " + "; ".join(requirement_errors))
            from . import schema as _schema
            if _schema.is_structured(ns["DATA_SCHEMA"]):
                # variable_keys must name the REAL ids: integer positions passed on the example (ids 0..5) and the
                # delivered sites could not be read as villages (smoke18, 2026-10-12)
                key_errors = _schema.variable_key_errors(ns["DATA_SCHEMA"], bundle["data"], ns["build_ir"](bundle["data"]))
                if key_errors:
                    raise ValueError("VARIABLE_KEYS on the real data: " + "; ".join(key_errors)
                                     + " (use the set's ids from the data, not positions)")
            inspection = inspect_problem(ir, settings.get("preflight_time_s", 2))
            if validator is not None and inspection["feasibility"] not in ("invalid", "infeasible"):
                # independent Checker vs model on random valid / invalid decisions (evidence, never a hard stop)
                raw_ir = ns["build_ir"](bundle["data"])
                req_issues = validator.preflight(bundle["data"], raw_ir)
                store.event("requirement_check", "issues" if req_issues else "ok", version=version,
                            report=validator.last, issues=req_issues)
                compared = validator.last.get("valid", 0)
                log(f"[需求檢查] {'不一致：' + '; '.join(req_issues)[:300] if req_issues else '一致' if compared else '無法比較（沒有可對照的方案）'} {validator.last}")
        except Exception as e:
            ir = None
            inspection = {"feasibility": "invalid", "issues": [f"{type(e).__name__}: {e}"],
                          "baseline_objective": None, "baseline_solution": None, "proven_optimal": False}
        store.event("preflight", inspection["feasibility"], version=version, inspection=inspection)
        log(f"[流程檢查] {inspection['feasibility']}；簡單基準 {inspection.get('baseline_objective')}")
        result = None
        can_repair = repair is not None and attempts < max_repairs
        if req_issues and can_repair and inspection["feasibility"] not in ("invalid", "infeasible"):
            # repair the model before spending solver calls on a version the Checker disagrees with
            issue = {"stage": "model", "evidence": req_issues, "version": version, "from_requirement_check": True}
            delivery = {"status": "no_solution", "message": "需求檢查與模型不一致，先修正模型", "next_stage": "model",
                        "evidence": req_issues, "quality_gap": None,
                        "baseline_objective": inspection.get("baseline_objective")}
        elif inspection["feasibility"] in ("invalid", "infeasible"):
            data_issue = any("DATA_CONTRACT:" in e for e in inspection["issues"])
            issue = {"stage": "data" if data_issue else "model" if inspection["feasibility"] == "invalid" else "diagnose",
                     "evidence": inspection["issues"], "version": version}
            delivery = {"status": "no_solution", "message": "資料／模型檢查未通過，未消耗退火機額度",
                        "next_stage": issue["stage"], "evidence": inspection["issues"], "quality_gap": None,
                        "baseline_objective": inspection.get("baseline_objective")}
        else:
            local = copy.deepcopy(saved_context["config"] if saved_context else cfg)
            local["loop"]["max_solver_calls"] = limit - calls
            local.setdefault("workflow_baseline", {"objective": inspection.get("baseline_objective"),
                                           "note": "feasible conventional baseline, not necessarily optimal"}
                             )
            local.setdefault("workflow_prior_calls", calls)
            local.setdefault("workflow_attempt", attempts)
            local.setdefault("workflow_run_id", run_id)
            # Exact values are optional; heuristic baselines must never be labelled optima.
            if inspection["proven_optimal"]:
                ir.reference_objective = inspection["baseline_objective"]
            try:
                checkpoint = resume_checkpoint if saved_context else store.root / "checkpoints" / f"{run_id}_{version}_{attempts}.json"
                if saved_context:
                    from .encodings import ProblemIR
                    saved_ir = ProblemIR.model_validate(saved_context["ir"])
                    if ir.model_dump(exclude={"reference_objective"}) != saved_ir.model_dump(exclude={"reference_objective"}):
                        raise ValueError("checkpoint 與目前模型／資料版本不符，請還原相同版本或開始新工作")
                    ir = saved_ir
                store.event("solve", "running", version=version, checkpoint=str(checkpoint))
                result = run_loop(None, backend, policy_factory(), local, budget=budget, simulate=simulate,
                                  rag=rag, ir=ir, root=store.root, log=log,
                                  encodings=saved_context["encodings"] if saved_context else encodings if attempts == 0 else None,
                                  checkpoint=checkpoint, resume=bool(saved_context),
                                  seed=saved_context["seed"] if saved_context else 0)
                saved_context = None
                raw_head = list(result.best_solution[:ir.n_variables]) if result.best_solution is not None else None
                raw_feasible = raw_head is not None and all(c.satisfied(raw_head) for c in ir.constraints)
                solver_raw = {"feasible": raw_feasible,
                              "objective": ir.objective(raw_head) if raw_feasible else None, "polished": False}
                if result.best_solution is not None and ir.quadratic:
                    # classical polish of the annealer's answer with constraint-preserving exchanges (same moves as
                    # the baseline); the solver's own result stays in the loop record (Melbourne tour, 2026-10-21:
                    # 6809 m from SA -> 5311 m after exchanges)
                    from .assessment import _exchange_improve
                    head = list(result.best_solution[:ir.n_variables])
                    sign = 1 if ir.sense == "minimize" else -1
                    if all(c.satisfied(head) for c in ir.constraints):
                        polished = _exchange_improve(ir, head, lambda y: all(c.satisfied(y) for c in ir.constraints), sign)
                        if sign * ir.objective(polished) < sign * ir.objective(head) - 1e-9:
                            log(f"[求解] 交換修整：{ir.objective(head):.6g} → {ir.objective(polished):.6g}")
                            store.event("solve", "polished", before=ir.objective(head), after=ir.objective(polished))
                            result.best_solution = polished + list(result.best_solution[ir.n_variables:])
                            result.best_objective = ir.objective(polished)
                            solver_raw["polished"] = True
                calls += result.solver_calls
                all_attempts.append({"version": version, "checkpoint": str(checkpoint), "loop": result.model_dump()})
                delivery = evaluate_delivery(ir, result, inspection, settings.get("quality_tolerance", 0.01))
                # what the solver itself returned, before the classical exchange polish: a delivery "by QAOA" must be
                # told apart from one QAOA started and the polish finished
                delivery["solver_raw"] = solver_raw
                sol_issues = []
                if validator is not None and delivery["status"] != "no_solution":
                    raw_ir = raw_ir or ns["build_ir"](bundle["data"])
                    sol_issues = validator.solution(bundle["data"], raw_ir, result.best_solution[:ir.n_variables])
                    store.event("requirement_check", "solution_issues" if sol_issues else "solution_ok",
                                version=version, issues=sol_issues)
                suspect = [r.diagnosis for r in result.rounds if r.diagnosis
                           and "formulation_suspect" in r.diagnosis.get("labels", [])]
                # A weak result alone is insufficient evidence to rewrite a model.
                # a Checker objection only sends the model back when a repaired version can still be solved;
                # otherwise a good annealer solution was thrown away for a baseline (2026-10-11, a Checker that
                # invented a fixed start station)
                route_checker = bool(sol_issues) and calls < limit
                issue = {"stage": "model" if suspect or route_checker else delivery["next_stage"],
                         "evidence": suspect or (sol_issues if route_checker else delivery["evidence"]), "version": version}
                if sol_issues:
                    delivery["requirement_warnings"] = sol_issues
                    issue["from_requirement_check"] = route_checker
                if result is not None and result.best_solution is not None and delivery["status"] != "no_solution":
                    best_so_far = {"version": version, "solution": result.best_solution[:ir.n_variables],
                                   "delivery": dict(delivery, solution_source="annealer"), "inspection": inspection,
                                   "loop": result.model_dump()}
                if not suspect and issue["stage"] == "solve":
                    issue["evidence"] = ["同一模型已有獨立可行基準；目前證據支持繼續搜尋或交付基準，尚不足以改資料／模型"]
            except Exception as e:
                # A crash has unknown hardware cost: do not retry and risk exceeding the budget.
                delivery = {"status": "no_solution", "message": f"求解階段失敗：{type(e).__name__}: {e}",
                            "next_stage": "solve", "evidence": [str(e)], "quality_gap": None,
                            "baseline_objective": inspection.get("baseline_objective")}
                issue = {"stage": "solve", "evidence": [str(e)], "version": version}
        diagnoses.append(issue)
        store.event("diagnosis", issue["stage"], version=version, evidence=issue["evidence"])
        store.event("delivery", delivery["status"], version=version, delivery=delivery, calls=calls)
        if issue["stage"] in ("data", "model", "diagnose") and repair is not None and attempts < max_repairs:
            attempts += 1
            store.event("recovery", "started", issue=issue, attempt=attempts)
            log(f"[回復] 回查 {issue['stage']}：{issue['evidence']}")
            try:
                decision = repair(issue, bundle)
            except Exception as e:
                decision = {"status": "needs_input", "message": f"修正失敗：{e}"}
            repairs.append(decision.get("message", decision.get("status", "unknown")))
            if decision.get("status") == "repaired" and calls < limit:
                code, data = decision["code"], decision["data"]
                try:
                    candidate_ns = modeler.load_module(code)
                    candidate_ir = modeler.to_ir(candidate_ns["build_ir"](data), "repair_check")
                    from .contracts import changed_semantics, check_requirements
                    for key, rules in ((ns or {}).get("DATA_CONTRACT") or {}).items():
                        if rules.get("source_kind") == "user" and key in bundle["data"] and data.get(key) != bundle["data"][key]:
                            raise ValueError(f"修正改變使用者設定 {key!r}，需要重新確認規格")
                    candidate_metadata = {**metadata, **decision.get("metadata", {})}
                    candidate_errors, candidate_review = check_data(data, candidate_ns.get("DATA_CONTRACT"),
                        candidate_metadata.get("report", {}).get("data_metadata", {}),
                        candidate_metadata.get("report", {}).get("sources", []))
                    if candidate_errors:
                        raise ValueError("; ".join(candidate_errors))
                    if candidate_metadata.get("source") == "real" and candidate_review and not candidate_metadata.get("data_review_approved"):
                        raise ValueError("資料修正仍有待確認項目：" + "; ".join(candidate_review))
                    if metadata.get("confirmed_spec"):
                        requirement_errors = check_requirements(metadata["confirmed_spec"], candidate_ns,
                                                                candidate_ns["build_ir"](data))
                        if requirement_errors:
                            raise ValueError("; ".join(requirement_errors))
                    if ir is not None and code != bundle["code"]:
                        # Compare both builders on the SAME original data. Changing input data
                        # must not hide a code-level relaxation of a bound.
                        candidate_on_old = modeler.to_ir(candidate_ns["build_ir"](bundle["data"]), "repair_same_data")
                        changes = changed_semantics(ir, candidate_on_old)
                        if changes and issue.get("from_requirement_check") and validator is not None:
                            # the change was asked for by the independent requirement check: accept it only if
                            # that check now agrees; settings, constraint names and direction are checked below
                            remaining = validator.preflight(data, candidate_ns["build_ir"](data))
                            if remaining:
                                raise ValueError("修正後仍與需求檢查不一致：" + "; ".join(remaining))
                            decision.setdefault("metadata", {})["formula_changes"] = changes
                            log(f"[回復] 依需求檢查修改公式（需人工確認）：{'; '.join(changes)}")
                        elif changes:
                            raise ValueError("修正變更模型公式，需要重新確認規格：" + "; ".join(changes))
                    candidate_check = inspect_problem(candidate_ir, settings.get("preflight_time_s", 2))
                    if candidate_check["feasibility"] in ("invalid", "infeasible"):
                        raise ValueError("；".join(candidate_check["issues"]))
                    if ir is not None and (candidate_ir.sense != ir.sense or
                            not {c.name for c in ir.constraints}.issubset({c.name for c in candidate_ir.constraints})):
                        raise ValueError("修正改變目標方向或移除既有限制，需重新確認規格")
                    store.commit(code, data, "流程修正", {**bundle.get("metadata", {}),
                                 **decision.get("metadata", {}), "repair": issue})
                    store.event("recovery", "repaired", from_version=version, to_version=store.current()["version"])
                    continue
                except Exception as e:
                    decision = {"status": "needs_input", "message": f"修正未通過檢查，保留先前版本：{e}"}
                    repairs[-1] = decision["message"]
            delivery["recovery_message"] = decision.get("message", "修正未完成，保留先前版本")
            store.event("recovery", "needs_input", decision=decision)
            if issue.get("from_requirement_check"):
                continue            # not solved yet: try again, or solve with a warning once repairs run out
        if result is None and best_so_far is not None and best_so_far["version"] != version:
            # the repaired version could not be solved (no calls left): deliver the earlier version's solution,
            # with its model and data, rather than a weaker baseline of an unsolved version
            prev = best_so_far
            prev["delivery"]["message"] = (prev["delivery"].get("message", "") + "；修正後的版本沒有剩餘求解次數，"
                                           "交付前一版本的退火結果").lstrip("；")
            output = {"version": prev["version"], "delivery": prev["delivery"], "inspection": prev["inspection"],
                      "loop": prev["loop"], "solution": prev["solution"], "total_solver_calls": calls,
                      "repairs": repairs, "attempts": all_attempts, "diagnoses": diagnoses,
                      "unsolved_later_version": version}
            atomic_write(state_path, json.dumps(output, ensure_ascii=False, default=str))
            store.event("delivery", "complete", version=prev["version"], delivery=prev["delivery"],
                        result_path=str(state_path))
            return output
        # Keep a feasible conventional result if the annealer did not beat it. Attribute its source.
        fallback = None
        baseline = inspection.get("baseline_solution")
        if ir is not None and baseline is not None and (delivery["status"] == "no_solution" or
                (delivery.get("quality_gap") is not None and delivery["quality_gap"] > settings.get("quality_tolerance", 0.01))):
            fallback = baseline
            delivery.update(status="limited", message="退火結果未達簡單基準，交付已驗證可行的基準方案",
                            solution_source="baseline", objective=inspection["baseline_objective"],
                            annealer_quality_gap=delivery.get("quality_gap"), quality_gap=0.0)
        elif result is not None and result.best_solution is not None:
            delivery["solution_source"] = "annealer"
        delivered = fallback if fallback is not None else \
            result.best_solution[:ir.n_variables] if result and delivery["status"] != "no_solution" else None
        warnings = list(req_issues) + list(delivery.get("requirement_warnings", []))
        if validator is not None and fallback is not None:
            warnings += validator.solution(bundle["data"], raw_ir or ns["build_ir"](bundle["data"]), fallback)
        if warnings and delivered is not None:
            delivery["requirement_warnings"] = sorted(set(warnings))
            if delivery["status"] == "usable":
                delivery["status"] = "limited"
            delivery["message"] = delivery.get("message", "") + "；獨立需求檢查與模型不一致，需人工確認"
        if metadata.get("formula_changes"):
            delivery["formula_changes"] = metadata["formula_changes"]      # shown for human confirmation
            # an automatic repair changed the formulation: the answer stands, but nobody has confirmed the new formulas
            # yet (batch mode accepted one silently, Melbourne 2026-10-21) - flag it so reports can count it apart
            delivery["review_required"] = True
            delivery["message"] = (delivery.get("message", "") + "；自動修復改了模型公式，尚未經人工確認").lstrip("；")
        output = {"version": version, "delivery": delivery, "inspection": inspection,
                  "loop": result.model_dump() if result is not None else None,
                  "solution": fallback if fallback is not None else
                              result.best_solution[:ir.n_variables] if result and delivery["status"] != "no_solution" else None,
                  "total_solver_calls": calls, "repairs": repairs, "attempts": all_attempts, "diagnoses": diagnoses}
        atomic_write(state_path, json.dumps(output, ensure_ascii=False, default=str))
        store.event("delivery", "complete" if delivery["status"] in ("usable", "limited") else "needs_input",
                    version=version, delivery=delivery, result_path=str(state_path))
        return output
