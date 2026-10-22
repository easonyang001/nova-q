"""Live UCI acceptance pilot: confirmed spec -> source discovery -> LLM model/adapter -> SA -> result.

python scripts/run_real_acceptance.py --out results/acceptance/wine_live
This pilot uses a specified official source page, not unrestricted autonomous web search.
"""
import argparse
import json
import os
import sys
import time
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from qubo_auto.annealing import modeler
from qubo_auto.annealing.acquisition import prepare_real_data
from qubo_auto.annealing.artifacts import ArtifactStore, atomic_write
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.llm_client import LLMClient
from qubo_auto.annealing.workflow import run_workflow


def export_delivery(out, data, x, result, report):
    """Save a reviewable result independently of generated visualization code."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (out / "feature_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "selected", "abs_quality_correlation"])
        writer.writerows(zip(data["names"], x, data["relevance"]))
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.barh(data["names"], data["relevance"], color=["#176b91" if b else "#b9c3cc" for b in x])
    axis.set_xlabel("Absolute Pearson correlation with quality")
    axis.set_title("UCI red wine: selected features in blue (5 of 11)")
    fig.tight_layout()
    fig.savefig(out / "feature_selection.png", dpi=160)
    plt.close(fig)
    relevant = sum(r * b for r, b in zip(data["relevance"], x))
    redundant = 0.5 * sum(data["redundancy"][i][j] * x[i] * x[j]
        for i in range(len(x)) for j in range(i + 1, len(x)))
    text = ["# 真實紅酒資料：特徵選擇成果", "", f"使用 {report['original_rows']:,} 筆 UCI 紅酒紀錄，從 11 個理化特徵選出 5 個。",
            "", "選取：" + "、".join(report["selected_features"]), "",
            f"相關總和 {relevant:.10f} − 加權冗餘 {redundant:.10f} = {report['delivered_objective']:.10f}。",
            f"獨立枚舉最佳值：{report['oracle_objective']:.10f}；原需求檢查：恰 5 個，不包含 quality 或 ID。",
            f"交付來源：{result['delivery']['solution_source']}；求解次數：{result['total_solver_calls']}；版本：{result['version']}。", "",
            "[各特徵分數](feature_scores.csv) · [圖表](feature_selection.png) · [完整驗收紀錄](acceptance.json)", "",
            "所有相關數值重新由實際 CSV 計算核對；所有 2,048 組二元決策的目標與限制都做獨立檢查。這個目標衡量相關性與冗餘，不代表預測準確率。", ""]
    atomic_write(out / "delivery.md", "\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/acceptance/wine_live")
    parser.add_argument("--model-file", type=Path, help="Reuse a previously generated model; it is validated again")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "docs/acceptance_cases.json").read_text())
    case = next(c for c in cases if c["id"] == "wine_feature_selection")
    spec = {"problem_summary": case["prompt"] + "\nSOURCE PAGE: " + case["source_url"],
        "constraints": [{"name": "choose_five", "strength": "hard", "type": "cardinality",
                         "description": "Choose exactly five original physicochemical features; exclude quality and IDs"}],
        "objectives": [{"name": "relevance_minus_redundancy", "direction": "maximize",
                        "description": "sum(abs Pearson(feature,quality))*x - 0.5*sum(i<j,abs Pearson(feature_i,feature_j))*x_i*x_j"}],
        "assumptions": ["Use all red-wine records only. k=5 and redundancy weight=0.5 are confirmed user settings.",
                        "Required DATA_SCHEMA keys exactly: names (original feature names), relevance (absolute target correlations), redundancy (absolute feature correlations), k=5. Do not rename or omit these fields.",
                        "REQUIREMENT_MAP must map choose_five to an actual returned IR constraint name. names/relevance/redundancy are public-derived fields; k is a user setting. Metadata unit for correlations is dimensionless. Period/geography are unspecified: omit them.",
                        "Keep the model module under 120 lines; no long explanatory comments.",
                        "Correlation coefficients are dimensionless; no assumption about predictive accuracy."]}
    spec_text = json.dumps(spec, ensure_ascii=False)
    atomic_write(out / "confirmed_spec.json", spec_text)
    os.environ["LLM_ENABLE_THINKING"] = "false"
    client = LLMClient(out / "transcripts", "live", max_tokens=100_000, use_cache=False)
    client.client = client.client.with_options(timeout=180, max_retries=0)
    model = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
    started = time.time()
    report = {"case": case["id"], "mode": "live local LLM + official-source download + real-data adapter + local SA",
              "source_discovery": "specified official publisher page", "success": False}
    if (out / "acceptance.json").exists():
        atomic_write(out / f"acceptance_previous_{time.time_ns()}.json", (out / "acceptance.json").read_text())
    try:
        print("[1/4] Generate and validate model", flush=True)
        if args.model_file:
            code = args.model_file.read_text()
            ns = modeler.load_module(code)
            errors, validation = modeler.validate(ns)
            from qubo_auto.annealing.contracts import check_requirements
            errors += check_requirements(spec, ns, ns["build_ir"](ns["make_example_data"](11, 0)))
            report["reused_model"] = str(args.model_file.resolve())
        else:
            code, ns, errors, validation = modeler.build_model(spec_text, "[]", client, model, max_attempts=3, backend="local SA")
        atomic_write(out / "generated_model.py", code)
        if errors:
            raise ValueError("model: " + "; ".join(errors))
        store = ArtifactStore(out / "artifacts")
        store.commit(code, ns["make_example_data"](11, 0), "synthetic model check", {"source": "synthetic", "confirmed_spec": spec})
        print("[2/4] Discover/download source and run live adapter", flush=True)
        data, errors, prep = prepare_real_data(code, spec_text, client, model, store.root / "data_files",
                    notes="Use only winequality-red. Return names, relevance, redundancy and k=5. All rows; no mock values.",
                    search_rounds=2, max_downloads=2, log=lambda m: print(m, flush=True))
        report["preprocessing"] = prep
        # Only these exact descriptive notes can be discharged by the independent
        # case-specific audit below. Unknown metadata/assumptions still stop.
        verified_descriptions = {
            "names": "Original feature names from UCI Wine Quality dataset.",
            "relevance": "Absolute Pearson correlation between each physicochemical feature and quality.",
            "redundancy": "Absolute Pearson correlation between pairs of physicochemical features.",
            "k": "User specified to select exactly 5 features."}
        allowed_reviews = {f"{key}: {description}" for key, description in verified_descriptions.items()}
        unverified_reviews = [r for r in prep.get("review_required", []) if r not in allowed_reviews]
        if errors or unverified_reviews:
            raise ValueError("data needs correction/confirmation: " + "; ".join(errors + prep.get("review_required", [])))
        # Independent check of the requested statistics on the actual downloaded rows.
        from qubo_auto.annealing.datakit import load_tables
        import numpy as np
        files = [p for p in (store.root / "data_files").iterdir() if not p.name.startswith("_")]
        tables = {}
        for file in files:
            try:
                tables.update(load_tables([file]))
            except Exception:
                # Acquisition already records unreadable alternate candidates.
                # Audit the actual readable source rather than unrelated files.
                continue
        red = next(df for name, df in tables.items() if "red" in name.lower())
        features = [name for name in red.columns if name != "quality"]
        if set(data["names"]) != set(features) or len(data["names"]) != 11 or data.get("k", 5) != 5:
            raise ValueError("independent check: feature names/count/k mismatch")
        names = data["names"]
        relevance = red[names].corrwith(red["quality"]).abs().to_numpy()
        redundancy = red[names].corr().abs().to_numpy()
        if not np.allclose(data["relevance"], relevance) or not np.allclose(data["redundancy"], redundancy):
            raise ValueError("independent check: adapter correlations differ from original rows")
        ir = modeler.to_ir(ns["build_ir"](data), "independent")
        objective_sign = 1 if ir.sense == "maximize" else -1
        report["objective_convention"] = "original maximization" if objective_sign == 1 else "equivalent minimization of the negative objective"
        oracle_objective = float("-inf")
        for bits in itertools.product((0, 1), repeat=11):
            expected = sum(relevance[i] * bits[i] for i in range(11)) - 0.5 * sum(
                redundancy[i,j] * bits[i] * bits[j] for i in range(11) for j in range(i+1,11))
            if not np.isclose(objective_sign * ir.objective(bits), expected) or all(c.satisfied(bits) for c in ir.constraints) != (sum(bits) == 5):
                raise ValueError("independent check: generated model differs from the confirmed formula")
            if sum(bits) == 5:
                oracle_objective = max(oracle_objective, expected)
        report["confirmed_notes"] = prep.get("review_required", [])
        report["confirmation_basis"] = "case-specific independent source/statistic audit; complete 2048-assignment model check"
        store.commit(code, data, "validated real red-wine data", {"source": "real", "report": prep,
                     "confirmed_spec": spec, "data_review_approved": bool(prep.get("review_required"))})
        print("[3/4] Solve and assess delivery", flush=True)
        cfg = {"loop": {"max_solver_calls": 4, "patience": 4}, "workflow": {"max_repairs": 0},
               "penalty_search": {"compiler": "pyqubo"}, "param_caps": {"sa": {"num_reads": 80, "num_sweeps": 500}}}
        result = run_workflow(store, get_backend("sa"), lambda: make_policy("penalty_tpe"), cfg,
                              log=lambda m: print(m, flush=True))
        report["workflow"] = result
        x = result["solution"]
        report["success"] = x is not None and sum(x) == 5
        report["original_rows"] = len(red)
        report["selected_features"] = [name for name, bit in zip(names, x or []) if bit]
        report["independent_checks"] = "source correlations + all 2048 model/objective/feasibility assignments + original cardinality"
        report["oracle_objective"] = oracle_objective
        report["delivered_objective"] = objective_sign * ir.objective(x) if x is not None else None
        if report["success"]:
            export_delivery(out, data, x, result, report)
        print("[4/4] Saved delivery:", report["success"], report["selected_features"], flush=True)
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        print(report["error"], flush=True)
    report.update(wall_time_s=time.time()-started, llm_tokens=client.tokens_used, llm_cost=client.cost_used)
    report["recorded_session_llm_tokens"] = sum(json.loads(p.read_text()).get("usage", {}).get("total_tokens", 0)
        for p in (out / "transcripts").rglob("*.json"))
    atomic_write(out / "acceptance.json", json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
