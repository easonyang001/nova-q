"""Convert human-written test case descriptions into benchmark JSON files.

The user provides instance data (connections, weights, energies, optimal bitstrings)
in any plain-text format. Called after formulation so the LLM has full QUBO context
to correctly map instance keys and sign the optimal_value.

Supports multiple test cases in a single input string.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec
from ..verification.test_runner import MAX_BRUTE_FORCE_VARS

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "debugger_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def _formulation_context(spec: StructuredSpec, formulation: QUBOFormulation) -> str:
    return (
        f"objective direction : {spec.objective.direction}\n"
        f"n_variables         : {formulation.n_variables}\n"
        f"variable_mapping    : {json.dumps(formulation.variable_mapping)}\n"
        f"instance_parameters : {spec.instance_parameters}"
    )


def _extract_json(raw: str) -> list[dict]:
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nError: {e}\nRaw:\n{raw[:600]}")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array of test cases, got: {type(parsed)}")
    return parsed


def _validate_case(case: dict, index: int) -> list[str]:
    errors = []
    for field in ("n_variables", "instance", "ground_truth"):
        if field not in case:
            errors.append(f"case[{index}] missing '{field}'")
    n = case.get("n_variables")
    if isinstance(n, int) and n > MAX_BRUTE_FORCE_VARS:
        errors.append(f"case[{index}] has {n} variables; brute-force verification handles at most "
                      f"{MAX_BRUTE_FORCE_VARS}. Use a smaller instance (fewer items / nodes) for this case")
    if "ground_truth" in case:
        gt = case["ground_truth"]
        if "optimal_value" not in gt:
            errors.append(f"case[{index}] ground_truth missing 'optimal_value'")
        if "optimal_bitstrings" not in gt:
            errors.append(f"case[{index}] ground_truth missing 'optimal_bitstrings'")
        else:
            n = case.get("n_variables")
            for bs in gt["optimal_bitstrings"]:
                if n and len(bs) != n:
                    errors.append(
                        f"case[{index}] bitstring length {len(bs)} != n_variables {n}"
                    )
    return errors


def build_test_cases(
    benchmark_dir: Path,
    raw_input: str,
    client: LLMClient,
    model: str,
    spec: StructuredSpec,
    formulation: QUBOFormulation,
    temperature: float = 0.1,
    max_retries: int = 3,
    cases_dir: Path | None = None,
) -> list[Path]:
    """Parse raw_input and write one JSON file per test case into benchmark_dir/cases/.

    Args:
        benchmark_dir: path to benchmarks/<name>/ (must contain prompt.txt)
        raw_input:     plain-text description(s) of one or more test cases
        client:        LLMClient instance
        model:         model identifier string
        spec:          StructuredSpec from the planner_agent (provides objective direction)
        formulation:   QUBOFormulation from the math expert (provides variable mapping)
        temperature:   LLM sampling temperature
        max_retries:   retry attempts if LLM output is invalid

    Returns:
        list of paths to the written JSON files
    """
    problem_prompt = (benchmark_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    cases_dir = cases_dir if cases_dir is not None else benchmark_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    context = _formulation_context(spec, formulation)
    previous_feedback = ""

    for attempt in range(max_retries):
        prompt = _fill(
            _PROMPT_TEMPLATE,
            problem_prompt=problem_prompt,
            formulation_context=context,
            raw_input=raw_input,
            previous_feedback=previous_feedback,
        )
        raw = client.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            step_name="parse_test_cases",
        )
        try:
            cases = _extract_json(raw)
        except ValueError as e:
            previous_feedback = f"\nPREVIOUS ATTEMPT FAILED:\n{e}\nFix and return valid JSON array.\n"
            continue

        all_errors: list[str] = []
        for i, case in enumerate(cases):
            all_errors.extend(_validate_case(case, i))

        if all_errors:
            previous_feedback = (
                "\nPREVIOUS ATTEMPT HAD VALIDATION ERRORS — fix all before responding:\n"
                + "\n".join(f"  - {e}" for e in all_errors)
                + "\n"
            )
            continue

        written: list[Path] = []
        for case in cases:
            name = case.get("name") or f"case_{len(list(cases_dir.glob('*.json'))) + len(written) + 1}"
            out = {
                "name": name,
                "description": case.get("description", ""),
                "n_variables": case["n_variables"],
                "instance": case["instance"],
                "ground_truth": {
                    "optimal_value": float(case["ground_truth"]["optimal_value"]),
                    "optimal_bitstrings": [list(bs) for bs in case["ground_truth"]["optimal_bitstrings"]],
                },
            }
            path = cases_dir / f"{name}.json"
            path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            written.append(path)

        return written

    raise RuntimeError(
        f"Failed to parse test cases after {max_retries} attempts.\n"
        f"Last feedback: {previous_feedback}"
    )


# ---------------------------------------------------------------- problem-level cases (verification.mode = problem_level)
_REF_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "reference_cases.txt").read_text(encoding="utf-8")


def _shape(v):
    """JSON shape used to check that every reference solution follows one format."""
    if isinstance(v, dict):
        return ("object", tuple(sorted(v)))
    if isinstance(v, list):
        return ("array",)
    return (type(v).__name__,)


def _validate_reference(data, direction: str) -> list[str]:
    if not isinstance(data, dict):
        return ["expected one JSON object with 'solution_format' and 'cases'"]
    errors = []
    if not str(data.get("solution_format", "")).strip():
        errors.append("missing 'solution_format'")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["'cases' must be a non-empty list"]
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            errors.append(f"cases[{i}] is not an object")
            continue
        if not isinstance(c.get("instance"), dict):
            errors.append(f"cases[{i}] missing 'instance' object")
        try:
            float(c.get("optimal_value"))
        except (TypeError, ValueError):
            errors.append(f"cases[{i}] 'optimal_value' must be a number")
    refs = [(i, c["reference_solution"]) for i, c in enumerate(cases)
            if isinstance(c, dict) and c.get("reference_solution") is not None]
    if len({_shape(r) for _, r in refs}) > 1:
        errors.append("reference_solution values do not share one format: "
                      + "; ".join(f"cases[{i}] is {_shape(r)[0]}" + (f" with keys {list(_shape(r)[1])}" if len(_shape(r)) > 1 else "")
                                  for i, r in refs)
                      + ". Every reference_solution must follow solution_format exactly")
    fmt = str(data.get("solution_format", "")).lstrip()
    if refs and fmt.startswith("{") and not all(isinstance(r, dict) for _, r in refs):
        errors.append("solution_format describes a JSON object, but a reference_solution is not an object")
    if refs and fmt.startswith("[") and not all(isinstance(r, list) for _, r in refs):
        errors.append("solution_format describes a JSON array, but a reference_solution is not an array")
    return errors


def build_reference_cases(
    benchmark_dir: Path,
    raw_input: str,
    client: LLMClient,
    model: str,
    spec: StructuredSpec,
    temperature: float = 0.1,
    max_retries: int = 3,
    cases_dir: Path | None = None,
    feedback: str = "",
) -> list[Path]:
    """Transcribe the benchmark's human-written cases into problem-level reference cases (instance, stated
    optimal value, stated optimal solution) plus one shared solution format. No formulation is shown: the
    cases must not depend on how the problem is encoded."""
    from ..verification.reference import save_reference_cases
    problem_prompt = (benchmark_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    cases_dir = cases_dir if cases_dir is not None else benchmark_dir / "cases"
    previous_feedback = feedback
    for _ in range(max_retries):
        prompt = _fill(_REF_PROMPT_TEMPLATE, problem_prompt=problem_prompt, direction=spec.objective.direction,
                       instance_parameters=", ".join(spec.instance_parameters) or "(none given)",
                       raw_input=raw_input, previous_feedback=previous_feedback)
        raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="parse_reference_cases")
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            previous_feedback = f"\nPREVIOUS ATTEMPT WAS NOT VALID JSON ({e}). Return one JSON object.\n"
            continue
        errors = _validate_reference(data, spec.objective.direction)
        if errors:
            previous_feedback = ("\nPREVIOUS ATTEMPT HAD ERRORS — fix all of them:\n"
                                 + "\n".join(f"  - {e}" for e in errors) + "\n")
            continue
        return save_reference_cases(data["cases"], str(data["solution_format"]), cases_dir)
    raise RuntimeError(f"Failed to parse reference cases after {max_retries} attempts.\n"
                       f"Last feedback: {previous_feedback}")
