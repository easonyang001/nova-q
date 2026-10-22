"""Checker agent: writes evaluate(instance, solution) -> (feasible, objective) from the problem statement alone.

It is the referee of problem-level verification (verification/reference.py), so it sees neither the spec's
formulation nor the Coder's code, and it is trusted on a case only after it reproduces the benchmark's stated
optimal value on the benchmark's stated optimal solution.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from ..llm_client import LLMClient
from .coder_agent import _fill

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "checker_agent.txt").read_text(encoding="utf-8")


def generate_checker(
    problem_prompt: str,
    direction: str,
    solution_format: str,
    sample_instance: dict,
    client: LLMClient,
    model: str,
    temperature: float,
    previous_issues: list[str] | None = None,
) -> str:
    feedback = ""
    if previous_issues:
        feedback = ("\nYOUR PREVIOUS CHECKER DISAGREED WITH THE BENCHMARK — re-read the problem statement and fix "
                    "the feasibility or objective logic (the benchmark's reference solutions are stated to be "
                    "optimal):\n" + "\n".join(f"- {i}" for i in previous_issues) + "\n")
    prompt = _fill(_PROMPT_TEMPLATE, problem_prompt=problem_prompt.strip(), direction=direction,
                   solution_format=solution_format.strip(), sample_instance=json.dumps(sample_instance, indent=2),
                   previous_feedback=feedback)
    raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="checker")
    return _extract(raw)


def _extract(raw: str) -> str:
    import re
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    code = m.group(1).strip() if m else re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    if "def evaluate" not in code:
        raise ValueError(f"Could not find `def evaluate` in checker response:\n{raw[:500]}")
    return code


def compile_checker(code: str) -> Callable:
    namespace: dict = {}
    exec(compile(code, "<checker>", "exec"), namespace)  # noqa: S102
    if "evaluate" not in namespace:
        raise ValueError("Checker did not define `evaluate`")
    return namespace["evaluate"]
