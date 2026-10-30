from __future__ import annotations
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def generate_latex(
    nl_problem: str,
    spec: StructuredSpec,
    formulation: QUBOFormulation,
    client: LLMClient,
    model: str,
    temperature: float = 0.2,
) -> str:
    prompt = _fill(
        _PROMPT_TEMPLATE,
        nl_problem=nl_problem,
        spec_json=spec.model_dump_json(indent=2),
        formulation_json=formulation.model_dump_json(indent=2),
        problem_name=spec.problem_name,
    )
    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="latex")
    return raw.strip()
