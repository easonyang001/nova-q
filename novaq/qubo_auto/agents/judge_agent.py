from __future__ import annotations
import json
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec, VerificationResult


_SPEC_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge_agent_spec.txt"
_SPEC_PROMPT_TEMPLATE = _SPEC_PROMPT_PATH.read_text(encoding="utf-8")

_QUBO_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge_agent.txt"
_QUBO_PROMPT_TEMPLATE = _QUBO_PROMPT_PATH.read_text(encoding="utf-8")


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def verify_spec(
    nl_problem: str,
    spec: StructuredSpec,
    client: LLMClient,
    model: str,
    temperature: float,
) -> VerificationResult:
    spec_json = spec.model_dump_json(indent=2)
    prompt = _fill(_SPEC_PROMPT_TEMPLATE,
                   nl_problem=nl_problem, spec_json=spec_json)
    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="verify_spec", json_mode=True)
    data = json.loads(raw)
    return VerificationResult.model_validate(data)


def verify_formulation(
    spec: StructuredSpec,
    formulation: QUBOFormulation,
    client: LLMClient,
    model: str,
    temperature: float,
) -> VerificationResult:
    spec_json = spec.model_dump_json(indent=2)
    formulation_json = formulation.model_dump_json(indent=2)
    prompt = _fill(_QUBO_PROMPT_TEMPLATE,
                   spec_json=spec_json, formulation_json=formulation_json)
    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="verify_formulation", json_mode=True)
    data = json.loads(raw)
    return VerificationResult.model_validate(data)
