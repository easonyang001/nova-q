from __future__ import annotations
import json
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "formulizer_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_SCHEMA_JSON = json.dumps(QUBOFormulation.model_json_schema(), indent=2)


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def _sanitize(data: dict) -> dict:
    for pt in data.get("penalty_terms", []):
        if isinstance(pt.get("penalty_weight"), str):
            pt["penalty_weight"] = 1.0
    return data


def formulate(
    spec: StructuredSpec,
    client: LLMClient,
    model: str,
    temperature: float,
    previous_issues: list[str] | None = None,
    feedback_source: str = "verifier",
) -> QUBOFormulation:
    spec_json = spec.model_dump_json(indent=2)
    prompt = _fill(_PROMPT_TEMPLATE, spec_json=spec_json, schema_json=_SCHEMA_JSON)

    if previous_issues:
        issues_block = "\n".join(f"  - {i}" for i in previous_issues)
        prompt += (
            f"\n\n--- FEEDBACK FROM PREVIOUS ATTEMPT [{feedback_source.upper()}] ---\n"
            f"Your last formulation was rejected. Issues found:\n"
            f"{issues_block}\n"
            "Fix ALL of them. Do not repeat the same mistakes."
        )

    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="formulate", json_mode=True)
    data = _sanitize(json.loads(raw))
    return QUBOFormulation.model_validate(data)
