from __future__ import annotations
import json
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import StructuredSpec


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_SCHEMA_JSON = json.dumps(StructuredSpec.model_json_schema(), indent=2)


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def restructure(
    nl_problem: str,
    client: LLMClient,
    model: str,
    temperature: float,
    previous_issues: list[str] | None = None,
) -> StructuredSpec:
    prompt = _fill(_PROMPT_TEMPLATE, nl_problem=nl_problem, schema_json=_SCHEMA_JSON)

    if previous_issues:
        issues_block = "\n".join(f"  - {i}" for i in previous_issues)
        prompt += (
            "\n\n--- FEEDBACK FROM PREVIOUS ATTEMPT ---\n"
            "The verifier rejected your last structured spec with these issues:\n"
            f"{issues_block}\n"
            "Fix ALL of them before responding."
        )

    raw = client.call(model=model, prompt=prompt, temperature=temperature,
                      step_name="restructure", json_mode=True)
    data = json.loads(raw)
    return StructuredSpec.model_validate(data)
