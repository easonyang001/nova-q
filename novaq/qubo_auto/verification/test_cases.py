"""Loader for benchmark test cases stored as JSON files.

Each benchmark folder has structure:
    benchmarks/<name>/
        prompt.txt
        cases/
            *.json     (one test case per file)

Test case JSON schema:
    {
        "name": "human-readable identifier",
        "description": "optional description",
        "n_variables": <int>,
        "instance": { ... benchmark-specific dict passed to build_qubo ... },
        "ground_truth": {
            "optimal_value": <float>,
            "optimal_bitstrings": [[0,1,0,...], ...],   // required
            "all_strings": [                              // optional, for debug
                {"x": [0,1,0,...], "objective": <float>, "feasible": <bool>},
                ...
            ]
        }
    }
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    name: str
    description: str
    n_variables: int
    instance: dict
    optimal_value: float
    optimal_bitstrings: list[tuple[int, ...]]   # set of optimal x's
    all_strings: list[dict] | None              # full enumeration if provided


def load_test_cases(benchmark_dir: Path, cases_dir: Path | None = None) -> list[TestCase]:
    cases_dir = cases_dir if cases_dir is not None else benchmark_dir / "cases"
    if not cases_dir.exists():
        raise FileNotFoundError(f"No 'cases/' folder in {cases_dir}")

    cases: list[TestCase] = []
    for json_path in sorted(cases_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        gt = data["ground_truth"]
        cases.append(TestCase(
            name=data.get("name", json_path.stem),
            description=data.get("description", ""),
            n_variables=data["n_variables"],
            instance=data["instance"],
            optimal_value=float(gt["optimal_value"]),
            optimal_bitstrings=[tuple(b) for b in gt["optimal_bitstrings"]],
            all_strings=gt.get("all_strings"),
        ))
    if not cases:
        raise FileNotFoundError(f"No JSON files found in {cases_dir}")
    return cases


def load_prompt(benchmark_dir: Path) -> str:
    prompt_path = benchmark_dir / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"No prompt.txt in {benchmark_dir}")
    return prompt_path.read_text(encoding="utf-8").strip()
