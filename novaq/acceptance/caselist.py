"""The acceptance cases: acceptance/cases.json plus, when present, held-out cases kept out of the public
repository in acceptance/local/cases_extra.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load() -> dict:
    data = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    extra = HERE / "local" / "cases_extra.json"
    if extra.exists():
        data["cases"] = data["cases"] + json.loads(extra.read_text(encoding="utf-8"))["cases"]
    return data
