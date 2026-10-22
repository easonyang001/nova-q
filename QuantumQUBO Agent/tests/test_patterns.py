"""Modelling-pattern retrieval for the Modeling Agent."""
import json

from qubo_auto.annealing import modeler


def test_library_covers_the_problem_categories_and_cites_sources():
    lib = modeler.load_patterns()
    assert len(lib) >= 15 and len({p["id"] for p in lib}) == len(lib)
    tags = {t for p in lib for t in p["tags"]}
    for category in ("scheduling", "routing", "location", "assignment", "knapsack", "portfolio", "graph",
                     "set_cover", "clustering", "network_design", "resource_allocation", "feature_selection"):
        assert category in tags, category
    assert all(p["source"] for p in lib)


def test_selection_keeps_known_ids_only_and_returns_their_text():
    class Fake:
        def call(self, model, prompt, **kw):
            assert "mp-position-tour" in prompt and "{catalogue}" not in prompt
            return json.dumps({"ids": ["mp-position-tour", "made-up", "mp-cardinality", "mp-capacity"]})
    ids, text = modeler.select_patterns('{"problem_summary": "tour"}', Fake(), "m", k=2)
    assert ids == ["mp-position-tour", "mp-cardinality"] and "NO subtour" in text


def test_patterns_reach_the_modelling_prompt_and_are_recorded():
    from test_schema import MODEL
    good = MODEL.replace("%EXTRA%", "")
    prompts = []

    class Fake:
        def call(self, model, prompt, step_name="", **kw):
            prompts.append((step_name, prompt))
            if step_name == "pattern_pick":
                return json.dumps({"ids": ["mp-coverage-once"]})
            if step_name == "visualizer":
                return "```python\ndef visualize(data, x):\n    return {'type': 'selection', 'items': []}\n```"
            return f"```python\n{good}```"
    code, ns, errors, report = modeler.build_model("{}", "[]", Fake(), "m", patterns=True)
    model_prompt = next(p for s, p in prompts if s == "model")
    assert "GENERAL MODELLING PATTERNS" in model_prompt and "counts once" in model_prompt
    assert report["patterns"] == ["mp-coverage-once"]
    prompts.clear()
    modeler.build_model("{}", "[]", Fake(), "m")                       # off by default: no retrieval call
    assert not any(s == "pattern_pick" for s, _ in prompts)
    assert "GENERAL MODELLING PATTERNS" not in next(p for s, p in prompts if s == "model")
