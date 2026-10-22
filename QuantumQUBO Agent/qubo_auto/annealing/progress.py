"""Durable stage status and concrete next action across UI restarts."""
import json
from pathlib import Path
from .artifacts import atomic_write


def record_progress(root, stage, status, **details):
    path = Path(root) / "task_state.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    action = {"data": "prepare_data", "model": "build_model", "preflight": "solve",
              "solve": "resume_solver", "recovery": "review_recovery", "delivery": "view_result"}.get(stage, stage)
    state = {"stage": stage, "status": status, "next_action": action,
             "version": details.get("version", previous.get("version")), "details": details}
    if stage == "recovery" and status == "needs_input":
        state["next_action"] = "confirm_requirements_or_data"
    atomic_write(path, json.dumps(state, ensure_ascii=False, default=str))
    return state


def load_progress(root):
    path = Path(root) / "task_state.json"
    return json.loads(path.read_text()) if path.exists() else {}
