"""Immutable model/data bundles and an atomic current pointer for resume/rollback."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def commit(self, code: str, data: dict, label: str, metadata: dict | None = None) -> str:
        text = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        # Preserve any pre-existing working pair before its first replacement.
        if not (self.root / "active.json").exists() and (self.root / "model.py").exists() and (self.root / "data.json").exists():
            old_code = (self.root / "model.py").read_text(encoding="utf-8")
            old_data = json.loads((self.root / "data.json").read_text(encoding="utf-8"))
            if old_code != code or old_data != data:
                self._save(old_code, old_data, "先前版本", {})
        version = self._save(code, data, label, metadata or {})
        atomic_write(self.root / "active.json", json.dumps({"version": version}))
        # Compatibility exports; readers use the atomic active pointer.
        atomic_write(self.root / "model.py", code)
        atomic_write(self.root / "data.json", text)
        return version

    def _save(self, code, data, label, metadata):
        text = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        provenance = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
        version = hashlib.sha256((code + "\0" + text + "\0" + provenance).encode()).hexdigest()[:24]
        dest = self.root / "versions" / version
        if not (dest / "bundle.json").exists():
            atomic_write(dest / "bundle.json", json.dumps({"version": version, "code": code, "data": data,
                         "label": label, "metadata": metadata, "created": time.time()}, ensure_ascii=False, allow_nan=False))
        return version

    def current(self) -> dict:
        active = self.root / "active.json"
        if active.exists():
            return self.read(json.loads(active.read_text())["version"])
        return {"version": "legacy", "code": (self.root / "model.py").read_text(encoding="utf-8"),
                "data": json.loads((self.root / "data.json").read_text(encoding="utf-8")), "metadata": {}}

    def read(self, version: str) -> dict:
        if len(version) != 24 or any(c not in "0123456789abcdef" for c in version):
            raise ValueError("invalid version")
        return json.loads((self.root / "versions" / version / "bundle.json").read_text(encoding="utf-8"))

    def restore(self, version: str):
        bundle = self.read(version)
        atomic_write(self.root / "active.json", json.dumps({"version": version}))
        atomic_write(self.root / "model.py", bundle["code"])
        atomic_write(self.root / "data.json", json.dumps(bundle["data"], ensure_ascii=False))
        return bundle

    def versions(self) -> list[dict]:
        return sorted([json.loads(p.read_text(encoding="utf-8")) for p in (self.root / "versions").glob("*/bundle.json")],
                      key=lambda b: b["created"], reverse=True)

    def event(self, stage: str, status: str, **details):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "workflow.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), "stage": stage, "status": status, **details}, ensure_ascii=False, default=str) + "\n")
        from .progress import record_progress
        record_progress(self.root, stage, status, **details)
