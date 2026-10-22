"""Source facts recorded by code, with content fingerprints and unknown licences."""
import hashlib
from datetime import datetime, timezone


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(candidate, path):
    return {"title": candidate.get("title", path.name),
            "url": candidate.get("page") or candidate.get("url", ""),
            "resource_url": candidate.get("url", ""),
            "publisher": candidate.get("source", ""),
            "license": candidate.get("license_from_page") or "待確認",
            "license_evidence": candidate.get("license_evidence", ""),
            "retrieved": datetime.now(timezone.utc).isoformat(),
            "sha256": file_digest(path), "bytes": path.stat().st_size}
