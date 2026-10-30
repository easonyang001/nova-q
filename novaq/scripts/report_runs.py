"""Cost, repair and provenance report over saved end-to-end runs (results/**/record.json).

Answers three reviewer questions from files that already exist:
  * what does one run cost (LLM tokens, wall clock, solver calls, and dollars at a price you give)?
  * how many passes depended on an automatic repair, and how many of those changed formulas without review?
  * which solver backend produced each delivery, and which batches have no manifest (frozen code hash) or no
    recorded backend, so a reader cannot tell what ran?

    python scripts/report_runs.py [results_dir ...] [--usd-per-mtok 0.0] [--out report.md]
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(paths):
    for base in paths:
        for f in sorted(Path(base).rglob("record.json")):
            try:
                yield f, json.loads(f.read_text())
            except json.JSONDecodeError:
                print(f"skipped (unreadable): {f}")


def judged(rec):
    v = rec.get("verdict")
    return isinstance(v, dict) and all(isinstance(v.get(k), dict) for k in ("data", "model", "delivery"))


def passed(rec):
    return judged(rec) and all(rec["verdict"][k].get("ok") is True for k in ("data", "model", "delivery"))


def batch_of(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return rel.parts[0]


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def summarize(rows, usd_per_mtok):
    out = ["| Batch | Runs | Judged | Passed | Pass w/ repair | Pass w/ formula change | Review flag | Median min | Median ktok | USD/run |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for batch, recs in sorted(rows.items()):
        j = [r for r in recs if judged(r)]
        p = [r for r in j if passed(r)]
        rep = [r for r in p if r.get("repairs")]
        fc = [r for r in p if (r.get("delivery") or {}).get("formula_changes")]
        rv = [r for r in recs if (r.get("delivery") or {}).get("review_required")]
        mins = [r["seconds"] / 60 for r in recs if isinstance(r.get("seconds"), (int, float))]
        toks = [(r.get("cost") or {}).get("llm_tokens") for r in recs]
        toks = [t / 1000 for t in toks if isinstance(t, (int, float))]
        usd = f"{median(toks) * usd_per_mtok / 1000:.2f}" if usd_per_mtok and toks else "–"
        out.append(f"| {batch} | {len(recs)} | {len(j)} | {len(p)} | {len(rep)} | {len(fc)} | {len(rv)} | "
                   f"{median(mins):.0f} | {median(toks):.0f} | {usd} |")
    return out


def provenance(rows, base):
    lines = ["| Batch | Backends recorded | Without backend field | manifest.json |", "|---|---|--:|---|"]
    for batch, recs in sorted(rows.items()):
        backends = Counter(str(r["backend"]) for r in recs if r.get("backend"))
        missing = sum(1 for r in recs if not r.get("backend"))
        manifest = (base / batch / "manifest.json").exists()
        lines.append(f"| {batch} | {dict(backends) or '–'} | {missing} | {'yes' if manifest else '**no**'} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=[str(ROOT / "results")])
    ap.add_argument("--usd-per-mtok", type=float, default=0.0, help="blended price per million tokens; 0 hides dollars")
    ap.add_argument("--out", help="write the report here instead of stdout")
    a = ap.parse_args()
    rows = defaultdict(list)
    base = Path(a.dirs[0])
    for f, rec in load(a.dirs):
        rows[batch_of(f, base)].append(rec)
    allrecs = [r for recs in rows.values() for r in recs]
    p = [r for r in allrecs if passed(r)]
    text = ["# Run report", "",
            f"{len(allrecs)} runs in {len(rows)} batches; {len(p)} passed the judge; "
            f"{sum(1 for r in p if r.get('repairs'))} of those passes went through an automatic repair, "
            f"{sum(1 for r in p if (r.get('delivery') or {}).get('formula_changes'))} with changed formulas.", "",
            "## Cost, repairs and review flags", "", *summarize(rows, a.usd_per_mtok), "",
            "## Provenance", "", *provenance(rows, base), "",
            "Runs without a `backend` field predate its recording: they cannot show which solver delivered. "
            "A batch without `manifest.json` has no frozen code hash.", ""]
    report = "\n".join(text)
    if a.out:
        Path(a.out).write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
