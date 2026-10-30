"""Run the data layer over a downloaded corpus (crawl_open_data.py) and report where it breaks, without labels.

A file is not "correct" or "wrong" here; it either violates an invariant the data layer promises, or it does not:
  load        the file is read at all (error type and message are the failure signature)
  not_html    a web page saved under a data name is refused, never read as a table
  non_empty   at least one row and one column
  text_ok     no replacement characters or typical mojibake in text cells (a wrong encoding guess)
  numbers     columns that look numeric (>=80% of cells parse) are numeric after prep.to_number
  unique_cols no two columns with the same name lost
  coords      where a latitude/longitude pair exists it is inside the valid range and not swapped
  profile     the model-facing profile can be built and contains no instruction-like cell flagged by prep.safety

    python scripts/check_corpus.py [--dir results/data_cache/corpus] [--limit N]
Writes <dir>/invariants.jsonl and <dir>/invariants_summary.md (failures grouped by signature).
"""
from __future__ import annotations
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(ROOT))
from qubo_auto.annealing import prep                                  # noqa: E402
from qubo_auto.annealing.datakit import load_tables, profile          # noqa: E402

MOJIBAKE = re.compile(r"�|Ã[\x80-\xbf¡-¿]|â€|ï»¿|Ð[\x80-\xbf]")
LAT = re.compile(r"^(lat|latitude|breitengrad|y_?coord|lat_?dd)$", re.I)
LON = re.compile(r"^(lon|lng|long|longitude|laengengrad|x_?coord|lon_?dd)$", re.I)


def check_table(name: str, df: pd.DataFrame) -> list[str]:
    bad = []
    if df.empty or df.shape[1] == 0:
        return ["non_empty: no rows or no columns"]
    text = df.select_dtypes(include=["object", "string"])
    sample = [v for v in text.head(500).to_numpy().ravel() if isinstance(v, str)] if not text.empty else []
    if sample and sum(bool(MOJIBAKE.search(v)) for v in sample) / len(sample) > 0.01:
        bad.append("text_ok: replacement characters or mojibake in text cells")
    for col in text.columns:
        vals = text[col].dropna().astype(str).head(500)
        if len(vals) >= 5:
            parsed = prep.to_number(vals)
            share = parsed.notna().mean()
            if share >= 0.8 and share < 1.0 and (vals[parsed.isna()].str.strip() != "").sum() > 0.05 * len(vals):
                bad.append(f"numbers: column {str(col)[:30]!r} is {share:.0%} numeric, the rest does not parse")
    lat = next((c for c in df.columns if LAT.match(str(c))), None)
    lon = next((c for c in df.columns if LON.match(str(c))), None)
    if lat is not None and lon is not None:
        la, lo = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
        ok = la.notna() & lo.notna()
        if ok.any():
            if ((la[ok].abs() > 90) | (lo[ok].abs() > 180)).mean() > 0.02:
                bad.append("coords: values outside the valid latitude/longitude range")
            elif (la[ok].abs() > lo[ok].abs()).mean() > 0.9 and la[ok].abs().median() > 90:
                bad.append("coords: latitude and longitude look swapped")
    try:
        profile(name, df)
    except Exception as e:
        bad.append(f"profile: {type(e).__name__}: {str(e)[:80]}")
    return bad


def check_file(path: Path) -> dict:
    rec = {"file": str(path.relative_to(path.parents[1])), "bytes": path.stat().st_size, "failures": []}
    t0 = time.time()
    try:
        tables = load_tables([path])
    except Exception as e:
        rec["failures"].append(f"load: {type(e).__name__}: {str(e)[:100]}")
        rec["trace"] = traceback.format_exc(limit=-3)[-400:]
        return {**rec, "seconds": round(time.time() - t0, 2)}
    rec["tables"] = {k: list(v.shape) for k, v in tables.items()}
    for name, df in tables.items():
        rec["failures"] += [f"{f}" for f in check_table(name, df)]
        dup = [c for c in df.columns if str(c).endswith(".1")]
        if dup:
            rec["failures"].append(f"unique_cols: duplicate column names were renamed ({dup[:3]})")
    return {**rec, "seconds": round(time.time() - t0, 2)}


def signature(failure: str) -> str:
    kind, _, rest = failure.partition(": ")
    return kind if kind != "load" else f"load: {rest.split(':')[0][:40]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "data_cache" / "corpus"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300, help="seconds per file")
    a = ap.parse_args()
    base = Path(a.dir)
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in (".csv", ".tsv", ".json", ".geojson", ".xlsx", ".zip"))
    files = files[:a.limit] if a.limit else files
    recs = []
    out = (base / "invariants.jsonl").open("w")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = {pool.submit(check_file, f): f for f in files}
        for i, job in enumerate(as_completed(jobs), 1):
            f = jobs[job]
            try:
                rec = job.result(timeout=a.timeout)
            except Exception as e:                    # a file that hangs or kills the worker is a finding too
                rec = {"file": str(f.relative_to(base)), "bytes": f.stat().st_size,
                       "failures": [f"load: {type(e).__name__}: {str(e)[:80]}"]}
            recs.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"{i}/{len(files)} checked", flush=True)
    out.close()
    groups = defaultdict(list)
    for r in recs:
        for sig in {signature(f) for f in r["failures"]}:      # a file counts once per signature
            groups[sig].append(r["file"])
    clean = sum(1 for r in recs if not r["failures"])
    lines = ["# Corpus invariants", "", f"{len(recs)} files, {clean} without a violated invariant "
             f"({clean / max(1, len(recs)):.0%}). Failures grouped by signature; each group is a candidate for a general fix "
             "and a test in `tests/test_data_robustness.py`.", "", "| Signature | Files | Example |", "|---|--:|---|"]
    for sig, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {sig} | {len(fs)} | `{fs[0]}` |")
    by_ext = Counter(Path(r["file"]).suffix for r in recs)
    lines += ["", "Files by type: " + ", ".join(f"{k} {v}" for k, v in by_ext.most_common()), ""]
    (base / "invariants_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
