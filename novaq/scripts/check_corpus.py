"""Run the data layer over a downloaded corpus (crawl_open_data.py) and report where it breaks, without labels.

A file is not "correct" or "wrong" here; it either violates an invariant the data layer promises, or it does not.
Version 2 (2026-10-31) states each invariant as the promise it checks; version 1 also counted correct behaviour as
violations (code columns such as bus route '53A' as unparsed numbers, a repeated column renamed '.1' as lost). The
version 1 results are kept in docs/data_portal_coverage.md for comparison.
  load        the file is read, or refused with a reason when it holds no table (archives of PDFs or images, web
              pages): refusals are reported as their own outcome, neither read nor failed
  non_empty   the file yields at least one table with rows and columns
  text_ok     no replacement characters or typical mojibake in text cells (a wrong encoding guess)
  numbers     a value written purely as a number (digits, separators, sign, currency, percent, footnote marks; no
              letters) is parsed by prep.to_number in a column that is mostly numeric
  coords      a column named latitude/longitude holds degrees, or the profile warns that it does not
  profile     the model-facing profile can be built
Columns renamed because the source repeats a name are listed (info), not counted: nothing is lost.

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
LAT = re.compile(r"^(lat|latitude|breitengrad|lat_?dd)$", re.I)        # degree names only: x_coord may be metres
LON = re.compile(r"^(lon|lng|long|longitude|laengengrad|lon_?dd)$", re.I)
# a value that is nothing but a number: no letters, so codes like '53A', '800C' or 'V58.43' never count
# commas only as thousands groups (1,234,567) or one decimal comma (12,5): '1,3,4' is a list of three numbers
PURE_NUMBER = re.compile(r"^[\s(+\-−$€£¥]*(\d{1,3}([,'’\s\u00a0\u202f]\d{3})+|\d+)([.,]\d+)?\)?\s*(%|[*†‡§]+)?\s*$")


def check_table(name: str, df: pd.DataFrame) -> list[str]:
    bad = []
    if df.empty or df.shape[1] == 0:
        return ["non_empty: no rows or no columns"]
    text = df.select_dtypes(include=["object", "string"])
    sample = [v for v in text.head(500).to_numpy().ravel() if isinstance(v, str)] if not text.empty else []
    damaged_at_source = any("replacement characters" in w for w in df.attrs.get("load_warnings", []))
    if sample and sum(bool(MOJIBAKE.search(v)) for v in sample) / len(sample) > 0.01 and not damaged_at_source:
        bad.append("text_ok: replacement characters or mojibake in text cells")
    for col in text.columns:
        vals = text[col].dropna().astype(str).head(500)
        vals = vals[~vals.map(prep.is_missing)]          # '-', '*', 'n/a' are missing values, not parse failures
        if len(vals) >= 5:
            parsed = prep.to_number(vals)
            if parsed.notna().mean() >= 0.5:
                missed = sorted({v for v in vals[parsed.isna()] if PURE_NUMBER.match(v)})
                if missed:
                    bad.append(f"numbers: column {str(col)[:30]!r}: number-shaped values not parsed, e.g. {missed[:3]}")
    try:
        text_profile = profile(name, df)
    except Exception as e:
        return bad + [f"profile: {type(e).__name__}: {str(e)[:80]}"]
    lat = next((c for c in df.columns if LAT.match(str(c))), None)
    lon = next((c for c in df.columns if LON.match(str(c))), None)
    if lat is not None and lon is not None:
        la, lo = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
        ok = la.notna() & lo.notna()
        if ok.any() and ((la[ok].abs() > 90) | (lo[ok].abs() > 180)).mean() > 0.02 \
                and "named like a coordinate in degrees" not in text_profile:
            bad.append("coords: degree-named columns outside the valid range and no warning in the profile")
        elif ok.any() and (la[ok].abs() > lo[ok].abs()).mean() > 0.9 and la[ok].abs().median() > 90:
            bad.append("coords: latitude and longitude look swapped")
    return bad


REFUSAL = re.compile(r"archive contains no readable table|this file is a web page")


def check_file(path: Path) -> dict:
    rec = {"file": str(path.relative_to(path.parents[1])), "bytes": path.stat().st_size, "failures": [], "info": []}
    t0 = time.time()
    try:
        tables = load_tables([path])
    except Exception as e:
        message = f"{type(e).__name__}: {str(e)[:160]}"
        if isinstance(e, ValueError) and REFUSAL.search(str(e)):
            rec["outcome"], rec["info"] = "refused", [message]      # no table in the file, and the loader says so
        else:
            rec["failures"].append(f"load: {message}")
            rec["trace"] = traceback.format_exc(limit=-3)[-400:]
        return {**rec, "seconds": round(time.time() - t0, 2)}
    rec["tables"] = {k: list(v.shape) for k, v in tables.items()}
    filled = {k: v for k, v in tables.items() if not (v.empty or v.shape[1] == 0)}
    if not filled:
        if any(v.shape[1] > 0 for v in tables.values()) and path.stat().st_size < 4096:
            rec["outcome"], rec["info"] = "empty_source", ["the published file has a header and no rows"]
            return {**rec, "seconds": round(time.time() - t0, 2)}
        rec["failures"].append("non_empty: no table with rows and columns")
    for name, df in filled.items():
        rec["failures"] += check_table(name, df)
        dup = [c for c in df.columns if re.search(r"\.\d+$", str(c))]
        if dup:
            rec["info"].append(f"renamed repeated columns: {dup[:3]}")
    rec["outcome"] = "violation" if rec["failures"] else "clean"
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
    n = max(1, len(recs))
    outcome = Counter(r.get("outcome", "violation") for r in recs)
    clean, refused = outcome["clean"], outcome["refused"] + outcome["empty_source"]
    lines = ["# Corpus invariants (check version 2)", "",
             f"{len(recs)} files: {clean} read without a violated invariant ({clean / n:.1%}), {refused} without a table "
             f"({refused / n:.1%}: refused with a reason, or published with a header and no rows), {len(recs) - clean - refused} with a violation or a load failure "
             f"({(len(recs) - clean - refused) / n:.1%}). Handled correctly (read cleanly or refused with a reason): "
             f"{(clean + refused) / n:.1%}. Of the files that hold a table: {clean / max(1, len(recs) - refused):.1%} read cleanly.",
             "", "| Signature | Files | Example |", "|---|--:|---|"]
    for sig, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {sig} | {len(fs)} | `{fs[0]}` |")
    info = Counter(i.split(":")[0] for r in recs for i in r.get("info", []) if r.get("outcome") != "refused")
    reasons = Counter(("only documents or images" if "archive" in i else "web page") if r["outcome"] == "refused"
                      else "header without rows" for r in recs if r.get("outcome") in ("refused", "empty_source")
                      for i in r["info"])
    by_ext = Counter(Path(r["file"]).suffix for r in recs)
    lines += ["", "Refusals: " + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "none"),
              "Information (not failures): " + (", ".join(f"{k} {v}" for k, v in info.most_common()) or "none"),
              "Files by type: " + ", ".join(f"{k} {v}" for k, v in by_ext.most_common()), ""]
    (base / "invariants_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
