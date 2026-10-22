"""Detect, by code, which known data situations the tables contain, and which toolkit tool must handle each.

The Data Adapter is REQUIRED to call these tools (checked at run time, not by reading its code): in a real-data run
the local model ignored the toolkit, re-implemented the steps by hand and wrote licence information that was not in
the sources (2026-10-11). Detection is deliberately conservative: a rule fires only on clear evidence.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import pandas as pd

from .table import TOTAL_WORDS, admin_level
from .text import normalize_text

PERIOD_COLUMNS = re.compile(r"^(年|年份|年度|月|月份|期間|資料年月|統計年月|year|month|period)$", re.I)
AREA_CODE = re.compile(r"(區|鄉鎮|行政|村里|area|district|borough|county|municipal|gemeinde|bezirk).*(代碼|代號|碼|code|id|nr|kennziffer|schlüssel)", re.I)
CATEGORY_TOTALS = {"計", "合計", "總計", "小計"}
CATEGORY_PARTS = {"男", "女"}


@dataclass
class Situation:
    table: str
    column: str
    kind: str
    tools: tuple[str, ...]          # any one of these satisfies the requirement
    evidence: str

    def text(self) -> str:
        return (f"table {self.table!r}, column {self.column!r}: {self.kind} ({self.evidence}) -> must call "
                + " or ".join(f"prep.{t}" for t in self.tools))


def detect(tables: dict[str, pd.DataFrame]) -> list[Situation]:
    found: list[Situation] = []
    for name, df in tables.items():
        for col in df.columns:
            s = df[col]
            if s.dtype == object or pd.api.types.is_string_dtype(s):    # pandas >= 3 has a separate string dtype
                vals = {normalize_text(v) for v in s.dropna().astype(str).unique()[:5000]}
                if vals & CATEGORY_TOTALS and vals & CATEGORY_PARTS and len(vals) <= 6:
                    found.append(Situation(name, col, "rows split by category", ("keep_category",),
                                           f"values {sorted(vals)[:5]}"))
                    continue
                levels = {admin_level(v) for v in vals} - {""}
                has_total = any(TOTAL_WORDS.match(v) for v in vals)
                # a column of place NAMES, not of facility names that happen to end in 區/里 (AED sites, 2026-10-11)
                admin_share = sum(bool(admin_level(v)) or bool(TOTAL_WORDS.match(v)) for v in vals) / max(1, len(vals))
                short = sorted(len(v) for v in vals)[len(vals) // 2] <= 8 if vals else False
                if len(levels) >= 2 and ({"里", "村"} & levels) and admin_share >= 0.6 and short:
                    found.append(Situation(name, col, "administrative levels mixed in one column",
                                           ("keep_level",), f"levels {sorted(levels)}" + (", plus totals" if has_total else "")
                                           + "; if you will join on these names, first prep.attach_parent to keep each "
                                             "row's parent (e.g. district), THEN keep_level"))
                elif has_total:
                    found.append(Situation(name, col, "total / subtotal rows mixed with data",
                                           ("drop_total_rows", "keep_level"), "contains 總計/合計/小計"))
                sample = [v for v in s.dropna().astype(str).head(200)]
                if sample and sum(bool(re.fullmatch(r"[\s$NT,，\d.%()\-]+", x)) and any(c.isdigit() for c in x) for x in sample) / len(sample) > 0.8 \
                        and any("," in x or "%" in x or "$" in x for x in sample):
                    found.append(Situation(name, col, "numbers stored as text", ("to_number",),
                                           f"e.g. {sample[0]!r}"))
            if PERIOD_COLUMNS.match(normalize_text(col)) and s.nunique() > 1:
                found.append(Situation(name, col, "several periods stacked in one table", ("latest_period",),
                                       f"{s.nunique()} distinct values"))
        # areas given only as CODES next to WGS84 coordinates: a code list written from memory kept the wrong
        # districts in two runs (Taipei AED, 2026-10-12); the real boundaries decide instead
        code_cols = [c for c in df.columns if AREA_CODE.search(str(c)) and
                     pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9 and df[c].nunique() > 1]
        lat_cols = [c for c in df.columns if re.search(r"^(lat|latitude|緯度|y_?wgs84)$", str(c).strip(), re.I)]
        if code_cols and lat_cols:
            found.append(Situation(name, code_cols[0], "areas identified only by codes",
                                   ("in_area",), f"{df[code_cols[0]].nunique()} codes, coordinates in {lat_cols[0]!r}; "
                                   "ONLY IF you select areas by this column: filter them by their names with "
                                   f"prep.in_area(df, lat, lon, [names], code_column={code_cols[0]!r}) instead of "
                                   "guessing which code is which. A value the user states for another column "
                                   "(a district number, a year) is kept with keep"))
        # projected coordinates (TWD97 / TWD67 metres or a .prj saying so)
        prj = str(df["crs_prj"].iloc[0]) if "crs_prj" in df.columns and len(df) else ""
        # coordinates are often numbers stored as text ("302864.7812"): coerce before looking (2026-10-11)
        num = df.apply(lambda c: c if pd.api.types.is_numeric_dtype(c) else pd.to_numeric(c, errors="coerce")).select_dtypes("number")
        num = num.loc[:, num.notna().mean() > 0.5] if len(num.columns) else num
        named = lambda c, axis: bool(re.search(rf"(^|_){axis}($|_)|{axis}_?coord", str(c).lower()))
        xs = sorted([c for c in num.columns if 140_000 <= num[c].median() <= 360_000], key=lambda c: not named(c, "x"))
        ys = sorted([c for c in num.columns if 2_400_000 <= num[c].median() <= 2_800_000], key=lambda c: not named(c, "y"))
        if ("TWD" in prj.upper() or "TM" in prj.upper()) or (xs and ys):
            found.append(Situation(name, f"{(xs or ['x'])[0]}/{(ys or ['y'])[0]}", "projected coordinates (metres)",
                                   ("to_wgs84",), prj[:40] or "values in the TWD97 TM2 range"))
    # several tables alone no longer require a join: two lists of the same kind (or the sheets of one workbook) are
    # appended, not joined; a forced join of two cities' lists matched nothing (2026-10-12). A hand-written merge is
    # still forbidden (FORBIDDEN below), so every real join goes through prep.match.
    return found


FORBIDDEN = [
    (re.compile(r"\.head\(\s*\d+|\.iloc\[\s*:\s*\d+|\.sample\(\s*(n\s*=\s*)?\d+|\.nlargest\(|\.nsmallest\("),
     "shrinking data by taking the first / a random N rows; use prep.reduce_points (representative points) instead"),
    (re.compile(r"\blog\.source\("), "writing source records; sources are recorded by the system, not by you"),
    (re.compile(r"(\bpd\.merge\(|\.merge\()"),
     "a hand-written pandas merge; join tables with prep.match (it reports unmatched and duplicate keys)"),
    # different venues in one building share coordinates: a coordinate-only dedupe merged two airlines and a cinema
    # into their neighbours (Taipei AED, smoke19 and smoke20, 2026-10-12)
    (re.compile(r"drop_duplicates\(\s*(subset\s*=\s*)?\[\s*['\"][^'\"]*(lat|緯)[^'\"]*['\"]\s*,\s*['\"][^'\"]*(lon|lng|經)[^'\"]*['\"]\s*\]",
                re.I),
     "deduplicating by coordinates alone; different places in one building share coordinates. Use "
     "prep.merge_same_place(df, name_col, lat_col, lon_col) (same name AND close together)"),
]


def forbidden_patterns(code: str) -> list[str]:
    return [msg for rx, msg in FORBIDDEN if rx.search(code)]
