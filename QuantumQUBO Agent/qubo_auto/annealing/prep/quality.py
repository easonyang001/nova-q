"""The preprocessing record (docs/data_landscape.md §4) and a data-quality summary.

Every prep tool can write to a PrepLog; the record is shown to the user before they approve the data, so a filter
that silently dropped half the rows cannot go unnoticed.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PrepLog:
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)       # problems a tool found that must be fixed, not just noted

    def step(self, what: str, rows_before: int | None = None, rows_after: int | None = None, note: str = "") -> None:
        self.steps.append({"what": what, "before": rows_before, "after": rows_after, "note": note})

    def source(self, title: str, url: str = "", publisher: str = "", license: str = "", retrieved: str = "") -> None:
        self.sources.append({"title": title, "url": url, "publisher": publisher, "license": license,
                             "retrieved": retrieved})

    def markdown(self) -> str:
        out = []
        if self.sources:
            out += ["**資料來源**", ""] + [
                f"- {s['title']}" + (f"（{s['publisher']}）" if s["publisher"] else "") +
                (f" <{s['url']}>" if s["url"] else "") + (f"，授權：{s['license']}" if s["license"] else "") +
                (f"，取得 {s['retrieved']}" if s["retrieved"] else "") for s in self.sources] + [""]
        if self.steps:
            out += ["**前處理紀錄**", "", "| 步驟 | 前 | 後 | 說明 |", "|---|---|---|---|"]
            for s in self.steps:
                out.append(f"| {s['what']} | {'' if s['before'] is None else s['before']} | "
                           f"{'' if s['after'] is None else s['after']} | {s['note']} |")
        return "\n".join(out)


def report(df: pd.DataFrame, name: str = "table", max_cols: int = 30) -> str:
    """Missing rates, numeric ranges and duplicate rows, as short text."""
    lines = [f"{name}: {len(df)} rows, {df.shape[1]} columns, {int(df.duplicated().sum())} duplicate rows"]
    for col in list(df.columns)[:max_cols]:
        s = df[col]
        miss = s.isna().mean()
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
            lines.append(f"- {col}: {miss:.0%} missing, range {s.min():.6g} … {s.max():.6g}"
                         + (f", {outliers} extreme values" if outliers else ""))
        else:
            lines.append(f"- {col}: {miss:.0%} missing, {s.nunique()} distinct")
    return "\n".join(lines)
