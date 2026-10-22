"""Table-structure fixes for real-world files (docs/data_landscape.md §3.3).

Government spreadsheets put notes above the header, total rows inside the data, the same entity in several rows
(by gender, by year) and one column per month. These tools undo that, and report what they did.
"""
from __future__ import annotations
import re
from typing import Any

import pandas as pd

from .text import normalize_text

TOTAL_WORDS = re.compile(r"^(總計|合計|小計|總和|全區|全市|全縣|全國|共計|total|subtotal|grand total|all|sum)$", re.I)


def sample_rows(df, n, seed=0, log=None):
    """Explicit fixed-seed sampling requested by the confirmed specification."""
    if not isinstance(n, int) or n < 1 or n > len(df):
        raise ValueError("sample size must be a positive integer no larger than available rows")
    result = df.sample(n=n, random_state=seed).copy()
    if log is not None:
        log.step("fixed-seed sampling", len(df), len(result), f"n={n}, seed={seed}; original row indices retained")
    return result


def find_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    """Index of the real header row in a sheet read with header=None: the first row whose cells are mostly
    non-empty text and that is followed by rows with numbers (notes and titles above it are skipped)."""
    best, best_score = 0, -1.0
    for i in range(min(max_scan, len(raw) - 1)):
        row = raw.iloc[i]
        filled = row.notna().mean()
        texty = sum(isinstance(v, str) and not re.fullmatch(r"[\d,.\s%]+", v) for v in row) / max(1, len(row))
        below = raw.iloc[i + 1:i + 4]
        numeric_below = below.apply(lambda c: pd.to_numeric(c, errors="coerce").notna().mean()).mean()
        score = filled * 0.4 + texty * 0.4 + numeric_below * 0.2
        if score > best_score:
            best, best_score = i, score
    return best


def apply_header(raw: pd.DataFrame, row: int, combine_next: bool = False) -> pd.DataFrame:
    """Use row `row` as the header (optionally merged with the next row for two-level headers)."""
    head = raw.iloc[row].ffill()
    if combine_next:
        sub = raw.iloc[row + 1]
        head = [f"{a}_{b}" if pd.notna(b) and str(b) != str(a) else str(a) for a, b in zip(head, sub)]
        body = raw.iloc[row + 2:]
    else:
        body = raw.iloc[row + 1:]
    df = body.copy()
    df.columns = [normalize_text(c) or f"col{i}" for i, c in enumerate(head)]
    return df.dropna(how="all").reset_index(drop=True)


def drop_total_rows(df: pd.DataFrame, columns: list[str] | None = None, log=None) -> pd.DataFrame:
    """Remove rows whose label cells say total / subtotal (總計, 合計, 小計, 全區 ...), which double-count."""
    cols = columns or [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    mask = df[cols].apply(lambda c: c.map(lambda v: bool(TOTAL_WORDS.match(normalize_text(v))))).any(axis=1) \
        if cols else pd.Series(False, index=df.index)
    out = df[~mask].reset_index(drop=True)
    if log is not None:
        log.step("移除總計／小計列", len(df), len(out), f"移除 {int(mask.sum())} 列")
    return out


def keep_category(df: pd.DataFrame, column: str, value: str = "計", log=None) -> pd.DataFrame:
    """Rows split by a category (性別 計/男/女): keep one value, usually the total."""
    norm = df[column].map(normalize_text)
    out = df[norm == normalize_text(value)].reset_index(drop=True)
    if log is not None:
        log.step(f"只保留 {column}={value}", len(df), len(out))
    return out


def find_records(obj: Any) -> list[dict]:
    """The largest list of dicts inside a nested JSON / API response ({"result": {"records": [...]}} etc.)."""
    best: list[dict] = []

    def walk(o):
        nonlocal best
        if isinstance(o, list):
            if o and all(isinstance(x, dict) for x in o[:20]) and len(o) > len(best):
                best = o
            for x in o[:5]:
                walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
    walk(obj)
    return best


def dedupe(df: pd.DataFrame, keys: list[str], log=None) -> pd.DataFrame:
    keys = [keys] if isinstance(keys, str) else list(keys)        # one column name or a list
    norm = df[keys].apply(lambda c: c.map(normalize_text) if (c.dtype == object or pd.api.types.is_string_dtype(c)) else c)
    out = df[~norm.duplicated()].reset_index(drop=True)
    if log is not None:
        log.step(f"依 {keys} 去除重複", len(df), len(out))
    return out


def wide_to_long(df: pd.DataFrame, id_cols: list[str], value_name: str = "value", var_name: str = "period") -> pd.DataFrame:
    """One column per month / year -> one row per (entity, period)."""
    id_cols = [id_cols] if isinstance(id_cols, str) else list(id_cols)
    return df.melt(id_vars=id_cols, var_name=var_name, value_name=value_name)


def admin_level(name) -> str:
    """The administrative level a place name ends with (市/縣/區/鄉/鎮/村/里), '' if none."""
    t = normalize_text(name)
    return t[-1] if t and t[-1] in "市縣區鄉鎮村里" else ""


def attach_parent(df: pd.DataFrame, column: str, parent_levels: str = "區鄉鎮市", new: str = "parent", log=None) -> pd.DataFrame:
    """Hierarchical tables list a district row followed by its villages. Add the most recent parent-level name
    above each row as column `new` (e.g. each 里 gets its 區)."""
    out = df.copy()
    parent = out[column].map(lambda v: normalize_text(v) if admin_level(v) and admin_level(v) in parent_levels else None)
    out[new] = parent.ffill()
    if log is not None:
        log.step(f"補上上層行政區（{new}）", len(df), len(out), f"{int(out[new].isna().sum())} 列找不到上層")
    if len(out) and out[new].isna().mean() > 0.5:
        # called after keep_level had already removed the district rows: every 里 got no 區 (2026-10-12)
        msg = (f"attach_parent on {column!r} found no parent row ({parent_levels}) above {int(out[new].isna().sum())} of "
               f"{len(out)} rows: call attach_parent on the ORIGINAL table, BEFORE keep_level or any filter that "
               f"removes the parent rows")
        if log is None:
            raise ValueError(msg)
        log.flags.append(msg)
    return out


def keep_level(df: pd.DataFrame, column: str, levels: str = "里村", log=None) -> pd.DataFrame:
    """Keep only rows at the given administrative level (drops city / district totals mixed into the table)."""
    # `"" in "里村"` is True in Python: without the first test the city-total row ("總計", no suffix) was kept as a
    # village and doubled Taipei's population (2026-10-11)
    out = df[df[column].map(lambda v: bool(admin_level(v)) and admin_level(v) in levels)].reset_index(drop=True)
    if log is not None:
        log.step(f"只保留行政層級「{levels}」", len(df), len(out), f"移除 {len(df) - len(out)} 列上層彙總")
        if len(df) >= 20 and len(out) < 0.05 * len(df):
            log.flags.append(f"keep_level on {column!r} kept only {len(out)} of {len(df)} rows: this column does not "
                             f"look like administrative-unit names (村/里/區); do not filter it by level")
    return out


def latest_period(df: pd.DataFrame, columns: list[str], log=None) -> pd.DataFrame:
    """Keep the most recent period (e.g. ['年份', '月份']), compared numerically. A single column name is accepted
    too (an adapter passed "年份" and the error was unreadable, 2026-10-12)."""
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    def comparable(c):
        num = pd.to_numeric(c, errors="coerce")
        if num.notna().mean() >= 0.5:
            return num
        t = pd.to_datetime(c, errors="coerce")                      # timestamps such as "2026-10-12 10:05:00"
        return pd.Series(t.astype("int64"), index=c.index).where(t.notna()) if t.notna().mean() >= 0.5 else num
    key = df[columns].apply(comparable).apply(tuple, axis=1)
    last = key.max()
    out = df[key == last].reset_index(drop=True)
    if log is not None:
        log.step(f"只取最新一期 {dict(zip(columns, last))}", len(df), len(out))
        # stacked periods repeat every entity per period (few distinct values); a column with a different time on
        # almost every row is each row's own update time (YouBike 'mday'): keeping its maximum kept 1 of 1815 rows
        if len(df) >= 20 and key.nunique() > 0.5 * len(df) and len(out) < 0.2 * len(df):
            log.flags.append(f"latest_period on {columns}: {key.nunique()} distinct values in {len(df)} rows looks like "
                             f"each row's own update time, not stacked periods; keep all rows (do not filter by it)")
    return out
