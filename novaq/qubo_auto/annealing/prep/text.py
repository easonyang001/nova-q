"""Text and number cleaning for real-world tables (docs/data_landscape.md §3.4).

Every function returns plain pandas objects and never guesses silently: what it changed is reported through the
optional `log` (a prep.quality.PrepLog).
"""
from __future__ import annotations
import re
import unicodedata

import pandas as pd

MISSING_MARKERS = {"", "-", "--", "—", "–", "－", "…", "...", "na", "n/a", "nan", "none", "null", "x", "*", "＊", "無", "不詳"}
_NUM_JUNK = re.compile(r"[,\s，]|(?<=\d)['’ʼ](?=\d{3}(?!\d))|NT\$|US\$|\$|元|円|€|£|CHF|Fr\.")   # 1'234 is a Swiss thousands group


def normalize_text(s) -> str:
    """Full-width -> half-width (NFKC), 台 -> 臺 (official form in Taiwanese place names), trimmed spaces."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = t.replace("台", "臺")
    return re.sub(r"\s+", " ", t)


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or normalize_text(v).lower() in MISSING_MARKERS


_COMMA_NUM = re.compile(r"^[-+]?\d+,(\d+)$")


_MAGNITUDE = re.compile(r"^([-+]?\d+(?:\.\d+)?)(k|m|bn|b|thousand|million|billion)$", re.I)
_MULTIPLIER = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_CURRENCY = re.compile(r"\$|€|£|元|円")


def decimal_comma(values) -> bool:
    """True when the comma in these numbers is a decimal comma: some value has a comma not followed by exactly three
    digits ('52,5368', '0,5'). Thousands groups ('1,234') alone keep the comma as a separator. Berlin's toilet file
    mixes '52,5368' and '52.4451' in one column (2026-10-25)."""
    tails = [m.group(1) for v in values if isinstance(v, str)
             for m in [_COMMA_NUM.match(unicodedata.normalize("NFKC", v).strip())] if m]
    return any(len(t) != 3 for t in tails)


def to_number(series: pd.Series, log=None, name: str | None = None) -> pd.Series:
    """'1,234' / 'NT$300' / '12.5%' / '（123）' / missing markers -> float. Percent strings become 12.5 (not 0.125);
    the column name in the log says so. Values that still fail become NaN and are counted."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    pct = False
    dec_comma = decimal_comma(series.dropna().tolist())

    def conv(v):
        nonlocal pct
        if is_missing(v):
            return float("nan")
        t = unicodedata.normalize("NFKC", str(v)).strip()
        if t.endswith("%"):
            pct, t = True, t[:-1]
        if dec_comma and "," in t:
            t = t.replace(".", "").replace(",", ".") if "." in t and t.rfind(",") > t.rfind(".") else t.replace(",", ".")
        neg = t.startswith("(") and t.endswith(")")            # accounting negatives: (123)
        money = bool(_CURRENCY.search(t))
        t = _NUM_JUNK.sub("", t.strip("()")).rstrip("?")
        try:
            x = float(t)
        except ValueError:
            # '$2.846M' / '$1.6 million': a magnitude word counts only next to a currency symbol or as a full word
            # ('5m' in a length column is metres, not five million)
            m = _MAGNITUDE.match(t)
            if not m or not (money or len(m.group(2)) > 2):
                return float("nan")
            x = float(m.group(1)) * _MULTIPLIER[m.group(2).lower()]
        return -x if neg else x
    out = series.map(conv).astype(float)
    if log is not None:
        bad = int(out.isna().sum() - sum(is_missing(v) for v in series))
        log.step(f"數字清理 {name or series.name}", len(series), len(series),
                 f"{'含百分比（保留為 0–100）；' if pct else ''}{bad} 個值無法轉成數字" if bad or pct else "")
    return out


_ROC = re.compile(r"^\s*(\d{2,3})\s*[年/.\-]\s*(\d{1,2})\s*(?:[月/.\-]\s*(\d{1,2})\s*日?)?\s*$")
_ROC_PACKED = re.compile(r"^\s*(\d{2,3})(\d{2})(\d{2})?\s*$")


def roc_to_date(series: pd.Series) -> pd.Series:
    """ROC (民國) dates -> Timestamp: '113年7月31日', '113/07/31', '113.07', '1130731', '11307'.
    Western dates pass through pandas. Year < 200 is taken as ROC (+1911)."""
    def conv(v):
        if is_missing(v):
            return pd.NaT
        t = unicodedata.normalize("NFKC", str(v)).strip()
        m = _ROC.match(t) or (_ROC_PACKED.match(t) if t.isdigit() and len(t) in (5, 6, 7) else None)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            if y < 200:
                y += 1911
            try:
                return pd.Timestamp(year=y, month=mo, day=d)
            except ValueError:
                return pd.NaT
        try:
            return pd.Timestamp(t)
        except (ValueError, TypeError):
            return pd.NaT
    return series.map(conv)
