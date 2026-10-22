"""Web lookup for the Clarification Agent (DuckDuckGo, no API key).

The LLM only proposes queries; this module runs them, fetches the top pages and returns short excerpts with
their URLs. Findings are stored in the clarification session, so re-running a session reuses them instead of
searching again (the confirmed spec stays reproducible). No other agent uses the web.
"""
from __future__ import annotations
import re
import time
from urllib.parse import urlparse

MAX_RESULTS = 5           # search hits per query
MAX_FETCH = 3             # pages read per query
EXCERPT_CHARS = 2500      # text kept per page
FETCH_TIMEOUT_S = 15
# official and reference sites first: a regulation should come from the regulator, not from a blog
PREFERRED = (".gov.tw", ".gov", ".edu", ".edu.tw", "laws.moj.gov.tw", "law.moj.gov.tw", "wikipedia.org")


def _rank(url: str) -> int:
    host = urlparse(url).netloc.lower()
    return 0 if any(host.endswith(d) or d in host for d in PREFERRED) else 1


def _session():
    """requests session that still verifies certificates but without Python 3.13's VERIFY_X509_STRICT:
    Taiwanese government sites (mol.gov.tw, mohw.gov.tw, ...) use certificates without a Subject Key Identifier,
    which the strict mode rejects ("Missing Subject Key Identifier"); curl and browsers accept them."""
    import ssl
    import requests
    from requests.adapters import HTTPAdapter

    class _Adapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = ssl.create_default_context()
            ctx.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

    s = requests.Session()
    s.mount("https://", _Adapter())
    s.headers["User-Agent"] = "Mozilla/5.0 (research clarification agent)"
    return s


def _page_text(url: str) -> str:
    import io
    from bs4 import BeautifulSoup
    import socket
    import urllib3.util.connection as conn
    # this machine has no IPv6 route, and urllib3 gives up after the IPv6 attempt for some hosts
    # (law.moj.gov.tw: "Network is unreachable" while curl falls back to IPv4); fetch over IPv4 only
    old, conn.allowed_gai_family = conn.allowed_gai_family, (lambda: socket.AF_INET)
    try:
        for attempt in range(3):            # some government servers refuse a share of connections
            try:
                r = _session().get(url, timeout=FETCH_TIMEOUT_S)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    finally:
        conn.allowed_gai_family = old
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        from pypdf import PdfReader                   # government notices are often PDFs
        reader = PdfReader(io.BytesIO(r.content))
        text = " ".join((page.extract_text() or "") for page in reader.pages[:8])
        return re.sub(r"\s+", " ", text).strip()
    if "html" not in ctype and "text" not in ctype:
        return ""
    r.encoding = r.apparent_encoding or r.encoding
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _terms(query: str) -> list[str]:
    """Search terms of a query: words, and for CJK runs their 2-character pieces (no word boundaries there)."""
    terms = []
    for w in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query):
        if re.match(r"[\u4e00-\u9fff]", w) and len(w) > 2:
            terms += [w[i:i + 2] for i in range(len(w) - 1)]
        elif len(w) >= 2:
            terms.append(w)
    return list(dict.fromkeys(terms))


def excerpt(text: str, query: str, budget: int = EXCERPT_CHARS, window: int = 350) -> str:
    """The parts of a long page that mention the query's terms (a law page lists every article; the relevant
    ones are rarely at the top). Falls back to the beginning when nothing matches."""
    if len(text) <= budget:
        return text
    terms = _terms(query)
    hits = sorted({m.start() for t in terms for m in re.finditer(re.escape(t), text)})
    if not hits:
        return text[:budget]
    # score windows by the query terms they contain, rare terms weighing more (on a law page "勞動" is
    # everywhere, "輪班" only where it matters); keep the best non-overlapping windows
    weight = {t: 1.0 / max(1, text.count(t)) for t in terms}
    scored = []
    for h in hits:
        a, b = max(0, h - window // 2), min(len(text), h + window // 2)
        scored.append((sum(w for t, w in weight.items() if t in text[a:b]), a, b))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen, used = [], 0
    for _, a, b in scored:
        if used >= budget:
            break
        if all(b <= c or a >= d for c, d in chosen):
            chosen.append((a, b))
            used += b - a
    return " … ".join(text[a:b] for a, b in sorted(chosen))[:budget]


def lookup(query: str, region: str = "tw-tzh") -> dict:
    """{"query", "time", "results": [{"title", "url", "excerpt"}], "error"}; never raises."""
    out = {"query": query, "time": time.time(), "results": [], "error": ""}
    try:
        from ddgs import DDGS
        hits = list(DDGS().text(query, region=region, max_results=MAX_RESULTS))
    except Exception as e:
        out["error"] = f"search failed: {type(e).__name__}: {str(e)[:200]}"
        return out
    hits.sort(key=lambda h: _rank(h.get("href", "")))
    for h in hits[:MAX_FETCH]:
        url, snippet = h.get("href", ""), h.get("body", "")
        try:
            text = excerpt(_page_text(url), query)
        except Exception:
            text = ""
        out["results"].append({"title": h.get("title", ""), "url": url, "excerpt": text or snippet})
    return out


def render_findings(findings: list[dict], max_chars: int = 12000) -> str:
    """Findings as prompt text, newest last, trimmed to a budget so the prompt stays small."""
    if not findings:
        return "(none yet)"
    parts = []
    for f in findings:
        parts.append(f"QUERY: {f['query']}" + (f"  [{f['error']}]" if f.get("error") else ""))
        for r in f["results"]:
            parts.append(f"- {r['title']} <{r['url']}>\n  {r['excerpt'][:1500]}")
    text = "\n".join(parts)
    return text if len(text) <= max_chars else text[-max_chars:]
