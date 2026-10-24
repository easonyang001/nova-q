"""Clarification Agent: pin down a real-world problem with the user BEFORE any QUBO is formulated.

A session is a JSON file holding the description, every question/answer round and the latest draft spec.
Nothing downstream may use a session that the user has not explicitly confirmed (`confirmed: true`).
"""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "clarifier.txt").read_text(encoding="utf-8")

BACKEND_LIMITS = {
    "fujitsu_da": "Fujitsu Digital Annealer v3c: up to 100,000 binary variables; 64-bit integer quadratic coefficients "
                  "(very wide dynamic range is fine); time_limit_sec per job; account quota 300 DA-minutes per month.",
    "dwave_qpu": "D-Wave Advantage QPU: about 5,000 qubits but sparse connectivity; densely coupled problems embed only "
                 "about 150-180 logical variables; limited coefficient precision (auto-scaled to h/J ranges).",
    "cga": "CGA GPU annealer (Socket.IO server): dense QUBOs are fine; runs a few thousand variables (measured "
           "here up to 3,600); solution quality on routing was near the reference up to ~1,000 variables and drops "
           "noticeably above ~1,500; only a per-job time limit is tunable.",
    "sa": "Local simulated annealing: no hard size limit; practical up to a few thousand variables.",
    "qaoa_cudaq": "QAOA on NVIDIA CUDA-Q (GPU state-vector simulation): at most 30 binary variables in total, including "
                  "slack bits for inequalities (one L40S holds 32 qubits); only small instances fit.",
}


class Variable(BaseModel):
    name: str
    meaning: str
    index_sets: str = ""
    domain: Literal["binary", "integer", "continuous"] = "binary"
    count_estimate: str = ""


class Objective(BaseModel):
    name: str
    description: str
    direction: Literal["minimize", "maximize"]
    formula_sketch: str = ""
    weight: str = ""


class ConstraintSpec(BaseModel):
    name: str
    description: str
    strength: Literal["hard", "soft"]
    type: Literal["equality", "inequality", "cardinality", "exclusion", "implication", "other"]
    formula_sketch: str = ""


class DataItem(BaseModel):
    name: str
    description: str
    provided: bool = False


class SizeEstimate(BaseModel):
    decision_variables: str = ""
    auxiliary_variables: str = ""
    total_qubo_variables: str = ""
    fits_backend: str = ""


class Source(BaseModel):
    claim: str                     # what the draft relies on (e.g. "rest between shifts >= 11 h")
    url: str                       # where it was read (must come from the web findings)


class DraftSpec(BaseModel):
    problem_summary: str
    decision_variables: list[Variable] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    data_required: list[DataItem] = Field(default_factory=list)
    data_scope: list[str] = Field(default_factory=list, description=(
        "every rule about WHICH records count, stated as the user gave it: area, categories to exclude, period, "
        "level, one entity per real place, ..."))
    size_estimate: SizeEstimate = Field(default_factory=SizeEstimate)
    assumptions: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class Question(BaseModel):
    id: str
    question: str
    why_it_matters: str = ""
    options: list[str] = Field(default_factory=list)
    suggested_default: str = ""
    blocking: bool = True


CHECKLIST = ["scope", "decision_units", "objectives", "tradeoffs", "hard_constraints", "soft_constraints",
             "spatial_rules", "time_horizon", "equity", "existing_state", "data", "scale", "output", "evaluation"]


class Coverage(BaseModel):
    status: Literal["confirmed", "assumed", "unknown", "not_applicable"]
    note: str = ""


class ClarifierOut(BaseModel):
    draft: DraftSpec
    questions: list[Question] = Field(default_factory=list)
    coverage: dict[str, Coverage] = Field(default_factory=dict)
    ready_to_confirm: bool = False
    web_queries: list[str] = Field(default_factory=list)   # facts to look up before finalising (laws, standards)


MAX_QUESTIONS = 5          # per round: the user wants a few precise questions; the agent decides the rest
MAX_WEB_QUERIES = 3        # per step; the Clarification Agent is the only agent allowed to use the web


def open_items(coverage: dict[str, "Coverage"]) -> list[str]:
    """Checklist items nobody has thought through yet (assumptions are allowed: the user accepts them at confirm)."""
    return [k for k in CHECKLIST if k not in coverage or coverage[k].status == "unknown"]


class Session(BaseModel):
    id: str
    created: float
    backend: str
    language: str = "Traditional Chinese (繁體中文)"
    description: str
    rounds: list[dict] = Field(default_factory=list)        # {"questions": [...], "answers": str}
    draft: DraftSpec | None = None
    pending_questions: list[Question] = Field(default_factory=list)
    coverage: dict[str, Coverage] = Field(default_factory=dict)
    ready_to_confirm: bool = False
    confirmed: bool = False
    confirmed_at: float | None = None
    web_findings: list[dict] = Field(default_factory=list)  # every lookup made for this session (reused on reruns)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Session":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def new_session(description: str, backend: str) -> Session:
    if backend not in BACKEND_LIMITS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKEND_LIMITS)}")
    return Session(id=time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4], created=time.time(),
                   backend=backend, description=description.strip())


def step(session: Session, client, model: str, answers: str | None = None, temperature: float = 0.0,
         retries: int = 1, web: bool | None = None, log=None) -> Session:
    """Record the user's answers to the pending questions (if any) and ask the agent for the next draft.
    With web on (default; CLARIFY_WEB=0 turns it off), queries the agent asks for are looked up once and the
    agent drafts again with the findings."""
    import os
    if web is None:
        web = os.environ.get("CLARIFY_WEB", "1") != "0"
    if session.confirmed:
        raise ValueError("session already confirmed; start a new one to change the problem")
    if answers is not None:
        session.rounds.append({"questions": [q.model_dump() for q in session.pending_questions], "answers": answers})
    out = _draft(session, client, model, temperature, retries, web)
    asked = {f["query"] for f in session.web_findings}
    new = [q for q in out.web_queries if q.strip() and q not in asked][:MAX_WEB_QUERIES]
    if web and new:
        from .websearch import lookup
        for q in new:
            if log:
                log(f"web lookup: {q}")
            session.web_findings.append(lookup(q))
        out = _draft(session, client, model, temperature, retries, web)    # rethink with the findings
    if session.rounds:
        clash = contradictions(session, out.draft, client, model)
        if clash:
            # a draft that overrides what the user said (an area-defining station turned into the start "as an
            # assumption" after the user answered "any station can be the start", Dublin 2026-10-24) is redrafted
            if log:
                log(f"draft contradicts the user's answers: {clash}")
            out = _draft(session, client, model, temperature, retries, web, note=(
                "YOUR DRAFT CONTRADICTS WHAT THE USER SAID. Follow the user's words; change these items:\n- "
                + "\n- ".join(f"{c.get('item')} (user said: {c.get('user_said')})" for c in clash)))
    _check_sources(out.draft, session.web_findings)
    session.draft, session.pending_questions, session.coverage = out.draft, out.questions[:MAX_QUESTIONS], out.coverage
    session.ready_to_confirm = (out.ready_to_confirm and not any(q.blocking for q in out.questions)
                                and not open_items(out.coverage))
    return session


def _check_sources(draft: DraftSpec, findings: list[dict]) -> None:
    """A cited URL must be a page that was actually fetched. LLMs garble long URLs (seen: dl-47213 written as
    dl-472913), so a near-identical one is corrected; anything else is marked unverified for the user."""
    import difflib
    fetched = [r["url"] for f in findings for r in f.get("results", [])]
    for src in draft.sources:
        if src.url in fetched:
            continue
        close = difflib.get_close_matches(src.url, fetched, n=1, cutoff=0.9)
        if close:
            src.url = close[0]
        elif not src.claim.startswith("（未驗證"):
            src.claim = f"（未驗證：網址不在查詢結果中）{src.claim}"


_CONTRADICTION_PROMPT = """The user described a problem and answered questions. A draft specification was written.
List every item of the draft (summary, objective, constraint, data scope or assumption) that CONTRADICTS something
the user said explicitly. Only direct contradictions: an item the user did not mention is not a contradiction.

USER'S REQUEST:
{description}

USER'S ANSWERS:
{answers}

DRAFT:
{draft}

Return JSON only: {{"contradictions": [{{"item": "<draft item, quoted>", "user_said": "<the user's words, quoted>"}}]}}
(an empty list when there is none)."""


def contradictions(session: Session, draft: DraftSpec, client, model: str) -> list[dict]:
    """Draft items that contradict the user's own words (checked separately from drafting; [] when none or when
    the check itself fails, which never blocks the conversation)."""
    answers = "\n".join(r["answers"] for r in session.rounds if r.get("answers"))
    view = draft.model_dump(include={"problem_summary", "objectives", "constraints", "data_scope", "assumptions"})
    prompt = _CONTRADICTION_PROMPT.format(description=session.description, answers=answers,
                                          draft=json.dumps(view, ensure_ascii=False, indent=1))
    try:
        found = json.loads(client.call(model=model, prompt=prompt, temperature=0.0, step_name="clarify_check",
                                       json_mode=True)).get("contradictions") or []
        return [c for c in found if isinstance(c, dict) and c.get("item")]
    except Exception:
        return []


def _draft(session: Session, client, model: str, temperature: float, retries: int, web: bool,
           note: str = "") -> ClarifierOut:
    from .websearch import render_findings
    qa = [{"questions": [q["question"] for q in r["questions"]], "answers": r["answers"]} for r in session.rounds]
    prompt = (_PROMPT.replace("{description}", session.description)
              .replace("{web_findings}", render_findings(session.web_findings) if web else "(web lookup disabled)")
              .replace("{qa_history}", json.dumps(qa, ensure_ascii=False, indent=1) if qa else "(none yet)")
              .replace("{previous_draft}", session.draft.model_dump_json(indent=1) if session.draft else "(none)")
              .replace("{backend}", f"{session.backend}: {BACKEND_LIMITS[session.backend]}")
              .replace("{language}", session.language)
              .replace("{schema}", json.dumps(ClarifierOut.model_json_schema(), indent=1)))
    error = ""
    for _ in range(retries + 1):
        raw = client.call(model=model, prompt=prompt + (f"\n\n{note}" if note else "")
                          + (f"\n\nYOUR PREVIOUS ANSWER WAS INVALID: {error}" if error else ""),
                          temperature=temperature, step_name="clarify", json_mode=True)
        try:
            out = ClarifierOut.model_validate(json.loads(raw))
            break
        except Exception as e:                      # schema / JSON problems: retry with the error
            error = str(e)[:800]
    else:
        raise ValueError(f"clarifier output invalid: {error}")
    return out


def spec_for_data(draft: "DraftSpec", limit: int = 2500, session: "Session | None" = None) -> str:
    """The confirmed specification as the data agents get it: the data-scope rules come first and verbatim, so a
    category exclusion or "one entity per place" is not lost in a long JSON (an AED run ignored both, 2026-10-12).
    With the session, the user's own words follow: a drafted rule can drift from them ("same name is one place"
    became "same location is one place", and checker and data agent then both counted the wrong places, Basel
    2026-10-24), and the user's words win."""
    scope = "\n".join(f"- {r}" for r in draft.data_scope)
    words = ""
    if session is not None:
        answers = "\n".join(f"- {r['answers']}" for r in session.rounds if r.get("answers"))
        words = ("\nTHE USER'S OWN WORDS (where the specification differs from them, follow the user):\n"
                 f"- {session.description}" + (f"\n{answers}" if answers else ""))
    return (draft.problem_summary + ("\nDATA SCOPE (every rule must hold for the prepared data):\n" + scope if scope else "")
            + words + "\n" + draft.model_dump_json()[:limit])


def confirm(session: Session) -> Session:
    """The user's explicit approval. Refused while blocking questions remain."""
    if session.draft is None:
        raise ValueError("nothing to confirm yet")
    blocking = [q.question for q in session.pending_questions if q.blocking]
    if blocking:
        raise ValueError("blocking questions are still open:\n- " + "\n- ".join(blocking))
    missing = open_items(session.coverage)
    if missing:
        raise ValueError(f"checklist items not confirmed yet: {missing}")
    for item in session.coverage.values():          # confirming = accepting every listed assumption
        if item.status == "assumed":
            item.status, item.note = "confirmed", f"{item.note} (assumption accepted at confirmation)".strip()
    session.confirmed, session.confirmed_at = True, time.time()
    return session


def render(session: Session) -> str:
    """Human-readable view of the current draft and open questions (Markdown)."""
    d = session.draft
    if d is None:
        return "(no draft yet)"
    out = [f"# 問題確認 · {session.id} · backend: {session.backend}" + (" · ✅ 已確認" if session.confirmed else ""),
           "", f"**理解摘要**：{d.problem_summary}", "", "## 決策變數"]
    out += [f"- **{v.name}**（{v.domain}，{v.count_estimate}）：{v.meaning}" + (f"；索引：{v.index_sets}" if v.index_sets else "")
            for v in d.decision_variables]
    out += ["", "## 目標"] + [f"- **{o.name}**（{o.direction}）：{o.description}"
                              + (f"；式子：`{o.formula_sketch}`" if o.formula_sketch else "") + (f"；權重：{o.weight}" if o.weight else "")
                              for o in d.objectives]
    out += ["", "## 限制"] + [f"- **{c.name}**［{'硬性' if c.strength == 'hard' else '軟性'}・{c.type}］：{c.description}"
                              + (f"；式子：`{c.formula_sketch}`" if c.formula_sketch else "") for c in d.constraints]
    out += ["", "## 需要的資料"] + [f"- {'✅' if x.provided else '❓'} **{x.name}**：{x.description}" for x in d.data_required]
    s = d.size_estimate
    out += ["", f"## 規模估計\n決策變數 {s.decision_variables}；輔助變數 {s.auxiliary_variables}；QUBO 總變數 {s.total_qubo_variables}；"
                f"是否符合 backend：{s.fits_backend}"]
    if d.data_scope:
        out += ["", "## 資料範圍（哪些資料要算）"] + [f"- {r}" for r in d.data_scope]
    if d.assumptions:
        out += ["", "## 目前的假設（需要你同意或修正）"] + [f"- {a}" for a in d.assumptions]
    if d.sources:
        out += ["", "## 參考來源（agent 上網查到的，請核對）"] + [f"- {x.claim}：<{x.url}>" for x in d.sources]
    if session.web_findings:
        out += ["", f"<small>已上網查詢：{'；'.join(f['query'] for f in session.web_findings)}</small>"]
    labels = {"confirmed": "✅ 已確認", "assumed": "⚠️ 假設", "unknown": "❓ 未知", "not_applicable": "— 不適用"}
    if session.coverage:
        out += ["", "## 確認清單"] + [f"- {labels.get(session.coverage[k].status, '?') if k in session.coverage else '❓ 未知'} "
                                       f"**{k}**：{session.coverage[k].note if k in session.coverage else ''}" for k in CHECKLIST]
    if session.pending_questions:
        out += ["", "## 需要你回答的問題"]
        for q in session.pending_questions:
            out.append(f"- **[{q.id}]{'（必答）' if q.blocking else ''}** {q.question}")
            if q.why_it_matters:
                out.append(f"  - 為什麼重要：{q.why_it_matters}")
            if q.options:
                out.append(f"  - 選項：{' / '.join(q.options)}")
            if q.suggested_default:
                out.append(f"  - 建議預設：{q.suggested_default}")
    out += ["", "**可以確認了**（確認即代表接受上面所有「⚠️ 假設」；不同意的請直接說要改哪一項）"
            if session.ready_to_confirm else "**還不能確認：仍有必答問題或未知項目**"]
    return "\n".join(out)
