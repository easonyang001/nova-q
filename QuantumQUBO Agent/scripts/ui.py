"""Test UI: clarify a problem -> build the model -> solve with the agent loop on SA / D-Wave / Fujitsu.

    /research/conda-envs/qubo-ui/bin/python scripts/ui.py          # then open http://localhost:7860
                                                                    # (VS Code forwards the port automatically)
Needs the local vLLM server for every agent step. Artifacts go to results/clarify/ and results/ui/<session id>/.

The page is a guided flow: ① describe -> ② answer questions -> ③ model & data -> ④ solve -> ⑤ results.
A step only unlocks once the previous one is done, and finishing a step jumps to the next one.
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import pandas as pd
import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing import clarify as C
from qubo_auto.annealing import modeler as M
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.encodings import compile_problem, encode, options_for, verify_encoding
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.tools import qubo_summary
from qubo_auto.annealing.artifacts import ArtifactStore, atomic_write
from qubo_auto.annealing.assessment import inspect_problem
from qubo_auto.annealing.acquisition import prepare_real_data
from qubo_auto.annealing.workflow import run_workflow
from qubo_auto.llm_client import LLMClient

MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
# per-role models ("provider:model" picks another server/company, e.g. "openrouter:<vendor>/<model>")
CLARIFY_MODEL = os.environ.get("LLM_MODEL_CLARIFY") or MODEL
MODELER_MODEL = os.environ.get("LLM_MODEL_MODELER") or MODEL
DATA_MODEL = os.environ.get("LLM_MODEL_DATA") or MODEL
CLARIFY_DIR, UI_DIR = ROOT / "results" / "clarify", ROOT / "results" / "ui"
BACKENDS = [("Fujitsu Digital Annealer（真機，耗額度）", "fujitsu_da"),
            ("本地模擬退火 SA（免費、最快）", "sa"),
            ("D-Wave 量子退火（真機，耗額度）", "dwave_qpu"),
            ("D-Wave Hybrid（混合求解，每次最多 10 秒）", "dwave_hybrid"),
            ("CGA GPU 退火機（無額度限制，主力）", "cga")]
BACKEND_NAME = {v: k.split("（")[0] for k, v in BACKENDS}
STOP_REASON = {"gap_target": "已達到目標品質", "no_improvement": "連續幾輪沒有進步，自動停止",
               "budget": "用完 solver 呼叫次數上限", "hardware_budget": "硬體額度用完",
               "backend_error": "solver 出錯", "no_valid_proposal": "agent 提不出合法的新設定",
               "preflight": "先回查資料／模型，尚未提交退火機", "feasible_count": "已找到指定輪數的可行方案",
               "policy_stop:penalty_grid": "已檢查完設定範圍內的權重組合",
               "policy_stop:penalty_cem": "權重搜尋結束", "policy_stop:penalty_tpe": "權重搜尋結束"}
OTHER = "其他（在下方填寫）"
STEPS = ["① 描述題目", "② 釐清問題", "③ 建模與資料", "④ 求解", "⑤ 結果紀錄"]
LLM_WAIT = "⏳ 正在請本地 LLM 分析，通常要 1–3 分鐘，請稍候…"


def _client(run_id: str) -> LLMClient:
    return LLMClient(transcript_dir=ROOT / "transcripts", run_id=run_id, max_tokens=500_000, use_cache=False)


# ---------------------------------------------------------------- sessions & progress
def _load_session(sid: str) -> C.Session:
    return C.Session.load(CLARIFY_DIR / f"{sid}.json")


def _model_dir(sid: str) -> Path:
    return UI_DIR / sid


def _stage(sid: str | None) -> int:
    """Index of the furthest unlocked step (0..4)."""
    if not sid or not (CLARIFY_DIR / f"{sid}.json").exists():
        return 0
    if not _load_session(sid).confirmed:
        return 1
    d = _model_dir(sid)
    if not ((d / "model.py").exists() and (d / "data.json").exists()):
        return 2
    return 4 if (d / "runs").exists() and any((d / "runs").glob("*.json")) else 3


def _session_label(sid: str) -> str:
    try:
        s = _load_session(sid)
    except Exception:
        return sid
    title = " ".join(s.description.split())[:24] + ("…" if len(s.description) > 24 else "")
    mark = ["", "💬", "📐", "▶️", "✅"][_stage(sid)]
    if _stage(sid) == 4:
        mark = "📋"
        state = _model_dir(sid) / "workflow_state.json"
        if state.exists():
            delivery = json.loads(state.read_text(encoding="utf-8")).get("delivery", {})
            mark = {"usable": "✅", "limited": "⚠️", "no_solution": "❌"}.get(delivery.get("status"), "📋")
    return f"{mark} {title}（{datetime.fromtimestamp(s.created):%m/%d %H:%M}）"


def _session_choices() -> list[tuple[str, str]]:
    ids = sorted((p.stem for p in CLARIFY_DIR.glob("*.json")), reverse=True) if CLARIFY_DIR.exists() else []
    return [(_session_label(i), i) for i in ids]


def _progress(sid: str | None) -> str:
    st = _stage(sid)
    cells = []
    for i, name in enumerate(STEPS):
        if i < st:
            cells.append(f"<span class='step done'>✔ {name}</span>")
        elif i == st:
            cells.append(f"<span class='step now'>{name}</span>")
        else:
            cells.append(f"<span class='step todo'>{name}</span>")
    hint = ["先用一般的話描述你的題目，按「開始分析」。",
            "回答 agent 的問題；全部清楚後按「確認規格」。",
            "產生數學模型，並準備資料（可以先用模擬資料）。",
            "選 solver，按「開始求解」。",
            "可以看結果，或換個 solver / 設定再跑一次。"][st]
    return f"<div class='steps'>{'<span class=arrow>›</span>'.join(cells)}</div><div class='hint'>下一步：{hint}</div>"


def _refresh_labels():
    return gr.update(choices=_session_choices())


def gate(sid: str | None):
    """Progress bar + which tabs are open; called after every action."""
    st = _stage(sid)
    return [_progress(sid)] + [gr.update(interactive=st >= i) for i in range(1, 5)]


def _budget_text() -> str:
    cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))["hardware_budget"]
    b = HardwareBudget(ROOT / cfg["ledger"], {k: v for k, v in cfg.items() if k != "ledger"})
    rows = []
    for name in ("fujitsu_da", "dwave_qpu", "dwave_hybrid", "cga"):
        u, lim = b.usage(name), cfg.get(name, {})
        rows.append(f"{BACKEND_NAME[name]}：已用 {u['calls']} / {lim.get('max_calls', 0)} 次、"
                    f"{u['solver_time_s']:.1f} / {lim.get('max_solver_time_s', 0)} 秒")
    return "**硬體額度**　" + "　｜　".join(rows)


# ---------------------------------------------------------------- ① describe / ② clarify
def _draft_view(s: C.Session) -> str:
    """The agent's understanding, without the question list (questions are shown as cards)."""
    return C.render(s.model_copy(update={"pending_questions": []}))


MAX_Q = C.MAX_QUESTIONS


def question_slots(qs: list[dict]):
    """Updates for the fixed question cards: header, MAX_Q x (box, text, radio, input), submit button."""
    out = [gr.update(visible=bool(qs), value=f"### agent 的問題（{len(qs)} 題）\n沒把握的可以留空，會採用建議預設。")]
    for i in range(MAX_Q):
        if i >= len(qs):
            out += [gr.update(visible=False), gr.update(), gr.update(), gr.update()]
            continue
        q = qs[i]
        title = f"**{'🔴 必答　' if q.get('blocking') else ''}{q['question']}**"
        if q.get("why_it_matters"):
            title += f"\n\n<small>為什麼要問：{q['why_it_matters']}</small>"
        default, opts = q.get("suggested_default", ""), q.get("options") or []
        out += [gr.update(visible=True), gr.update(value=title),
                gr.update(choices=opts + [OTHER], value=default if default in opts else None, visible=bool(opts)),
                gr.update(value="", placeholder=(f"建議預設：{default}" if default else "你的回答") + ("（可補充）" if opts else ""))]
    out.append(gr.update(visible=bool(qs)))
    return out


def _questions(s: C.Session) -> list[dict]:
    return [] if s.confirmed else [q.model_dump() for q in s.pending_questions]


def _clarify_outputs(s: C.Session, msg: str):
    ready = s.ready_to_confirm and not s.confirmed
    return (_draft_view(s), msg, _questions(s), gr.update(visible=ready),
            gr.update(visible=not s.confirmed and not ready))


def start_session(description: str, backend: str):
    if not description.strip():
        raise gr.Error("請先輸入題目描述。")
    s = C.new_session(description, backend)
    try:
        s = C.step(s, _client(f"clarify_{s.id}"), CLARIFY_MODEL)
    except Exception as e:
        raise gr.Error(f"分析失敗（本地 LLM 有開嗎？scripts/serve_vllm.sh）：{e}")
    s.save(CLARIFY_DIR / f"{s.id}.json")
    return gr.update(choices=_session_choices(), value=s.id), gr.Tabs(selected=1)


def submit_answers(sid: str, questions: list[dict], *values):
    s = _load_session(sid)
    if s.confirmed:
        raise gr.Error("這題已經確認過，不能再改；要改題目請按「＋ 新題目」。")
    lines, missing = [], []
    for q, choice, extra in zip(questions, values[0::2], values[1::2]):
        extra = (extra or "").strip()
        if choice and choice != OTHER:
            ans = choice + (f"；補充：{extra}" if extra else "")
        elif extra:
            ans = extra
        elif q.get("suggested_default"):
            ans = f"（採用建議預設）{q['suggested_default']}"
        elif q.get("blocking"):
            missing.append(q["question"])
            continue
        else:
            ans = "（沒有特別要求，由你決定）"
        lines.append(f"[{q['id']}] {q['question']}\n答：{ans}")
    if missing:
        raise gr.Error("這些必答題還沒回答：" + "；".join(missing))
    try:
        s = C.step(s, _client(f"clarify_{s.id}"), CLARIFY_MODEL, answers="\n\n".join(lines))
    except Exception as e:
        raise gr.Error(f"分析失敗：{e}")
    s.save(CLARIFY_DIR / f"{s.id}.json")
    msg = ("✅ 問題都清楚了。請看看上面的理解和假設，沒問題就按「確認規格」。" if s.ready_to_confirm
           else f"已更新。agent 還有 {len(s.pending_questions)} 個問題。")
    return _clarify_outputs(s, msg)


def confirm_session(sid: str):
    s = _load_session(sid)
    try:
        s = C.confirm(s)
    except ValueError as e:
        raise gr.Error(f"還不能確認：{e}")
    s.save(CLARIFY_DIR / f"{s.id}.json")
    return (*_clarify_outputs(s, "✅ 規格已確認。"), gr.Tabs(selected=2))


def show_session(sid: str):
    if not sid:
        return "", "", [], gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update()
    s = _load_session(sid)
    msg = "✅ 規格已確認，可以去「③ 建模與資料」。" if s.confirmed else ""
    tab = [0, 1, 2, 3, 4][min(_stage(sid), 4)]
    return (*_clarify_outputs(s, msg), gr.update(value=s.backend), gr.Tabs(selected=tab))


# ---------------------------------------------------------------- ③ model & data
def _load_model(sid: str):
    store = ArtifactStore(_model_dir(sid))
    code = store.current()["code"] if (store.root / "active.json").exists() else (store.root / "model.py").read_text(encoding="utf-8")
    return code, M.load_module(code)


def _load_ir(sid: str):
    bundle = ArtifactStore(_model_dir(sid)).current()
    ns, data = M.load_module(bundle["code"]), bundle["data"]
    return ns, data, M.to_ir(ns["build_ir"](data), f"ui_{sid}")


def _model_summary(sid: str, report: dict | None = None) -> str:
    _, ns = _load_model(sid)
    from qubo_auto.annealing import schema as S
    ds = ns["DATA_SCHEMA"]
    schema = ("```\n" + S.describe(ds) + "\n```" if S.is_structured(ds)
              else "\n".join(f"- **{k}**：{v}" for k, v in ds.items()))
    out = ["### ✅ 模型已建立並通過驗證", "", "**模型需要的資料**", schema]
    if (report or {}).get("model_choice"):
        out += ["", f"**Modeling Agent 的建模選擇**：{report['model_choice']}"]
    for w in (report or {}).get("warnings", []):
        out += ["", f"> ⚠️ {w}"]
    if report:
        out += ["", f"**限制**：{', '.join(report.get('constraints', [])) or '無'}",
                f"**驗證用小型實例**：{report.get('n_variables')} 個變數，窮舉最佳值 {report.get('small_example_optimum')}"]
    return "\n".join(out)


def build_model(sid: str):
    s = _load_session(sid)
    if not s.confirmed:
        raise gr.Error("規格還沒確認：請先在「② 釐清問題」按「確認規格」。")
    qa = json.dumps(s.rounds, ensure_ascii=False, indent=1)
    try:
        code, ns, errors, report = M.build_model(s.draft.model_dump_json(indent=1), qa, _client(f"model_{sid}"), MODELER_MODEL,
                                                 backend=f"{s.backend}: {C.BACKEND_LIMITS.get(s.backend, '')}")
    except Exception as e:
        raise gr.Error(f"建模失敗：{e}")
    d = _model_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    if errors:
        atomic_write(d / "rejected_model.py", code)
        return ("### ❌ 模型驗證沒通過\n- " + "\n- ".join(errors) + "\n\n可以再按一次「產生模型」重試。",
                code, gr.update(visible=False), "", "", [])     # msg, code, data box, overview, data, encodings
    data = ns["make_example_data"](20, 0)          # ready-to-solve default; the user can regenerate below
    store = ArtifactStore(d)
    store.commit(code, M.to_jsonable(data), "初始模擬資料", {"source": "synthetic", "confirmed_spec": s.draft.model_dump()})
    store.event("model", "ready", version=store.current()["version"])
    return (_model_summary(sid, report), code, gr.update(visible=True), *_ir_overview(sid, "模擬資料（規模 20、seed 0）"))


def make_data(sid: str, source: str, size: int, seed: int, upload):
    try:
        _, ns = _load_model(sid)
    except Exception:
        raise gr.Error("請先按「產生模型」。")
    if source == "上傳真實資料":
        if upload is None:
            raise gr.Error("請先選擇要上傳的 JSON 檔。")
        try:
            data = json.loads(Path(upload if isinstance(upload, str) else upload.name).read_text(encoding="utf-8"))
        except Exception as e:
            raise gr.Error(f"讀不了這個檔案（需要 JSON）：{e}")
        label = "上傳的資料"
    else:
        data = ns["make_example_data"](int(size), int(seed))
        label = f"模擬資料（規模 {int(size)}、seed {int(seed)}）"
    code, _ = _load_model(sid)
    M.to_ir(ns["build_ir"](M.to_jsonable(data)), "data_check")
    store = ArtifactStore(_model_dir(sid))
    store.commit(code, M.to_jsonable(data), label,
                 {**store.current().get("metadata", {}), "report": {}, "data_review_approved": False,
                  "source": "uploaded" if source == "上傳真實資料" else "synthetic"})
    return _ir_overview(sid, label)


# ---------------------------------------------------------------- ③b real data (upload / open data -> adapter -> approve)
def _files_dir(sid: str) -> Path:
    return _model_dir(sid) / "data_files"


def _spec_text(sid: str) -> str:
    s = _load_session(sid)
    return C.spec_for_data(s.draft)


def _files_overview(sid: str) -> str:
    from qubo_auto.annealing.datakit import load_tables, profiles
    files = sorted(p for p in _files_dir(sid).glob("*") if p.is_file() and not p.name.startswith("_"))
    if not files:
        return "（還沒有任何資料檔）"
    try:
        tables = load_tables(files)
    except Exception as e:
        return f"❌ 讀檔失敗：{e}"
    head = "**目前的資料檔**：" + "、".join(f"`{p.name}`" for p in files)
    return head + "\n\n<details><summary>資料摘要（agent 只會看到這些，不會讀整份資料）</summary>\n\n```\n" + \
        profiles(tables) + "\n```\n</details>"


def add_files(sid: str, files):
    import shutil
    if not files:
        raise gr.Error("請先選擇檔案。")
    d = _files_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = Path(f if isinstance(f, str) else f.name)
        shutil.copy(src, d / src.name)
    return _files_overview(sid)


def clear_files(sid: str):
    import shutil
    shutil.rmtree(_files_dir(sid), ignore_errors=True)
    return _files_overview(sid), gr.update(choices=[], value=[]), "", gr.update(visible=False)


def scout_data(sid: str):
    from qubo_auto.annealing import opendata as O
    try:
        _, ns = _load_model(sid)
    except Exception:
        raise gr.Error("請先產生模型。")
    client = _client(f"scout_{sid}")
    out = O.scout(_spec_text(sid), ns["DATA_SCHEMA"], client, DATA_MODEL)
    ranked = O.rank_candidates(_spec_text(sid), ns["DATA_SCHEMA"], out["candidates"], client, DATA_MODEL)
    d = _files_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_candidates.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = ["| | 來源 | 資料集 | 可提供 | 理由 |", "|---|---|---|---|---|"]
    for c in ranked:
        flag = ("✔" if c["relevant"] else "✘") + ("" if c["readable"] else "（格式讀不了）")
        title = f"[{c['title'][:40]}]({c['page']})" if c.get("page") else c["title"][:40]
        rows.append(f"| {flag} | {c['source']} | {title} | {c['feeds']} | {c['reason'][:60]} |")
    plan = out["plan"]
    md = (f"**agent 的搜尋計畫**：{plan.get('explanation', '')}\n\n地區：{plan.get('area') or '—'}；"
          f"搜尋關鍵字：{'、'.join(plan.get('govtw_queries') or []) or '—'}\n\n"
          f"找到 {len(ranked)} 筆，agent 判斷相關的有 {sum(c['relevant'] for c in ranked)} 筆（✔ 在前）：\n\n" + "\n".join(rows))
    choices = [(f"{'✔' if c['relevant'] else '✘'} {c['source']}｜{c['title'][:40]}", c["id"]) for c in ranked if c["readable"]]
    return md, gr.update(choices=choices, value=[])


def fetch_selected(sid: str, ids: list[str]):
    from qubo_auto.annealing import opendata as O
    if not ids:
        raise gr.Error("請先勾選要下載的資料。")
    cands = {c["id"]: c for c in json.loads((_files_dir(sid) / "_candidates.json").read_text(encoding="utf-8"))}
    done, failed = [], []
    src_path = _files_dir(sid) / "_sources.json"
    sources = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    for i in ids:
        try:
            path = O.fetch_candidate(cands[i], _files_dir(sid))
            done.append(path.name)
            c = cands[i]                                   # provenance, shown in the preprocessing record
            from qubo_auto.annealing.provenance import source_record
            sources[path.name] = source_record(c, path)
        except Exception as e:
            failed.append(f"{cands[i]['title'][:30]}：{str(e)[:80]}")
    src_path.write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"已下載：{'、'.join(done) or '無'}" + (f"\n\n⚠️ 失敗：\n- " + "\n- ".join(failed) if failed else ""))
    return msg + "\n\n" + _files_overview(sid)


def adapt_data(sid: str, notes: str):
    from qubo_auto.annealing.adapter import build_adapter
    from qubo_auto.annealing.datakit import load_tables
    try:
        _, ns = _load_model(sid)
    except Exception:
        raise gr.Error("請先產生模型。")
    files = sorted(p for p in _files_dir(sid).glob("*") if p.is_file() and not p.name.startswith("_"))
    if not files:
        raise gr.Error("請先上傳檔案或下載開放資料。")
    tables = load_tables(files)
    src_path = _files_dir(sid) / "_sources.json"
    known = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    sources = [known.get(p.name, {"title": p.name, "publisher": "使用者上傳"}) for p in files]
    code, data, errors, report = build_adapter(ns, tables, _spec_text(sid), _client(f"adapter_{sid}"), DATA_MODEL,
                                               notes=notes or "", sources=sources)
    d = _files_dir(sid)
    (d / "_pending.json").unlink(missing_ok=True)
    (d / "_adapter.py").write_text(code, encoding="utf-8")
    if errors:
        return ("### ❌ 資料轉換沒成功\n- " + "\n- ".join(e[:300] for e in errors) +
                "\n\n可以補充說明（例如哪個欄位是什麼）後再試一次。"), gr.update(visible=False)
    (d / "_pending.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    atomic_write(d / "_pending_meta.json", json.dumps({"version": ArtifactStore(_model_dir(sid)).current()["version"],
                                                       "report": report}, ensure_ascii=False, default=str))
    limit = C.BACKEND_LIMITS.get(_load_session(sid).backend, "")
    flags = _adapter_flags(report.get("notes", "")) + report.get("review_required", [])
    return (f"### ✅ 轉換完成，等你核准\n\n**題目規模**：{report['n_variables']} 個變數、{report['n_constraints']} 條限制"
            f"（目標硬體：{limit[:80]}）\n\n" + "".join(f"> ⚠️ {f}\n>\n" for f in flags) +
            f"\n**agent 說明它做了什麼**：{report.get('notes') or '（沒有說明）'}\n\n"
            f"{report.get('prep_log', '')}\n\n"
            f"<details><summary>轉換程式</summary>\n\n```python\n{code[:6000]}\n```\n</details>"), gr.update(visible=True)


def _adapter_flags(notes: str) -> list[str]:
    """Things in the adapter's own notes the user must not miss before approving."""
    n = notes.lower()
    flags = []
    if any(w in n for w in ("synthetic", "random", "generated", "simulat", "placeholder", "dummy", "模擬", "隨機", "產生的")):
        flags.append("部分資料**不是真實資料**（agent 自己產生或模擬的），請確認可以接受，或補上真實資料再轉換一次。")
    if any(w in n for w in ("default", "assum", "預設", "假設")):
        flags.append("有欄位用了**預設值或假設**，請看下面的說明確認數值合理。")
    if any(w in n for w in ("first ", "sample", "subset", "前 ", "抽樣", "只取")):
        flags.append("只用了**部分資料**（例如只取前幾筆或抽樣），請確認選取方式合理。")
    if any(w in n for w in ("ignor", "drop", "忽略", "刪除", "略過")):
        flags.append("有資料被**忽略或刪除**，請確認這是你要的。")
    return flags


def auto_prepare_data(sid: str, notes: str):
    """Resume from readable files, search alternate sources, and preserve a working bundle."""
    code, _ = _load_model(sid)
    store = ArtifactStore(_model_dir(sid))
    store.event("data", "started")
    logs = []
    (_files_dir(sid) / "_pending.json").unlink(missing_ok=True)
    cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8")).get("workflow", {})
    try:
        data, errors, report = prepare_real_data(code, _spec_text(sid), _client(f"data_workflow_{sid}"), DATA_MODEL,
            _files_dir(sid), notes=notes, search_rounds=cfg.get("data_search_rounds", 2),
            max_downloads=cfg.get("max_downloads", 4), log=logs.append)
    except Exception as e:
        data, errors, report = {}, [str(e)], {}
    if errors:
        store.event("data", "needs_input", errors=errors)
        return ("### 尚未備妥資料，已保留原版本\n\n" + "\n".join(logs) + "\n\n- " + "\n- ".join(errors),
                gr.update(visible=False), _files_overview(sid), gr.update(), gr.update(), gr.update())
    flags = _adapter_flags(report.get("notes", "")) + report.get("review_required", [])
    inspection = inspect_problem(M.to_ir(M.load_module(code)["build_ir"](data), "prepared"))
    if inspection["feasibility"] in ("invalid", "infeasible"):
        flags.extend(inspection["issues"])
    if flags:
        atomic_write(_files_dir(sid) / "_pending.json", json.dumps(data, ensure_ascii=False, allow_nan=False))
        atomic_write(_files_dir(sid) / "_pending_meta.json", json.dumps({"version": store.current()["version"],
                                                                      "report": report}, ensure_ascii=False, default=str))
        store.event("data", "needs_input", flags=flags)
        return ("### 資料已轉換，有下列假設／問題需要確認\n\n- " + "\n- ".join(flags) + "\n\n" + report.get("prep_log", ""),
                gr.update(visible=True), _files_overview(sid), gr.update(), gr.update(), gr.update())
    store.commit(code, data, "自動備妥的真實資料", {**store.current().get("metadata", {}), "source": "real", "report": report,
                                            "data_review_approved": False})
    store.event("data", "ready", version=store.current()["version"])
    return ("### ✅ 真實資料已備妥，可進入求解\n\n" + "\n".join(logs) + "\n\n" + report.get("prep_log", ""),
            gr.update(visible=False), _files_overview(sid), *_ir_overview(sid, "自動備妥的真實資料"))


def _recover_workflow(sid: str, issue: dict, bundle: dict, logs: list[str]):
    stage = issue["stage"]
    reason = json.dumps(issue["evidence"], ensure_ascii=False)
    if stage == "diagnose":
        from qubo_auto.annealing.adapter import _shape
        prompt = ("Route this failure using evidence. Return JSON {stage: data|model|needs_input, message: string}. "
                  "Infeasibility alone does not prove bad data or bad modeling. Choose data/model only when there is "
                  "a concrete implementation/data error; otherwise needs_input. Never relax confirmed constraints, "
                  "change units without evidence, fabricate missing data, or change the objective.\nCONFIRMED SPEC:\n"
                  + _spec_text(sid) + "\nFAILURE:\n" + reason + "\nDATA SHAPE AND EXAMPLES:\n"
                  + json.dumps(_shape(bundle["data"]), ensure_ascii=False, default=str)[:5000]
                  + "\nPREPROCESSING RECORD:\n" + json.dumps(bundle.get("metadata", {}), ensure_ascii=False, default=str)[:5000]
                  + "\nMODEL:\n" + bundle["code"][:16000])
        raw = _client(f"recovery_{sid}").call(model=MODELER_MODEL, prompt=prompt, temperature=0.0, step_name="workflow_route")
        route = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        stage = route.get("stage")
        reason += "\n" + str(route.get("message", ""))
        if stage not in ("data", "model"):
            return {"status": "needs_input", "message": reason}
    logs.append(f"[修正] 回到 {stage}：{reason}")
    code = bundle["code"]
    if stage == "model":
        s = _load_session(sid)
        code, ns, errors, report = M.build_model(s.draft.model_dump_json(indent=1), json.dumps(s.rounds, ensure_ascii=False),
            _client(f"repair_model_{sid}"), MODELER_MODEL, max_attempts=2, previous_code=code, repair_feedback=reason,
            backend=f"{s.backend}: {C.BACKEND_LIMITS.get(s.backend, '')}")
        if errors:
            return {"status": "needs_input", "message": "建模修正未通過：" + "；".join(errors)}
        try:
            M.to_ir(ns["build_ir"](bundle["data"]), "repair")
            return {"status": "repaired", "message": reason, "code": code, "data": bundle["data"]}
        except Exception:
            stage = "data"
    if stage == "data":
        data, errors, report = prepare_real_data(code, _spec_text(sid), _client(f"repair_data_{sid}"), DATA_MODEL,
            _files_dir(sid), notes=reason, search_rounds=1, max_downloads=2, log=logs.append)
        flags = _adapter_flags(report.get("notes", "")) + report.get("review_required", [])
        if errors or flags:
            return {"status": "needs_input", "message": "；".join(errors + flags)}
        return {"status": "repaired", "message": reason, "code": code, "data": data,
                "metadata": {"source": "real", "report": report}}
    return {"status": "needs_input", "message": "需要補充資料或確認需求：" + reason}


def version_choices(sid: str):
    return gr.update(choices=[(f"{b['label']} · {b['version'][:8]}", b["version"])
                              for b in ArtifactStore(_model_dir(sid)).versions()])


def restore_version(sid: str, version: str):
    if not version:
        raise gr.Error("請先選擇版本。")
    bundle = ArtifactStore(_model_dir(sid)).restore(version)
    return (f"已還原：{bundle['label']}；可直接接續求解。", *_ir_overview(sid, bundle["label"]))


def approve_data(sid: str):
    pending = _files_dir(sid) / "_pending.json"
    if not pending.exists():
        raise gr.Error("沒有待核准的資料。")
    store = ArtifactStore(_model_dir(sid))
    meta = json.loads((_files_dir(sid) / "_pending_meta.json").read_text(encoding="utf-8"))
    bundle = store.current()
    if meta["version"] != bundle["version"]:
        raise gr.Error("模型或資料版本已變更，請重新轉換資料，避免使用舊模型的轉換結果。")
    store.commit(bundle["code"], json.loads(pending.read_text(encoding="utf-8")), "核准的真實資料",
                 {**bundle.get("metadata", {}), "source": "real", "report": meta["report"], "data_review_approved": True})
    pending.unlink()
    return (*_ir_overview(sid, f"真實資料（{datetime.now():%m/%d %H:%M} 核准）"), gr.update(visible=False))


def _ir_overview(sid: str, source: str):
    try:
        _, data, ir = _load_ir(sid)
    except Exception as e:
        raise gr.Error(f"資料和模型對不起來（build_ir 失敗）：{e}")
    inspection = inspect_problem(ir)
    ref = inspection.get("baseline_objective")
    kinds: dict[str, int] = {}
    for c in ir.constraints:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    text = (f"**目前資料：{source}**　→　{ir.n_variables} 個決策變數、{len(ir.constraints)} 條限制"
            f"（{'、'.join(f'{k} ×{n}' for k, n in kinds.items()) or '無'}），目標：{'最小化' if ir.sense == 'minimize' else '最大化'}\n\n"
            f"簡單方法的可行基準：{ref if ref is not None else '尚未找到'}；檢查結果：{inspection['feasibility']}\n\n"
            + "\n".join(inspection["issues"]))
    return text, json.dumps(M.to_jsonable(data), ensure_ascii=False, indent=1)[:20000], \
        [[c.name, c.kind, ", ".join(options_for(c)), ""] for c in ir.constraints]


ENC_HEADERS = ["限制", "類型", "可用編碼", "選用編碼"]


def _df_rows(df) -> list[list]:
    if df is None:
        return []
    return df.values.tolist() if hasattr(df, "values") else list(df)


def _encodings_from(df) -> dict:
    """Manual encodings from the ③ table; an unknown name is an error, an empty cell keeps the default."""
    out = {}
    for name, _kind, opts, choice in _df_rows(df):
        choice = str(choice or "").strip()
        if not choice:
            continue
        allowed = [o.strip() for o in str(opts).split(",")]
        if choice not in allowed:
            raise gr.Error(f"「{name}」的編碼「{choice}」不存在，可用：{', '.join(allowed)}")
        out[name] = {"encoding": choice, "params": {}}
    return out


def check_encodings(sid: str, df):
    enc = _encodings_from(df)
    try:
        _, _, ir = _load_ir(sid)
        p = compile_problem(ir, {k: v["encoding"] for k, v in enc.items()}, {k: {} for k in enc})
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")
    s = qubo_summary(p)
    out = [f"QUBO 總變數 **{p.n_variables}**（決策 {ir.n_variables} + 輔助 {p.n_variables - ir.n_variables}），"
           f"二次項 {s['n_quadratic_terms']}，係數動態範圍 {s['coef_dynamic_range']}", "",
           "| 限制 | 編碼 | 檢查 |", "|---|---|---|"]
    nxt = ir.n_variables
    for c in ir.constraints:
        name = enc.get(c.name, {}).get("encoding") or next(iter(options_for(c)))
        en = encode(c, name, nxt)
        nxt += len(en.aux)
        r = verify_encoding(c, en, ir.n_variables)
        status = ("略過（變數太多，無法窮舉）" if not r["checked"] else
                  "✅ 精確，雙向驗證通過" if r["exact"] and r["passed"] else
                  "⚠️ 近似（不保證最佳解不變）" if not r["exact"] else f"❌ {r['errors'][:2]}")
        out.append(f"| {c.name} | {name} | {status} |")
    return "\n".join(out)


# ---------------------------------------------------------------- ④ solve / ⑤ results
def _fmt(v) -> str:
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


def _describe(obj, depth: int = 0) -> str:
    """describe_solution() output as readable Markdown (dicts -> bullets, lists of dicts -> tables)."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                cols = list(dict.fromkeys(c for x in v for c in x))
                out += [f"\n**{k}**（{len(v)} 筆{'，只顯示前 50 筆' if len(v) > 50 else ''}）\n",
                        "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                out += ["| " + " | ".join(_fmt(x.get(c, "")) for c in cols) + " |" for x in v[:50]]
                out.append("")
            elif isinstance(v, (dict, list)) and not (isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)):
                out.append(f"{'  ' * depth}- **{k}**：\n{_describe(v, depth + 1)}")
            elif isinstance(v, list):
                out.append(f"{'  ' * depth}- **{k}**：{', '.join(map(_fmt, v)) if v else '（無）'}")
            else:
                out.append(f"{'  ' * depth}- **{k}**：{_fmt(v)}")
        return "\n".join(out)
    if isinstance(obj, list):
        return "\n".join(f"{'  ' * depth}- {_describe(x, depth + 1) if isinstance(x, (dict, list)) else _fmt(x)}" for x in obj)
    return _fmt(obj)


def _render_result(record: dict):
    """(summary markdown, solution markdown, rounds table, objective chart) for a saved run."""
    sm, loop = record["summary"], record.get("loop") or {}
    delivery = record.get("delivery", {})
    gap, ref = sm.get("gap"), sm.get("reference")
    mark = {"usable": "✅", "limited": "⚠️", "no_solution": "❌"}.get(delivery.get("status"),
              "✅" if sm.get("best_objective") is not None else "❌")
    head = (f"### {mark} "
            f"{BACKEND_NAME.get(loop.get('backend'), loop.get('backend', ''))}{'（模擬）' if loop.get('simulated') else ''}"
            f" · policy {loop.get('policy', '')}")
    cards = [
        ("交付方案目標值", _fmt(sm.get("best_objective")) if sm.get("best_objective") is not None else "沒有可行解"),
        ("參考最佳值", _fmt(ref) if ref is not None else "—"),
        ("與最佳差距", f"{gap:.1%}" if gap is not None else "—"),
        ("簡單可行基準", _fmt(delivery.get("baseline_objective"))),
        ("交付方案來源", {"baseline": "簡單基準", "annealer": "退火求解器"}.get(delivery.get("solution_source"), "—")),
        ("退火樣本可行率", f"{sm['feasibility_rate']:.0%}" if sm.get("feasibility_rate") is not None else "—"),
        ("solver 呼叫", str(sm.get("solver_calls", "—"))),
        ("停止原因", STOP_REASON.get(sm.get("stop_reason", ""), sm.get("stop_reason", ""))),
    ]
    card_html = "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in cards) + "</div>"
    warn = "\n\n" + delivery.get("message", "")
    if delivery.get("recovery_message"):
        warn += "\n\n下一步：" + delivery["recovery_message"]
    best = loop.get("best_solution")
    if best is not None and not any(best):
        warn += "\n\n方案是全 0（什麼都沒選）；是否合理請依需求和基準判斷。"
    rows, chart = [], []
    for r in loop.get("rounds", []):
        fb = r.get("feedback") or {}
        obj = fb.get("best_feasible_objective")
        rows.append([r["index"] + 1, r.get("source", ""), f"{fb['feasibility_rate']:.0%}" if "feasibility_rate" in fb else "—",
                     _fmt(obj) if obj is not None else "—", f"{fb['gap']:.1%}" if fb.get("gap") is not None else "—",
                     ", ".join(f"{name}: {rate:.0%}" for name, rate in fb.get("constraint_violation_rates", {}).items() if rate > 0) or "—",
                     ", ".join(f"{name}: {weight:.3g}" for name, weight in r["config"].get("penalty_weights", {}).items()) or "—",
                     "⬆️" if r.get("improved") else "", (r.get("error") or (r.get("rationale") or ""))[:160]])
        if obj is not None:
            chart.append({"輪次": r["index"] + 1, "目標值": obj, "系列": "本輪最佳可行解"})
            if ref is not None:
                chart.append({"輪次": r["index"] + 1, "目標值": ref, "系列": "參考最佳值"})
    sol = sm.get("solution")
    sol_md = _describe(sol) if sol is not None else "（沒有可行解可以說明）"
    return head + "\n\n" + card_html + warn, sol_md, rows, pd.DataFrame(chart, columns=["輪次", "目標值", "系列"])


ROUND_HEADERS = ["輪次", "誰提的設定", "可行率", "最佳可行目標值", "gap", "原限制違反率", "懲罰權重", "進步", "說明 / 錯誤"]
EMPTY_CHART = pd.DataFrame(columns=["輪次", "目標值", "系列"])


def checkpoint_choices(sid):
    root = _model_dir(sid)
    version = ArtifactStore(root).current()["version"]
    choices = []
    for path in sorted((root / "checkpoints").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        saved = json.loads(path.read_text())
        context = saved.get("context", {})
        if context.get("policy", "").startswith("penalty_") and context.get("ir") and context["ir"]["instance_id"].endswith(version):
            choices.append((f"{context['policy']} · 已用 {saved['solver_calls']} 次 · {saved.get('stop_reason') or '中斷'}", str(path)))
    from qubo_auto.annealing.progress import load_progress
    state = load_progress(root)
    stages = {"data": "準備資料", "model": "建模", "preflight": "求解前檢查", "solve": "求解", "recovery": "修復", "delivery": "交付"}
    statuses = {"running": "執行中", "ready": "已就緒", "complete": "完成", "needs_input": "需要補充資訊"}
    actions = {"prepare_data": "準備資料", "build_model": "建模", "solve": "開始求解", "resume_solver": "接續求解", "review_recovery": "檢查修復結果", "view_result": "查看成果", "confirm_requirements_or_data": "確認需求或資料"}
    text = f"目前階段：{stages.get(state.get('stage'), '尚未開始')}；狀態：{statuses.get(state.get('status'), state.get('status', '—'))}；下一步：{actions.get(state.get('next_action'), '—')}"
    return text, gr.update(choices=choices, value=choices[0][1] if choices else None)


def resume_solve(sid, backend, simulate, max_calls, policy_name, mode, use_rag, use_enc_agent, fujitsu_tl, enc_table, resume_path):
    if not resume_path:
        raise gr.Error("請先重新整理進度並選擇可接續的紀錄。")
    yield from solve(sid, backend, simulate, max_calls, policy_name, mode, use_rag, use_enc_agent, fujitsu_tl, enc_table, resume_path)


def solve(sid, backend, simulate, max_calls, policy_name, mode, use_rag, use_enc_agent, fujitsu_tl, enc_table, resume_path=None):
    try:
        ns, data, ir = _load_ir(sid)
    except Exception as e:
        raise gr.Error(f"請先完成「③ 建模與資料」（{e}）")
    cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))
    cfg["loop"]["max_solver_calls"] = int(max_calls)
    cfg["agents"]["mode"] = mode
    cfg["agents"]["encoding_agent"] = bool(use_enc_agent)
    cfg["rag"]["enabled"] = bool(use_rag)
    cfg["param_caps"].setdefault("fujitsu_da", {})["time_limit_sec"] = int(fujitsu_tl)
    saved_context = None
    if resume_path:
        path = Path(resume_path).resolve()
        if not path.is_relative_to((_model_dir(sid) / "checkpoints").resolve()):
            raise gr.Error("請選擇此工作自己的 checkpoint。")
        saved_context = json.loads(path.read_text())["context"]
        cfg = saved_context["config"]
        cfg["loop"]["max_solver_calls"] = int(max_calls)
        backend, simulate, policy_name = saved_context["backend"], not saved_context["real"], saved_context["policy"]
        use_rag = cfg.get("rag", {}).get("enabled", False)
    rag = None
    if use_rag:
        from qubo_auto.annealing.rag import KnowledgeBase
        rag = KnowledgeBase.from_config(cfg["rag"], ROOT)
    hb = cfg["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    store = ArtifactStore(_model_dir(sid))
    bundle = store.current()
    store.commit(bundle["code"], bundle["data"], "求解前版本", bundle.get("metadata", {}))
    manual = _encodings_from(enc_table)

    logs: list[str] = [f"問題：{ir.n_variables} 個決策變數；"
                       f"{BACKEND_NAME[backend]}{'（模擬）' if simulate else ''}；policy={policy_name}"]
    result: dict = {}

    def work():
        try:
            fixed = manual if not (policy_name == "agent" and use_enc_agent) else None
            result["workflow"] = run_workflow(store, get_backend(backend),
                lambda: make_policy(policy_name, seed=saved_context["policy_seed"] if saved_context else 0,
                                    cfg=cfg, root=ROOT, rag=rag), cfg,
                simulate=simulate, budget=budget, rag=rag, log=logs.append, encodings=fixed,
                repair=lambda issue, bundle: _recover_workflow(sid, issue, bundle, logs),
                resume_checkpoint=Path(resume_path) if resume_path else None)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    t0 = time.time()
    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        time.sleep(1.5)
        yield (f"⏳ 求解中…已經 {time.time() - t0:.0f} 秒（agent 每輪都要想一下，請稍候）", "\n".join(logs),
               gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), _budget_text(), gr.update())
    if "error" in result:
        yield (f"❌ 求解失敗：{result['error']}", "\n".join(logs), gr.update(), gr.update(), gr.update(), gr.update(),
               gr.update(), _budget_text(), gr.update())
        return
    workflow = result["workflow"]
    loop = workflow["loop"] or {}
    delivery = workflow["delivery"]
    summary = {"best_objective": delivery.get("objective"), "reference": loop.get("reference_objective"),
               "gap": loop.get("best_gap") if delivery.get("solution_source") == "annealer" else None,
               "feasibility_rate": loop.get("best_feasibility_rate"), "stop_reason": loop.get("stop_reason", "preflight"),
               "solver_calls": workflow["total_solver_calls"], "llm_tokens": loop.get("llm_tokens", 0),
               "version": workflow["version"]}
    run_dir = _model_dir(sid) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{backend}_{policy_name}_{workflow['version'][:8]}"
    record = {"summary": summary, "loop": loop, "delivery": delivery, "workflow": workflow}
    report_path = run_dir / f"{run_name}.json"
    atomic_write(report_path, json.dumps(record, ensure_ascii=False, default=str))
    if workflow["solution"] is not None:
        bundle = store.read(workflow["version"])
        ns, data, best = M.load_module(bundle["code"]), bundle["data"], workflow["solution"]
        try:
            summary["solution"] = ns["describe_solution"](data, best)
            fig = M.figure(ns, data, best, run_dir / f"{run_name}.png")
            if fig:
                summary["figure"] = str(fig)
        except Exception as e:
            summary["solution"] = {"decision_bits": best, "report_issue": str(e)}
            logs.append(f"[成果] 說明／圖表生成失敗，已保存可行方案：{e}")
        atomic_write(report_path, json.dumps(record, ensure_ascii=False, default=str))
    head, sol_md, rows, chart = _render_result(json.loads(json.dumps(record, default=str)))
    mark = {"usable": "✅", "limited": "⚠️", "no_solution": "❌"}[delivery["status"]]
    yield (f"{mark} {delivery['message']}，用了 {time.time() - t0:.0f} 秒。", "\n".join(logs), head, summary.get("figure"),
           sol_md, rows, chart, _budget_text(), json.dumps(summary, ensure_ascii=False, indent=1, default=str))


def _run_label(name: str) -> str:
    parts = name.removesuffix(".json").split("_")
    try:
        when = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S").strftime("%m/%d %H:%M")
    except (ValueError, IndexError):
        return name
    rest = "_".join(parts[2:])
    backend = next((b for b in BACKEND_NAME if rest.startswith(b + "_")), "")
    return f"{when} · {BACKEND_NAME.get(backend, '')} · {rest[len(backend) + 1:] if backend else rest}"


def list_runs(sid: str):
    d = _model_dir(sid) / "runs" if sid else None
    names = sorted((p.name for p in d.glob("*.json")), reverse=True) if d and d.exists() else []
    return gr.update(choices=[(_run_label(n), n) for n in names], value=names[0] if names else None)


def show_run(sid: str, name: str):
    if not sid or not name:
        return "", None, "", [], EMPTY_CHART, ""
    record = json.loads((_model_dir(sid) / "runs" / name).read_text(encoding="utf-8"))
    fig = record["summary"].get("figure")
    head, sol_md, rows, chart = _render_result(record)
    return (head, fig if fig and Path(fig).exists() else None, sol_md, rows, chart,
            json.dumps(record["summary"], ensure_ascii=False, indent=1, default=str))


# ---------------------------------------------------------------- layout
CSS = """
.steps {display:flex; flex-wrap:wrap; gap:6px; align-items:center; font-size:15px}
.step {padding:4px 12px; border-radius:999px; border:1px solid var(--border-color-primary)}
.step.done {background:var(--color-accent-soft); color:var(--body-text-color)}
.step.now {background:var(--color-accent); color:white; font-weight:600}
.step.todo {opacity:.5}
.arrow {opacity:.4}
.hint {margin-top:6px; opacity:.8}
.cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin:8px 0}
.card {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px}
.card .k {font-size:12px; opacity:.7} .card .v {font-size:20px; font-weight:600; margin-top:2px}
.qcard {border:1px solid var(--border-color-primary); border-radius:10px; padding:4px 8px; margin-bottom:8px}
"""

with gr.Blocks(title="QUBO Annealing Agent") as app:
    gr.Markdown("# QUBO Annealing Agent\n用一般的話描述最佳化問題 → agent 釐清、建模 → 在模擬退火 / Fujitsu / D-Wave 上求解。")
    with gr.Row():
        sid = gr.Dropdown(choices=_session_choices(), value=None, label="目前的題目", scale=5,
                          info="💬 待釐清　📐 待建模　▶️ 可求解　✅ 已有結果")
        btn_newq = gr.Button("＋ 新題目", scale=1)
    progress = gr.HTML(_progress(None))
    questions = gr.State([])

    with gr.Tabs(selected=0) as tabs:
        # ---------------- ①
        with gr.Tab(STEPS[0], id=0):
            desc = gr.Textbox(lines=7, label="題目描述",
                              placeholder="例如：我想在台北市 20 個候選地點中選幾個設置充電站，預算有限，"
                                          "希望涵蓋最多人口，兩個站之間至少要隔 500 公尺…")
            backend_c = gr.Radio(BACKENDS, value="cga", label="之後打算用哪個 solver？",
                                 info="會影響 agent 估計規模是否放得進去；之後求解時還可以換。")
            btn_new = gr.Button("開始分析", variant="primary")
            status0 = gr.Markdown()

        # ---------------- ②
        with gr.Tab(STEPS[1], id=1, interactive=False) as tab1:
            with gr.Row():
                with gr.Column(scale=3):
                    status1 = gr.Markdown()

                    q_header = gr.Markdown(visible=False)
                    q_slots = []                  # fixed slots: gr.render events fail in this Gradio version
                    for _ in range(MAX_Q):
                        with gr.Group(visible=False, elem_classes="qcard") as box:
                            md = gr.Markdown()
                            radio = gr.Radio([], show_label=False)
                            text = gr.Textbox(show_label=False, lines=1)
                        q_slots.append((box, md, radio, text))
                    btn_ans = gr.Button("送出回答", variant="primary", visible=False)

                    btn_conf = gr.Button("✅ 確認規格（接受所有「⚠️ 假設」）", variant="primary", visible=False)
                    not_ready = gr.Markdown("還有問題沒回答完，回答後 agent 會更新理解。", visible=False)
                with gr.Column(scale=4):
                    draft_view = gr.Markdown(label="agent 目前的理解")

        # ---------------- ③
        with gr.Tab(STEPS[2], id=2, interactive=False) as tab2:
            btn_model = gr.Button("產生模型（Modeling Agent）", variant="primary")
            model_msg = gr.Markdown()
            with gr.Accordion("模型程式碼", open=False):
                model_code = gr.Code(language="python")
            with gr.Group(visible=False) as data_box:
                gr.Markdown("### 資料")
                ir_view = gr.Markdown()
                with gr.Accordion("換一份模擬資料", open=False):
                    src = gr.Radio(["模擬資料", "上傳真實資料"], value="模擬資料", show_label=False)
                    with gr.Row():
                        size = gr.Slider(4, 300, value=20, step=1, label="規模（主要決策單位數，例如候選地點數）")
                        seed = gr.Number(value=0, label="隨機種子", precision=0)
                    upload = gr.File(label="真實資料（JSON，欄位依「模型需要的資料」）", file_types=[".json"])
                    btn_data = gr.Button("套用這份資料")
                with gr.Accordion("使用真實資料（上傳檔案或讓 agent 找開放資料）", open=False):
                    gr.Markdown("流程：加入資料檔 → Data Adapter 轉換成模型需要的格式 → **你核准後才會使用**。"
                                "agent 只看資料摘要，不讀整份資料。")
                    with gr.Row():
                        real_files = gr.File(label="上傳檔案（CSV / Excel / JSON / GeoJSON / Shapefile zip）",
                                             file_count="multiple", scale=3,
                                             file_types=[".csv", ".tsv", ".txt", ".json", ".geojson", ".xlsx", ".xls", ".zip"])
                        with gr.Column(scale=1):
                            btn_add = gr.Button("加入這些檔案")
                            btn_clear = gr.Button("清除所有資料檔")
                    btn_scout = gr.Button("🔎 讓 agent 找開放資料（政府資料開放平台、OpenStreetMap）")
                    cand_md = gr.Markdown()
                    cand_pick = gr.CheckboxGroup(choices=[], label="勾選要下載的資料（✔ = agent 判斷相關）")
                    btn_fetch = gr.Button("下載勾選的資料")
                    files_md = gr.Markdown()
                    adapt_notes = gr.Textbox(label="補充說明給 Data Adapter（選填，例如：只用大安區、人口欄位叫 xx）", lines=2)
                    btn_adapt = gr.Button("產生資料轉換（Data Adapter）", variant="primary")
                    btn_auto_data = gr.Button("自動備妥真實資料（搜尋、下載、預處理、失敗換來源）")
                    adapt_md = gr.Markdown()
                    btn_approve = gr.Button("✅ 核准使用這份資料", variant="primary", visible=False)
                with gr.Accordion("版本紀錄與還原", open=False):
                    versions = gr.Dropdown(choices=[], label="模型與資料版本")
                    btn_versions = gr.Button("更新版本紀錄")
                    btn_restore = gr.Button("還原選取版本")
                    restore_msg = gr.Markdown()
                with gr.Accordion("資料內容", open=False):
                    data_view = gr.Code(language="json")
                with gr.Accordion("進階：手動指定每條限制的編碼（預設由 Encoding Agent 決定）", open=False):
                    enc_df = gr.Dataframe(headers=ENC_HEADERS, datatype=["str", "str", "str", "str"],
                                          interactive=True, wrap=True, label="在最後一欄填編碼名稱；留空＝預設",
                                          static_columns=[0, 1, 2])
                    btn_enc = gr.Button("檢查這組編碼")
                    enc_view = gr.Markdown()
            btn_go_solve = gr.Button("下一步：去求解 →", visible=False)

        # ---------------- ④
        with gr.Tab(STEPS[3], id=3, interactive=False) as tab3:
            with gr.Row():
                backend_s = gr.Radio(BACKENDS, value="cga", label="Solver", scale=3)
                with gr.Column(scale=2):
                    simulate = gr.Checkbox(value=False, label="真機改用本地替身模擬（不耗額度，測流程用）")
                    max_calls = gr.Slider(1, 10, value=6, step=1, label="最多呼叫 solver 幾次")
            budget_view = gr.Markdown(_budget_text())
            with gr.Accordion("進階設定", open=False):
                with gr.Row():
                    policy = gr.Dropdown([("Agent（LLM 依回饋調參）", "agent"), ("規則式", "rule"), ("預設值（先跑這個；有違反限制或品質不足再調參）", "default"),
                                          ("隨機", "random"), ("Optuna（含 solver 參數）", "optuna"),
                                          ("Penalty：Optuna TPE", "penalty_tpe"), ("Penalty：Grid Search", "penalty_grid"),
                                          ("Penalty：CEM", "penalty_cem"), ("Penalty：逐步增加", "penalty_sequential")],
                                         value="default", label="誰來調參數")
                    mode = gr.Dropdown([("多 agent", "multi"), ("單 agent", "single")], value="multi", label="Agent 模式")
                    fujitsu_tl = gr.Slider(1, 10, value=2, step=1, label="Fujitsu 每次最多跑幾秒")
                with gr.Row():
                    use_rag = gr.Checkbox(value=True, label="使用知識庫（RAG）")
                    use_enc = gr.Checkbox(value=True, label="由 Encoding Agent 選編碼（關掉就用 ③ 手動指定的）")
            btn_solve = gr.Button("開始求解", variant="primary", size="lg")
            with gr.Accordion("工作進度與接續", open=False):
                task_status = gr.Markdown()
                btn_progress = gr.Button("查看進度／更新可接續搜尋")
                checkpoint_pick = gr.Dropdown([], label="同模型／資料版本的 penalty 搜尋")
                btn_resume = gr.Button("接續選定搜尋（沿用原 backend 與設定）")
            solve_status = gr.Markdown()
            with gr.Accordion("即時紀錄", open=True):
                log_view = gr.Textbox(lines=12, show_label=False, autoscroll=True)

        # ---------------- ⑤
        with gr.Tab(STEPS[4], id=4, interactive=False) as tab4:
            with gr.Row():
                runs = gr.Dropdown(choices=[], label="這題的求解紀錄（最新在上）", scale=5)
                btn_runs = gr.Button("重新整理", scale=1)
            res_head = gr.HTML()
            fig_view = gr.Image(label="成果圖（由模型指定內容、程式繪製）", type="filepath", interactive=False)
            chart = gr.LinePlot(EMPTY_CHART, x="輪次", y="目標值", color="系列", title="每一輪的目標值", height=260)
            gr.Markdown("### 每一輪做了什麼")
            rounds_tbl = gr.Dataframe(headers=ROUND_HEADERS, interactive=False, wrap=True)
            gr.Markdown("### 最佳解說明")
            sol_view = gr.Markdown()
            with gr.Accordion("原始 JSON", open=False):
                raw_view = gr.Code(language="json")

    gate_out = [progress, tab1, tab2, tab3, tab4]
    slot_out = [q_header] + [c for slot in q_slots for c in slot] + [btn_ans]
    answer_inputs = [c for _, _, radio, text in q_slots for c in (radio, text)]
    questions.change(question_slots, questions, slot_out)
    btn_ans.click(lambda: LLM_WAIT, None, status1).then(
        submit_answers, [sid, questions, *answer_inputs], [draft_view, status1, questions, btn_conf, not_ready]).then(
        gate, sid, gate_out).then(lambda m: '' if m == LLM_WAIT else m, status1, status1)
    btn_enc.click(check_encodings, [sid, enc_df], enc_view)

    # -------- wiring
    def _new_question():
        return gr.update(value=None), "", gr.Tabs(selected=0)

    btn_newq.click(_new_question, None, [sid, desc, tabs])
    btn_new.click(lambda: LLM_WAIT, None, status0).then(
        start_session, [desc, backend_c], [sid, tabs]).then(lambda: "", None, status0)
    sid.change(show_session, sid, [draft_view, status1, questions, btn_conf, not_ready, backend_s, tabs]).then(
        gate, sid, gate_out).then(list_runs, sid, runs)
    btn_conf.click(confirm_session, sid, [draft_view, status1, questions, btn_conf, not_ready, tabs]).then(
        gate, sid, gate_out).then(_refresh_labels, None, sid)

    btn_model.click(lambda: LLM_WAIT, None, model_msg).then(
        build_model, sid, [model_msg, model_code, data_box, ir_view, data_view, enc_df]).then(
        gate, sid, gate_out).then(lambda s: gr.update(visible=_stage(s) >= 3), sid, btn_go_solve).then(
        lambda m: '' if m == LLM_WAIT else m, model_msg, model_msg).then(
        _refresh_labels, None, sid)
    btn_data.click(make_data, [sid, src, size, seed, upload], [ir_view, data_view, enc_df]).then(gate, sid, gate_out)
    btn_add.click(add_files, [sid, real_files], files_md)
    btn_clear.click(clear_files, sid, [files_md, cand_pick, adapt_md, btn_approve])
    btn_scout.click(lambda: "⏳ agent 正在規劃搜尋並篩選結果，約 1–3 分鐘…", None, cand_md).then(
        scout_data, sid, [cand_md, cand_pick])
    btn_fetch.click(lambda: "⏳ 下載中…", None, files_md).then(fetch_selected, [sid, cand_pick], files_md)
    btn_adapt.click(lambda: "⏳ Data Adapter 正在寫轉換程式並在完整資料上檢查，約 1–5 分鐘…", None, adapt_md).then(
        adapt_data, [sid, adapt_notes], [adapt_md, btn_approve])
    btn_approve.click(approve_data, sid, [ir_view, data_view, enc_df, btn_approve]).then(gate, sid, gate_out)
    btn_auto_data.click(lambda: "⏳ 正在備妥資料；會保留現有版本並嘗試替代來源…", None, adapt_md).then(
        auto_prepare_data, [sid, adapt_notes], [adapt_md, btn_approve, files_md, ir_view, data_view, enc_df]).then(
        gate, sid, gate_out)
    btn_versions.click(version_choices, sid, versions)
    btn_restore.click(restore_version, [sid, versions], [restore_msg, ir_view, data_view, enc_df]).then(gate, sid, gate_out)
    btn_go_solve.click(lambda: gr.Tabs(selected=3), None, tabs)

    btn_solve.click(
        solve, [sid, backend_s, simulate, max_calls, policy, mode, use_rag, use_enc, fujitsu_tl, enc_df],
        [solve_status, log_view, res_head, fig_view, sol_view, rounds_tbl, chart, budget_view, raw_view]).success(
        gate, sid, gate_out).then(list_runs, sid, runs).then(_refresh_labels, None, sid).then(
        lambda m: gr.Tabs(selected=4 if m.startswith(("✅", "⚠️", "❌")) else 3), solve_status, tabs)
    btn_progress.click(checkpoint_choices, sid, [task_status, checkpoint_pick])
    btn_resume.click(
        resume_solve, [sid, backend_s, simulate, max_calls, policy, mode, use_rag, use_enc, fujitsu_tl, enc_df, checkpoint_pick],
        [solve_status, log_view, res_head, fig_view, sol_view, rounds_tbl, chart, budget_view, raw_view]).success(
        gate, sid, gate_out).then(list_runs, sid, runs)
    btn_runs.click(list_runs, sid, runs)
    runs.change(show_run, [sid, runs], [res_head, fig_view, sol_view, rounds_tbl, chart, raw_view])

    def _restore_model(s):
        """When reopening a session that already has a model, show it instead of an empty tab."""
        d = _model_dir(s) if s else None
        if not d or not (d / "model.py").exists():
            return "", "", gr.update(visible=False), "", "", [], gr.update(visible=False)
        try:
            code, _ = _load_model(s)
            ov = _ir_overview(s, "上次使用的資料") if (d / "data.json").exists() else ("", "", [])
            return (_model_summary(s), code, gr.update(visible=True), *ov, gr.update(visible=_stage(s) >= 3))
        except Exception as e:
            return f"❌ 讀取既有模型失敗：{e}", "", gr.update(visible=False), "", "", [], gr.update(visible=False)

    sid.change(_restore_model, sid, [model_msg, model_code, data_box, ir_view, data_view, enc_df, btn_go_solve])

if __name__ == "__main__":
    # UI_SHARE=1 creates a public *.gradio.live link; it is only allowed together with a login (UI_AUTH_USER/PASS)
    share = os.environ.get("UI_SHARE") == "1"
    user, pw = os.environ.get("UI_AUTH_USER"), os.environ.get("UI_AUTH_PASS")
    if share and not (user and pw):
        raise SystemExit("UI_SHARE=1 requires UI_AUTH_USER and UI_AUTH_PASS (in .env)")
    app.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("UI_PORT", 7860)), share=share,
                       auth=(user, pw) if user and pw else None, css=CSS, allowed_paths=[str(UI_DIR)])
