"""Report real D-Wave Hybrid results, with caller-requested and measured time separated."""
import json, statistics, shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/experiments/dwave_hybrid_10s_20261011'
OUT = ROOT / 'results/paper_evidence_20261011'
PAPER = ROOT.parent / 'paper'
rows = json.loads((RUN / 'results.json').read_text())
audit = list(map(json.loads, (RUN / 'timing_audit.jsonl').read_text().splitlines())) if (RUN / 'timing_audit.jsonl').exists() else []
calls = sum(r['solver_calls'] for r in rows)
charge = sum(r['timing_us'].get('charge_time', 0) for r in audit) / 1e6
qpu_access = sum(r['timing_us'].get('qpu_access_time', 0) for r in audit) / 1e6
qpu_positive = sum(r['timing_us'].get('qpu_access_time', 0) > 0 for r in audit)
groups = []
lines = ['', '## D-Wave Hybrid：每次 10 秒', '',
    '實際 Leap Hybrid BQM 服務；不等同直接 QPU，也未使用 SA 代替。模型與先前 SA/CGA 比較相同。',
    f'完成 {len(rows)}/27 個條件，累計 {calls} 次求解提交；每次明確傳入 time_limit=10 秒，每條件最多 3 次。',
    '整批上限 81 次、810 秒。每次提交前檢查模型最低時間與剩餘預算；不自動加時。',
    f'另外唯讀取得 {len(audit)} 筆已完成答案的 timing，服務 charge_time 合計 {charge:.3f} 秒；這些查詢沒有新增求解提交。',
    f'同一批答案回報 QPU access time 合計 {qpu_access:.3f} 秒，{qpu_positive}/{len(audit)} 筆大於零；這是 Hybrid 內部使用量，不是直接 QPU 的比較實驗。',
    'Timing 欄位按微秒換算，定義見 [D-Wave 官方計時說明](https://docs.dwavequantum.com/en/latest/industrial_optimization/index_hybrid.html)。',
    '原始 ledger 使用每次 10 秒保守保留；實測 timing 另存 timing_audit.jsonl。端到端時間包含網路排隊與 LLM。', '',
    '| 問題 | 策略 | 可行／重複 | 平均 gap | 平均 calls | 平均端到端秒數 |',
    '|---|---|---:|---:|---:|---:|']
for case in ['wine', 'knapsack60', 'select80']:
    for policy in ['default', 'penalty_tpe', 'agent']:
        group = [r for r in rows if r['case'] == case and r['policy'] == policy]
        if not group: continue
        gaps = [r['gap'] for r in group if r['gap'] is not None]
        s = {'case': case, 'policy': policy, 'n': len(group),
             'feasible': sum(r['status'] == 'feasible' for r in group),
             'mean_gap': statistics.mean(gaps) if gaps else None,
             'mean_calls': statistics.mean(r['solver_calls'] for r in group),
             'mean_wall_s': statistics.mean(r['wall_s'] for r in group)}
        groups.append(s)
        gap = f"{s['mean_gap']:.4f}" if gaps else '—'
        lines.append(f"| {case} | {policy} | {s['feasible']}/{len(group)} | {gap} | {s['mean_calls']:.2f} | {s['mean_wall_s']:.1f} |")
lines += ['', '![D-Wave Hybrid](dwave_hybrid_comparison.png)', '',
    '每條件三次外部重複提交，不是可控 hardware RNG seed。Hybrid 的 10 秒與 CGA 的 3 秒、SA 的 reads/sweeps 不代表相同算力；不能据此宣稱硬體優勢。',
    '這是固定小型模型的求解／调參比較，沒有證明任意資料來源、模型生成或跨後端 RAG 轉移能力。', '']
text = '\n'.join(lines).replace('据此', '據此').replace('调參', '調參')
(OUT / 'dwave_hybrid_section.md').write_text(text)
stats = {'completed': len(rows), 'expected': 27, 'solver_calls': calls, 'timing_answers': len(audit),
         'service_charge_s': charge, 'qpu_access_s': qpu_access, 'qpu_positive_answers': qpu_positive,
         'requested_limit_s': 10, 'groups': groups}
(OUT / 'dwave_hybrid_statistics.json').write_text(json.dumps(stats, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
for ax, case in zip(axes, ['wine', 'knapsack60', 'select80']):
    vals = [next((s['mean_gap'] for s in groups if s['case'] == case and s['policy'] == p), np.nan)
            for p in ['default', 'penalty_tpe', 'agent']]
    vals = [v if v is not None else np.nan for v in vals]
    ax.bar(range(3), vals, color='#217e8b')
    ax.set_title(case); ax.set_xticks(range(3), ['default', 'TPE', 'LLM']); ax.set_xlabel('policy')
    for i, val in enumerate(vals):
        if val is not None and np.isfinite(val): ax.text(i, val + .002, f'{val:.4f}', ha='center', fontsize=9)
axes[0].set_ylabel('Mean gap to exact reference')
finite_gaps = [s['mean_gap'] for s in groups if s['mean_gap'] is not None]
axes[0].set_ylim(0, max(.05, max(finite_gaps, default=0) * 1.25 + .005))
fig.suptitle('Real D-Wave Hybrid BQM: time_limit = 10 seconds per submission')
fig.tight_layout(); fig.savefig(OUT / 'dwave_hybrid_comparison.png', dpi=170); fig.savefig(OUT / 'dwave_hybrid_comparison.pdf'); plt.close(fig)
shutil.copyfile(OUT / 'dwave_hybrid_comparison.pdf', PAPER / 'figures/dwave_hybrid_comparison.pdf')
tex = [r'\subsection{D-Wave Hybrid BQM with an explicit ten-second limit}', r'\label{sec:dwavehybrid}',
    'We submitted the same three frozen models to the real Leap Hybrid BQM service, under default,',
    'penalty TPE and single-agent policies. This service is distinct from direct QPU sampling.',
    f'The experiment completed {len(rows)} of 27 conditions and used {calls} solver submissions.',
    'Each condition allowed at most three calls. Every call explicitly set the service time limit',
    'to ten seconds; a model requiring a larger minimum was refused before submission rather than',
    'given a longer limit. Sequential execution was bounded by 81 calls and 810 service seconds.',
    f'A read-only retrieval of {len(audit)} completed answers reported total service charge time',
    f'of {charge:.3f} seconds. The original ledger conservatively reserved ten seconds per call;',
    f'the answers also report {qpu_access:.3f} seconds of aggregate internal QPU access,',
    f'positive for {qpu_positive} of {len(audit)} answers. These are hybrid-service timings, not direct-QPU runs.',
    'measured service timings are preserved separately. Network and inference time are additional.',
    r'\begin{table}[t]\centering\small',
    r'\caption{Real Hybrid BQM comparison. Gaps use exact references for the three frozen models; wall time includes network and inference.}',
    r'\begin{tabular}{@{}llrrrr@{}}\toprule Model & Policy & Feasible & Gap & Calls & Wall s \\\midrule']
for s in groups:
    gap = f"{s['mean_gap']:.4f}" if s['mean_gap'] is not None else '--'
    p = s['policy'].replace('_', r'\_')
    tex.append(f"{s['case']} & {p} & {s['feasible']}/{s['n']} & {gap} & {s['mean_calls']:.2f} & {s['mean_wall_s']:.1f} " + r'\\')
tex += [r'\bottomrule\end{tabular}\end{table}',
    r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/dwave_hybrid_comparison.pdf}',
    r'\caption{Real hybrid-service results with an explicit ten-second limit.}\end{figure}',
    'The three repetitions are external resubmissions, without exposed hardware RNG seeds.',
    'The ten-second Hybrid ceiling and earlier three-second CGA ceiling are different resource',
    'allocations. They do not establish a hardware advantage, nor a held-out knowledge-transfer result.']
(PAPER / 'sections/dwave_hybrid_experiments.tex').write_text('\n'.join(tex) + '\n')
print(json.dumps(stats, indent=2))
