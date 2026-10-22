"""Append encoding, routing, acceptance evidence and export a browsable local report."""
import csv,html,json,statistics,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qubo_auto.annealing.artifacts import atomic_write
OUT=ROOT/'results/paper_evidence_20261011'
report=(OUT/'report.md').read_text()
lines=['','## 編碼消融（78 次）','','同一 knapsack 原問題、固定 penalty 與 solver 預算。SA 每條件 10 seeds，CGA 每條件 3 次提交，每條件最多 3 calls。unbalanced 是近似編碼，不承諾保留原問題最佳解。','','| 後端 | 物件数 | 編碼 | QUBO 變數 | 可行／次數 | 平均 gap |','|---|---:|---|---:|---:|---:|']
path=ROOT/'results/experiments/encoding_ablation_20261011/results.json';rows=json.loads(path.read_text());summaries=[]
for b in ['sa','cga']:
 for n in [12,30]:
  for e in ['binary_slack','unary_slack','unbalanced']:
   group=[r for r in rows if r['backend']==b and r['n']==n and r['encoding']==e];gaps=[r['gap'] for r in group if r['gap'] is not None]
   r={'backend':b,'n':n,'encoding':e,'n_runs':len(group),'feasible':sum(r['feasible'] for r in group),'variables':group[0].get('encoded_variables'),'mean_gap':statistics.mean(gaps) if gaps else None};summaries.append(r)
   lines.append(f"| {b} | {n} | {e} | {r['variables']} | {r['feasible']}/{len(group)} | {r['mean_gap']:.4f} |")
atomic_write(OUT/'encoding_statistics.json',json.dumps(summaries,indent=2))
fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
for ax,n in zip(axes,[12,30]):
 for index,b in enumerate(['sa','cga']):
  vals=[next(r['mean_gap'] for r in summaries if r['backend']==b and r['n']==n and r['encoding']==e) for e in ['binary_slack','unary_slack','unbalanced']];ax.bar(np.arange(3)+(index-.5)*.34,vals,width=.34,label=b)
 ax.set_title(f'Knapsack: {n} items');ax.set_xticks(range(3),['binary','unary','unbalanced']);ax.set_xlabel('encoding')
axes[0].set_ylabel('Mean gap to exact optimum');axes[-1].legend();fig.tight_layout();fig.savefig(OUT/'encoding_comparison.png',dpi=170);fig.savefig(OUT/'encoding_comparison.pdf');plt.close(fig)
lines+=['','![Encoding](encoding_comparison.png)','','SA 上 unary 增加的輔助變數伴隨較差品質；unbalanced 減少變數，但 CGA 的 12-item 結果未達最佳值。不能把「變數更少」等同「一定更好」。僅兩個固定 instance，不對其他問題泛化。','']
path=ROOT/'results/experiments/large_routing_20261011/results.json';routing=json.loads(path.read_text()) if path.exists() else [];manifest=json.loads((path.parent/'manifest.json').read_text())
lines+=['## 40 站實際座標路線','','Taipei YouBike 真實快照、手寫且凍結的 TSP IR，40 站、1,600 決策變數。這部分測試求解及調參，不是 LLM 產生路線模型的端到端成功率。距離為大圓公里，不是道路距離。',f"20-restart 2-opt 可行基準：{manifest['baseline_km']:.6f} km。此值未證明最佳；下表 gap 是與 heuristic 的相對差。",'', '| 後端 | 策略 | 可行／次數 | 平均路線 km（可行執行） | 平均 heuristic gap |','|---|---|---:|---:|---:|'];rs=[]
for b in ['sa','cga']:
 for p in ['default','penalty_tpe','agent']:
  group=[r for r in routing if r['backend']==b and r['policy']==p]
  if not group:continue
  good=[r for r in group if r['feasible']];r={'backend':b,'policy':p,'n':len(group),'feasible':len(good),'mean_km':statistics.mean(r['objective_km'] for r in good) if good else None,'mean_gap':statistics.mean(r['gap_to_heuristic'] for r in good) if good else None};rs.append(r)
  lines.append(f"| {b} | {p} | {len(good)}/{len(group)} | {r['mean_km']:.4f} | {r['mean_gap']:.4f} |" if good else f'| {b} | {p} | 0/{len(group)} | — | — |')
atomic_write(OUT/'routing_statistics.json',json.dumps({'completed':len(routing),'expected':18,'baseline_km':manifest['baseline_km'],'groups':rs},indent=2))
if rs:
 fig,ax=plt.subplots(figsize=(7,4));labels=[r['backend']+' / '+r['policy'] for r in rs];vals=[r['mean_km'] if r['mean_km'] is not None else np.nan for r in rs];ax.bar(range(len(rs)),vals,color=['#577f9c' if r['backend']=='sa' else '#1f9a85' for r in rs]);ax.axhline(manifest['baseline_km'],linestyle='--',color='#c05f30',label='2-opt feasible baseline');ax.set_xticks(range(len(rs)),labels,rotation=25);ax.set_ylabel('Route km (feasible runs only)');ax.legend();fig.tight_layout();fig.savefig(OUT/'routing_comparison.png',dpi=170);fig.savefig(OUT/'routing_comparison.pdf');plt.close(fig)
 lines+=['','![Routing](routing_comparison.png)','']
lines+=['## 真實案例與重複驗收','','| 案例 | 實際交付 | 證據範圍 |','|---|---|---|','| 紅酒特徵選擇 | 1,599 筆、恰 5 個特徵、原目標 0.7497088247 | 先前一次全新流程成功；此次三個生成結果經修正的驗收器重查，3/3 可交付 |','| 批發客戶平衡分群 | 24 人、3 組各 8 | 重用模型、重新下載資料與 LLM adapter；獨立核對抽樣與矩陣 |']
case=ROOT/'results/acceptance/karate_live_20261011_v4/acceptance.json'
if case.exists():
 d=json.loads(case.read_text());lines.append(f"| 社交網路平衡分割 | {'成功，原目標 '+str(d.get('requested_objective')) if d['success'] else '尚未通過：'+str(d.get('error'))} | 保存官方原檔、欄位排序錯誤與後续修復紀錄 |")
lines+=['','紅酒初次三次驗收停止，原因是驗收器誤拒等價的負目標、強求非必要 k 欄位，以及對不相關壞檔一起讀取；修正後用保存的模型／資料重查。不把事後修正算成預先固定驗收方案的 100% 自主成功率。','','[紅酒成果](../acceptance/wine_live_20261011_fresh2/delivery.md) · [批發分組成果](../acceptance/wholesale_live_20261011_v3/delivery.md) · [社團分組成果](../acceptance/karate_live_20261011_v4/delivery.md)','','## 此次實作與檢查','','120 項自動測試通過，UI 可行交付、不可能需求零次求解，以及累計 checkpoint 接續通過。補上 GML graph ZIP 讀取、指定抽樣工具、大小可變的限制家族對應、同名下載不覆蓋，以及生成程式 import prep 時仍保留工具使用紀錄。','','尚未完成：Fujitsu 外部額度恢復後的實測、全部 10 題固定條件重複驗收、一般 LLM 語意修復成功率、held-out backend transfer 的 RAG 知識轉移比較。現有結果不足以主張這些能力已獲證明。','']
dwave_section=OUT/'dwave_hybrid_section.md'
if dwave_section.exists():lines.append(dwave_section.read_text())
text=(report+'\n'.join(lines)).replace('物件数','物件數').replace('後续','後續');atomic_write(OUT/'report.md',text)
# Local static report, all plots are standalone PNG/PDF artifacts.
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables','fenced_code'])
except ImportError:
 body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>論文實驗與實作成果</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#20313d}h1,h2{color:#136b75}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dce5e9;text-align:left}th{background:#eef5f7}img{max-width:100%}pre{white-space:pre-wrap}a{color:#086e99}</style><body>'+body+'</body></html>'
atomic_write(OUT/'report.html',page);print(OUT/'report.html')
