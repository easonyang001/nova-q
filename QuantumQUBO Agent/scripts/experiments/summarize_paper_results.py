"""Create traceable statistics, figures and a reviewable report from saved experiment records."""
import json,csv,statistics,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/paper_evidence_20261011';OUT.mkdir(parents=True,exist_ok=True)

def extended():
 path=ROOT/'results/experiments/penalty_extended_20261011/runs.csv'
 rows=list(csv.DictReader(path.open()));policies=list(dict.fromkeys(r['policy'] for r in rows))
 instances=sorted({r['instance_id'] for r in rows});seeds=sorted({int(r['seed']) for r in rows})
 matrices={}
 for p in policies:
  lookup={(r['instance_id'],int(r['seed'])):float(r['gap']) for r in rows if r['policy']==p and r['gap'] not in ('','None')}
  if len(lookup)!=len(instances)*len(seeds):raise ValueError('Incomplete feasible pair grid; report failures separately')
  matrices[p]=np.array([[lookup[i,s] for s in seeds] for i in instances])
 # Resample instances and then seeds within each selected instance (two-level bootstrap).
 rng=np.random.default_rng(42);i=rng.integers(0,len(instances),(10000,len(instances)));s=rng.integers(0,len(seeds),(10000,len(instances),len(seeds)))
 stats=[];tests=[]
 for p in policies:
  group=[r for r in rows if r['policy']==p];m=matrices[p];boot=m[i[:,:,None],s].mean(axis=(1,2));lo,hi=np.quantile(boot,[.025,.975])
  stats.append({'policy':p,'runs':len(group),'mean_gap':m.mean(),'ci95':[float(lo),float(hi)],'sd_across_instance_means':float(m.mean(axis=1).std(ddof=1)),'feasible':sum(r['best_objective'] not in ('','None') for r in group),'optimum':sum(float(r['gap'])==0 for r in group),'mean_calls':statistics.mean(int(r['solver_calls']) for r in group),'errors':sum(bool(r['error']) for r in group)})
  if p.startswith('penalty_'):
   d=(m-matrices['default']).mean(axis=1);pv=float(wilcoxon(d).pvalue) if np.any(d) else 1.;tests.append({'comparison':p+' vs default','independent_instance_pairs':len(d),'mean_gap_difference':float(d.mean()),'raw_p':pv})
 order=sorted(range(len(tests)),key=lambda k:tests[k]['raw_p']);running=0
 for rank,k in enumerate(order):running=max(running,min(1.,tests[k]['raw_p']*(len(order)-rank)));tests[k]['holm_p']=running
 atomic={'design':'8 instances x 10 seeds x 8 policies; max 6 calls, 60 reads, 300 sweeps. Same ceiling; actual calls differ due early stop.', 'bootstrap':'95% percentile interval, two-level resampling of instances and within-instance seeds; 10000 draws, seed42. Small eight-instance scope.', 'inference':'Wilcoxon on 8 per-instance seed-averaged differences; Holm correction across 4 penalty comparisons. No 80-independent-instance claim.', 'summaries':stats,'tests':tests}
 (OUT/'penalty_statistics.json').write_text(json.dumps(atomic,indent=2))
 means=[r['mean_gap'] for r in stats];lower=[r['mean_gap']-r['ci95'][0] for r in stats];upper=[r['ci95'][1]-r['mean_gap'] for r in stats]
 fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(stats)),means,color=['#547890']*4+['#169a8c']*4);ax.errorbar(range(len(stats)),means,yerr=[lower,upper],fmt='none',ecolor='black',capsize=3);ax.set_xticks(range(len(stats)),[r['policy'].replace('penalty_','') for r in stats],rotation=20);ax.set_ylabel('Mean gap to exact reference (lower better)');ax.set_title('Penalty search: 640 SA runs; hierarchical bootstrap 95% intervals');fig.tight_layout();fig.savefig(OUT/'penalty_comparison.png',dpi=170);fig.savefig(OUT/'penalty_comparison.pdf');plt.close(fig)
 return atomic

def backends():
 path=ROOT/'results/experiments/paper_backends_20261011/results.json'
 rows=json.loads(path.read_text()) if path.exists() else []
 groups=[]
 for b in ['sa','cga']:
  for case in ['wine','knapsack60','select80']:
   for p in ['default','penalty_tpe','agent']:
    group=[r for r in rows if r['backend']==b and r['case']==case and r['policy']==p]
    if not group:continue
    gaps=[r['gap'] for r in group if r['gap'] is not None]
    groups.append({'backend':b,'case':case,'policy':p,'n':len(group),'feasible':sum(r['status']=='feasible' for r in group),'mean_gap':statistics.mean(gaps) if gaps else None,'mean_wall_s':statistics.mean(r['wall_s'] for r in group),'mean_calls':statistics.mean(r['solver_calls'] for r in group),'mean_llm_tokens':statistics.mean(r['tokens'] for r in group),'errors':sum(bool(r['error']) for r in group)})
 (OUT/'backend_statistics.json').write_text(json.dumps({'completed_conditions':len(rows),'expected_conditions':54,'groups':groups},indent=2))
 if groups:
  fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
  for ax,case in zip(axes,['wine','knapsack60','select80']):
   for k,b in enumerate(['sa','cga']):
    selected=[next((g for g in groups if g['backend']==b and g['case']==case and g['policy']==p),None) for p in ['default','penalty_tpe','agent']]
    vals=[g['mean_gap'] if g and g['mean_gap'] is not None else np.nan for g in selected];ax.bar(np.arange(3)+(k-.5)*.34,vals,width=.34,label=b)
   ax.set_xticks(range(3),['default','TPE','LLM']);ax.set_title(case);ax.set_xlabel('policy')
  axes[0].set_ylabel('Mean gap to exact reference');axes[-1].legend();fig.suptitle('Fixed-model solver comparison (3 repetitions per condition)');fig.tight_layout();fig.savefig(OUT/'backend_comparison.png',dpi=170);fig.savefig(OUT/'backend_comparison.pdf');plt.close(fig)
 return groups,len(rows)

def main():
 ext=extended();groups,n=backends();lines=['# 論文實驗與實作成果（2026-10-11）','','原始紀錄、設定和圖表均保存；既有論文的舊結果與此次新增實驗分開報告。','','## 640 次 penalty 比較','','8 個固定合成 instance × 10 seeds × 8 策略。每次最多 6 calls、60 SA reads、300 sweeps；提早停止會造成實際 calls 不同。95% 區間以 instance/seed 兩層 bootstrap 計算。','','| 策略 | 可行交付 | 平均 gap | 95% CI | 最佳解次數 | 平均 calls |','|---|---:|---:|---|---:|---:|']
 for r in ext['summaries']:lines.append(f"| {r['policy']} | {r['feasible']}/{r['runs']} | {r['mean_gap']:.4f} | [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {r['optimum']} | {r['mean_calls']:.2f} |")
 lines+=['','以 8 個 instance 各自的 seed 平均差做配對檢定，不把 80 次執行當成 80 個獨立問題：','','| 比較 | 平均 gap 差（正值較差） | Holm p |','|---|---:|---:|']
 for r in ext['tests']:lines.append(f"| {r['comparison']} | {r['mean_gap_difference']:+.4f} | {r['holm_p']:.4f} |")
 lines+=['','![Penalty 比較](penalty_comparison.png)','','## 真實求解器與固定模型比較','',f'完成 {n}/54 個條件。每條件最多 3 calls；CGA 搜尋上限每次 3 秒，SA 為 60 reads × 300 sweeps。外部 repetition 是重複提交，不是可控 hardware RNG seed。不同後端秒數不等於同等算力。','','| 後端 | 問題 | 策略 | 可行／重複 | 平均 gap | 平均端到端秒數 |','|---|---|---|---:|---:|---:|']
 for r in groups:lines.append(f"| {r['backend']} | {r['case']} | {r['policy']} | {r['feasible']}/{r['n']} | {r['mean_gap']:.4f}"+f" | {r['mean_wall_s']:.1f} |" if r['mean_gap'] is not None else f"| {r['backend']} | {r['case']} | {r['policy']} | {r['feasible']}/{r['n']} | — | {r['mean_wall_s']:.1f} |")
 lines+=['','![Backend 比較](backend_comparison.png)','','Fujitsu DA 已實際提交一次，但服務端 HTTP 400 回覆 Monthly usage exceeds specified metering limit；屬外部用量阻擋，不能當作模型無可行解，也沒有以 SA 模擬冒充 DA。拒絕紀錄保存在 backend_probe_20261011。','','## 實作成果','','- PyQUBO placeholder/constraint 診斷、Optuna TPE、Grid/CEM/sequential policy。','- 資料來源與雜湊、資料／需求契約、獨立案例核對、版本回退、逾時、checkpoint 接續。','- 可用／品質有限／無方案交付區分，保留可行 baseline 並標註来源。','- [真實紅酒完整成果](../acceptance/wine_live_20261011_fresh2/delivery.md)。','- 真實案例重複驗收與更多題型的結果需讀取對應 acceptance.json；不能以功能測試取代連線成功率。','','## 審稿仍需注意','','同一組硬體條件僅三次重複，不足以支持廣泛的顯著優勢；題型數量仍有限。調 penalty 不代表改善自然語言模型的語意。資料來源指定、确认规格人工提供；所有失敗和驗收器修正必須一起報告。Fujitsu 需帳戶服務端額度恢復才可補齊。','']
 (OUT/'report.md').write_text('\n'.join(lines).replace('来源','來源').replace('确认规格','確認規格'))
 print(OUT/'report.md')
if __name__=='__main__':main()
