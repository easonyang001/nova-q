"""Export human-readable independently audited live-case delivery artifacts."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qubo_auto.annealing.artifacts import ArtifactStore
from qubo_auto.annealing.datakit import load_tables
for name in ['wholesale_live_20261011_v3','karate_live_20261011_v4']:
 out=ROOT/'results/acceptance'/name
 if not (out/'acceptance.json').exists():continue
 report=json.loads((out/'acceptance.json').read_text())
 if not report['success']:continue
 bundle=ArtifactStore(out/'artifacts').current();data=bundle['data'];x=report['workflow']['solution']
 if report['case']=='wholesale':
  assignments=np.array(x).reshape(24,3).argmax(axis=1);rows=list(zip(data['ids'],assignments.tolist()));header=['original_csv_row_index','group']
  order=np.argsort(assignments,kind='stable');matrix=np.array(data['distances'])[np.ix_(order,order)]
  fig,ax=plt.subplots(figsize=(6,5));image=ax.imshow(matrix,cmap='viridis');fig.colorbar(image,ax=ax,label='Squared standardized spending distance');ax.set_title('Wholesale: 24 customers in 3 groups of 8');ax.set_xlabel('Customers ordered by assigned group');ax.set_ylabel('Customers ordered by assigned group')
  for line in [7.5,15.5]:ax.axhline(line,color='white');ax.axvline(line,color='white')
 else:
  import networkx as nx
  adjacency=np.array(data['adjacency']);graph=nx.from_numpy_array(adjacency);assignments=np.array(x);header=['node_index','group'];rows=list(enumerate(x))
  fig,ax=plt.subplots(figsize=(7,5));nx.draw_networkx(graph,pos=nx.spring_layout(graph,seed=0),node_color=['#278db3' if b else '#da9245' for b in x],node_size=230,font_size=7,ax=ax);ax.set_title('Real karate network: balanced groups (17 / 17)');ax.set_axis_off()
 with (out/'assignments.csv').open('w',newline='') as handle:
  writer=csv.writer(handle);writer.writerow(header);writer.writerows(rows)
 fig.tight_layout();fig.savefig(out/'delivery.png',dpi=160);plt.close(fig)
 baseline=report['workflow']['delivery']['baseline_objective'];objective=report['requested_objective'];improvement=(baseline-objective)/abs(baseline)*100 if baseline else 0
 lines=['# '+report['case']+' 真實資料最佳化成果','','原始資料規模：'+json.dumps(report['original_size'],ensure_ascii=False)+'。',f"原目標：{objective:.6f}；可行基準：{baseline:.6f}；相對基準改善：{improvement:.1f}%。",f"交付來源：{report['workflow']['delivery']['solution_source']}；4 次 SA 求解。",'', '[分組 CSV](assignments.csv) · [圖表](delivery.png) · [完整紀錄](acceptance.json)','','| 組別 | 成員 ID／原始列索引 |','|---|---|']
 for group in sorted(set(assignments)):lines.append(f"| {int(group)} | "+', '.join(str(a) for a,b in rows if b==group)+' |')
 lines+=['','已對原始資料和生成模型做獨立核對。這個基準是可行比較方案，沒有宣稱為最佳值。成功執行沿用前次生成模型，重新連線取得資料並產生 adapter；不能當成一次全新 LLM 流程的成功率。','']
 (out/'delivery.md').write_text('\n'.join(lines));print(out/'delivery.md')
