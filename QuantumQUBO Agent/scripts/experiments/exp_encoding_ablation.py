"""Fixed-weight encoding ablation; original objective/constraints and solver ceilings held constant.
Unbalanced is approximate. References are exact dynamic-programming knapsack optima.
"""
import copy,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
import yaml
from qubo_auto.annealing.artifacts import atomic_write
from qubo_auto.annealing.examples import knapsack_ir
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop,build_context
from exp_paper_backends import AbortOnExternalError
OUT=ROOT/'results/experiments/encoding_ablation_20261011';OUT.mkdir(parents=True,exist_ok=True)
base=yaml.safe_load((ROOT/'config_loop.yaml').read_text());base['loop'].update(max_solver_calls=3,patience=10,gap_target=-1);base['param_caps']={'sa':{'num_reads':60,'num_sweeps':300},'cga':{'timeout':3}}
base['rag']['enabled']=False
jobs=[(b,n,e,s) for b in ['sa','cga'] for n in [12,30] for e in ['binary_slack','unary_slack','unbalanced'] for s in range(10 if b=='sa' else 3)]
atomic_write(OUT/'manifest.json',json.dumps({'config':base,'jobs':jobs,'meaning':'frozen default weights, same original constraints; no automatic encoding choice or penalty tuning; unbalanced approximate'},indent=2))
def run(job):
 b,n,e,s=job;dest=OUT/'runs'/f'{b}_{n}_{e}_{s}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 cfg=copy.deepcopy(base);cfg['logging']={'backend_log':str(dest/'backend.jsonl'),'loop_log':str(dest/'rounds.jsonl')}
 ir=knapsack_ir(n=n,seed=0);enc={'capacity':{'encoding':e,'params':{'l1':1.,'l2':1.} if e=='unbalanced' else {}}}
 hb=cfg['hardware_budget'];budget=HardwareBudget(ROOT/hb['ledger'],{k:v for k,v in hb.items() if k!='ledger'});backend=AbortOnExternalError(get_backend(b),budget,3)
 start=time.time();row={'backend':b,'n':n,'encoding':e,'repeat':s,'gap':None,'feasible':False,'error':None}
 try:
  ctx=build_context(None,backend,cfg,ir=ir,seed=s,encodings=enc);row['encoded_variables']=ctx.problem.n_variables
  with (dest/'execution.log').open('w') as log:result=run_loop(None,backend,make_policy('default'),cfg,ir=ir,seed=s,encodings=enc,budget=budget,root=ROOT,checkpoint=dest/'checkpoint.json',log=lambda m:(log.write(m+'\n'),log.flush()))
  row.update(feasible=result.best_solution is not None,gap=result.best_gap,objective=result.best_objective,calls=result.solver_calls,error=backend.error)
  atomic_write(dest/'loop.json',result.model_dump_json(indent=2))
 except Exception as err:row['error']=f'{type(err).__name__}: {err}'
 row['wall_s']=time.time()-start;atomic_write(dest/'result.json',json.dumps(row,indent=2));return row
rows=[]
with ThreadPoolExecutor(4) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):
  row=future.result();rows.append(row);atomic_write(OUT/'results.json',json.dumps(rows,indent=2));print(f"[{len(rows)}/{len(jobs)}] {row['backend']} n{row['n']} {row['encoding']} gap={row['gap']} error={row['error']}",flush=True)
