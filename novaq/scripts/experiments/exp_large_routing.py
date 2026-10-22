"""40-stop real-coordinate routing: frozen verified TSP, tuning on SA and real CGA.
Distances are haversine kilometres. Fixed-source acquisition and hand-written IR are explicit.
Reference is feasible 2-opt, not a certified optimum. No graph/model generation is attributed to LLM.
"""
import copy,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np, requests,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from qubo_auto.annealing.artifacts import atomic_write
from qubo_auto.annealing.encodings import ProblemIR,LinearConstraint
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.prep import haversine_matrix
from qubo_auto.annealing.provenance import source_record
from exp_paper_backends import AbortOnExternalError
OUT=ROOT/'results/experiments/large_routing_20261011';OUT.mkdir(parents=True,exist_ok=True)
rawfile=OUT/'youbike_snapshot.json'
u='https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json'
if not rawfile.exists():
 response=requests.get(u,timeout=30);response.raise_for_status();rawfile.write_bytes(response.content)
 atomic_write(OUT/'source.json',json.dumps(source_record({'title':'Taipei YouBike snapshot','source':'Taipei transport','page':'https://data.nat.gov.tw/dataset/147580','url':u},rawfile),indent=2))
rows=sorted([r for r in json.loads(rawfile.read_text()) if str(r.get('act'))=='1'],key=lambda r:r['sno'])[:40]
n=len(rows)
if n!=40:raise ValueError('Need exactly 40 active stations')
d=haversine_matrix([float(r['latitude']) for r in rows],[float(r['longitude']) for r in rows])/1000
rng=np.random.default_rng(0);best_route=None;baseline=float('inf')
for restart in range(20):
 route=list(rng.permutation(n));improve=True
 while improve:
  improve=False
  for i in range(1,n-1):
   for j in range(i+1,n):
    a,b,c,e=route[i-1],route[i],route[j],route[(j+1)%n]
    if d[a,c]+d[b,e]<d[a,b]+d[c,e]-1e-10:route[i:j+1]=reversed(route[i:j+1]);improve=True
 cost=sum(d[route[t],route[(t+1)%n]] for t in range(n))
 if cost<baseline:baseline=cost;best_route=route
q={}
for t in range(n):
 for i in range(n):
  for j in range(n):
   if i!=j:
    a,b=i*n+t,j*n+(t+1)%n;key=(min(a,b),max(a,b));q[key]=q.get(key,0.)+float(d[i,j])
cons=[LinearConstraint(name=f'station_{i}',coeffs={i*n+t:1 for t in range(n)},sense='eq',rhs=1) for i in range(n)]
cons += [LinearConstraint(name=f'position_{t}',coeffs={i*n+t:1 for i in range(n)},sense='eq',rhs=1) for t in range(n)]
ir=ProblemIR(name='real_coordinate_tsp40',instance_id='youbike40_frozen',n_variables=n*n,sense='minimize',quadratic=q,constraints=cons)
# Independent valid tours: objective and both one-hot families must agree with direct route distance.
for _ in range(12):
 route=rng.permutation(n);x=[0]*(n*n)
 for t,i in enumerate(route):x[i*n+t]=1
 assert all(c.satisfied(x) for c in cons)
 assert np.isclose(ir.objective(x),sum(d[route[t],route[(t+1)%n]] for t in range(n)))
base=yaml.safe_load((ROOT/'config_loop.yaml').read_text());base['loop'].update(max_solver_calls=3,patience=10,gap_target=-1)
base['param_caps']={'sa':{'num_reads':30,'num_sweeps':300},'cga':{'timeout':3}}
base['penalty_search']['compiler']='native' # avoid repeated large symbolic compilations
base['rag']['enabled']=False;base['agents'].update(mode='single',encoding_agent=False,max_tokens_per_run=60000)
atomic_write(OUT/'manifest.json',json.dumps({'n_stations':n,'n_variables':n*n,'source':'real YouBike coordinates, fixed snapshot','model':'hand-written frozen TSP IR, independently checked on 12 tours','baseline_type':'20-restart 2-opt, feasible heuristic not optimum','baseline_km':baseline,'config':base,'station_ids':[r['sno'] for r in rows]},indent=2))
atomic_write(OUT/'baseline_route.json',json.dumps([rows[i]['sna'] for i in best_route],ensure_ascii=False))
def run(job):
 b,p,s=job;dest=OUT/'runs'/f'{b}_{p}_{s}';dest.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(base);cfg['logging']={'loop_log':str(dest/'rounds.jsonl'),'backend_log':str(dest/'backend.jsonl')};hb=cfg['hardware_budget'];budget=HardwareBudget(ROOT/hb['ledger'],{k:v for k,v in hb.items() if k!='ledger'});backend=AbortOnExternalError(get_backend(b),budget,3);start=time.time();record={'backend':b,'policy':p,'repetition':s,'feasible':False,'objective_km':None,'gap_to_heuristic':None,'error':None}
 try:
  with (dest/'execution.log').open('w') as log:result=run_loop(None,backend,make_policy(p,seed=s,cfg=cfg,root=dest),cfg,ir=ir,seed=s,budget=budget,root=ROOT,log=lambda m:(log.write(m+'\n'),log.flush()))
  atomic_write(dest/'loop.json',result.model_dump_json(indent=2));x=result.best_solution
  if x is not None:
   bits=x[:n*n];feasible=all(c.satisfied(bits) for c in cons)
   if feasible:
    route=np.array(bits).reshape(n,n).argmax(axis=0).tolist();objective=sum(d[route[t],route[(t+1)%n]] for t in range(n));assert np.isclose(objective,ir.objective(bits));record.update(feasible=True,objective_km=objective,gap_to_heuristic=(objective-baseline)/baseline,route_station_ids=[rows[i]['sno'] for i in route])
  record.update(calls=result.solver_calls,tokens=result.llm_tokens,error=backend.error)
 except Exception as e:record['error']=f'{type(e).__name__}: {e}'
 record['wall_s']=time.time()-start;atomic_write(dest/'result.json',json.dumps(record,indent=2));return record
jobs=[(b,p,s) for b in ['sa','cga'] for p in ['default','penalty_tpe','agent'] for s in range(3)];records=[]
with ThreadPoolExecutor(3) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):
  rec=future.result();records.append(rec);atomic_write(OUT/'results.json',json.dumps(records,indent=2));print(f"[{len(records)}/18] {rec['backend']} {rec['policy']} r{rec['repetition']} feasible={rec['feasible']} heuristic_gap={rec['gap_to_heuristic']} error={rec['error']}",flush=True)
