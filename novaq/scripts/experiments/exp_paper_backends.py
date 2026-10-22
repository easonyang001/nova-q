"""Frozen-model backend/policy comparison; real calls only, bounded by existing ledger.
Defaults: wine + two harder generated problems, three repetitions, three-call ceiling.
Within-backend policy budgets match; compute seconds across technologies are not equivalent.
"""
import argparse, copy, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
import yaml
from qubo_auto.annealing.artifacts import ArtifactStore, atomic_write
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.guardrail import HardwareBudget, GuardrailError
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.modeler import load_module, to_ir
from qubo_auto.annealing.examples import knapsack_random, select_k_random

class AbortOnExternalError:
    """Do not re-submit an externally uncertain failed call during this experiment."""
    def __init__(self, backend, budget, time_cap):
        self.backend,self.budget,self.time_cap,self.error=backend,budget,time_cap,None
    def __getattr__(self,name): return getattr(self.backend,name)
    def run(self,*args,**kwargs):
        if self.backend.name=='dwave_hybrid':
            used=self.budget.usage(self.backend.name)
            limit=self.budget.limits[self.backend.name]['max_solver_time_s']
            if used['solver_time_s']+self.time_cap>limit+1e-9:
                raise GuardrailError(['insufficient D-Wave Hybrid budget for the full bounded call'])
        try: return self.backend.run(*args,**kwargs)
        except GuardrailError:
            raise  # local pre-submission refusal consumes no external allowance
        except Exception as e:
            self.error=f'{type(e).__name__}: {e}'
            if not self.backend.is_simulator:
                self.budget.record(self.backend.name,self.time_cap,note='uncertain failure; conservative time reservation, not reported billing')
            raise GuardrailError(['external call failed; stop without resubmission: '+self.error]) from e

def run_one(job,base,out):
    backend_name,case,policy_name,repetition=job
    name=f'{backend_name}_{case}_{policy_name}_{repetition}'
    destination=out/'runs'/name
    if (destination/'result.json').exists():
        return json.loads((destination/'result.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(base)
    cfg['logging']={'loop_log':str(destination/'rounds.jsonl'),'backend_log':str(destination/'backend.jsonl')}
    cfg['rag'].update(enabled=False,write_experience=False)
    cfg['agents'].update(mode='single',encoding_agent=False,max_tokens_per_run=80000)
    params={'sa':{'num_reads':60,'num_sweeps':300},'cga':{'timeout':3},'fujitsu_da':{'time_limit_sec':3},'dwave_hybrid':{'time_limit':10.0}}
    cfg['param_caps']=params
    cfg['loop'].update(max_solver_calls=base['loop']['max_solver_calls'],patience=10,gap_target=-1)
    if case=='wine':
        store=ArtifactStore(ROOT/'results/acceptance/wine_live_20261011_fresh2/artifacts');bundle=store.current()
        ns=load_module(bundle['code']);ir=to_ir(ns['build_ir'](bundle['data']),'frozen_wine')
        report=json.loads((store.root.parent/'acceptance.json').read_text());ir.reference_objective=report['oracle_objective']
        problem=None
        snapshot={'version':bundle['version'],'ir':ir.model_dump(mode='json')}
    else:
        generator=knapsack_random if case=='knapsack60' else select_k_random
        problem=generator(n=60 if case=='knapsack60' else 80,seed=0,**({} if case=='knapsack60' else {'k':12}))
        ir=None;snapshot={'instance_id':problem.instance_id,'generator':case,'instance_seed':0,'reference':problem.reference_objective}
    atomic_write(destination/'config.json',json.dumps(cfg,indent=2))
    atomic_write(destination/'instance.json',json.dumps(snapshot,indent=2))
    hb=cfg['hardware_budget'];budget=HardwareBudget(ROOT/hb['ledger'],{k:v for k,v in hb.items() if k!='ledger'})
    backend=get_backend(backend_name)
    if backend_name=='fujitsu_da': backend.max_queue_wait_s=120
    wrapper=AbortOnExternalError(backend,budget,10 if backend_name=='dwave_hybrid' else 3)
    started=time.time()
    row={'backend':backend_name,'case':case,'policy':policy_name,'repetition':repetition,'real_external':not backend.is_simulator,
         'status':'failed','best_objective':None,'gap':None,'solver_calls':0,'tokens':0,'error':None}
    try:
        policy=make_policy(policy_name,seed=repetition,cfg=cfg,root=destination,rag=None)
        with (destination/'execution.log').open('w') as log:
            result=run_loop(problem,wrapper,policy,cfg,budget=budget,seed=repetition,ir=ir,root=ROOT,
                            log=lambda m: (log.write(m+'\n'),log.flush()),checkpoint=destination/'checkpoint.json')
        atomic_write(destination/'loop.json',result.model_dump_json(indent=2))
        row.update(status='feasible' if result.best_solution is not None else 'no_feasible',best_objective=result.best_objective,
                   gap=result.best_gap,solver_calls=result.solver_calls,tokens=result.llm_tokens,
                   stop_reason=result.stop_reason,error=wrapper.error,
                   fallback_rounds=sum(r.source.startswith('fallback') for r in result.rounds))
        logs=[json.loads(l) for l in (destination/'backend.jsonl').read_text().splitlines()] if (destination/'backend.jsonl').exists() else []
        row['backend_metadata']=[r.get('result',{}).get('metadata',{}) for r in logs]
    except Exception as e: row['error']=f'{type(e).__name__}: {e}'
    row['wall_s']=time.time()-started
    atomic_write(destination/'result.json',json.dumps(row,ensure_ascii=False,indent=2))
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/experiments/paper_backends_20261011')
    p.add_argument('--backends',nargs='+',default=['sa','cga','fujitsu_da'])
    p.add_argument('--cases',nargs='+',default=['wine','knapsack60','select80'])
    p.add_argument('--policies',nargs='+',default=['default','penalty_tpe','agent'])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--calls',type=int,default=3)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load((ROOT/'config_loop.yaml').read_text());base['loop']['max_solver_calls']=a.calls
    jobs=[(b,c,p,r) for r in range(a.repetitions) for c in a.cases for p in a.policies for b in a.backends]
    planned={b:sum(j[0]==b for j in jobs)*a.calls for b in a.backends}
    manifest={'design':'fixed models; budget ceiling matched within backend; no data/model regeneration',
              'jobs':jobs,'planned_max_calls':planned,'time_caps_s':{'cga':3,'fujitsu_da':3},
              'repetitions_meaning':'SA RNG seeds; external backends repeated submissions (no exposed solver seed)',
              'full_config':base}
    atomic_write(a.out/'manifest.json',json.dumps(manifest,indent=2))
    print('Planned upper-bound calls:',planned,flush=True)
    rows=[]
    with ThreadPoolExecutor(a.workers) as pool:
        futures=[pool.submit(run_one,j,base,a.out) for j in jobs]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            atomic_write(a.out/'results.json',json.dumps(rows,ensure_ascii=False,indent=2))
            print(f"[{len(rows)}/{len(jobs)}] {row['backend']} {row['case']} {row['policy']} r{row['repetition']}: {row['status']} gap={row['gap']} calls={row['solver_calls']} error={row['error']}",flush=True)
    fields=['backend','case','policy','repetition','status','best_objective','gap','solver_calls','tokens','wall_s','error']
    with (a.out/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

if __name__=='__main__': main()
