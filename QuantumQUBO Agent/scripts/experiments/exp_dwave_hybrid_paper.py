"""Sequential real Leap Hybrid comparison. Each call is explicitly limited to 10 seconds.

No direct-QPU claim, no stand-in, no larger time limit on model-size refusal.
Three frozen models, three policies, three submissions per condition at most.
Stop the batch after an external error rather than burn repeated uncertain submissions.
"""
import json
from pathlib import Path
import yaml
from exp_paper_backends import ROOT, run_one, atomic_write

OUT = ROOT / 'results/experiments/dwave_hybrid_10s_20261011'
OUT.mkdir(parents=True, exist_ok=True)
base = yaml.safe_load((ROOT / 'config_loop.yaml').read_text())
base['loop']['max_solver_calls'] = 3
jobs = [('dwave_hybrid', case, policy, repeat)
        for repeat in range(3) for case in ['wine', 'knapsack60', 'select80']
        for policy in ['default', 'penalty_tpe', 'agent']]
atomic_write(OUT / 'manifest.json', json.dumps({
    'design': 'fixed models shared with SA/CGA study; real hybrid BQM service, sequential',
    'jobs': jobs, 'time_limit_s': 10, 'max_calls_per_condition': 3,
    'planned_max_calls': 81, 'planned_max_service_time_s': 810,
    'time_limit_is_not_wall_timeout': True,
    'hardware_repetition': 'repeated submissions; hardware RNG seed not exposed',
    'full_config': base,
}, indent=2))
rows = []
for job in jobs:
    row = run_one(job, base, OUT)
    rows.append(row)
    atomic_write(OUT / 'results.json', json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"[{len(rows)}/{len(jobs)}] {row['case']} {row['policy']} r{row['repetition']}: "
          f"{row['status']} gap={row['gap']} calls={row['solver_calls']} error={row['error']}", flush=True)
    if row['error'] or row.get('stop_reason') == 'hardware_budget' or (row['solver_calls'] == 0 and row['status'] != 'feasible'):
        print('Batch stopped after failed/refused condition. Inspect saved records before resubmission.', flush=True)
        break
