"""Retrieve completed answers by ID, without submitting any optimization jobs.

Retain original run logs. The first experiment process reserved 10 seconds per call because
the SDK reports hybrid timing at the top level of SampleSet.info, unlike QPU timing.
This separate audit records the actual returned charge/run/QPU timing for publication.
"""
import json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from dwave.cloud import Client

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
OUT = ROOT / 'results/experiments/dwave_hybrid_10s_20261011'
target = OUT / 'timing_audit.jsonl'
existing = {r['problem_id']: r for r in map(json.loads, target.read_text().splitlines())} if target.exists() else {}
jobs = {}
for path in sorted((OUT / 'runs').glob('*/backend.jsonl')):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        meta = row['result']['metadata']
        pid = meta.get('problem_id')
        if pid:
            jobs[pid] = {'condition': path.parent.name, 'round': row['round'],
                         'requested_limit_s': meta['requested_time_limit_s']}
with Client.from_config(token=os.environ['DWAVE_API_TOKEN'], request_timeout=30, polling_timeout=120) as client:
    def retrieve(pid):
        if pid in existing:
            return existing[pid]
        answer = client.retrieve_answer(pid).result()
        info = dict(answer['sampleset'].info)
        return {'problem_id': pid, **jobs[pid], 'retrieval_only': True,
                'timing_us': {k: info[k] for k in ['charge_time', 'run_time', 'qpu_access_time'] if k in info}}
    with ThreadPoolExecutor(4) as pool:
        for row in pool.map(retrieve, jobs):
            existing[row['problem_id']] = row
            target.write_text(''.join(json.dumps(r) + '\n' for r in existing.values()))
print(json.dumps({'completed_answers_retrieved': len(existing), 'new_submissions': 0,
                  'service_charge_s': sum(r['timing_us'].get('charge_time', 0) for r in existing.values()) / 1e6}))
