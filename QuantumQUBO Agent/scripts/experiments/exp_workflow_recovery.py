"""Controlled fault experiment; synthetic known-truth fixtures, not live-data validation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qubo_auto.annealing.artifacts import ArtifactStore, atomic_write
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.workflow import run_workflow

CODE = '''DATA_SCHEMA = {"values": "list", "capacity": "integer"}
DATA_CONTRACT = {"values": {"source_kind": "mock", "unit": "points"}, "capacity": {"source_kind": "user", "value": 2}}
REQUIREMENT_MAP = {"capacity": ["capacity"]}
def make_example_data(size, seed):
    return {"values": [3, 2, 1], "capacity": 2}
def build_ir(data):
    return {"name": "selection", "variable_names": [str(i) for i in range(len(data["values"]))],
            "sense": "maximize", "linear": {i: v for i, v in enumerate(data["values"])},
            "constraints": [{"name": "capacity", "kind": "linear", "coeffs": {i: 1 for i in range(len(data["values"]))}, "sense": "le", "rhs": data["capacity"]}]}
def describe_solution(data, x):
    return {"selected": [i for i, b in enumerate(x) if b]}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'results/experiments/workflow_recovery_20261011')
    args = p.parse_args()
    rows = []
    cfg = {'loop': {'max_solver_calls': 2, 'patience': 3}, 'workflow': {'max_repairs': 1},
           'param_caps': {'sa': {'num_reads': 10, 'num_sweeps': 100}}, 'penalty_search': {'compiler': 'pyqubo'}}
    for seed in range(3):
        data = {'values': np.random.default_rng(seed).integers(1, 20, 6).tolist(), 'capacity': 2}
        metadata = {'source': 'synthetic', 'report': {'data_metadata': {'values': {'unit': 'points'}}},
                    'confirmed_spec': {'constraints': [{'name': 'capacity', 'strength': 'hard'}]}}
        cases = [
            ('model_exception', CODE.replace('return {"name":', 'raise ValueError("injected column error")\n    return {"name":'), data, metadata, True),
            ('wrong_unit_metadata', CODE, data, {**metadata, 'report': {'data_metadata': {'values': {'unit': 'kg'}}}}, True),
            ('missing_input', CODE, {'capacity': 2}, metadata, True),
            ('missing_constraint', CODE.replace('"constraints": [{"name": "capacity", "kind": "linear", "coeffs": {i: 1 for i in range(len(data["values"]))}, "sense": "le", "rhs": data["capacity"]}]', '"constraints": []'), data, metadata, True),
            ('unauthorized_setting_change', CODE, {**data, 'capacity': -1}, metadata, False),
        ]
        for name, broken_code, broken_data, broken_metadata, recoverable in cases:
            for enabled in (False, True):
                store = ArtifactStore(args.out / 'runs' / f'{name}_{seed}_{int(enabled)}')
                original = store.commit(broken_code, broken_data, 'injected fault', broken_metadata)
                def repair(issue, bundle):
                    return {'status': 'repaired', 'code': CODE, 'data': data, 'metadata': metadata, 'message': 'restore known fixture truth'}
                result = run_workflow(store, get_backend('sa'), lambda: make_policy('penalty_tpe', seed=seed), cfg,
                                      repair=repair if enabled else None, log=lambda _: None)
                delivered = result['solution'] is not None
                correctly_handled = delivered if recoverable else not delivered and store.current()['version'] == original
                rows.append({'case': name, 'seed': seed, 'recovery_enabled': enabled, 'recoverable': recoverable,
                             'status': result['delivery']['status'], 'delivered': delivered, 'correctly_handled': correctly_handled,
                             'solver_calls': result['total_solver_calls'], 'diagnoses': result['diagnoses'], 'repairs': result['repairs']})
    atomic_write(args.out / 'runs.json', json.dumps(rows, ensure_ascii=False, indent=2))
    lines = ['# Controlled workflow recovery experiment', '', 'Synthetic known-truth fault injection; this is not evidence that arbitrary real-world faults can be diagnosed automatically.', '',
             '| Recovery | Repairable cases delivered | Unauthorized setting changes rejected | Correctly handled | Solver calls |',
             '|---|---:|---:|---:|---:|']
    for enabled in (False, True):
        group = [r for r in rows if r['recovery_enabled'] == enabled]
        lines.append(f"| {'on' if enabled else 'off'} | {sum(r['delivered'] for r in group if r['recoverable'])}/12 | {sum(r['correctly_handled'] for r in group if not r['recoverable'])}/3 | {sum(r['correctly_handled'] for r in group)}/15 | {sum(r['solver_calls'] for r in group)} |")
    lines += ['', 'Repairs restore an explicitly known fixture, not LLM-generated diagnoses. Three seeds vary the six objective coefficients. Both conditions retain validation; the difference is whether a correction callback is available.', '']
    atomic_write(args.out / 'summary.md', '\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
