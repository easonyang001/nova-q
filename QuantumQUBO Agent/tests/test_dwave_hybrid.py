"""Offline tests of the externally charged hybrid submission boundaries."""
import dimod
import pytest
from types import SimpleNamespace
from qubo_auto.annealing.backends import DWaveHybridBackend
from qubo_auto.annealing.guardrail import GuardrailError


def test_hybrid_refuses_model_minimum_before_submit(monkeypatch):
    monkeypatch.setenv('DWAVE_API_TOKEN', 'offline-test')
    class Sampler:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def min_time_limit(self, bqm): return 11
        def sample(self, *args, **kwargs): pytest.fail('must not submit or increase limit')
    monkeypatch.setattr('dwave.system.LeapHybridSampler', Sampler)
    with pytest.raises(GuardrailError, match='not submitted'):
        DWaveHybridBackend().run({(0, 0): -1}, 0, 1, {'time_limit': 10})


def test_hybrid_passes_exact_limit_and_normalizes_charge(monkeypatch):
    monkeypatch.setenv('DWAVE_API_TOKEN', 'offline-test')
    class Sampler:
        solver = SimpleNamespace(name='offline-hybrid')
        def __init__(self, **kwargs): assert kwargs['polling_timeout'] == 120
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def min_time_limit(self, bqm): return 3
        def sample(self, bqm, **kwargs):
            assert kwargs['time_limit'] == 10
            return dimod.SampleSet.from_samples({0: 1}, dimod.BINARY, energy=-1,
                    info={'problem_id': 'offline-id', 'timing': {'charge_time': 10_000_000, 'qpu_access_time': 123}})
    monkeypatch.setattr('dwave.system.LeapHybridSampler', Sampler)
    result = DWaveHybridBackend().run({(0, 0): -1}, 0, 1, {'time_limit': 10})
    assert result.solver_time_s == 10  # service charge time, not embedded QPU use
    assert result.samples == [[1]] and result.energies == [-1]
    assert result.metadata['requested_time_limit_s'] == 10
    assert not result.simulated


def test_hybrid_missing_timing_reserves_requested_limit():
    samples = dimod.SampleSet.from_samples({0: 0}, dimod.BINARY, energy=0)
    _, _, elapsed, metadata = DWaveHybridBackend().normalize((samples, 'offline', 3, 10), 1)
    assert elapsed == 10
    assert metadata['ledger_time_basis'] == 'requested_limit_reservation'


def test_hybrid_top_level_service_timing():
    samples = dimod.SampleSet.from_samples({0: 0}, dimod.BINARY, energy=0,
                info={'run_time': 9_950_000, 'qpu_access_time': 123})
    _, _, elapsed, metadata = DWaveHybridBackend().normalize((samples, 'offline', 3, 10), 1)
    assert elapsed == 9.95
    assert metadata['timing_us']['qpu_access_time'] == 123
    assert metadata['ledger_time_basis'] == 'run_time'
