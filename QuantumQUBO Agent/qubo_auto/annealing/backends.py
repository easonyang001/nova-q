"""Backend adapters: one interface for every annealer and its local stand-in.

    describe()             tunable parameters + ranges, precision limits, units
    submit(bqm, params)    run on the backend, return its raw result
    normalize(raw, n)      raw result -> samples / occurrences / solver time / metadata
    simulate(...)          same call, executed on local simulated annealing
    run(...)               guardrail check -> submit -> normalize -> SolverRunResult

Energies are always recomputed locally from (qubo, offset), so they are comparable across
backends regardless of how a backend reports energy (e.g. Fujitsu drops the offset).
Times are reported in seconds; the backend's own timing fields are kept in metadata.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any

import dimod
import numpy as np
from pydantic import BaseModel, Field

from .guardrail import GuardrailError, ParamSpec, check_params
from .problem import QUBO, qubo_energy


class SolverRunResult(BaseModel):
    """One backend execution. Not to be confused with schemas.RunResult (a pipeline run)."""
    backend: str
    simulated: bool = False
    params: dict[str, Any]
    samples: list[list[int]]              # unique samples, variables 0..n-1
    num_occurrences: list[int]
    energies: list[float]                 # local x^T Q x + offset, same order as samples
    wall_time_s: float
    solver_time_s: float | None = None    # backend-reported compute time, seconds
    metadata: dict[str, Any] = Field(default_factory=dict)


def _to_bqm(qubo: QUBO, offset: float, n: int) -> dimod.BinaryQuadraticModel:
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
    for i in range(n):                    # keep variables whose coefficients are all zero
        bqm.add_variable(i)
    return bqm


def _aggregate(sampleset: dimod.SampleSet, n: int) -> tuple[list[list[int]], list[int]]:
    counts: dict[tuple[int, ...], int] = {}
    for sample, occ in sampleset.data(fields=["sample", "num_occurrences"], sorted_by=None):
        x = tuple(int(sample.get(i, 0)) for i in range(n))
        counts[x] = counts.get(x, 0) + int(occ)
    return [list(x) for x in counts], list(counts.values())


class Backend:
    name = "base"
    is_simulator = False

    def param_specs(self) -> dict[str, ParamSpec]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "simulator": self.is_simulator,
                "params": {k: v.model_dump() for k, v in self.param_specs().items()}}

    def submit(self, bqm: dimod.BinaryQuadraticModel, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def normalize(self, raw: Any, n: int) -> tuple[list[list[int]], list[int], float | None, dict]:
        raise NotImplementedError

    def run(self, qubo: QUBO, offset: float, n: int, params: dict[str, Any] | None = None,
            simulate: bool = False) -> SolverRunResult:
        clean = check_params(params or {}, self.param_specs())
        if simulate and not self.is_simulator:
            return self.simulate(qubo, offset, n, clean)
        t0 = time.time()
        raw = self.submit(_to_bqm(qubo, offset, n), clean)
        wall = time.time() - t0
        samples, occ, solver_time, meta = self.normalize(raw, n)
        return SolverRunResult(
            backend=self.name, simulated=self.is_simulator, params=clean, samples=samples,
            num_occurrences=occ, energies=[qubo_energy(qubo, offset, x) for x in samples],
            wall_time_s=wall, solver_time_s=solver_time, metadata=meta,
        )

    def simulate(self, qubo: QUBO, offset: float, n: int, params: dict[str, Any]) -> SolverRunResult:
        """Local stand-in: simulated annealing with the same number of reads."""
        sa_params = {k: params[k] for k in ("num_reads",) if k in params}
        result = SimulatedAnnealingBackend().run(qubo, offset, n, sa_params)
        result.backend, result.simulated = self.name, True
        result.metadata["stand_in"] = "simulated_annealing"
        return result


class SimulatedAnnealingBackend(Backend):
    """Development backend and the default local stand-in (dwave-samplers SA)."""
    name = "sa"
    is_simulator = True

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            "num_reads": ParamSpec(type="int", min=1, max=10_000, default=100,
                                   description="independent annealing runs"),
            "num_sweeps": ParamSpec(type="int", min=1, max=1_000_000, default=1000,
                                    description="Metropolis sweeps per run"),
            "beta_min": ParamSpec(type="float", min=1e-6, max=1e6,
                                  description="initial inverse temperature (with beta_max)"),
            "beta_max": ParamSpec(type="float", min=1e-6, max=1e6,
                                  description="final inverse temperature (with beta_min)"),
            "seed": ParamSpec(type="int", min=0, max=2**31 - 1, description="RNG seed"),
        }

    def submit(self, bqm, params):
        from dwave.samplers import SimulatedAnnealingSampler
        kwargs = {k: v for k, v in params.items() if k not in ("beta_min", "beta_max")}
        if "beta_min" in params or "beta_max" in params:
            if not ("beta_min" in params and "beta_max" in params):
                raise ValueError("beta_min and beta_max must be given together")
            kwargs["beta_range"] = (params["beta_min"], params["beta_max"])
        t0 = time.time()
        sampleset = SimulatedAnnealingSampler().sample(bqm, **kwargs)
        return sampleset, time.time() - t0

    def normalize(self, raw, n):
        sampleset, elapsed = raw
        samples, occ = _aggregate(sampleset, n)
        return samples, occ, elapsed, {"beta_range": list(sampleset.info.get("beta_range", []))}


class DWaveQPUBackend(Backend):
    """D-Wave QPU through minor embedding (Ocean SDK). Token: DWAVE_API_TOKEN."""
    name = "dwave_qpu"

    def __init__(self, solver: str | None = None):
        self.solver = solver or os.environ.get("DWAVE_SOLVER")   # None = let Leap choose

    def param_specs(self) -> dict[str, ParamSpec]:
        # Ranges from Advantage_system4 properties (2026-10-04).
        return {
            "num_reads": ParamSpec(type="int", min=1, max=10_000, default=100,
                                   description="QPU reads (each costs QPU access time)"),
            "annealing_time": ParamSpec(type="float", min=0.5, max=2000.0, default=20.0,
                                        description="anneal duration, microseconds"),
            "chain_strength": ParamSpec(type="float", min=1e-3, max=1e4,
                                        description="unset = uniform torque compensation"),
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["precision"] = {"h_range": [-4.0, 4.0], "j_range": [-1.0, 1.0],
                             "note": "Ocean auto-scales the embedded Ising problem into these ranges"}
        info["billing"] = "QPU access time (microseconds) per call"
        return info

    def submit(self, bqm, params):
        from dwave.system import DWaveSampler, EmbeddingComposite
        token = os.environ.get("DWAVE_API_TOKEN")
        if not token:
            raise RuntimeError("DWAVE_API_TOKEN is not set")
        qpu = DWaveSampler(token=token, solver=self.solver) if self.solver else DWaveSampler(token=token)
        sampler = EmbeddingComposite(qpu)
        sampleset = sampler.sample(bqm, return_embedding=True, label="qubo_auto", **params)
        sampleset.resolve()
        return sampleset, qpu.solver.name

    def normalize(self, raw, n):
        sampleset, solver_name = raw
        cbf = np.asarray(sampleset.record.chain_break_fraction, dtype=float)
        occ_all = np.asarray(sampleset.record.num_occurrences, dtype=float)
        ctx = sampleset.info.get("embedding_context", {})
        chains = ctx.get("embedding", {}) or {}
        chain_lengths = [len(c) for c in chains.values()]
        timing = dict(sampleset.info.get("timing", {}))
        samples, occ = _aggregate(sampleset, n)
        meta = {
            "solver": solver_name,
            "problem_id": sampleset.info.get("problem_id"),
            "timing_us": timing,
            "chain_strength": ctx.get("chain_strength"),
            "chain_break_method": ctx.get("chain_break_method"),
            "chain_break_fraction_mean": float((cbf * occ_all).sum() / occ_all.sum()) if occ_all.size else None,
            "samples_with_chain_breaks": float(occ_all[cbf > 0].sum() / occ_all.sum()) if occ_all.size else None,
            "num_qubits_used": int(sum(chain_lengths)),
            "max_chain_length": max(chain_lengths, default=0),
        }
        qpu_us = timing.get("qpu_access_time")
        return samples, occ, (qpu_us / 1e6 if qpu_us is not None else None), meta


class DWaveHybridBackend(Backend):
    """Leap Hybrid BQM service; time_limit is a strict caller-selected ceiling in seconds.

    This is a quantum-classical service, not direct QPU sampling. Never raise the requested
    time limit to accommodate a larger model: refuse before submission instead.
    """
    name = "dwave_hybrid"

    def __init__(self, solver: str | None = None):
        self.solver = solver or os.environ.get("DWAVE_HYBRID_SOLVER", "hybrid_binary_quadratic_model_version2p")

    def param_specs(self):
        return {"time_limit": ParamSpec(type="float", min=1, max=600, default=10,
                                         description="maximum hybrid service runtime, seconds; model minimum checked before submit")}

    def submit(self, bqm, params):
        from dwave.system import LeapHybridSampler
        token = os.environ.get("DWAVE_API_TOKEN")
        if not token:
            raise RuntimeError("DWAVE_API_TOKEN is not set")
        with LeapHybridSampler(token=token, solver=self.solver, request_timeout=30,
                               polling_timeout=120) as sampler:
            requested = params["time_limit"]
            minimum = sampler.min_time_limit(bqm)
            if requested < minimum:
                raise GuardrailError([f"D-Wave Hybrid model minimum {minimum:.3f}s exceeds requested ceiling {requested:.3f}s; not submitted"])
            sampleset = sampler.sample(bqm, time_limit=requested, label="qubo_auto_bounded_hybrid")
            sampleset.resolve()
            return sampleset, sampler.solver.name, minimum, requested

    def normalize(self, raw, n):
        sampleset, solver_name, minimum, requested = raw
        samples, occ = _aggregate(sampleset, n)
        timing = dict(sampleset.info.get("timing", {}))
        for field in ("charge_time", "run_time", "qpu_access_time"):
            if field in sampleset.info:
                timing.setdefault(field, sampleset.info[field])
        # Leap timing fields are microseconds. Prefer service charge time for the quota ledger.
        field = "charge_time" if timing.get("charge_time") is not None else "run_time"
        measured = timing.get(field)
        elapsed = float(measured) / 1e6 if measured is not None else requested
        return samples, occ, elapsed, {
            "solver": solver_name, "problem_id": sampleset.info.get("problem_id"),
            "service_kind": "quantum_classical_hybrid_bqm", "timing_us": timing,
            "minimum_time_limit_s": minimum, "requested_time_limit_s": requested,
            "ledger_time_basis": field if measured is not None else "requested_limit_reservation",
        }


class FujitsuDABackend(Backend):
    """Fujitsu Digital Annealer v3c (REST, async job). Key: FUJITSU_API_KEY.

    POST /async/qubo/solve -> poll /async/jobs/result/{id} -> always DELETE the job
    (the account has a job-count limit, so finished jobs must not pile up).
    Parameters are validated only when the job runs (status "Error" + message), and a job that is
    already running can be neither cancelled nor deleted: it runs to its time limit. Hence the
    guardrail must catch bad values before submission.
    The offset is not sent; DA energies therefore exclude it (kept in metadata).
    """
    name = "fujitsu_da"

    def __init__(self, api_url: str | None = None, poll_interval_s: float = 2.0, max_queue_wait_s: float = 1800.0,
                 orphan_log: Path | None = None):
        self.api_url = (api_url or os.environ.get("FUJITSU_API_URL", "https://api.aispf.global.fujitsu.com")).rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.max_queue_wait_s = max_queue_wait_s       # jobs can sit in "Waiting" for minutes when the queue is busy
        self.orphan_log = orphan_log or Path(__file__).resolve().parents[2] / "results" / "fujitsu_orphan_jobs.jsonl"

    def param_specs(self) -> dict[str, ParamSpec]:
        # fujitsuDA3 fields. Lower/upper limits probed against v3c on 2026-10-04 (the API validates
        # asynchronously: bad values only show up as a job with status "Error" and a message).
        # Upper limits marked "probe" were accepted but are not the true maximum.
        return {
            "time_limit_sec": ParamSpec(type="int", min=1, max=1800, default=10,
                                        description="search time limit, integer seconds (1801 also accepted)"),
            "num_output_solution": ParamSpec(type="int", min=1, max=1024,
                                             description="number of solutions returned (server default 5)"),
            "num_run": ParamSpec(type="int", min=1, max=128, description="parallel annealing runs (>=129 accepted)"),
            "num_group": ParamSpec(type="int", min=1, max=16, description="parallel groups"),
            "gs_level": ParamSpec(type="int", min=0, max=100, description="global search level"),
            "gs_cutoff": ParamSpec(type="int", min=0, max=1_000_000,
                                   description="global search cutoff (upper limit not probed)"),
            "target_energy": ParamSpec(type="float", min=-1e15, max=1e15,
                                       description="stop early when reached (excludes offset)"),
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["billing"] = ("DA solve time against a 300 min/month quota; finished jobs count against a "
                           "job limit until deleted")
        info["notes"] = "offset not sent; returned energies exclude it"
        return info

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("FUJITSU_API_KEY")
        if not key:
            raise RuntimeError("FUJITSU_API_KEY is not set")
        return {"Content-Type": "application/json", "X-Api-Key": key}

    def submit(self, bqm, params):
        import requests
        base, hdr = f"{self.api_url}/da/v3c", self._headers()
        qubo, offset = bqm.to_qubo()
        terms = [{"coefficient": float(c), "polynomials": [int(i)] if i == j else [int(i), int(j)]}
                 for (i, j), c in qubo.items() if c != 0]
        body = {"fujitsuDA3": params, "binary_polynomial": {"terms": terms}}
        r = requests.post(f"{base}/async/qubo/solve", headers=hdr, json=body, timeout=60)
        if not r.ok:
            raise RuntimeError(f"Fujitsu submit failed: HTTP {r.status_code} {r.text[:500]}")
        job_id = r.json()["job_id"]
        limit = params.get("time_limit_sec", 10)
        submitted, started = time.time(), None
        status = None
        try:
            while True:
                time.sleep(self.poll_interval_s)
                res = requests.get(f"{base}/async/jobs/result/{job_id}", headers=hdr, timeout=60).json()
                status = res.get("status")
                if status == "Done":
                    res["_queue_wait_s"] = (started or time.time()) - submitted
                    return res, job_id, offset
                if status == "Running" and started is None:
                    started = time.time()
                if status not in ("Waiting", "Running"):
                    raise RuntimeError(f"Fujitsu job {job_id} status {status!r}: {res.get('message', str(res)[:500])}")
                if started is None and time.time() - submitted > self.max_queue_wait_s:
                    raise TimeoutError(f"Fujitsu job {job_id} still waiting in queue after {self.max_queue_wait_s:.0f}s")
                if started is not None and time.time() - started > limit + 300:
                    raise TimeoutError(f"Fujitsu job {job_id} running > time_limit_sec + 300s")
        finally:
            # Waiting/Running jobs cannot be deleted; remember them so they can be cleaned up later.
            deleted = False
            try:
                deleted = requests.delete(f"{base}/async/jobs/result/{job_id}", headers=hdr,
                                          timeout=30).json().get("status") == "Deleted"
            except Exception:
                pass
            if not deleted and status != "Done":
                self.orphan_log.parent.mkdir(parents=True, exist_ok=True)
                with self.orphan_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"job_id": job_id, "time": time.time(), "last_status": status}) + "\n")

    def cleanup_orphans(self) -> dict[str, int]:
        """Delete jobs this adapter left behind (only ids recorded in the orphan log, never other jobs)."""
        import requests
        if not self.orphan_log.exists():
            return {"deleted": 0, "pending": 0}
        base, hdr = f"{self.api_url}/da/v3c", self._headers()
        pending, deleted = [], 0
        for line in self.orphan_log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            try:
                st = requests.delete(f"{base}/async/jobs/result/{rec['job_id']}", headers=hdr, timeout=30).json().get("status")
            except Exception:
                st = None
            if st in ("Waiting", "Running") or st is None:   # not deletable yet, or request failed: retry later
                pending.append(line)
            else:                                             # "Deleted", or the job no longer exists
                deleted += 1
        self.orphan_log.write_text("".join(l + "\n" for l in pending), encoding="utf-8")
        return {"deleted": deleted, "pending": len(pending)}

    def normalize(self, raw, n):
        res, job_id, offset = raw
        sol = res["qubo_solution"]
        counts: dict[tuple[int, ...], int] = {}
        da_energy: dict[tuple[int, ...], float] = {}
        for s in sol.get("solutions", []):
            cfg = {int(k): int(v) for k, v in s["configuration"].items()}
            x = tuple(cfg.get(i, 0) for i in range(n))
            counts[x] = counts.get(x, 0) + int(s.get("frequency", 1))
            da_energy[x] = float(s["energy"])
        timing = sol.get("timing", {})
        solve_ms = timing.get("solve_time")
        meta = {
            "job_id": job_id,
            "timing_raw": timing,                         # DA reports milliseconds
            "da_energies_excluding_offset": [da_energy[x] for x in counts],
            "offset_not_sent": offset,
            "result_status": sol.get("result_status"),
        }
        return ([list(x) for x in counts], list(counts.values()),
                float(solve_ms) / 1000 if solve_ms is not None else None, meta)


def _cga_index(key) -> int | None:
    """CGA renames variables: plain ints, 'c[j]', 'y[j]' or 'j' all mean variable j."""
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, str):
        k = key
        for prefix in ("c[", "y["):
            if k.startswith(prefix) and k.endswith("]"):
                k = k[len(prefix):-1]
        try:
            return int(k)
        except ValueError:
            return None
    return None


def _upper_triangular(qubo: dict) -> dict[tuple[int, int], float]:
    """Keys as (i, j) with i <= j, duplicates merged, zeros dropped. The CGA server answers any (i, j) with
    i > j (which dimod's to_qubo() produces) with "Something Wrong ... upload your qubo again" (2026-10-06)."""
    upper: dict[tuple[int, int], float] = {}
    for (i, j), c in qubo.items():
        key = (min(int(i), int(j)), max(int(i), int(j)))
        upper[key] = upper.get(key, 0.0) + float(c)
    return {k: c for k, c in upper.items() if c != 0}


class CGABackend(Backend):
    """CGA GPU annealer behind a Socket.IO server (protocol from the AED / catalyst-hpc-qml projects).

    emit "upload qubo" (pickled {"qubo", "offset", "timeout"}) -> "job token" -> "job finished"
    -> emit "request solution" -> "return solution" (pickled payload holding a dimod SampleSet).
    Errors arrive as "job error" / "return message". The client waits with a hard deadline instead of
    sio.wait(), which can block forever. No "variables" field is sent: the server's QUBO check crashes on
    integer variable names. Address: CGA_URI.
    """
    name = "cga"

    def __init__(self, uri: str | None = None, grace_s: float = 120.0):
        # measured 2026-10-06: a 10 s job came back after 66 s (queue + server overhead), so wait generously
        self.uri = uri or os.environ.get("CGA_URI", "")        # the server address lives only in .env
        self.grace_s = grace_s

    def param_specs(self) -> dict[str, ParamSpec]:
        return {"timeout": ParamSpec(type="int", min=1, max=600, default=30,
                                     description="server-side search time limit, integer seconds")}

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["notes"] = "only the time limit is tunable; the server returns a dimod SampleSet"
        return info

    def submit(self, bqm, params):
        # user-controlled pause (2026-10-14): while results/CGA_PAUSED exists, no job is sent to the Compal service
        pause = Path(__file__).resolve().parents[2] / "results" / "CGA_PAUSED"
        if pause.exists():
            raise RuntimeError(f"CGA is paused by the user ({pause} exists); use the local 'sa' backend or remove the file")
        # the CGA service takes ONE job at a time: parallel runs that submitted together timed out on the server
        # ("no solution generated in timeout", 2026-10-13). A cross-process file lock queues them here instead.
        import fcntl
        lock_path = Path(os.environ.get("CGA_LOCK_FILE", str(Path(__file__).resolve().parents[2] / "results" / ".cga.lock")))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock:
            waited = time.time()
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.last_queue_wait_s = round(time.time() - waited, 1)
            try:
                return self._submit_locked(bqm, params)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _submit_locked(self, bqm, params):
        import pickle
        import socketio
        qubo, offset = bqm.to_qubo()
        request = {"qubo": _upper_triangular(qubo),
                   "offset": float(offset), "timeout": int(params.get("timeout", 30))}
        upload = pickle.dumps(request, protocol=2)
        # the first HTTP poll can take ~4 s, over the client's default 5 s read timeout
        sio = socketio.Client(logger=False, engineio_logger=False, request_timeout=30)
        state: dict[str, Any] = {"token": None, "status": "pending", "payload": None}

        @sio.event
        def connect():
            sio.emit("upload qubo", upload)

        @sio.on("job token")
        def on_token(token):
            state["token"] = token

        @sio.on("job finished")
        def on_finished(data):
            sio.emit("request solution", data.get("job_token", state["token"]))

        @sio.on("job error")
        def on_job_error(data):
            state["status"] = f"job error: {data.get('message', data) if isinstance(data, dict) else data}"

        @sio.on("return message")
        def on_message(data):
            state["status"] = f"server message: {pickle.loads(data)}"

        @sio.on("return solution")
        def on_solution(data):
            state["payload"], state["status"] = pickle.loads(data), "ok"

        if not self.uri:
            raise RuntimeError("CGA_URI is not set: put the server address in .env")
        sio.connect(self.uri, wait_timeout=30)
        deadline = time.time() + request["timeout"] + self.grace_s
        try:
            while state["status"] == "pending" and time.time() < deadline:
                time.sleep(0.5)
        finally:
            try:
                sio.disconnect()
            except Exception:
                pass
        if state["status"] == "pending":
            raise TimeoutError(f"CGA returned nothing within timeout + {self.grace_s:.0f}s")
        if state["status"] != "ok":
            raise RuntimeError(f"CGA {state['status']}"[:500])
        return state["payload"], state["token"]

    def normalize(self, raw, n):
        payload, token = raw
        ss = next((payload[k] for k in ("sampleset", "solution", "samples", "result") if k in payload), None)
        if ss is None:
            raise RuntimeError(f"CGA payload has no sample set (keys {sorted(payload)})")
        counts: dict[tuple[int, ...], int] = {}
        for sample, occ in ss.data(fields=["sample", "num_occurrences"], sorted_by=None):
            x = [0] * n
            for k, v in sample.items():
                j = _cga_index(k)
                if j is not None and 0 <= j < n and v == 1:
                    x[j] = 1
            counts[tuple(x)] = counts.get(tuple(x), 0) + int(occ)
        rt = payload.get("run_time")
        if isinstance(rt, (int, float)):           # the server's unit varies: us, ms or s
            rt = rt / 1e6 if rt > 1e6 else rt / 1e3 if rt > 1e3 else float(rt)
        else:
            rt = None
        return [list(x) for x in counts], list(counts.values()), rt, {"job_token": token, "run_time_raw": payload.get("run_time")}


class SQABackend(Backend):
    """Simulated quantum annealing (path-integral Monte Carlo, transverse-field Ising), OpenJij SQASampler."""
    name = "sqa"
    is_simulator = True

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            "num_reads": ParamSpec(type="int", min=1, max=10_000, default=100, description="independent runs"),
            "num_sweeps": ParamSpec(type="int", min=1, max=1_000_000, default=1000, description="Monte Carlo sweeps per run"),
            "trotter": ParamSpec(type="int", min=2, max=256, default=8, description="Trotter slices"),
            "beta": ParamSpec(type="float", min=1e-6, max=1e6, description="inverse temperature (unset = OpenJij default)"),
            "gamma": ParamSpec(type="float", min=1e-6, max=1e6, description="initial transverse field (unset = default)"),
            "seed": ParamSpec(type="int", min=0, max=2**31 - 1, description="RNG seed (makes all reads identical)"),
        }

    def submit(self, bqm, params):
        # OpenJij's beta/gamma are absolute, so the model is rescaled to max |coefficient| = 1 first (the argmin is
        # unchanged; energies are recomputed locally). A fixed `seed` makes every read identical in OpenJij 0.12:
        # leave it unset for independent reads.
        import openjij as oj
        scale = max([abs(v) for v in bqm.linear.values()] + [abs(v) for v in bqm.quadratic.values()] + [1e-12])
        scaled = bqm.copy()
        scaled.scale(1.0 / scale)
        t0 = time.time()
        sampleset = oj.SQASampler().sample(scaled, **params)
        return sampleset, time.time() - t0

    def normalize(self, raw, n):
        sampleset, elapsed = raw
        samples, occ = _aggregate(sampleset, n)
        return samples, occ, elapsed, {}


class QAOABackend(Backend):
    """QAOA on NVIDIA CUDA-Q (GPU state-vector simulation), run in its own environment as a subprocess
    (qaoa_cudaq.py) so this process never imports cudaq. Interpreter: CUDAQ_PYTHON. Memory grows as 2^n, so the
    guard refuses more than QAOA_MAX_QUBITS (default 30) variables; reduce the instance first."""
    name = "qaoa_cudaq"
    is_simulator = True

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            "layers": ParamSpec(type="int", min=1, max=12, default=2, description="QAOA depth p"),
            "shots": ParamSpec(type="int", min=1, max=1_000_000, default=10000, description="samples of the final state"),
            "maxiter": ParamSpec(type="int", min=1, max=5000, default=200, description="COBYLA iterations per restart"),
            "restarts": ParamSpec(type="int", min=1, max=20, default=3, description="optimiser restarts"),
            "seed": ParamSpec(type="int", min=0, max=2**31 - 1, default=0, description="RNG seed"),
            "cvar_alpha": ParamSpec(type="float", min=0.01, max=1.0, default=1.0,
                                    description="CVaR fraction of best shots optimised (1 = expectation)"),
        }

    def submit(self, bqm, params):
        import subprocess
        import tempfile
        n = len(bqm.variables)
        limit = int(os.environ.get("QAOA_MAX_QUBITS", "30"))
        if n > limit:
            raise ValueError(f"QAOA state-vector simulation needs 2^{n} amplitudes; limit is {limit} qubits")
        qubo, offset = bqm.to_qubo()
        spec = {"n": n, "qubo": [[int(i), int(j), float(c)] for (i, j), c in qubo.items()], "offset": float(offset),
                "params": {**params, "target": os.environ.get("QAOA_TARGET", "nvidia")}}   # nvidia, nvidia-mgpu, qpp-cpu
        py = os.environ.get("CUDAQ_PYTHON", "/home/var/miniconda3/envs/cudaq-env/bin/python")
        script = Path(__file__).with_name("qaoa_cudaq.py")
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
            src.write_text(json.dumps(spec))
            proc = subprocess.run([py, "-I", str(script), str(src), str(dst)], capture_output=True, text=True,
                                  timeout=int(os.environ.get("QAOA_TIMEOUT_S", "3600")))
            if proc.returncode != 0:
                raise RuntimeError(f"CUDA-Q QAOA failed: {proc.stderr[-800:]}")
            return json.loads(dst.read_text())

    def normalize(self, raw, n):
        meta = {k: raw[k] for k in ("theta", "layers", "shots", "target", "timing", "expectation_scaled", "n_terms")}
        return raw["samples"], raw["counts"], raw["timing"]["optimise_s"] + raw["timing"]["sample_s"], meta


BACKENDS: dict[str, type[Backend]] = {
    "qaoa_cudaq": QAOABackend,
    "sqa": SQABackend,
    "sa": SimulatedAnnealingBackend,
    "dwave_qpu": DWaveQPUBackend,
    "dwave_hybrid": DWaveHybridBackend,
    "fujitsu_da": FujitsuDABackend,
    "cga": CGABackend,
}


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
