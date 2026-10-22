"""Time-bounded execution of generated code in disposable processes.

This isolates hangs and crashes, not filesystem/network permissions. It is not a
security sandbox. A fresh interpreter avoids inheriting UI threads or importing
the Gradio entry point on every generated function call.
"""
from __future__ import annotations
import importlib
import os
import pickle
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


class StageTimeout(TimeoutError):
    pass


def _worker(module, function, args, kwargs):
    try:
        fn = getattr(importlib.import_module(module), function)
        return True, fn(*args, **kwargs)
    except BaseException as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"


def run_task(module: str, function: str, *args, timeout: float = 30, **kwargs):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen([sys.executable, "-m", __name__], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=env, start_new_session=True)
    try:
        try:
            output, _ = process.communicate(pickle.dumps((module, function, args, kwargs)), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise StageTimeout(f"{function} 超過 {timeout:g} 秒；已停止這次工作，可重試或回到前一步") from e
        try:
            ok, value = pickle.loads(output)
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"{function} worker exited without a result") from e
        if not ok:
            raise RuntimeError(value)
        return value
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        process.communicate()


def _generated(code, function, args=(), kwargs=None):
    import types
    import numpy as np
    ns = {"np": np, "__source__": code}
    exec(compile(code, "<model>", "exec"), ns)
    if function is None:
        return {"values": {k: ns[k] for k in ("DATA_SCHEMA", "DATA_CONTRACT", "REQUIREMENT_MAP", "MODEL_CHOICE") if k in ns},
                "functions": [k for k, v in ns.items() if isinstance(v, types.FunctionType)
                              and v.__code__.co_filename == "<model>"]}
    return ns[function](*args, **(kwargs or {}))


@dataclass
class GeneratedFunction:
    code: str
    name: str
    timeout: float = 30

    def __call__(self, *args, **kwargs):
        return run_task(__name__, "_generated", self.code, self.name, args, kwargs, timeout=self.timeout)


def generated_module(code: str, timeout: float = 30) -> dict:
    meta = run_task(__name__, "_generated", code, None, timeout=timeout)
    return {**meta["values"], "__source__": code,
            **{name: GeneratedFunction(code, name, timeout) for name in meta["functions"]}}


if __name__ == "__main__":
    from contextlib import redirect_stdout
    task = pickle.loads(sys.stdin.buffer.read())
    output = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        result = _worker(*task)
    output.write(pickle.dumps(result))
    output.flush()
