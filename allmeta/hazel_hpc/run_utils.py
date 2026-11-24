"""Per-run bookkeeping for Slurm jobs on Hazel: manifest, metrics, results, checkpoints.

Copy this file next to your training code (it has no dependencies beyond the stdlib; torch and
numpy are used if installed). Typical use:

    from run_utils import RunContext, slurm_cpus

    torch.set_num_threads(slurm_cpus())
    with RunContext(args.run_dir, config=cfg, seed=cfg["seed"]) as ctx:
        state = ctx.load_checkpoint() or {"step": 0, "X": [], "Y": []}
        for step in range(state["step"], n_steps):
            ...                                   # one unit of work
            ctx.log(step=step, loss=loss, hv=hv)  # appends to metrics.jsonl
            state["step"] = step + 1
            if ctx.should_checkpoint():           # every `checkpoint_every_s` seconds
                ctx.save_checkpoint(state)
            if ctx.stop_requested:                # SIGUSR1 (time-limit warning) or SIGTERM (preemption)
                ctx.save_checkpoint(state)
                ctx.interrupt()                   # exits with code 75 (resumable)
        ctx.finish({"best_y": best, "hypervolume": hv})   # writes result.json atomically

Layout written into run_dir:
    manifest.json   config, seed, git commit, host, GPU, Slurm ids, attempts[], status
    metrics.jsonl   one JSON object per ctx.log() call, with wall time
    result.json     final scalar results (only when the run finished)
    ckpt.pt         latest checkpoint (ckpt.prev.pt = previous one)

Status values: running | finished | interrupted | failed. A run directory without result.json
did not finish; collect_results.py reports those.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pickle
import platform
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

EXIT_INTERRUPTED = 75  # EX_TEMPFAIL: "try again" — resumable stop
SLURM_KEYS = [
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS", "SLURM_JOB_ACCOUNT", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_MEM_PER_NODE",
    "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_RESTART_COUNT", "SLURM_SUBMIT_DIR", "CUDA_VISIBLE_DEVICES",
]


def slurm_cpus(default: int = 1) -> int:
    """Cores allocated to this task. Never use os.cpu_count() on a shared node."""
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit():
        return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return default


def cap_threads(n: int | None = None) -> int:
    """Set thread env vars and torch threads to the allocation. Call before heavy imports if possible."""
    n = n or slurm_cpus()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    return n


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def array_seed(base: int = 0) -> int:
    """Deterministic seed from the array index (0 when not an array job)."""
    return base + int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _to_jsonable(o: Any):
    if hasattr(o, "item") and callable(o.item):        # numpy / torch scalars
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_to_jsonable, sort_keys=False))


def _git_info(cwd: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        if not commit:
            return {}
        return {"commit": commit, "dirty": bool(run("status", "--porcelain")), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}
    except Exception:
        return {}


def _gpu_info() -> list[dict]:
    try:
        import torch
        if torch.cuda.is_available():
            return [{"index": i, "name": torch.cuda.get_device_name(i),
                     "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                    for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


class RunContext:
    def __init__(self, run_dir: str | os.PathLike, config: dict | None = None, seed: int | None = None,
                 checkpoint_every_s: float = 1800, code_dir: str | os.PathLike | None = None,
                 install_signal_handlers: bool = True):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if str(self.run_dir).startswith("/usr/local/usrapps"):
            raise ValueError("run_dir under /usr/local/usrapps: compute nodes can't write there. Use /share/$GROUP/$USER/...")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.seed = seed
        self.checkpoint_every_s = checkpoint_every_s
        self.code_dir = Path(code_dir or os.getcwd())
        self.stop_requested = False
        self.stop_signal: str | None = None
        self._last_ckpt = time.monotonic()
        self._t0 = time.monotonic()
        self._finished = False
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.result_path = self.run_dir / "result.json"
        self.ckpt_path = self.run_dir / "ckpt.pt"
        if install_signal_handlers:
            for sig in (signal.SIGUSR1, signal.SIGTERM):
                signal.signal(sig, self._on_signal)
        if seed is not None:
            set_seed(seed)

    # ---- lifecycle
    def __enter__(self):
        m = self._read_manifest()
        attempt = {"start": _now(), "host": socket.gethostname(),
                   "slurm": {k: os.environ[k] for k in SLURM_KEYS if k in os.environ}, "gpus": _gpu_info()}
        if not m:
            m = {"run_dir": str(self.run_dir), "created": _now(), "config": self.config, "seed": self.seed,
                 "git": _git_info(self.code_dir), "python": sys.version.split()[0], "platform": platform.platform(),
                 "argv": sys.argv, "attempts": []}
        else:
            if self.config and m.get("config") and m["config"] != self.config:
                print(f"[run_utils] WARNING: config differs from existing manifest in {self.run_dir}", file=sys.stderr)
        m["status"] = "running"
        m["attempts"].append(attempt)
        self._manifest = m
        self._write_manifest()
        return self

    def __exit__(self, exc_type, exc, tb):
        att = self._manifest["attempts"][-1]
        att["end"] = _now()
        att["elapsed_s"] = round(time.monotonic() - self._t0, 1)
        if self._finished:
            self._manifest["status"] = "finished"
        elif exc_type is SystemExit and getattr(exc, "code", None) == EXIT_INTERRUPTED:
            self._manifest["status"] = "interrupted"
            att["signal"] = self.stop_signal
        elif exc_type is not None:
            self._manifest["status"] = "failed"
            att["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()[-2000:]
        else:
            self._manifest["status"] = "interrupted"  # left the block without finish()
        self._write_manifest()
        return False

    # ---- signals
    def _on_signal(self, signum, frame):
        self.stop_requested = True
        self.stop_signal = signal.Signals(signum).name
        print(f"[run_utils] {self.stop_signal} received at {_now()}; will checkpoint and stop", file=sys.stderr, flush=True)

    def interrupt(self):
        """Stop now with a 'resumable' exit code (75). Call after saving a checkpoint."""
        raise SystemExit(EXIT_INTERRUPTED)

    # ---- logging & results
    def log(self, **metrics):
        rec = {"t": round(time.monotonic() - self._t0, 3), **metrics}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_to_jsonable) + "\n")

    def finish(self, result: dict):
        payload = {"finished": _now(), "result": result}
        atomic_write_json(self.result_path, payload)
        self._finished = True

    # ---- checkpoints
    def should_checkpoint(self) -> bool:
        return time.monotonic() - self._last_ckpt >= self.checkpoint_every_s

    def save_checkpoint(self, state: Any) -> None:
        tmp = self.ckpt_path.with_name("ckpt.pt.tmp")
        try:
            import torch
            torch.save(state, tmp)
        except ImportError:
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
        if self.ckpt_path.exists():
            os.replace(self.ckpt_path, self.run_dir / "ckpt.prev.pt")
        os.replace(tmp, self.ckpt_path)
        self._last_ckpt = time.monotonic()
        self._manifest["last_checkpoint"] = _now()
        self._write_manifest()

    def load_checkpoint(self, map_location: str | None = None) -> Any | None:
        for p in (self.ckpt_path, self.run_dir / "ckpt.prev.pt"):
            if not p.exists():
                continue
            try:
                try:
                    import torch
                    return torch.load(p, map_location=map_location, weights_only=False)
                except ImportError:
                    with open(p, "rb") as f:
                        return pickle.load(f)
            except Exception as e:  # corrupt latest -> fall back to previous
                print(f"[run_utils] could not load {p.name}: {e}", file=sys.stderr)
        return None

    # ---- manifest io
    def _read_manifest(self) -> dict | None:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _write_manifest(self):
        atomic_write_json(self.manifest_path, self._manifest)

    @property
    def is_done(self) -> bool:
        return self.result_path.exists()
