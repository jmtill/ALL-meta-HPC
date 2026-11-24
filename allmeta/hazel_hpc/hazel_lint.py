#!/usr/bin/env python3
"""Static checks for Slurm batch scripts against Hazel (NC State HPC) rules.

Usage:
    python hazel_lint.py job.sbatch [more.sbatch ...]
    python hazel_lint.py job.sbatch -- --array=0-9 --qos=short   # extra args = sbatch CLI overrides
    python hazel_lint.py --json job.sbatch

Exit code: 1 if any ERROR, else 0. Warnings and info never fail.

Rules are based on hpc.ncsu.edu Slurm docs (retrieved 2026-10-09). Limits change during the
LSF->Slurm migration; update the tables below if `sqos` on the cluster says otherwise.
Stdlib only, so it runs on a login node without activating an env.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- Hazel tables
GPU_TYPES = {  # type: (vram_gb, gpus_per_node, fp64_full_rate)
    "h200": (141, 4, True), "rtx_6000_pro": (96, 8, False), "h100": (80, 4, True),
    "l40s": (48, 4, False), "l40": (48, 4, False), "a100": (40, 4, True),
    "a30": (24, 2, True), "a10": (24, 2, False), "p100": (16, 2, True),
    "rtx_2080": (8, 4, False), "gtx_1080": (8, 4, False),
}
PARTITION_QOS = {
    "compute": {"normal", "long"},
    "compute_partners": {"short", "scavenger"},       # + p_<group>
    "gpu": {"gpu"},
    "gpu_partners": {"short_gpu", "scavenger_gpu"},   # + p_<group>_gpu
    "xfer": {"xfer"},
}
DEFAULT_QOS = {"compute": "normal", "gpu": "gpu", "xfer": "xfer"}
QOS_MAX_MIN = {  # max wall time, minutes
    "normal": 4 * 1440, "long": 10 * 1440, "gpu": 4 * 1440, "short": 120, "short_gpu": 120,
    "scavenger": 4 * 1440, "scavenger_gpu": 4 * 1440, "xfer": 1440,
}
GPU_PARTITIONS = {"gpu", "gpu_partners"}
XFER_MAX_CPUS, XFER_MAX_MEM_MB = 4, 24 * 1024
MAX_NODE_MEM_MB = 1536 * 1024
CPU_CONSTRAINTS = {"turin", "genoa", "sapphirerapids", "icelake_6326", "icelake_8358",
                   "cascadelake", "skylake", "broadwell", "haswell", "intel", "amd", "nvidia",
                   "avx", "avx2", "avx512", "sse4_1", "sse4_2", "ib"}

SHORT_OPTS = {"-p": "partition", "-t": "time", "-J": "job-name", "-o": "output", "-e": "error",
              "-n": "ntasks", "-c": "cpus-per-task", "-N": "nodes", "-q": "qos", "-a": "array",
              "-C": "constraint", "-G": "gpus", "-A": "account", "-d": "dependency"}
NETWORK_CMDS = [
    (r"\bpip3?\s+install\b", "pip install"), (r"\bconda\s+(install|create|env\s+create|update)\b", "conda install/create"),
    (r"\bmamba\s+(install|create)\b", "mamba install/create"), (r"\bwget\b", "wget"), (r"\bcurl\b", "curl"),
    (r"\bgit\s+(clone|pull|fetch|push)\b", "git network operation"), (r"\bhuggingface-cli\s+download\b", "huggingface-cli download"),
    (r"\bhf\s+download\b", "hf download"), (r"\brclone\b", "rclone"), (r"\baws\s+s3\b", "aws s3"),
    (r"\bwandb\s+sync\b", "wandb sync"),
]


@dataclass
class Finding:
    level: str  # ERROR | WARN | INFO
    rule: str
    line: int | None
    msg: str


@dataclass
class Script:
    path: str
    lines: list[str]
    opts: dict[str, tuple[str, int | None]] = field(default_factory=dict)  # name -> (value, line)
    body: list[tuple[int, str]] = field(default_factory=list)              # non-directive lines


# ---------------------------------------------------------------- parsing helpers
def parse_time_min(s: str) -> float | None:
    """Slurm time formats -> minutes. None if unparseable or UNLIMITED."""
    s = s.strip()
    if not s or s.upper() in {"UNLIMITED", "INFINITE"}:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        if not d.isdigit():
            return None
        days = int(d)
        parts = s.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        h = nums[0]; m = nums[1] if len(nums) > 1 else 0; sec = nums[2] if len(nums) > 2 else 0
        return days * 1440 + h * 60 + m + sec / 60
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] + nums[1] / 60
    if len(nums) == 3:
        return nums[0] * 60 + nums[1] + nums[2] / 60
    return None


def parse_mem_mb(s: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", s, re.I)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2).upper()
    return v * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[u]


def parse_option_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t[2:].split("=", 1)
            else:
                k = t[2:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    v = tokens[i + 1]; i += 1
                else:
                    v = ""
            out.append((k, v))
        elif t.startswith("-") and len(t) >= 2:
            k = SHORT_OPTS.get(t[:2], t[1:2])
            if len(t) > 2:
                v = t[2:].lstrip("=")
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                v = tokens[i + 1]; i += 1
            else:
                v = ""
            out.append((k, v))
        i += 1
    return out


def load_script(path: str, overrides: list[str]) -> Script:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sc = Script(path=path, lines=text.splitlines())
    in_header = True
    for n, raw in enumerate(sc.lines, 1):
        line = raw.strip()
        if line.startswith("#SBATCH"):
            if not in_header:
                # Slurm stops reading #SBATCH after the first command line
                sc.opts.setdefault("__late__", (str(n), n))
                continue
            rest = line[len("#SBATCH"):].split("#", 1)[0].strip()
            try:
                toks = shlex.split(rest)
            except ValueError:
                toks = rest.split()
            for k, v in parse_option_tokens(toks):
                sc.opts[k] = (v, n)
            continue
        if line and not line.startswith("#"):
            in_header = False
        sc.body.append((n, raw))
    for k, v in parse_option_tokens(overrides):
        sc.opts[k] = (v, None)
    return sc


def opt(sc: Script, *names: str):
    for nme in names:
        if nme in sc.opts:
            return sc.opts[nme]
    return (None, None)


def code_lines(sc: Script):
    """Body lines with comments stripped (naive: ignores # inside quotes)."""
    for n, raw in sc.body:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        yield n, re.sub(r"\s+#.*$", "", raw)


# ---------------------------------------------------------------- checks
def lint(sc: Script) -> list[Finding]:
    F: list[Finding] = []
    add = lambda lvl, rule, line, msg: F.append(Finding(lvl, rule, line, msg))
    body_code = list(code_lines(sc))
    all_code = "\n".join(c for _, c in body_code)

    # --- leftovers / placeholders
    for n, raw in enumerate(sc.lines, 1):
        if raw.strip().startswith("#BSUB"):
            add("ERROR", "lsf-directive", n, "#BSUB is LSF (retired). Translate to #SBATCH (see cluster-facts §10).")
        if re.search(r"<[a-z_][a-z0-9_ ]*>", raw) and not raw.strip().startswith("# "):
            add("ERROR", "placeholder", n, f"unfilled template placeholder: {raw.strip()[:80]}")
    for n, c in body_code:
        if re.search(r"\b(bsub|bjobs|bkill|bqueues|bhist)\b", c):
            add("ERROR", "lsf-command", n, "LSF command in script; use sbatch/squeue/scancel/sacct.")
        if re.search(r"\$\{?LSB_|\$\{?LS_SUBCWD", c):
            add("ERROR", "lsf-env", n, "LSF env var; use $SLURM_JOB_ID / $SLURM_ARRAY_TASK_ID / $SLURM_SUBMIT_DIR.")
    if "__late__" in sc.opts:
        n = sc.opts["__late__"][1]
        add("ERROR", "late-sbatch", n, "#SBATCH after the first command is ignored by Slurm; move it into the header.")

    # --- resources
    part, pl = opt(sc, "partition")
    part = part or "compute"
    qos, ql = opt(sc, "qos")
    time_s, tl = opt(sc, "time")
    gres, gl = opt(sc, "gres")
    gpus, gpl = opt(sc, "gpus", "gpus-per-node", "gpus-per-task")
    cpt = int(opt(sc, "cpus-per-task")[0] or 1) if (opt(sc, "cpus-per-task")[0] or "1").isdigit() else 1
    ntasks_s = opt(sc, "ntasks")[0] or "1"
    ntasks = int(ntasks_s) if ntasks_s.isdigit() else 1
    nodes_s = (opt(sc, "nodes")[0] or "1").split("-")[0]
    nodes = int(nodes_s) if nodes_s.isdigit() else 1
    mem, ml = opt(sc, "mem")
    mempc, mpl = opt(sc, "mem-per-cpu")

    if part not in PARTITION_QOS:
        add("ERROR", "partition", pl, f"unknown partition '{part}'. Hazel: {', '.join(PARTITION_QOS)}.")

    eff_qos = qos or DEFAULT_QOS.get(part)
    if qos and part in PARTITION_QOS:
        ok = qos in PARTITION_QOS[part] or (qos.startswith("p_") and (
            (part == "compute_partners" and not qos.endswith("_gpu")) or (part == "gpu_partners" and qos.endswith("_gpu"))))
        if not ok:
            lvl = "WARN" if part in {"gpu", "xfer"} else "ERROR"
            fix = " (cluster will reset it)" if lvl == "WARN" else ""
            add(lvl, "partition-qos", ql, f"QOS '{qos}' not allowed on '{part}'{fix}. Allowed: {sorted(PARTITION_QOS[part])}"
                + (" or your p_<group>" if part.endswith("_partners") else "") + ".")
    if part in {"compute_partners", "gpu_partners"} and not qos:
        add("WARN", "partners-qos", pl, f"'{part}' without --qos: you get short/short_gpu (2 h cap) unless you're in a partner project. Set it explicitly.")
        eff_qos = "short" if part == "compute_partners" else "short_gpu"

    # GPU
    gpu_req = None  # (type or None, count)
    if gres:
        for item in gres.split(","):
            if not item.startswith("gpu"):
                continue
            parts = item.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                add("ERROR", "gpu-untyped", gl, f"--gres={item} has no GPU type; Hazel rejects it. Use gpu:<type>:{parts[1]}.")
                gpu_req = (None, int(parts[1]))
            elif len(parts) == 1:
                add("ERROR", "gpu-untyped", gl, "--gres=gpu without type/count. Use gpu:<type>:N.")
                gpu_req = (None, 1)
            else:
                gtype = parts[1].lower()
                cnt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                gpu_req = (gtype, cnt)
                if gtype not in GPU_TYPES:
                    add("ERROR", "gpu-type", gl, f"unknown GPU type '{gtype}'. Valid: {', '.join(GPU_TYPES)}.")
                elif cnt > GPU_TYPES[gtype][1]:
                    add("ERROR", "gpu-per-node", gl, f"{cnt}×{gtype} per node requested; {gtype} nodes have {GPU_TYPES[gtype][1]}.")
    if gpus:
        typed = ":" in gpus
        if not typed and eff_qos != "scavenger_gpu":
            add("ERROR", "gpu-untyped", gpl, "--gpus without a type is rejected (except under scavenger_gpu). Use --gres=gpu:<type>:N.")
        gpu_req = gpu_req or ((gpus.split(":")[0].lower() if typed else None), int(re.sub(r"\D", "", gpus.split(":")[-1]) or 1))

    if gpu_req and part not in GPU_PARTITIONS:
        add("ERROR", "gpu-partition", gl or gpl, f"GPU requested on partition '{part}'. Use --partition=gpu (or gpu_partners).")
    if part in GPU_PARTITIONS and not gpu_req:
        add("WARN", "gpu-unused", pl, f"partition '{part}' but no GPU requested: you're holding a GPU node's CPUs. Use compute unless you need a GPU.")
    if gpu_req and gpu_req[0] in GPU_TYPES and not GPU_TYPES[gpu_req[0]][2]:
        if re.search(r"botorch|gpytorch|float64|double", all_code, re.I):
            add("WARN", "gpu-fp64", gl, f"{gpu_req[0]} has low FP64 throughput; BoTorch/GPyTorch default to float64. Prefer a30/a100/h100/h200 or CPU.")
    if gpu_req and cpt < 2:
        add("INFO", "gpu-cpus", gl, "GPU job with 1 CPU: data loading usually needs 4–8 --cpus-per-task per GPU.")

    # time
    if not time_s:
        add("WARN", "no-time", None, "no --time. Set an accurate limit (helps backfill; avoids surprise defaults).")
    else:
        tmin = parse_time_min(time_s)
        if tmin is None:
            add("ERROR", "time-format", tl, f"can't parse --time={time_s}. Use HH:MM:SS or D-HH:MM:SS.")
        elif eff_qos in QOS_MAX_MIN and tmin > QOS_MAX_MIN[eff_qos]:
            add("ERROR", "time-limit", tl, f"--time={time_s} exceeds {eff_qos} max ({QOS_MAX_MIN[eff_qos] / 60:g} h).")
        elif tmin and tmin > 2 * 1440 and not re.search(r"ckpt|checkpoint|resume", all_code, re.I):
            add("WARN", "no-checkpoint", tl, "run > 2 days with no sign of checkpoint/resume in the script.")

    # memory
    if not mem and not mempc:
        add("WARN", "no-mem", None, "no --mem / --mem-per-cpu. Request measured peak × 1.3 (seff).")
    if mem:
        mmb = parse_mem_mb(mem)
        if mmb is None and mem != "0":
            add("ERROR", "mem-format", ml, f"can't parse --mem={mem}.")
        elif mmb and mmb > MAX_NODE_MEM_MB:
            add("ERROR", "mem-too-big", ml, f"--mem={mem} exceeds the largest node (1536 GB).")
        elif mmb and mmb >= 0.75 * 192 * 1024 and "exclusive" not in sc.opts:
            add("INFO", "mem-large", ml, "large memory request: AUP asks for --exclusive if you need most of a node's memory.")

    # xfer
    if part == "xfer":
        if cpt * ntasks > XFER_MAX_CPUS:
            add("ERROR", "xfer-cpus", None, f"xfer jobs are capped at {XFER_MAX_CPUS} cores (1 is usually right).")
        mmb = parse_mem_mb(mem) if mem else None
        if mmb and mmb > XFER_MAX_MEM_MB:
            add("ERROR", "xfer-mem", ml, "xfer jobs are capped at 24G memory.")
        if re.search(r"\bpython\b.*\b(train|fit|model)", all_code):
            add("WARN", "xfer-compute", None, "xfer partition is for moving data, not computing on it.")

    # topology
    sw, swl = opt(sc, "switches")
    if sw and sw.split("@")[0].isdigit() and int(sw.split("@")[0]) > 1 and "ib" not in (opt(sc, "constraint")[0] or ""):
        add("ERROR", "switches-ib", swl, "--switches>1 without --constraint=ib can spread the job over unconnected Ethernet switches.")
    con, cl = opt(sc, "constraint")
    if con:
        for tok in re.split(r"[&|,\[\]()*0-9]+", con):
            if tok and tok.lower() not in CPU_CONSTRAINTS:
                add("WARN", "constraint", cl, f"unknown constraint '{tok}'.")

    # array
    arr, al = opt(sc, "array")
    if arr:
        m = re.findall(r"(\d+)-(\d+)", arr)
        span = sum(int(b) - int(a) + 1 for a, b in m)
        if span > 100 and "%" not in arr:
            add("WARN", "array-throttle", al, f"array of {span} tasks with no %K throttle; add e.g. %40.")
        outp = opt(sc, "output")[0] or ""
        if outp and "%a" not in outp and "%j" not in outp:
            add("WARN", "array-output", opt(sc, "output")[1], "array job output without %a/%j: tasks will overwrite one log.")

    # requeue
    if "no-requeue" in sc.opts and eff_qos in {"scavenger", "scavenger_gpu"}:
        add("ERROR", "no-requeue", sc.opts["no-requeue"][1], "--no-requeue with a preemptable QOS: preemption will cancel the job.")
    if eff_qos in {"scavenger", "scavenger_gpu"} and not re.search(r"ckpt|checkpoint|resume", all_code, re.I):
        add("WARN", "scavenger-ckpt", ql, "preemptable QOS but no checkpoint/resume visible; the script reruns from line 1 on requeue.")

    # --- environment/body checks
    uses_python = re.search(r"\bpython3?\b|\btorchrun\b", all_code) is not None
    activate_lines = [n for n, c in body_code if re.search(r"\bconda\s+activate\b|\bsource\s+activate\b", c)]
    if activate_lines:
        first = activate_lines[0]
        has_init = any(n < first and re.search(r"source\s+(~|\$HOME|\$\{HOME\})/\.bashrc|conda\s+shell\.bash\s+hook", c)
                       for n, c in body_code)
        if not has_init:
            add("ERROR", "conda-init", first, "conda activate without `source ~/.bashrc` first (OIT: required in batch scripts).")
    for n, c in body_code:
        if re.search(r"module\s+load\s+anaconda3|\bsource\s+activate\b", c):
            add("WARN", "conda-legacy", n, "old conda pattern (module load anaconda3 / source activate); use `source ~/.bashrc` then `conda activate <prefix>`.")
        if re.search(r"\bconda\s+activate\s+(?!/)[\w.-]+\s*$", c):
            add("INFO", "conda-name", n, "env activated by name; OIT layout uses a prefix path under /usr/local/usrapps/$GROUP/$USER/.")
    if uses_python and (cpt > 1 or ntasks > 1) and not re.search(r"OMP_NUM_THREADS", all_code):
        add("WARN", "threads", None, "multi-core Python job without OMP_NUM_THREADS (+MKL/OPENBLAS) set to $SLURM_CPUS_PER_TASK; libraries will spawn one thread per node core.")
    if uses_python and ntasks > 1 and not re.search(r"\bsrun\b|\bmpirun\b|\btorchrun\b", all_code):
        add("WARN", "ntasks-python", opt(sc, "ntasks")[1], f"--ntasks={ntasks} but no srun/mpirun: a single Python process only uses its own cores. Use --ntasks=1 --cpus-per-task=N.")
    if re.search(r"n_jobs\s*=\s*-1", all_code):
        add("WARN", "njobs-all", None, "n_jobs=-1 uses every core on the node; pass $SLURM_CPUS_PER_TASK.")
    for n, c in body_code:
        if re.search(r"\bexport\s+CUDA_VISIBLE_DEVICES=", c):
            add("ERROR", "cuda-visible", n, "don't set CUDA_VISIBLE_DEVICES; Slurm sets it to your allocated GPUs.")

    # network on compute nodes
    if part != "xfer":
        for n, c in body_code:
            for pat, name in NETWORK_CMDS:
                if re.search(pat, c):
                    add("ERROR", "network", n, f"{name} in a compute job: compute nodes have no internet. Do it on a login node or in an xfer job.")
        if uses_python and re.search(r"from_pretrained|load_dataset|wandb", all_code) and not re.search(r"OFFLINE|WANDB_MODE", all_code):
            add("WARN", "offline", None, "HF/W&B usage without offline env vars (HF_HUB_OFFLINE=1, WANDB_MODE=offline).")

    # where outputs go
    out_targets = []
    for k in ("output", "error"):
        v, ln = opt(sc, k)
        if v:
            out_targets.append((ln, v))
    for n, c in body_code:
        for m in re.finditer(r"(?:>>?|\bmkdir\s+(?:-p\s+)?|\bcp\s+\S+\s+|\bmv\s+\S+\s+|--(?:run-dir|out(?:put)?-?dir|output|save-dir)[= ])\s*(\S+)", c):
            out_targets.append((n, m.group(1)))
    for ln, tgt in out_targets:
        t = tgt.strip("\"'")
        if "/usr/local/usrapps" in t:
            add("ERROR", "write-usrapps", ln, f"writing to {t}: compute nodes can't write to /usr/local/usrapps. Use /share/$GROUP/$USER/...")
        elif re.match(r"(/home/|~/|\$HOME|\$\{HOME\})", t) and not re.search(r"\.(log|txt)$", t):
            add("WARN", "write-home", ln, f"output under home ({t}): 15 GB / 10K-file quota. Put job outputs on /share/$GROUP/$USER.")
    outp = opt(sc, "output")
    if outp[0] and "/" in outp[0] and not outp[0].startswith("/"):
        d = outp[0].rsplit("/", 1)[0]
        if not re.search(rf"mkdir\s+(-p\s+)?{re.escape(d)}\b", "\n".join(sc.lines)):
            add("INFO", "log-dir", outp[1], f"--output in '{d}/': the directory must exist before sbatch, or the job fails silently. `mkdir -p {d}`.")

    if not uses_python and not re.search(r"\bsrun\b|\./|\brsync\b|\bnvidia-smi\b|\bwget\b|\bcurl\b|\brclone\b|\baws\s+s3\b|\bRscript\b|\bjulia\b", all_code):
        add("INFO", "no-command", None, "no obvious workload command found.")
    return F


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = []
    if "--" in argv:
        i = argv.index("--"); overrides = argv[i + 1:]; argv = argv[:i]
    args = ap.parse_args(argv)

    any_error, report = False, {}
    for p in args.scripts:
        sc = load_script(p, overrides)
        fs = lint(sc)
        any_error |= any(f.level == "ERROR" for f in fs)
        report[p] = fs
    if args.json:
        print(json.dumps({p: [f.__dict__ for f in fs] for p, fs in report.items()}, indent=2))
    else:
        order = {"ERROR": 0, "WARN": 1, "INFO": 2}
        for p, fs in report.items():
            counts = {k: sum(f.level == k for f in fs) for k in order}
            print(f"{p}: {counts['ERROR']} error(s), {counts['WARN']} warning(s), {counts['INFO']} info")
            for f in sorted(fs, key=lambda f: (order[f.level], f.line or 0)):
                loc = f"line {f.line}" if f.line else "script"
                print(f"  {f.level:5} [{f.rule}] {loc}: {f.msg}")
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
