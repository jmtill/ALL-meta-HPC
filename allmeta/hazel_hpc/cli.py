"""Command line for the factorial sweeps. Run from the repo root with PYTHONPATH=src.

  python -m allmeta.hazel_hpc.cli prepare  sweeps/main.toml            expand cells, write configs + job.sbatch, lint
  python -m allmeta.hazel_hpc.cli submit   sweeps/main.toml [--test 3] prepare, sbatch every block, then collect
  python -m allmeta.hazel_hpc.cli status   sweeps/main.toml            finished / failed / missing per block
  python -m allmeta.hazel_hpc.cli resubmit sweeps/main.toml            only the array elements with unfinished cells
  python -m allmeta.hazel_hpc.cli collect  sweeps/main.toml            summary.csv + predictions.csv.gz per block
  python -m allmeta.hazel_hpc.cli select   sweeps/tune.toml            tuned.json: best params per model/noise/outcome/target
  python -m allmeta.hazel_hpc.cli local    sweeps/smoke.toml --n 2     run the first n cells of each block here

--project-dir overrides the sweep's project_dir (e.g. a local folder on a laptop). submit, status and resubmit
need sbatch / a login node; without sbatch on PATH, submit prints what it would run and stops."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import sweep as sw

HERE = Path(__file__).resolve().parent


def _blocks(args):
    s = sw.load(args.sweep)
    if args.project_dir:
        s["sweep"]["project_dir"] = args.project_dir
    root = sw.expand_path(s["sweep"]["project_dir"])
    names = [b["name"] for b in s["block"] if not args.block or b["name"] in args.block]
    return s, root, names


def _cells(root, name, block):
    path = root / "sweeps" / name / block / "configs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _state(run_dir: Path) -> str:
    if (run_dir / "result.json").exists():
        return "finished"
    m = run_dir / "manifest.json"
    return json.loads(m.read_text()).get("status", "unknown") if m.exists() else "missing"


def _per_task(s, block):
    return next(b for b in s["block"] if b["name"] == block).get("per_task", 1)


def _compress(ids):
    ids, out, i = sorted(set(ids)), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def cmd_prepare(args):
    s, root, names = _blocks(args)
    tmp = Path(args.sweep)
    if args.project_dir:                                        # write a copy with the override applied
        import tomllib
        tmp = root / "_override.toml"
        root.mkdir(parents=True, exist_ok=True)
        text = Path(args.sweep).read_text()
        old = tomllib.loads(text)["sweep"]["project_dir"]
        tmp.write_text(text.replace(f'"{old}"', f'"{args.project_dir}"', 1))
    dirs = sw.prepare(tmp, only_blocks=args.block)
    bad = 0
    for d in dirs:
        (root / "logs" / s["sweep"]["name"] / d.name).mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(HERE / "hazel_lint.py"), str(d / "job.sbatch")], capture_output=True, text=True)
        bad += r.returncode != 0
        if r.returncode or args.verbose:
            print((r.stdout + r.stderr).strip())
    if bad:
        sys.exit(f"{bad} job script(s) failed hazel_lint; fix profiles.toml or the sweep before submitting")
    return s, root, dirs


def _sbatch(argv, dry):
    if dry:
        print("would run:", " ".join(argv))
        return "DRYRUN"
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]


def cmd_submit(args, array=None):
    s, root, dirs = cmd_prepare(args)
    dry = shutil.which("sbatch") is None
    ids, log = [], root / "sweeps" / s["sweep"]["name"] / "submissions.csv"
    for d in dirs:
        extra = []
        if args.test:
            extra = [f"--array=0-{args.test - 1}", "--partition=compute_partners", "--qos=short", "--time=01:55:00"]
        elif array and array.get(d.name):
            extra = [f"--array={array[d.name]}"]
        elif array is not None:
            continue
        job = _sbatch(["sbatch", "--parsable", *extra, str(d / "job.sbatch")], dry)
        ids.append(job)
        print(f"{d.name}: job {job} {' '.join(extra)}")
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([s["sweep"]["name"], d.name, job, " ".join(extra)])
    if ids and not args.test:
        cmd = (f"cd {s['sweep']['code_dir']} && PYTHONPATH=.:src python -m allmeta.hazel_hpc.cli collect {Path(args.sweep).resolve()}"
               + (f" --project-dir {args.project_dir}" if args.project_dir else ""))
        wrap = (f"source ~/.bashrc; conda activate {s['sweep']['env']}; {cmd}")
        job = _sbatch(["sbatch", "--parsable", f"--dependency=afterany:{':'.join(ids)}", "--partition=compute",
                       "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:30:00",
                       f"--job-name={s['sweep']['name']}-collect",
                       f"--output={root}/logs/{s['sweep']['name']}/collect.%j.out", f"--wrap={wrap}"], dry)
        print(f"collect: job {job} (after all blocks)")


def cmd_status(args):
    s, root, names = _blocks(args)
    rerun = {}
    for b in names:
        cells = _cells(root, s["sweep"]["name"], b)
        runs = root / "runs" / s["sweep"]["name"] / b
        states = [_state(runs / c["id"]) for c in cells]
        counts = {k: states.count(k) for k in sorted(set(states))}
        per = _per_task(s, b)
        todo = sorted({i // per for i, st in enumerate(states) if st != "finished"})
        rerun[b] = _compress(todo)
        print(f"{b:14s} {len(cells):5d} cells  {counts}" + (f"  rerun --array={rerun[b]}" if todo else ""))
        for c, st in zip(cells, states):
            if st == "failed" and args.verbose:
                err = json.loads((runs / c["id"] / "manifest.json").read_text())["attempts"][-1].get("error", "")
                print(f"   {c['id']}: {err[-200:]}")
    return rerun


def cmd_resubmit(args):
    rerun = cmd_status(args)
    args.test = None
    cmd_submit(args, array={b: spec for b, spec in rerun.items() if spec})


def cmd_collect(args):
    import pandas as pd
    s, root, names = _blocks(args)
    out_dir = root / "sweeps" / s["sweep"]["name"]
    summaries = []
    for b in names:
        rows, preds = [], []
        runs = root / "runs" / s["sweep"]["name"] / b
        for c in _cells(root, s["sweep"]["name"], b):
            d = runs / c["id"]
            row = {"block": b, "status": _state(d), **{k: v for k, v in c.items() if k != "params"},
                   "params": json.dumps(c.get("params", {}), sort_keys=True)}
            if row["status"] == "finished":
                row.update(json.loads((d / "result.json").read_text())["result"])
                if (d / "predictions.csv").exists():
                    preds.append(pd.read_csv(d / "predictions.csv").assign(id=c["id"]))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / b / "summary.csv", index=False)
        if preds:
            pd.concat(preds).to_csv(out_dir / b / "predictions.csv.gz", index=False)
        summaries.append(frame)
        print(f"{b:14s} {(frame.status == 'finished').sum()}/{len(frame)} finished -> {out_dir / b / 'summary.csv'}")
    pd.concat(summaries).to_csv(out_dir / "summary.csv", index=False)


def cmd_select(args):
    """Nested tuning: for each (model, noise, outcome, target) the params with the lowest mean inner NLPD."""
    import pandas as pd
    cmd_collect(args)
    s, root, _ = _blocks(args)
    out_dir = root / "sweeps" / s["sweep"]["name"]
    frame = pd.read_csv(out_dir / "summary.csv")
    done = frame[(frame.status == "finished") & frame["score"].notna()].copy()
    done["key"] = done.apply(lambda r: sw.tuned_key(r.to_dict()), axis=1)
    best = done.loc[done.groupby("key")["score"].idxmin()]
    tuned = {r.key: json.loads(r.params) for r in best.itertuples()}
    (out_dir / "tuned.json").write_text(json.dumps(tuned, indent=1, sort_keys=True))
    best[["key", "params", "score"]].to_csv(out_dir / "tuned.csv", index=False)
    missing = frame[frame.status != "finished"]
    print(f"tuned.json: {len(tuned)} keys from {len(done)} finished cells ({len(missing)} unfinished)")


def cmd_local(args):
    from .worker import main as worker
    s, root, dirs = cmd_prepare(args)
    for d in dirs:
        print(f"== {d.name}: first {args.n} cells")
        worker([str(d), "--local", str(args.n), "--runs", str(root / "runs" / s["sweep"]["name"] / d.name)])
    cmd_collect(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "submit", "status", "resubmit", "collect", "select", "local"])
    ap.add_argument("sweep")
    ap.add_argument("--block", action="append", help="only these blocks (repeatable)")
    ap.add_argument("--project-dir")
    ap.add_argument("--test", type=int, help="submit: only the first N array elements, on the short QOS")
    ap.add_argument("--n", type=int, default=1, help="local: cells per block")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status, "resubmit": cmd_resubmit,
     "collect": cmd_collect, "select": cmd_select, "local": cmd_local}[args.command](args)


if __name__ == "__main__":
    main()
