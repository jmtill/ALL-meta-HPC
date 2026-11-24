#!/usr/bin/env python3
"""Aggregate run directories written by run_utils.RunContext into one table.

    python collect_results.py /share/$GROUP/$USER/proj/runs/sweep1 -o summary.csv
    python collect_results.py runs/sweep1 -o summary.parquet           # needs pandas + pyarrow
    python collect_results.py runs/sweep1 --missing --expected 200     # list array tasks to rerun

Each row = one run directory (any directory containing manifest.json). Columns:
    run, status, job_id, array_job_id, task_id, attempts, host, gpu, elapsed_s, error,
    config.<key>..., result.<key>...   (nested dicts flattened with dots)

--missing prints the tasks that never finished as an sbatch --array spec, e.g. "3,7,10-12",
so you can resubmit only those. Stdlib only unless you ask for Parquet.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list) and len(d) <= 8 and all(not isinstance(x, (dict, list)) for x in d):
        out[prefix[:-1]] = json.dumps(d)
    elif isinstance(d, (dict, list)):
        out[prefix[:-1]] = json.dumps(d)[:500]
    else:
        out[prefix[:-1]] = d
    return out


def compress_ranges(ids):
    ids = sorted(set(ids))
    parts, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(parts)


def read_run(mpath: Path) -> dict:
    run_dir = mpath.parent
    try:
        m = json.loads(mpath.read_text())
    except Exception as e:
        return {"run": str(run_dir), "status": "unreadable-manifest", "error": str(e)}
    atts = m.get("attempts", [])
    last = atts[-1] if atts else {}
    sl = last.get("slurm", {})
    row = {
        "run": str(run_dir),
        "status": m.get("status"),
        "job_id": sl.get("SLURM_JOB_ID"),
        "array_job_id": sl.get("SLURM_ARRAY_JOB_ID"),
        "task_id": sl.get("SLURM_ARRAY_TASK_ID"),
        "attempts": len(atts),
        "host": last.get("host"),
        "gpu": ";".join(g.get("name", "") for g in last.get("gpus", [])) or None,
        "elapsed_s": round(sum(a.get("elapsed_s", 0) or 0 for a in atts), 1),
        "error": last.get("error"),
        "seed": m.get("seed"),
        "git_commit": (m.get("git") or {}).get("commit"),
    }
    row.update(flatten(m.get("config", {}), "config."))
    rpath = run_dir / "result.json"
    if rpath.exists():
        try:
            r = json.loads(rpath.read_text())
            row.update(flatten(r.get("result", r), "result."))
            row["finished"] = r.get("finished")
        except Exception as e:
            row["status"] = "unreadable-result"; row["error"] = str(e)
    elif row["status"] == "finished":
        row["status"] = "finished-without-result"
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory containing run directories (searched recursively)")
    ap.add_argument("-o", "--out", help="output .csv or .parquet (default: print summary only)")
    ap.add_argument("--missing", action="store_true", help="print unfinished/failed task ids as an --array spec")
    ap.add_argument("--expected", type=int, help="number of array tasks expected (0..N-1), to catch tasks that never started")
    args = ap.parse_args(argv)

    root = Path(args.root)
    rows = [read_run(p) for p in sorted(root.rglob("manifest.json"))]
    if not rows:
        print(f"no manifest.json under {root}", file=sys.stderr)
        return 1

    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"{len(rows)} runs under {root}: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items(), key=lambda kv: str(kv[0]))))

    if args.out:
        out = Path(args.out)
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        if out.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError:
                print("pandas not available; write .csv instead", file=sys.stderr)
                return 1
            pd.DataFrame(rows, columns=cols).to_parquet(out, index=False)
        else:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows, {len(cols)} columns)")

    if args.missing:
        done = {int(r["task_id"]) for r in rows if r["status"] == "finished" and r.get("task_id") not in (None, "")}
        bad = [r for r in rows if r["status"] != "finished"]
        bad_ids = {int(r["task_id"]) for r in bad if r.get("task_id") not in (None, "")}
        if args.expected:
            seen = {int(r["task_id"]) for r in rows if r.get("task_id") not in (None, "")}
            bad_ids |= set(range(args.expected)) - seen
            bad_ids -= done
        for r in bad:
            if r.get("task_id") in (None, ""):
                print(f"  not finished: {r['run']} ({r['status']}) {r.get('error') or ''}")
        if bad_ids:
            print(f"rerun with: --array={compress_ranges(bad_ids)}")
        else:
            print("no missing array tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
