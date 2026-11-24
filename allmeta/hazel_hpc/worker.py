"""Run one array element: cells [index * per_task, (index + 1) * per_task) of a block's configs.jsonl.

    python -m allmeta.hazel_hpc.worker <block dir> --index 3 --per-task 4 --runs <runs dir>
    python -m allmeta.hazel_hpc.worker <block dir> --local 2 --runs /tmp/runs      # first 2 cells, no Slurm

Each cell gets runs/<cell id>/ with manifest.json, result.json and predictions.csv (run_utils.RunContext).
A cell with result.json is skipped, so resubmitting a whole array is safe. A cell that raises is recorded as
failed and the next cell still runs. On USR1/TERM (time-limit warning, preemption) the current cell finishes
and the worker exits 75 before starting another. Exit: 0 all done, 1 any failed, 75 stopped early."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

from .run_utils import EXIT_INTERRUPTED, RunContext, cap_threads

STOP = {"requested": False}


def _on_signal(signum, frame):
    STOP["requested"] = True
    print(f"[worker] {signal.Signals(signum).name}: finishing the current cell, then stopping", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("block_dir")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--local", type=int, help="run the first N cells in this process (ignores --index)")
    ap.add_argument("--runs", required=True)
    args = ap.parse_args(argv)
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    n_threads = cap_threads()
    import torch
    torch.set_num_threads(n_threads)
    torch.set_default_dtype(torch.float64)
    from allmeta.experiment import run_cell

    cells = [json.loads(line) for line in Path(args.block_dir, "configs.jsonl").read_text().splitlines()]
    mine = cells[:args.local] if args.local else cells[args.index * args.per_task:(args.index + 1) * args.per_task]
    failed = 0
    for cfg in mine:
        if STOP["requested"]:
            return EXIT_INTERRUPTED
        run_dir = Path(args.runs) / cfg["id"]
        if (run_dir / "result.json").exists():
            print(f"[worker] done already: {cfg['id']}")
            continue
        t0 = time.time()
        try:
            with RunContext(run_dir, config=cfg, seed=cfg.get("seed", 0), install_signal_handlers=False) as ctx:
                result = run_cell(cfg, out_dir=run_dir)
                ctx.finish({**result, "seconds": round(time.time() - t0, 1), "threads": n_threads})
            print(f"[worker] {cfg['id']} finished in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:                                      # recorded in manifest.json by RunContext
            failed += 1
            print(f"[worker] {cfg['id']} FAILED: {e!r}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
