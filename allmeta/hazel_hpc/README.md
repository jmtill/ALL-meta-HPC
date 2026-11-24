# allmeta/hazel_hpc: the factorial on Hazel

Runs the Phase 16 / 17 factorial (target liquid × outcome × model variant × noise level × seed, each over every
context size k) as Slurm job arrays on NC State's Hazel cluster. The same code runs on a laptop (`cli local`).
The folder is `hazel_hpc`, not `hazel-hpc`, because Python cannot import a name with a hyphen.

## Files

| File | Role | Knows Slurm? |
|---|---|---|
| `../evaluation/tasks.py` | per-liquid trial stats in trial order; source / context / query split; leakage check | no |
| `../evaluation/experiment.py` | `run_cell(cfg)`: one cell, meta-fit once, every k → flat dict of scores + `predictions.csv` | no |
| `../evaluation/gp_metrics.py` | RMSE, MAE, NLPD, z², 95% coverage, CRPS, Spearman, regret | no |
| `../evaluation/verdict.py` | the pre-set verdict rule A on `main/summary.csv` | no |
| `../data/` | the GP inputs: `transfers.csv`, `conditions.csv`, `descriptors.csv`, `runs.csv` (selection audit), `build_data.py` | no |
| `sweep.py` | sweep TOML → `configs.jsonl` + `job.sbatch` per block | yes |
| `worker.py` | one array element: its slice of cells, skip finished, isolate failures, exit 75 on USR1/TERM | via env |
| `cli.py` | `prepare`, `submit`, `status`, `resubmit`, `collect`, `select`, `local` | yes |
| `profiles.toml` | resource classes (partition, QOS, cores, memory, time) with the measured timings | yes |
| `job.sbatch.in` | job script template (thread caps, signal forwarding) | yes |
| `sweeps/smoke.toml`, `tune.toml`, `main.toml` | the three stages | |
| `run_utils.py`, `collect_results.py`, `hazel_lint.py` | from the hazel-hpc skill, unchanged (stdlib only) | yes |
| `requirements.txt`, `environment.yml` | CPU env (BoTorch 0.18.1, GPyTorch 1.15.2); the YAML installs the requirements file | |

## Design

- **Unit of work = one cell**, not one (cell, k): a cell meta-fits once and reuses it for all six k.
- **Blocks by cost.** Cheap cells are packed many per array element (`per_task`), so Python start-up and imports
  don't dominate; PACOH-SVGD gets one cell per element on the `compute` partition. Each block is its own array.
- **Short QOS where it fits.** Every block except PACOH-SVGD finishes an element in < 2 h, so it runs on
  `compute_partners` + `short`, which uses idle partner nodes and usually starts sooner.
- **CPU only.** Everything is float64 with n ≤ ~1,600; a GPU would not help, and the cheaper consumer cards are
  slow at float64.
- **Data travel with the code.** `allmeta/data/` holds the complete, uncapped, first optimization run of each
  liquid (`runs.csv` says why every other run was dropped): 15 liquids with descriptors, 100 trials each.
- **Hyperparameters are tuned nested.** Every liquid is a scored target, so there are no spare tuning liquids. In `tune.toml` every outer target is removed from the data entirely, and `n_inner` = 4 of
  the remaining liquids take turns as the target. `cli select` keeps the setting with the lowest mean inner NLPD
  per model / noise / outcome / target / variant, and `main.toml` blocks with `tuned = "tune"` read it.
- **Model variants are factors, not hyperparameters**: ScaML `weight_source`, HyperBO+ `prior`, pooled
  `use_descriptors` and PACOH MAP vs SVGD are compared, not searched.
- **Valid cells only** (`experiment.valid`). The noise levels fixed / smoothed / hetGP apply to the single-task GP on
  `signed`; the log outcomes have known sampling noise, so they get homoskedastic or fixed. Transfer models use
  their own homoskedastic noise. 5-fold CV (Phase 16) is single-task only.
- **Raw-scale E|e| is not a separate cell.** Every cell saves its query predictions; combine `signed` + `log_sd`
  with `folded.py`, or exponentiate `log_abs`, after `collect`.

## Size (measured at 2 threads, before any Hazel slowdown)

| Stage | Cells | Array elements | CPU-hours |
|---|---|---|---|
| smoke | 26 | 5 | < 0.5 |
| tune: ScaML descriptor prior | 135 | 68 | ~4 |
| tune: PACOH MAP | 270 | 270 | ~75 |
| main: single-task CV + k-shot | 240 | 8 | < 1 |
| main: ScaML (3 variants) | 135 | 23 | ~3.5 |
| main: HyperBO+ (2), pooled (2) | 180 | 15 | ~2 |
| main: PACOH MAP × 3 seeds | 135 | 34 | ~10 |
| main: PACOH SVGD × 3 seeds | 135 | 135 | ~90 |

PACOH SVGD is not tuned on its own (that search alone was ~390 CPU-h): it reuses the `prior_factor` and
`feature_dim` tuned for PACOH MAP (`tuned_variant` in `main.toml`), since both use the same hyperprior.
At 60 elements at a time, `main` takes a few hours of wall time; PACOH-SVGD is the long pole.

## One-time setup on Hazel (login node)

The repo carries the code and the data (`allmeta/data/`, ~4 MB), so a clone is all Hazel needs.
```bash
git clone <ALL-meta-HPC remote> ~/ALL-meta-HPC
conda env create --prefix /usr/local/usrapps/$GROUP/$USER/envs/allmeta -f ~/ALL-meta-HPC/allmeta/hazel_hpc/environment.yml
mkdir -p /share/$GROUP/$USER/allmeta
```
The sweeps assume the clone is at `~/ALL-meta-HPC` (`code_dir` in each TOML). `allmeta/` may sit at the repo root or
under `src/`; the job script puts both on `PYTHONPATH`. `environment.yml` reads `requirements.txt` from its own folder,
so run `conda env create` with the path shown (or from inside `hazel_hpc/`).

## Running (from the repo root on a login node)

```bash
export PYTHONPATH=.:src          # either repo layout
C="python -m allmeta.hazel_hpc.cli"; S=allmeta/hazel_hpc/sweeps   # or src/allmeta/...
$C local  $S/smoke.toml --n 1 --project-dir /tmp/allmeta   # laptop / salloc: one cell per block
$C submit $S/smoke.toml --test 1                            # one element per block on the short QOS
seff <jobid>                                                # set profile times / memory from this
$C submit $S/tune.toml   &&  $C status $S/tune.toml         # stage 1
$C select $S/tune.toml                                      # -> sweeps/tune/tuned.json
$C submit $S/main.toml   &&  $C status $S/main.toml         # stage 2 (a collect job runs after it)
$C resubmit $S/main.toml                                    # only elements with unfinished cells
python -m allmeta.evaluation.verdict /share/$GROUP/$USER/allmeta/sweeps/main/summary.csv -o verdict.csv   # rule A
rsync -av <unity-id>@login.hpc.ncsu.edu:/share/<group>/<unity-id>/allmeta/sweeps/main ./results/   # on your laptop, before the 30-day purge
```

Outputs on `/share/$GROUP/$USER/allmeta/`: `sweeps/<stage>/<block>/{configs.jsonl, job.sbatch, summary.csv,
predictions.csv.gz}`, `sweeps/<stage>/summary.csv`, `runs/<stage>/<block>/<cell id>/{manifest.json, result.json,
predictions.csv}`, `logs/`. `result.json` holds `k{k}.{metric}` (or `cv.{metric}`, or the tuning `score`) and per-k
timings.

## Verdict (fixed before any result; `verdict.py`)

Primary: NLPD averaged over k = 3, 5, 10, paired per liquid against single-task with homoskedastic noise. A transfer
model variant helps if the median paired difference is below 0 and the one-sided sign test gives p < 0.05, i.e. it
is better on at least 12 of the 15 liquids. Secondary (reported, not used for the verdict): the same rule with RMSE.
At k = 0 single-task has no prediction, so k = 0 only ranks the transfer models against each other.

## Rebuilding the data

Only needed if new campaigns are added. From an ALL-meta-learning checkout with autonomous-liquid-learning beside it:
`PYTHONPATH=src:. python src/allmeta/data/build_data.py` (needs scikit-learn and openpyxl for the descriptor lookup).
