# Tasks

In order. Tick them off as you go.

## Now

- [ ] Fix the four bugs in `src/allmeta/task_gp.py` and `scaml.py` (`REVIEW_your_code.md`)
- [ ] Check BoTorch and GPyTorch versions in `flex_env` (`README.md`)
- [ ] Decide where the model code lives: `src/allmeta/` (where your copies are) or
      `gp_validation/meta_learning_gps/code/models/` (where the Phase 17 README planned it). Then register
      `ScaMLTransfer` in that package's `__init__.py`
- [ ] Turn `checks/check_reference.py` and `check_variants_and_toluene.py` into pytest (`tests/test_models.py`)

## Phase 17: data and tasks

- [ ] Decide the open items in `reference/tasks.py`:
  - outcome(s): `mean_abs_error` and `std_error` (what production optimized) vs signed `mean_error`
  - true search-space bounds and log scale for the rates, from the Ax experiment's search space
  - water's two optimization runs: concatenate or keep one
  - query set: all remaining trials, or only the BoTorch-phase trials (index ≥ 10)
  - strata (`primary`, `legacy_space`, `incomplete` in the Phase 17 README): where they come from, how to filter
  - context sizes k (suggested 0, 3, 5, 10, 20, 40)
- [ ] Write `tasks.py` (trial order from the trial number, never from `condition_id` string order)
- [ ] Rerun the toluene noise-bound check on the Phase 17 tables
- [ ] `fitting.py`: for each target and k, fit `single_task`, ScaML-GP (c) and ScaML-GP (b); log the fitted weights
- [ ] `scoring.py`: paired comparison against `single_task` per liquid and k (RMSE, NLPD, interval coverage)
- [ ] Fill the Phase 17 design checklist and config before scoring anything

## PACOH-GP

- [ ] Retype `reference/models/pacoh.py` into your model folder
- [ ] Run PACOH-SVGD (K = 10) at 6,000 iterations on the synthetic check; compare with MAP at small k
- [ ] Decide `prior_factor` (0.1 default) and `n_iter`; consider a held-out source liquid to stop training
- [ ] Decide whether to vectorize over particles (cost: ≈6.5 h for leave-one-liquid-out at K = 10, 6,000 steps)
- [ ] Mixture quantiles for K > 1 (`Prediction.quantiles`), so interval coverage is exact

## Next models

- [ ] Phase 16: hetGP paper card (Binois et al.; journal PDF optional, arXiv v2 read)
- [ ] Decide which model gets the log-normal GP
- [ ] Phase 17: HyperBO+ card (reuses `fit_task_gp` and `fit_map`)

## Later

- [ ] Ax integration for ScaML-GP (`DECISIONS.md`, "Known consequences"): batched `forward`, outcome transforms,
      one model per objective
