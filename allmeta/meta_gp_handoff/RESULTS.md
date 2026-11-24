# Results of checks run so far

Synthetic checks show mechanisms, not real-data performance. Scripts are in `checks/`.

## 1. Toy check of correction speed (`checks/toy_scaml_numpy.py`)

Numpy, 2 inputs, 8 sine-like sources, source hyperparameters fixed, one seed. Target context = Sobol points.
Weights refitted at every k (MAP, Gamma(1, 1)). Held-out RMSE, ScaML-GP vs single-task GP:

| Target | k=0 (w = 1/M) | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| similar (0.7·f1) | 0.33 vs 0.59 | 0.51 vs 0.59 | 0.25 vs 0.44 | 0.11 vs 0.08 | tie |
| average of sources | 0.02 vs 0.55 | 0.40 vs 0.69 | 0.10 vs 0.25 | 0.08 vs 0.05 | tie |
| unrelated | 1.00 vs 0.75 | 3.11 vs 1.12 | 1.07 vs 0.57 | 0.65 vs 0.49 | tie |
| anti-correlated (−f1) | 1.39 vs 0.90 | 1.05 vs 0.88 | 0.31 vs 0.62 | 0.28 vs 0.14 | 0.08 vs 0.06 |

Reading: the averaged prior acts only at k = 0. The failure is weight overfitting at small k (one weight reached 2-4
for the unrelated target). Dissimilar targets recover by k ≈ 20. 2-D understates transfer's value for the real
5-input search space.

## 2. Reference implementation (`checks/check_reference.py`)

| Check | Result |
|---|---|
| `fit_task_gp` on a known function (64 Sobol points, noise sd 0.05) | RMSE 0.021; noise variance 0.0029 (true 0.0025) |
| outputs in raw units | y + 100 shifts predictions by exactly 100.0 |
| sources frozen | no parameter has `requires_grad` |
| prior draws outside bounds | 0 of 200 |
| `source_moments` vs `posterior()` | mean and covariance match exactly; no noise term |
| `forward`, w = 0 | exactly a plain GP with k_t |
| `forward`, w = e1 and tiny k_t | exactly source 1's posterior (1e-4 residual = k_t's lower bound) |
| weight recovery, target 0.7·f1 | largest weight on f1 from k = 10 (0.60-0.64); RMSE 0.02 (offset effect: see `DECISIONS.md`) |
| k = 0 | mean equals the average of the sources exactly; obs variance ≥ latent |

## 3. Weight-prior variants, unrelated target (`checks/check_variants_and_toluene.py`)

| k | (c) Gamma(1,1) | (b) Gamma(2,2M) | single-task |
|---|---|---|---|
| 3 | 1.594 (max w 1.70) | 0.835 (max w 0.08) | 0.980 |
| 5 | 0.467 (max w 0.27) | 0.479 (max w 0.08) | 0.453 |
| 10 | 0.389 (max w 0.00) | 0.394 (max w 0.07) | 0.320 |
| 20 | 0.122 (max w 0.00) | 0.141 (max w 0.08) | 0.126 |

## 4. Real data: toluene noise bound (same script)

Toluene rows with `pipette_volume == 50`, n = 112, 6 inputs scaled by observed min/max, outcome `mean_error`.

| Noise bounds (standardized) | Fitted noise | Lengthscales |
|---|---|---|
| (1e-8, 1e-2), paper | 0.01 (**at the bound**) | 0.79, 0.52, 1.11, 0.46, **0.10, 0.17** |
| (1e-6, 1.0), D2a | 0.030 | 0.77, 1.33, 1.04, 0.95, 0.80, 0.43 |

Caveats, found afterwards: these 112 rows mix optimization and benchmarking campaigns, `mean_error` is the **signed**
error (the optimizer minimized `mean_abs_error`), and inputs were scaled by observed min/max, not the search space.
The conclusion (the paper's bound is too tight) holds; rerun on the Phase 17 tables once `tasks.py` exists.

## 5. PACOH-GP (`checks/check_pacoh.py`, `checks/check_pacoh_training_length.py`)

SVGD direction: K = 1 returns the score exactly; identical particles with zero score don't move; nearby particles
with zero score are pushed apart. All pass.

Paper's Sinusoid environment (App. E.1, 1-D, noise sd 0.1): 15 source tasks × 60 points, 5 new targets, one seed.
1,500 iterations. Held-out RMSE / NLL / 95% coverage of new observations:

| k | PACOH-MAP | PACOH-SVGD (K = 10) | single-task GP |
|---|---|---|---|
| 0 | 1.55 / 1.87 / 0.98 | 1.04 / 1.74 / 0.99 | – |
| 3 | 0.71 / 1.06 / 0.97 | 0.63 / 1.02 / 0.97 | 1.60 / 16.7 / 0.54 |
| 5 | 0.51 / 0.77 / 0.99 | 0.49 / 0.75 / 1.00 | 1.27 / 19.7 / 0.55 |
| 10 | 0.30 / 0.50 / 1.00 | 0.27 / 0.48 / 1.00 | 0.67 / 23.7 / 0.73 |
| 20 | 0.17 / 0.32 / 1.00 | 0.15 / 0.32 / 1.00 | 0.24 / −0.33 / 0.87 |

Training length, PACOH-MAP:

| Iterations | Learned noise (true 0.01) | k = 5 RMSE / NLL / cov | k = 20 RMSE / NLL / cov |
|---|---|---|---|
| 1,500 | 0.19 | 0.47 / 0.78 / 1.00 | 0.16 / 0.31 / 1.00 |
| 6,000 | 0.012 | 0.68 / 0.56 / 0.93 | 0.11 / −0.55 / 0.96 |

Reading: PACOH beats the single-task GP at every k, most at small k, where the single-task GP's intervals cover only
54-73%. SVGD is slightly better than MAP and has a much better k = 0 prior. At 1,500 steps the noise is badly
overestimated (intervals too wide). Longer MAP training fixes the noise but worsens k = 5 RMSE: meta-overfitting.
SVGD at 6,000 steps was not run (≈25 min). Cost: 32 ms/step (K = 1), 246 ms/step (K = 10).
