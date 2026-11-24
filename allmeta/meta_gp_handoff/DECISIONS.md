# Decisions

Each decision: what was chosen, the options that were on the table, why, and the evidence. "Paper" means
ScaML-GP (Tighineanu et al., arXiv 2312.00742); section numbers refer to it.

## ScaML-GP (Phase 17, first model)

| # | Decision | Options considered | Why | Evidence |
|---|---|---|---|---|
| D1 | Target context = the target liquid's first k trials in time order. Weight prior: **Gamma(1, 1)** as in the paper (main, "c"); **Gamma(2, 2M)**, mean 1/M (variant, "b") | (a) w = 1/M until k ≥ k_min; (b) tighter prior; (c) paper as written | Every 50 uL campaign starts with 1 centre point + 9 Sobol points (Ax Sobol, seed 42), so "first k trials" already is your Sobol suggestion for k ≤ 10. Run the paper's prior and test the variant rather than assume | Raw Ax snapshots; toy check (`RESULTS.md` §1, §3): under (c) weights blow up at k = 3 for an unrelated target, (b) prevents it at a ~0.01-0.02 RMSE cost at k ≥ 10 |
| D2 | Target noise learned, with **wider bounds (1e-6, 1.0)** in standardized units. Same bounds on the source GPs | (a) learned, wider bounds; (b) fixed to replicate variance s²/n | (a) matches the paper apart from the bound | Toluene: the paper's bound (1e-8, 1e-2) pinned the noise at 1e-2 and shrank two lengthscales to 0.10 and 0.17; with (1e-6, 1.0) noise = 0.030, not at the bound (`RESULTS.md` §4) |
| D3 | Source posteriors converted to raw units (BoTorch `Standardize` undoes itself in `posterior()`), then put in the target's **pooled** standardized units: mu → (mu − shift)/scale, Sigma → Sigma/scale² | (a) raw then pooled; (b) each source's own standardized units | (a) gives the weights a consistent meaning across liquids. The paper doesn't say which it did | – |
| D4 | One source task per liquid (all of that liquid's trials pooled) | (a) per liquid; (b) per campaign | Simpler, one weight per liquid | – |
| D5 | SE (RBF) kernel with ARD inside a `ScaleKernel`, with the paper's priors and bounds (App. C.5) | paper kernel vs production kernel | Start faithful. Production also uses RBF with ARD (per `gp_validation/code/model_factory.py`), so only the priors differ | – |
| D6 | GPyTorch `ExactGP` + BoTorch `GPyTorchModel`. **Weights live on the model** and `forward` computes both halves of Eq. 6 | (a) GPyTorch modules; (b) plain torch | Ax-compatible later. Weights on the model rather than in separate `Mean`/`Kernel` modules, because the same w enters the mean (w) and the kernel (w²) | – |

## Choices the paper doesn't make (ours, documented as ours)

| Choice | Why |
|---|---|
| **k = 0:** w = 1/M, k_t and noise at their prior **medians** | With no target data, Gamma(1, 1)'s mode (w = 0) switches transfer off and its mean (w = 1) sums all sources. Prior modes make k_t's outputscale ≈ 2e-5 → overconfident prior; medians keep honest variance. With w = 1/M the k = 0 mean is exactly the average of the sources (checked). |
| Source covariance is the **latent** posterior (no noise term) | App. A derivation conditions on source functions, not their measurement noise. Checked: diagonal < observation variance. |
| Sources are frozen (`requires_grad_(False)`) and held in a plain list (not submodules) | Assumption 3; restarts and `state_dict` never touch them; cost stays linear in M. |
| 5 restarts: start 0 from GPyTorch defaults, starts 1-4 drawn from the priors; best MAP objective kept | App. C uses 5 starts sampled from the priors. |
| Covariance in `forward` kept as a `LinearOperator` (`to_linear_operator`) | A dense tensor made PyTorch factorize the full train+query matrix, which is singular → Cholesky failure. Found and fixed during testing. |

## Known consequences to keep in mind

- **Pooled standardization adds an implicit offset:** ỹ_t = Σ w_m (mu_m − c)/s means, in raw units,
  y_t = Σ w_m mu_m + c·(1 − Σ w_m). That's why the toy target 0.7·f1 recovered w1 ≈ 0.6-0.64, not 0.7.
  Tests should check the largest weight lands on the right source, not its exact value.
- **Positive-only weights** (App. C.5): an anti-correlated liquid can't be used directly.
- **Ax integration (later):** feasible through `ModelConfig(botorch_model_class=..., model_options=...)`.
  Needs: frozen sources passed in `model_options`; one model per objective (error, stdev); a decision on Ax's
  outcome transforms (turn them off for this node and let the model standardize, or map sources into Ax's space);
  and `forward` handling batched inputs (b, q, d), which it doesn't yet.

## PACOH-GP (Phase 17, second model)

Paper: Rothfuss et al., ICML 2021. Settings the paper leaves to a hyperparameter search come from the authors' code
(github.com/jonasrothfuss/meta_learning_pacoh). Defaults chosen by Claude, open to change.

| # | Default | Source | Alternative |
|---|---|---|---|
| P1 | One class: `n_particles=1` is PACOH-MAP, `n_particles=10` is PACOH-SVGD | paper App. C.2 | – |
| P2 | Mean and feature nets 4 × 32 tanh; feature dim 2; SE kernel on features with learned lengthscales, outputscale fixed at 1 | paper (layers), code (rest) | paper's ½·exp(−‖F(x) − F(x')‖²) |
| P3 | Hyper-prior: weights N(0, 0.5²), biases N(0, 3²), raw lengthscale N(0, 1), raw noise N(−1, 1) | code | paper gives no σ_P |
| P4 | Objective `prior_factor`·ln P + Σ ln Z_i/(m_i + 1); `prior_factor = 0.1` | paper Alg. 2 + experiment default | 1.0 = paper exactly; authors tuned 5e-5 to 0.5 |
| P5 | x and y standardized with statistics pooled over the sources only; fixed for every target and k | code | ScaML-GP's unit cube + target-inclusive pooling |
| P6 | Adam 1e-3, ×0.98 per 1,000 steps; all tasks each step; SVGD bandwidth by median heuristic | code | task batch 2, bandwidth 0.1 (their experiments) |
| P7 | `n_iter = 6000` | tested (`RESULTS.md` §5) | authors: 100,000 |
| P8 | One noise variance shared by all liquids (part of φ) | paper | per-liquid noise is not in the method; flag |
| P9 | Plain PyTorch, not GPyTorch | – | SVGD needs flat parameter vectors and a custom hyper-prior; GPyTorch's training adds nothing here |

No k = 0 rule is needed: with no context, the prediction is the meta-learned prior itself.

## Phase 16: noise models

| # | Decision | Where |
|---|---|---|
| N1 | Fixed replicate noise (var / count); noise at a new x from N2's smoother | `noise.py` |
| N2 | Per-replicate variance smoothed by a GP on log SD (exact chi-square noise) | `noise.py` |
| N3 | hetGP (Binois et al. 2018) following the authors' R package, with the penalty scale frozen at its start value (profiled, it drove the fit to flat noise) | `hetgp.py` |
| N4 | E abs(e) = folded normal of the signed-error GP and the log-SD GP (main model); log-normal GP on the trial mean abs(e) as the baseline, `parameterization="median"` or `"mean"` | `folded.py`, `log_normal.py` |

## Factorial and verdict (decided 2026-11-24)

| # | Decision |
|---|---|
| F1 | Data: the complete, uncapped, first optimization run of each liquid (`data/runs.csv`; `data/build_data.py`). Uncapped = dated before 2026-09-16; complete = at least 95% of planned replicates; repeat runs dropped. 15 liquids with HSPiP descriptors; the glycerol-water mixture is kept in the data but has no descriptors, so it is outside the factorial |
| F2 | Inputs mapped to [0, 1] with the pipette's physical limits, linear scale (`tasks.BOUNDS`) |
| F3 | Context = first k trials (k = 0, 3, 5, 10, 20, 40); query = trials 40 and later, the same for every k |
| F4 | Hyperparameters tuned nested: the outer target is removed, 4 other liquids take turns as the target, lowest mean inner NLPD wins (`hazel_hpc/sweeps/tune.toml`). PACOH-SVGD reuses PACOH-MAP's tuned `prior_factor` and `feature_dim` |
| F5 | Verdict rule A (`verdict.py`): NLPD averaged over k = 3, 5, 10, paired per liquid against single-task homoskedastic; a model helps if the median difference is below 0 and the one-sided sign test gives p < 0.05 (at least 12 of 15 liquids). RMSE with the same rule is secondary. k = 0 only ranks transfer models against each other |
| F6 | Noise prior LogNormal(-4, 1.5) on the standardized scale for every GP with a learned noise (`task_gp.NOISE_PRIOR`), replacing the paper's LogNormal(-8, 2), whose 95% upper limit (0.017) is below the measured noise of the log outcomes. 5-fold CV NLPD on 15 liquids: log_sd 2.59 -> 1.80, log_abs 0.96 -> 0.93, signed -2.13 -> -2.16; k <= 10 essentially unchanged |
