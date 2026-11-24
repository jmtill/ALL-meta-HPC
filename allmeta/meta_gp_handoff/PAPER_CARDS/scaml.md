## ScaML-GP (scaml.py)

**Source.** Tighineanu, Grossberger, Baireuther, Skubch, Falkner, Vinogradska, Berkenkamp (Bosch),
"Scalable Meta-Learning with Gaussian Processes", arXiv 2312.00742v1 (Dec 2023). Sec. 4, Alg. 1, App. A-C.5.

**Idea.** A multi-task GP whose task covariance is restricted so the joint kernel splits into one GP per source task
plus one residual GP on the target. The target's prior is a weighted sum of the source posteriors.

**Assumptions.**
- A1: source tasks are independent in the prior, Cov(f_1:M, f_1:M) = I. Only A1 is needed for cost linear in M (App. C.3).
- A2: f_t = f~_t + sum_m f~_m, each f~_m perfectly (anti)correlated with f_m, f~_t independent of every source.
- A3: p(theta_m | D_m, D_t) = p(theta_m | D_m). Source hyperparameters never see target data.

**Model.**
- Joint kernel (Eq. 5): k((x,v),(x',v')) = [v=t][v'=t] k_t(x,x') + sum_m g_m(v) g_m(v') k_m(x,x'),
  g_m(v) = w_m if v = t, 1 if v = m, else 0.
- Target prior given the source data (Eq. 6, Theorem 1):
  - mean: m_S(x) = sum_m w_m mu_m(x)
  - kernel: k_S(x,x') = k_t(x,x') + sum_m w_m^2 Sigma_m(x,x')
  - mu_m, Sigma_m: source m's LATENT posterior (no noise term; App. A).
- Target posterior (Eqs. 1, 7): GP conditioning of that prior on D_t. Eq. 1 as printed has (y_t - m(x)); read m(X_t).

**Shared across liquids vs fitted on the target.**
- Once per source liquid, on its own data: theta_m (kernel hyperparameters, noise).
- On the target's k context trials: theta_t = (k_t hyperparameters, w_1..w_M, target noise) by Eq. 9.
- Nothing else is learned: no network, no shared parametric prior.

**Algorithm (Alg. 1).** Fit one GP per source; cache source moments at X_t; fit theta_t (Eq. 9); condition on D_t.

**Paper settings (App. C, C.5).** SE-ARD everywhere; L-BFGS-B with 5 starts drawn from the priors. Inputs in the unit
cube; each source standardized on its own data; target standardized with mean/std of y_t together with all y_1:M.

| Parameter | Source GPs | Target GP |
|---|---|---|
| lengthscale | Gamma(3, 6) | log l ~ N(0.5, 1.5) |
| outputscale | Gamma(2, 0.15) | log o ~ N(-2, 3) |
| noise variance | log n ~ N(-8, 2), bounds [1e-8, 1e-2] | same |
| weights w_m | - | Gamma(1, 1), w_m > 0 |

Lengthscale and outputscale bounded to [1e-4, 1e2]. Positive weights only, so anti-correlated sources are ignored.

**Our departures (see DECISIONS.md).** Noise bounds widened to (1e-6, 1.0); variant weight prior Gamma(2, 2M);
k = 0 rule (w = 1/M, k_t and noise at prior medians); source moments converted to raw units, then pooled units.

**Cost.** One O(n_m^3) fit per source, done once. Each target fit O(M k^2 + k^3).

**Known failure modes.**
- Small k: the weights overfit (toy check: one weight reached 2-4 at k = 3 for an unrelated target).
- A2 violated (no positive combination of sources fits): k_t must absorb it, which needs enough target data.
- The paper's noise bound is too tight for dispensing data (toluene check).
- Pooled standardization adds an implicit offset c·(1 - sum w).
