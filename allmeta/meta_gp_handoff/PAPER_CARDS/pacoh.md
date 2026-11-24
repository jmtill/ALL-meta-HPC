## PACOH-GP (pacoh.py)

**Source.** Rothfuss, Fortuin, Josifoski & Krause, "PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees",
ICML 2021 (PMLR 139). Sec. 4-5.2, Alg. 1-2, App. B, C, E. Settings the paper leaves to a hyperparameter search are
taken from the authors' code: github.com/jonasrothfuss/meta_learning_pacoh (`meta_learn/GPR_meta_svgd.py`,
`random_gp.py`, `svgd.py`, `experiments/meta_GPR_SVGD_base_exp.py`).

**Idea.** Learn a distribution over GP priors (mean function and kernel) from the source tasks, by minimizing a
PAC-Bayesian bound on the error on new tasks. The bound's minimizer has a closed form (Prop. 1, Eq. 6), approximated
with K SVGD particles. At test time, nothing is fitted: each particle's prior is conditioned on the target's data.

**Model.**
- Prior P_phi: GP(m_phi(x), k_phi(x, x')). m_phi is a neural network; k_phi is an SE kernel on neural-network
  features F_phi(x) (App. C). Both nets: 4 hidden layers of 32, tanh.
- Hyper-prior P(phi): Gaussian over all prior parameters (paper: N(0, sigma_P^2 I), sigma_P not reported).
- PAC-optimal hyper-posterior (Eq. 6): Q*(phi) ∝ P(phi) exp( lambda/(n beta + lambda) · sum_i ln Z(S_i, phi) ).
  With lambda = n, beta_i = m_i and the NLL loss, ln Z(S_i, phi) is the GP marginal log-likelihood (Eq. 48), so
  ln Q*(phi) = ln P(phi) + sum_i ln p(y_i | X_i, phi) / (m_i + 1) + const (Alg. 2).
- SVGD (Eq. 7, App. C.1): phi_k += eta/K · sum_l [ k(phi_l, phi_k) grad ln Q*(phi_l) + grad_{phi_l} k(phi_l, phi_k) ],
  RBF kernel over particles. The first term moves particles uphill; the second pushes them apart.
- Prediction (Eq. 49): equal-weight mixture of the K particles' GP posteriors. K = 1 is PACOH-MAP (one Gaussian).

**Shared across liquids vs fitted on the target.**
- Learned from the sources (meta-training): every phi_k, including the noise variance (one noise for all tasks).
- On the target: nothing is fitted. Context trials only condition each particle's GP.

**Settings (paper vs authors' code).**

| Setting | Paper | Authors' code / experiments | Ours |
|---|---|---|---|
| nets | 4 layers × 32, tanh | `num_layers=4`, `layer_size=32` | same |
| kernel | ½·exp(−‖F(x) − F(x')‖²) | SE on features with learned lengthscale, outputscale 1, feature dim 2 | code |
| hyper-prior | N(0, σ_P² I) | weights N(0, 0.5²), biases N(0, 3²), raw lengthscale N(0, 1), raw noise N(−1, 1) | code |
| prior weight | 1 | `prior_factor` 0.1 (experiments; tuned 5e-5 to 0.5) | 0.1 |
| per-task weight | 1/(m_i + 1) on the total MLL | m̄/(m̄ + n) on per-point MLLs | paper |
| normalization | – | x and y standardized with pooled training-task statistics | code |
| SVGD | RBF, lengthscale ℓ | RBF, bandwidth 0.1 (experiments) or median heuristic | median heuristic |
| optimizer | step size η | Adam 1e-3, ×0.98 per 1,000 steps, task batch 2, 100,000 steps | Adam 1e-3, ×0.98, all tasks, 6,000 steps |
| particles | K = 10 (SVGD), 1 (MAP) | 10 | 10 and 1 |

**Cost.** Each step: K particles × n tasks × one m×m Cholesky + network passes. Measured here (15 tasks × 60 points,
1-D): 32 ms/step for K = 1, 246 ms/step for K = 10. The authors vectorize over particles; we loop.

**Known failure modes.**
- Meta-overfitting (paper App. B, Fig. 4): an expressive prior fits the source tasks too closely. The hyper-prior
  and SVGD are the guards; MAP (K = 1) is more exposed. Seen here: longer MAP training worsened k = 5 RMSE.
- Under-training: the noise variance is learned slowly. At 1,500 steps it was 0.19 vs a true 0.01 (intervals too wide).
- One shared noise variance for all tasks: wrong when liquids differ in noise.
- The predictive for K > 1 is a mixture, not a Gaussian: ±1.96σ intervals are approximate.
