"""Heteroskedastic GP for replicated experiments (Binois, Gramacy & Ludkovski, JCGS 2018; arXiv 1611.05902v2).
Noise level N3: the mean GP and a latent log-noise process are fitted together by one penalized likelihood.
Follows the authors' R package hetGP (CRAN 1.x: logLikHet, mleHetGP defaults), which adds safeguards that the paper's
equations alone do not state, with one change of ours to the penalty (below).

Works from per-trial summaries (mean ybar_i, sample variance s2_i, replicate count a_i): for Gaussian replicates the
full-N likelihood depends on the replicates only through these (Lemma 3.2), so no N x N matrix is ever built.

  y_ij = beta0 + f(x_i) + e_ij,  f ~ GP(0, nu C_theta),  e_ij ~ N(0, nu lambda(x_i))
  log lambda_n = m_g + C_g K_g^-1 (Delta - m_g),  K_g = C_g + g A^-1,  m_g = kriging mean of Delta    (Eq. 14, hetGP)
  log L   = -N/2 log psi - 1/2 sum[(a_i - 1) log lambda_i + log a_i] - 1/2 log|C + A^-1 Lambda|              (Lemma 3.2)
  psi     = [sum (a_i - 1) s2_i / lambda_i + (ybar - beta0)' K^-1 (ybar - beta0)] / N    (nu profiled, beta0 GLS)
  penalty = Gaussian log density of Delta around m_g, covariance nu_g K_g                                    (Eq. 18)
penalty="fixed" (default, ours): nu_g frozen at its value at the residual-based start. The paper/hetGP profile nu_g
  instead ("profiled"); that term is unbounded as Delta flattens (nu_g -> 0), and on our test data the optimizer
  followed it to a flat-noise fit. "none" drops the penalty and overfits the noise. Six-seed check (sd = 0.06 + 0.32x,
  21 sites x 6 runs), mean |log error| of the noise SD: fixed 0.13, profiled 0.41, none 0.68, hetGPy package 0.12.
hetGP safeguards kept: noise lengthscales phi = k theta with k in [1, 100]; g in [1e-6, 1]; homoskedastic priming fit
and residual-based Delta start (Sec. 4.3); final fallback to the homoskedastic fit when its log L is not lower.
Ours: lengthscale bounds from each input's range in the data; gradients by autograd (paper: closed forms)."""
import math

import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from .base import Prediction

EPS = math.sqrt(torch.finfo(torch.float64).eps)
LENGTHSCALE_RANGE = (0.01, 2.0)      # mean-process lengthscale bounds, as multiples of each input's range
K_THETA_G = (1.0, 100.0)             # noise lengthscales = k * mean lengthscales (hetGP linkThetas = "joint")
NUGGET_BOUNDS = (1e-6, 1.0)          # g, hetGP default
HOM_LAMBDA_BOUNDS = (1e-6, 1e2)      # noise-to-signal ratio of the homoskedastic fit


def _bounded(raw, lo, hi):
    """Map an unconstrained tensor into (lo, hi) on the log scale (sigmoid); lo, hi floats or tensors."""
    lo, hi = torch.as_tensor(lo, dtype=torch.float64), torch.as_tensor(hi, dtype=torch.float64)
    return torch.exp(lo.log() + (hi.log() - lo.log()) * torch.sigmoid(raw))


def _unbounded(value, lo, hi):
    lo, hi = torch.as_tensor(lo, dtype=torch.float64), torch.as_tensor(hi, dtype=torch.float64)
    p = (torch.log(torch.as_tensor(value, dtype=torch.float64)) - lo.log()) / (hi.log() - lo.log())
    return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def _corr(x1, x2, lengthscale):
    """Squared-exponential correlation exp(-0.5 ||(x1 - x2) / l||^2) -> (a, b)."""
    return torch.exp(-0.5 * torch.cdist(x1 / lengthscale, x2 / lengthscale).square())


def _kriging_mean(K_chol, v):
    """GLS (ordinary kriging) constant mean of v under covariance K: 1'K^-1 v / 1'K^-1 1."""
    ones = torch.ones_like(v).unsqueeze(-1)
    Ki1 = torch.cholesky_solve(ones, K_chol).squeeze(-1)
    return (Ki1 * v).sum() / Ki1.sum()


def _log_likelihood(Cn, lam, ybar, s2, a):
    """Lemma 3.2 log-likelihood with nu profiled and beta0 by GLS. Returns (log L, psi, beta0, Cholesky of K)."""
    N = a.sum()
    K = Cn + torch.diag(lam / a + EPS)
    L = torch.linalg.cholesky(K)
    beta0 = _kriging_mean(L, ybar)
    r = ybar - beta0
    psi = (((a - 1) * s2 / lam).sum() + (r * torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)).sum()) / N
    logL = (-0.5 * N * torch.log(psi) - 0.5 * ((a - 1) * torch.log(lam) + torch.log(a)).sum()
            - L.diagonal().log().sum() - 0.5 * N * (1 + math.log(2 * math.pi)))
    return logL, psi, beta0, L


def _lbfgs(params, closure, steps):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")
    def step():
        opt.zero_grad()
        loss = -closure()
        loss.backward()
        return loss
    opt.step(step)


class HetGP:
    """Fitted heteroskedastic GP. posterior(X) is the latent f (raw units); noise(X) the variance of a new trial mean."""

    def __init__(self, X, ybar, s2, a, *, steps=300, penalty="fixed"):
        self.X, self.a, self.penalty = X, a, penalty
        N = a.sum()
        shift = ((a * ybar).sum() / N)
        total = ((a - 1) * s2).sum() + (a * (ybar - shift).square()).sum()
        self.scale = torch.sqrt(total / (N - 1)).item()                   # pooled replicate SD (beta0 handles level)
        self.ybar, self.s2 = ybar / self.scale, s2 / self.scale**2
        self.train_inputs = (X,)                                          # what ScaML-GP reads from a source model
        span = (X.max(0).values - X.min(0).values).clamp_min(1e-12)
        self.ls_bounds = (LENGTHSCALE_RANGE[0] * span, LENGTHSCALE_RANGE[1] * span)
        self._fit_homoskedastic(steps)
        self._fit_heteroskedastic(steps)

    # --- parameters ------------------------------------------------------------------------------------------------
    @property
    def lengthscale(self):
        return _bounded(self.raw_ls, *self.ls_bounds)

    @property
    def noise_lengthscale(self):
        return _bounded(self.raw_k, *K_THETA_G) * self.lengthscale

    @property
    def nugget(self):
        return _bounded(self.raw_g, *NUGGET_BOUNDS)

    def _noise_parts(self):
        """Cholesky of K_g, kriging mean m_g, and log lambda at the design sites (Eq. 14, hetGP form)."""
        Cg = _corr(self.X, self.X, self.noise_lengthscale)
        Lg = torch.linalg.cholesky(Cg + torch.diag(EPS + self.nugget / self.a))
        m_g = _kriging_mean(Lg, self.delta)
        log_lam = m_g + Cg @ torch.cholesky_solve((self.delta - m_g).unsqueeze(-1), Lg).squeeze(-1)
        return Lg, m_g, log_lam

    # --- fitting ---------------------------------------------------------------------------------------------------
    def _fit_homoskedastic(self, steps):
        """Sec. 4.3 priming fit: one noise-to-signal ratio for every site."""
        self.raw_ls = _unbounded(0.25 * self.ls_bounds[1] / LENGTHSCALE_RANGE[1], *self.ls_bounds).clone().requires_grad_(True)
        self.raw_lam = _unbounded(0.1, *HOM_LAMBDA_BOUNDS).clone().requires_grad_(True)
        def objective():
            lam = _bounded(self.raw_lam, *HOM_LAMBDA_BOUNDS).expand(len(self.a))
            return _log_likelihood(_corr(self.X, self.X, self.lengthscale), lam, self.ybar, self.s2, self.a)[0]
        _lbfgs([self.raw_ls, self.raw_lam], objective, steps)
        with torch.no_grad():
            self.hom_logL = objective().item()
            self.hom_lam = _bounded(self.raw_lam, *HOM_LAMBDA_BOUNDS).item()
            self.hom_ls = self.raw_ls.detach().clone()

    def _fit_heteroskedastic(self, steps):
        with torch.no_grad():
            Cn = _corr(self.X, self.X, self.lengthscale)
            _, psi, beta0, L = _log_likelihood(Cn, torch.full_like(self.a, self.hom_lam), self.ybar, self.s2, self.a)
            r = self.ybar - beta0
            mu = beta0 + Cn @ torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
            resid = (self.a - 1) / self.a * self.s2 + (mu - self.ybar).square()        # mean squared residual per site
            delta0 = torch.log(resid.clamp_min(1e-12) / psi)
        self.raw_k = _unbounded(1.5, *K_THETA_G).clone().requires_grad_(True)
        self.raw_g = _unbounded(0.1, *NUGGET_BOUNDS).clone().requires_grad_(True)
        self.delta = delta0.clone().requires_grad_(True)
        params = [self.raw_ls, self.raw_k, self.raw_g, self.delta]
        with torch.no_grad():                                                # scale of the Delta process at the start
            Lg, m_g, _ = self._noise_parts()
            dc = (self.delta - m_g).unsqueeze(-1)
            nu_g0 = (dc * torch.cholesky_solve(dc, Lg)).sum() / len(self.a)
        def objective(penalized=True):
            Lg, m_g, log_lam = self._noise_parts()
            logL = _log_likelihood(_corr(self.X, self.X, self.lengthscale), log_lam.exp(), self.ybar, self.s2, self.a)[0]
            if not penalized or self.penalty == "none":
                return logL
            dc = (self.delta - m_g).unsqueeze(-1)
            nu_g = (dc * torch.cholesky_solve(dc, Lg)).sum() / len(self.a)
            if self.penalty == "fixed":                                      # Gaussian log density of Delta, scale frozen
                return logL - 0.5 * len(self.a) * (nu_g / nu_g0) - Lg.diagonal().log().sum()
            if nu_g < EPS:                                                   # "profiled": hetGP's guards
                return logL
            pen = -0.5 * len(self.a) * torch.log(nu_g) - Lg.diagonal().log().sum() - 0.5 * len(self.a) * (1 + math.log(2 * math.pi))
            if logL < self.hom_logL and pen > 0:                             # hetGP: penalty must not prop up a bad fit
                return logL
            return logL + pen
        _lbfgs(params, objective, steps)
        with torch.no_grad():
            self.het_logL = objective(penalized=False).item()
        self.homoskedastic = self.hom_logL >= self.het_logL                  # Sec. 4.3 fallback
        if self.homoskedastic:
            with torch.no_grad():
                self.raw_ls.copy_(self.hom_ls)
                self.delta.copy_(torch.full_like(self.delta, math.log(self.hom_lam)))
        self._cache()

    def _cache(self):
        with torch.no_grad():
            self.Lg, self.m_g, log_lam = self._noise_parts()
            self.w_g = torch.cholesky_solve((self.delta - self.m_g).unsqueeze(-1), self.Lg).squeeze(-1)
            self.lam = log_lam.exp()
            _, self.nu, self.beta0, self.L = _log_likelihood(_corr(self.X, self.X, self.lengthscale), self.lam,
                                                             self.ybar, self.s2, self.a)
            self.alpha = torch.cholesky_solve((self.ybar - self.beta0).unsqueeze(-1), self.L).squeeze(-1)

    # --- prediction (Lemma 3.1), raw units ---------------------------------------------------------------------------
    def posterior(self, X, observation_noise=False):
        """Latent posterior of f at X as a BoTorch posterior (mean (q, 1), full covariance). Noise is never added here:
        use noise() or predict()."""
        with torch.no_grad():
            c = _corr(X, self.X, self.lengthscale)
            mean = self.beta0 + c @ self.alpha
            v = torch.linalg.solve_triangular(self.L, c.T, upper=False)
            cov = self.nu * (_corr(X, X, self.lengthscale) - v.T @ v) + EPS * torch.eye(len(X), dtype=X.dtype)
        return GPyTorchPosterior(MultivariateNormal(mean * self.scale, cov * self.scale**2))

    def noise(self, X, n_replicates=10):
        """Variance of a new trial mean of n_replicates replicates at X: nu lambda(x) / n (raw units)."""
        with torch.no_grad():
            log_lam = self.m_g + _corr(X, self.X, self.noise_lengthscale) @ self.w_g
            return self.nu * torch.exp(log_lam) * self.scale**2 / n_replicates

    def predict(self, X, n_replicates=10):
        post = self.posterior(X)
        latent = post.variance.squeeze(-1)
        return Prediction(mean=post.mean.squeeze(-1), latent_var=latent, obs_var=latent + self.noise(X, n_replicates))


def fit_hetgp(X, ybar, s2, count, *, steps=300, penalty="fixed"):
    """X (n, d), trial means ybar (n,), per-replicate sample variances s2 (n,, ddof = 1), counts (n,) >= 2."""
    return HetGP(X, ybar, s2, count.double(), steps=steps, penalty=penalty)
