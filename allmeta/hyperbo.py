import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy import stats

from .base import Prediction
from .task_gp import SCALE_BOUNDS

JITTER = 1e-6


def fit_liquid_gp(X, y, *, noise_bounds=(1e-6, 1.0)):
    """MLE of one liquid's theta = (constant mean, lengthscales (d,), signal variance, noise variance), Eq. 3-4.
    No priors: the paper's per-dataset step is plain maximum likelihood. y is already pooled-standardized."""
    likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_bounds))
    covar = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=X.shape[-1], lengthscale_constraint=Interval(*SCALE_BOUNDS)),
                        outputscale_constraint=Interval(*SCALE_BOUNDS))
    model = SingleTaskGP(X, y.unsqueeze(-1), likelihood=likelihood, covar_module=covar, outcome_transform=None)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(likelihood, model))
    return {"mean": model.mean_module.constant.item(),
            "lengthscale": model.covar_module.base_kernel.lengthscale.detach().squeeze(0),
            "signal": model.covar_module.outputscale.item(),
            "noise": likelihood.noise.item()}

def fit_universal_prior(thetas):
    """MLE of a (Sec. 4.1, App. C): Normal(mu, sd) for the constant mean; Gamma(shape, rate) for the signal and
    noise variances; ONE Gamma for all lengthscales, pooled over liquids and dimensions."""
    def gamma(values):
        shape, _, scale = stats.gamma.fit(values, floc=0.0)
        return shape, 1.0 / scale
    means = [t["mean"] for t in thetas]
    return {"mean": (float(torch.tensor(means).mean()), float(torch.tensor(means).std(unbiased=False))),
            "lengthscale": gamma(torch.cat([t["lengthscale"] for t in thetas]).numpy()),
            "signal": gamma([t["signal"] for t in thetas]),
            "noise": gamma([t["noise"] for t in thetas])}

def sample_prior(prior, R, d):
    """R draws of theta from p(theta; a): mean (R,), lengthscale (R, d), signal (R,), noise (R,)."""
    def gamma(shape, rate, size):
        return torch.distributions.Gamma(torch.tensor(shape), torch.tensor(rate)).sample(size).double()
    mu, sd = prior["mean"]
    # A fitted Gamma with shape < 1 puts most of its mass near 0, and some draws underflow to exactly 0 (a zero
    # lengthscale makes the kernel NaN). Clamp to the range the per-liquid MLE fits were allowed to reach.
    return {"mean": mu + sd * torch.randn(R, dtype=torch.float64),
            "lengthscale": gamma(*prior["lengthscale"], (R, d)).clamp(*SCALE_BOUNDS),
            "signal": gamma(*prior["signal"], (R,)).clamp(*SCALE_BOUNDS),
            "noise": gamma(*prior["noise"], (R,))}

def stack_thetas(thetas):
    """The fitted theta_i as R = N samples (the paper's x-HyperBO+: no MLE step, uniform over the empirical set)."""
    return {"mean": torch.tensor([t["mean"] for t in thetas], dtype=torch.float64),
            "lengthscale": torch.stack([t["lengthscale"] for t in thetas]),
            "signal": torch.tensor([t["signal"] for t in thetas], dtype=torch.float64),
            "noise": torch.tensor([t["noise"] for t in thetas], dtype=torch.float64)}

def matern32(x1, x2, lengthscale, signal):
    """Matern 3/2 for R hyperparameter samples at once. x1 (a, d), x2 (b, d), lengthscale (R, d), signal (R,)
    -> (R, a, b): signal * (1 + sqrt(3) r) exp(-sqrt(3) r), r = ||(x1 - x2) / lengthscale||."""
    z1, z2 = x1 / lengthscale[:, None, :], x2 / lengthscale[:, None, :]
    r = math.sqrt(3.0) * torch.cdist(z1, z2)
    return signal[:, None, None] * (1.0 + r) * torch.exp(-r)

def _factor(X, y, theta):
    K = matern32(X, X, theta["lengthscale"], theta["signal"])
    K = K + (theta["noise"] + JITTER)[:, None, None] * torch.eye(len(X), dtype=X.dtype)
    L = torch.linalg.cholesky(K)                                            # (R, k, k)
    r = (y - theta["mean"][:, None]).unsqueeze(-1)                          # (R, k, 1)
    return L, r, torch.cholesky_solve(r, L)

def log_marginal(X, y, theta):
    """ln p(y | X, theta_r) for every sample r, Eq. 3 with one sub-dataset -> (R,)."""
    L, r, alpha = _factor(X, y, theta)
    return (-0.5 * (r * alpha).sum((-2, -1)) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - 0.5 * len(y) * math.log(2 * math.pi))

def gp_predict(X, y, Xq, theta):
    """GP posterior under every sample: mean (R, q), latent variance (R, q). With no data: the prior."""
    R = len(theta["mean"])
    prior_var = theta["signal"][:, None].expand(R, len(Xq))
    if len(y) == 0:
        return theta["mean"][:, None].expand(R, len(Xq)), prior_var
    L, _, alpha = _factor(X, y, theta)
    Ks = matern32(X, Xq, theta["lengthscale"], theta["signal"])            # (R, k, q)
    mean = theta["mean"][:, None] + (Ks.transpose(-2, -1) @ alpha).squeeze(-1)
    v = torch.linalg.solve_triangular(L, Ks, upper=False)
    return mean, (prior_var - v.square().sum(-2)).clamp_min(0.0)

class HyperBOPlus:
    """Phase 17 interface. prior="gamma": the paper's HyperBO+; prior="empirical": its x-HyperBO+ variant.
    n_samples: the paper uses R = 100; 1000 keeps the effective sample size >= 30 at k = 40 for ~0.2 s."""

    def __init__(self, *, prior="gamma", n_samples=1000, noise_bounds=(1e-6, 1.0), seed=0):
        self.prior_kind, self.n_samples, self.noise_bounds, self.seed = prior, n_samples, noise_bounds, seed

    def meta_fit(self, sources):
        """sources: liquid -> (X (n_m, d), y (n_m,)). y standardized with statistics pooled over all sources."""
        y_all = torch.cat([y for _, y in sources.values()])
        self.y_shift, self.y_scale = y_all.mean().item(), y_all.std().item()
        self.thetas = [fit_liquid_gp(X, (y - self.y_shift) / self.y_scale, noise_bounds=self.noise_bounds)
                       for X, y in sources.values()]
        d = next(iter(sources.values()))[0].shape[-1]
        if self.prior_kind == "empirical":
            self.samples = stack_thetas(self.thetas)
        else:
            self.prior = fit_universal_prior(self.thetas)
            torch.manual_seed(self.seed)                     # one fixed sample set, reused for every target and k
            self.samples = sample_prior(self.prior, self.n_samples, d)

    def condition(self, X_t, y_t):
        """Posterior weights over the samples: w_r ∝ p(D_t | theta_r) (Eq. 5). k = 0: uniform."""
        self.X_t, self.y_t = X_t, (y_t - self.y_shift) / self.y_scale
        R = len(self.samples["mean"])
        log_w = log_marginal(self.X_t, self.y_t, self.samples) if len(y_t) else torch.zeros(R, dtype=torch.float64)
        self.weights = torch.softmax(log_w, 0)
        self.ess = 1.0 / self.weights.square().sum().item()     # effective sample size, a degeneracy check

    def predict(self, X_q):
        """Weighted mixture of the R GP posteriors, in raw units."""
        with torch.no_grad():
            means, latent = gp_predict(self.X_t, self.y_t, X_q, self.samples)
        w = self.weights[:, None]
        mean = (w * means).sum(0)
        spread = (w * (means - mean).square()).sum(0)
        noise = (w * self.samples["noise"][:, None]).sum(0)
        s = self.y_scale
        return Prediction(mean=mean * s + self.y_shift, latent_var=((w * latent).sum(0) + spread) * s**2,
                          obs_var=((w * latent).sum(0) + spread + noise) * s**2)