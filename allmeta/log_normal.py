"""Log-normal GP for a positive per-trial outcome (Phase 16 baseline for mean |e|; the folded normal in folded.py is
the main E|e| model). A GP on log of the trial mean with fixed log-scale noise (delta method, TrialStats.log_mean).

With g = log mu(x) and posterior g ~ N(m, v), the back-transform depends on what the point estimate should be:
  parameterization="median": point = exp(m)            (median of mu(x); unbiased on the log scale)
  parameterization="mean":   point = exp(m + v / 2)    (mean of mu(x); unbiased on the raw scale)
Variances and quantiles are those of the log-normal either way. A new trial mean of n replicates adds log-scale
noise sigma_rep^2(x) / (n mu(x)^2) (delta method), with sigma_rep^2 from N2's smoother fitted on the same trials."""
import torch

from .base import Prediction
from .noise import VarianceSmoother, fit_fixed_noise_gp

Z = {0.025: -1.959964, 0.05: -1.644854, 0.1: -1.281552, 0.5: 0.0, 0.9: 1.281552, 0.95: 1.644854, 0.975: 1.959964}


class LogNormalGP:
    def __init__(self, stats, *, parameterization="median", **fit_kw):
        if parameterization not in ("median", "mean"):
            raise ValueError(f"unknown parameterization {parameterization!r}")
        self.parameterization = parameterization
        y, noise = stats.log_mean()
        self.model = fit_fixed_noise_gp(stats.X, y, noise, **fit_kw)
        self.smoother = VarianceSmoother(stats, **fit_kw)
        self.train_inputs = self.model.train_inputs

    def posterior(self, X, observation_noise=False):
        """Latent posterior of log mu(x) (log scale). Use predict() for raw units."""
        return self.model.posterior(X)

    def log_noise(self, X, n_replicates=10):
        """Log-scale noise variance of a new trial mean at X."""
        with torch.no_grad():
            mu = torch.exp(self.posterior(X).mean.squeeze(-1))
            return self.smoother.replicate_var(X) / (n_replicates * mu.square())

    def predict(self, X, n_replicates=10, quantiles=(0.025, 0.5, 0.975)):
        with torch.no_grad():
            post = self.posterior(X)
            m, v = post.mean.squeeze(-1), post.variance.squeeze(-1)
            v_obs = v + self.log_noise(X, n_replicates)
        point = torch.exp(m) if self.parameterization == "median" else torch.exp(m + v / 2)
        lognormal_var = lambda s2: (torch.exp(s2) - 1) * torch.exp(2 * m + s2)
        return Prediction(mean=point, latent_var=lognormal_var(v), obs_var=lognormal_var(v_obs),
                          quantiles={q: torch.exp(m + Z[q] * v_obs.sqrt()) for q in quantiles})


def fit_log_normal_gp(stats, *, parameterization="median", **fit_kw):
    """stats: TrialStats of a positive outcome (e.g. from_replicates(..., value="abs")), count >= 2."""
    return LogNormalGP(stats, parameterization=parameterization, **fit_kw)
