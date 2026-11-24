"""E|e| from the signed error's mean mu and SD sigma: the mean of a folded normal distribution
(Leone, Nelson & Nottingham, Technometrics 1961). The main E|e| model: one GP for mu(x) on the signed trial means,
one log-normal GP for sigma(x), and |e| derived from both. Checked on 1,651 optimization trials: the formula
reproduces the measured mean |e| from each trial's own mean and SD to a median of 2.8%."""
import math

import torch

from .base import Prediction

QUANTILES = (0.025, 0.5, 0.975)


def folded_mean(mu, sigma):
    """E|e| for e ~ N(mu, sigma^2): sigma sqrt(2/pi) exp(-mu^2 / 2 sigma^2) + mu (1 - 2 Phi(-mu / sigma))."""
    z = mu / sigma
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    return sigma * math.sqrt(2 / math.pi) * torch.exp(-0.5 * z.square()) + mu * (1 - 2 * normal.cdf(-z))


def folded_variance(mu, sigma):
    """Var|e| for e ~ N(mu, sigma^2): E[e^2] - (E|e|)^2 = mu^2 + sigma^2 - folded_mean^2."""
    return (mu.square() + sigma.square() - folded_mean(mu, sigma).square()).clamp_min(0.0)


def abs_error_prediction(mu_samples, log_sd_samples, *, n_replicates=10, quantiles=QUANTILES):
    """Prediction of the trial-mean |e| from posterior samples of mu(x) and log sigma(x).
    mu_samples, log_sd_samples: (S, q) joint samples, e.g. posterior(X).rsample(torch.Size([S])).squeeze(-1).
    latent_var: posterior variance of E|e| at each x. obs_var: adds the sampling variance of a new trial's mean |e|
    from n_replicates replicates, E[Var|e|] / n_replicates. quantiles: of E|e| (it is not Gaussian)."""
    sigma = log_sd_samples.exp()
    values = folded_mean(mu_samples, sigma)                                   # (S, q)
    mean, latent_var = values.mean(0), values.var(0)
    obs_var = latent_var + folded_variance(mu_samples, sigma).mean(0) / n_replicates
    levels = torch.tensor(quantiles, dtype=values.dtype)
    return Prediction(mean=mean, latent_var=latent_var, obs_var=obs_var,
                      quantiles=dict(zip(quantiles, torch.quantile(values, levels, dim=0))))
