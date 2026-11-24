"""The three noise levels for a GP on per-trial means (Phase 16). All take a TrialStats and return a NoisyGP.

  N1 fixed     mean GP with each trial's own noise, var_i / count_i (train_Yvar). Noise at a new x: N2's smoother.
  N2 smoothed  a GP on log SD (fixed chi-square noise, TrialStats.log_sd) smooths the per-replicate variance;
               the mean GP then uses the smoothed variance / count_i at the training trials and at new x.
  N3 hetgp     mean and log-noise fitted jointly (hetgp.py, Binois et al. 2018).

Convention: per-replicate noise variance sigma^2(x) on y's scale; a new trial of n replicates has noise sigma^2(x) / n.
Inputs X are expected in the unit cube (tasks.py rescales), since the paper lengthscale prior assumes it."""
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from .base import Prediction
from .hetgp import fit_hetgp
from .task_gp import PAPER_PRIORS, fit_map, make_prior, se_ard_kernel


def fit_fixed_noise_gp(X, y, yvar, *, n_starts=5, seed=0):
    """GP with known per-point noise yvar (n,) and the paper kernel priors; latent posterior from .posterior()."""
    P = PAPER_PRIORS
    model = SingleTaskGP(X, y.unsqueeze(-1), train_Yvar=yvar.unsqueeze(-1),
                         covar_module=se_ard_kernel(X.shape[-1], make_prior(P["lengthscale"]), make_prior(P["outputscale"])),
                         outcome_transform=Standardize(m=1))
    fit_map(model, n_starts=n_starts, seed=seed)
    model.requires_grad_(False)
    return model


class VarianceSmoother:
    """GP on log SD; replicate_var(X) = E[sigma^2(x)] = exp(2 m + 2 v) for log sigma ~ N(m, v)."""

    def __init__(self, stats, **fit_kw):
        y, noise = stats.log_sd()
        self.model = fit_fixed_noise_gp(stats.X, y, noise, **fit_kw)

    def replicate_var(self, X):
        with torch.no_grad():
            post = self.model.posterior(X)
            return torch.exp(2 * post.mean.squeeze(-1) + 2 * post.variance.squeeze(-1))


class NoisyGP:
    """Mean model + noise model, one interface for N1-N3. posterior(X) is the latent f (what ScaML-GP reads)."""

    def __init__(self, mean_model, replicate_var, kind):
        self.mean_model, self.replicate_var, self.kind = mean_model, replicate_var, kind
        self.train_inputs = mean_model.train_inputs

    def posterior(self, X, observation_noise=False):
        return self.mean_model.posterior(X)

    def noise(self, X, n_replicates=10):
        """Noise variance of a new trial mean of n_replicates replicates at X."""
        return self.replicate_var(X) / n_replicates

    def predict(self, X, n_replicates=10):
        post = self.posterior(X)
        latent = post.variance.squeeze(-1)
        return Prediction(mean=post.mean.squeeze(-1), latent_var=latent, obs_var=latent + self.noise(X, n_replicates))


def fit_noise_model(stats, level, **fit_kw):
    """stats: TrialStats (count >= 2). level: "fixed" (N1), "smoothed" (N2) or "hetgp" (N3)."""
    y, yvar = stats.mean_and_noise()
    if level == "hetgp":
        het = fit_hetgp(stats.X, stats.mean, stats.var, stats.count)
        return NoisyGP(het, lambda X: het.noise(X, 1), level)
    smoother = VarianceSmoother(stats, **fit_kw)
    if level == "smoothed":
        yvar = smoother.replicate_var(stats.X) / stats.count
    elif level != "fixed":
        raise ValueError(f"unknown noise level {level!r}")
    return NoisyGP(fit_fixed_noise_gp(stats.X, y, yvar, **fit_kw), smoother.replicate_var, level)
