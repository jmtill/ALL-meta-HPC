import copy
import math

import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.utils import sample_all_priors
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

# ScaML-GP paper priors, except the noise prior (below)

# Noise prior, on the standardized scale (noise variance / variance of y). The paper's LogNormal(-8, 2) puts 95% of
# its mass below 0.017, made for near-noiseless benchmarks; our trial noise is 0.001-0.03 (signed error),
# 0.06-0.42 (log SD) and 0.006-0.13 (log mean |e|). LogNormal(-4, 1.5): median 0.018, 95% range 0.001-0.34.
# Tested 2026-11-24 on all 15 liquids: 5-fold CV NLPD log_sd 2.59 -> 1.80, log_abs 0.96 -> 0.93, signed -2.13 -> -2.16;
# k <= 10 unchanged (too few points to inform the noise). Used by every GP with a learned noise (task GPs,
# ScaML-GP's target, pooled features). The paper's value: PAPER_NOISE_PRIOR.
PAPER_NOISE_PRIOR = ('lognormal', -8.0, 2.0)
NOISE_PRIOR = ('lognormal', -4.0, 1.5)

PAPER_PRIORS = {
    'lengthscale': ('gamma', 3.0, 6.0),
    'outputscale': ('gamma', 2.0, 0.15),
    'noise': NOISE_PRIOR,
}

SCALE_BOUNDS = (1e-4, 1e2)

def make_prior(spec):
    kind, a, b = spec
    return GammaPrior(a, b) if kind=='gamma' else LogNormalPrior(a, b)

def se_ard_kernel(d, lengthscale_prior, outputscale_prior):
    return ScaleKernel(
        RBFKernel(ard_num_dims=d, lengthscale_prior=lengthscale_prior,
                  lengthscale_constraint=Interval(*SCALE_BOUNDS)),
        outputscale_prior=outputscale_prior, outputscale_constraint=Interval(*SCALE_BOUNDS))

def fit_map(model, *, n_starts=5, seed=0):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    best, best_state = -math.inf, None
    for s in range(n_starts):
        torch.manual_seed(seed+s)
        if s >0:
            sample_all_priors(model)
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:      # one bad random start (e.g. a non-PD covariance) must not end the fit
            continue
        model.train()
        with torch.no_grad():
            value= mll(model(*model.train_inputs), model.train_targets).item()
        if value > best:
            best, best_state = value, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ModelFittingError(f"all {n_starts} MAP starts failed")
    model.load_state_dict(best_state)
    model.eval()
    return best


def fit_task_gp(X: torch.Tensor,
                y: torch.Tensor,
                *, noise_bounds=(1e-6, 1.0),
                n_starts: int = 5,
                seed: int=0)->SingleTaskGP:

    P = PAPER_PRIORS
    likelihood = GaussianLikelihood(noise_prior=make_prior(P["noise"]),
                                    noise_constraint=Interval(*noise_bounds))

    model = SingleTaskGP(X, y.unsqueeze(-1), likelihood=likelihood,
                         covar_module=se_ard_kernel(X.shape[-1],
                                                    make_prior(P["lengthscale"]),
                                                    make_prior(P["outputscale"])),
                         outcome_transform=Standardize(m=1))
    fit_map(model, n_starts=n_starts, seed=seed)
    model.requires_grad_(False)
    return model

