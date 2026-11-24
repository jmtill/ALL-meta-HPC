"""One exact GP per task (one source liquid). Shared by ScaML-GP, HyperBO(+), RGPE."""
import copy
import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.utils import sample_all_priors
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

# ScaML-GP App. C.5, source-task GPs. Gamma = (concentration, rate); LogNormal = (loc, scale = std of log).
PAPER_SOURCE_PRIORS = {
    "lengthscale": ("gamma", 3.0, 6.0),      # 5-95%: 0.14-1.05
    "outputscale": ("gamma", 2.0, 0.15),     # 5-95%: 2.4-31.8
    "noise": ("lognormal", -8.0, 2.0),       # 5-95%: 1.25e-5-9e-3
}
SCALE_BOUNDS = (1e-4, 1e2)                   # lengthscale and outputscale, App. C.5


def make_prior(spec):
    kind, a, b = spec
    return GammaPrior(a, b) if kind == "gamma" else LogNormalPrior(a, b)


def se_ard_kernel(d, lengthscale_prior, outputscale_prior):
    """Squared-exponential ARD kernel with an output scale, bounded as in App. C.5."""
    return ScaleKernel(
        RBFKernel(ard_num_dims=d, lengthscale_prior=lengthscale_prior,
                  lengthscale_constraint=Interval(*SCALE_BOUNDS)),
        outputscale_prior=outputscale_prior, outputscale_constraint=Interval(*SCALE_BOUNDS))


def fit_map(model, *, n_starts=5, seed=0):
    """MAP fit: best of n_starts. Start 0 from the initial values, the rest drawn from the priors (App. C).
    Returns the best log marginal likelihood + log prior, per training point."""
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    best, best_state = -math.inf, None
    for s in range(n_starts):
        torch.manual_seed(seed + s)
        if s > 0:
            sample_all_priors(model)
        fit_gpytorch_mll(mll)
        model.train()
        with torch.no_grad():
            value = mll(model(*model.train_inputs), model.train_targets).item()
        if value > best:
            best, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return best


def fit_task_gp(X, y, *, noise_bounds=(1e-6, 1.0), n_starts=5, seed=0):
    """MAP fit of one SE-ARD GP to one task (App. C.2 step 1, priors App. C.5).

    X: (n, d) float64 in the shared unit cube. y: (n,) raw units, standardized inside by Standardize,
    so posterior() returns raw units (D3a). noise_bounds are in this task's standardized units (D2a).
    Returns the model in eval mode with every parameter frozen (Assumption 3).
    """
    P = PAPER_SOURCE_PRIORS
    likelihood = GaussianLikelihood(noise_prior=make_prior(P["noise"]),
                                    noise_constraint=Interval(*noise_bounds))
    model = SingleTaskGP(X, y.unsqueeze(-1), likelihood=likelihood,
                         covar_module=se_ard_kernel(X.shape[-1], make_prior(P["lengthscale"]),
                                                    make_prior(P["outputscale"])),
                         outcome_transform=Standardize(m=1))
    fit_map(model, n_starts=n_starts, seed=seed)
    model.requires_grad_(False)
    return model
