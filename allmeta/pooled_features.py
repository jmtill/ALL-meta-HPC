"""One GP over every source liquid's trials on the joint space (x, z): k((x, z), (x', z')) = o · k_x(x, x') · k_z(z, z')
(Bonilla, Agakov & Williams, AISTATS 2007, "combined" method; Krause & Ong, NeurIPS 2011). Fitted once on the
sources; the target's trials only condition it (refit=True: refit with them included). use_descriptors=False drops z:
every liquid pooled as if it were the same liquid, the descriptor-free counterpart."""
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior

from .base import Prediction
from .task_gp import PAPER_PRIORS, SCALE_BOUNDS, fit_map, make_prior

DESCRIPTOR_LENGTHSCALE_PRIOR = (0.0, 1.0)    # LogNormal(loc, scale): median 1, since z is standardized


class PooledFeaturesGP:
    """Phase 17 interface: meta_fit on the sources, then condition (no refit) and predict per target and k."""

    def __init__(self, *, use_descriptors=True, refit=False, noise_bounds=(1e-6, 1.0), n_starts=3, seed=0):
        self.use_descriptors, self.refit, self.noise_bounds = use_descriptors, refit, noise_bounds
        self.n_starts, self.seed = n_starts, seed

    def _inputs(self, X, z):
        """(n, d) -> (n, d + p): every row of one liquid gets that liquid's descriptors appended."""
        return torch.cat([X, z.unsqueeze(0).expand(len(X), -1)], -1) if self.use_descriptors else X

    def meta_fit(self, sources, descriptors=None):
        """sources: liquid -> (X (n_m, d), y (n_m,)); descriptors: liquid -> standardized z (p,)."""
        self.sources, self.descriptors = sources, descriptors
        self.model = self._fit(sources, descriptors)

    def _fit(self, sources, descriptors):
        """MAP fit of one GP on every liquid's trials (frozen afterwards)."""
        X = torch.cat([self._inputs(Xm, descriptors[name] if self.use_descriptors else None)
                       for name, (Xm, _) in sources.items()])
        y = torch.cat([ym for _, ym in sources.values()])
        d = next(iter(sources.values()))[0].shape[-1]
        P = PAPER_PRIORS
        kernel = RBFKernel(ard_num_dims=d, active_dims=list(range(d)), lengthscale_prior=make_prior(P["lengthscale"]),
                           lengthscale_constraint=Interval(*SCALE_BOUNDS))
        if self.use_descriptors:
            p = X.shape[-1] - d
            kernel = kernel * RBFKernel(ard_num_dims=p, active_dims=list(range(d, d + p)),
                                        lengthscale_prior=LogNormalPrior(*DESCRIPTOR_LENGTHSCALE_PRIOR),
                                        lengthscale_constraint=Interval(*SCALE_BOUNDS))
        covar = ScaleKernel(kernel, outputscale_prior=make_prior(P["outputscale"]),
                            outputscale_constraint=Interval(*SCALE_BOUNDS))
        likelihood = GaussianLikelihood(noise_prior=make_prior(P["noise"]), noise_constraint=Interval(*self.noise_bounds))
        model = SingleTaskGP(X, y.unsqueeze(-1), likelihood=likelihood, covar_module=covar,
                             outcome_transform=Standardize(m=1))
        fit_map(model, n_starts=self.n_starts, seed=self.seed)
        return model.requires_grad_(False)

    def condition(self, X_t, y_t, z_t=None):
        """Add the target's k trials to the GP without refitting its hyperparameters (or, refit=True, refit
        the GP with them included as one more liquid). k = 0 is allowed."""
        self.z_t = z_t
        if len(y_t) == 0:
            self.conditioned = self.model
            return
        if self.refit:
            descriptors = {**self.descriptors, "_target": z_t} if self.use_descriptors else None
            self.conditioned = self._fit({**self.sources, "_target": (X_t, y_t)}, descriptors)
            return
        X = self._inputs(X_t, z_t)
        self.model.posterior(X)                        # BoTorch needs one eval call before conditioning
        self.conditioned = self.model.condition_on_observations(X, y_t.unsqueeze(-1))

    def predict(self, X_q):
        """X_q: (q, d) -> Prediction in raw units (BoTorch undoes the outcome standardization)."""
        X = self._inputs(X_q, self.z_t)
        with torch.no_grad():
            latent = self.conditioned.posterior(X)
            observed = self.conditioned.posterior(X, observation_noise=True)
        return Prediction(mean=latent.mean.squeeze(-1), latent_var=latent.variance.squeeze(-1),
                          obs_var=observed.variance.squeeze(-1))