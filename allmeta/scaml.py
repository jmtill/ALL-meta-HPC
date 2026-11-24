"""ScaML-GP (Tighineanu et al., arXiv 2312.00742): target prior = weighted source posteriors (Thm 1, Eq. 6)."""
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator import to_linear_operator

from .liquid_features import descriptor_weights, tune_bandwidth
from .base import Prediction
from .task_gp import NOISE_PRIOR, fit_map, fit_task_gp, se_ard_kernel

TARGET_PRIORS = {"lengthscale": (0.5, 1.5), "outputscale": (-2.0, 3.0), "noise": NOISE_PRIOR[1:]}   # paper noise: (-8, 2)

def pooled_standardization(y_sources: list[torch.Tensor], y_target: torch.Tensor)-> tuple[float,float]:
    y_all = torch.cat([*y_sources, y_target])
    return y_all.mean().item(), y_all.std().item()


def source_moments(sources, x, *, shift: float, scale:float):

    with torch.no_grad():
        posts = [s.posterior(x) for s in sources]
        mu = torch.stack([p.mean.squeeze(-1) for p in posts])
        sigma = torch.stack([p.distribution.covariance_matrix for p in posts])
    return (mu - shift)/scale, sigma / scale**2

class ScaMLGP(ExactGP, GPyTorchModel):
    """Target GP with the ScaML-GP prior (Eq. 6), conditioned on D_t by ExactGP (Eq. 7).

    Fitted on the target, theta_t (Eq. 9):
      covar_t   ScaleKernel(RBFKernel(ard_num_dims=d)); log l ~ N(0.5, 1.5), log o ~ N(-2, 3); bounds SCALE_BOUNDS
      weights   (M,) > 0; prior Gamma(1, 1) as in the paper [D1c], or Gamma(a, a*M), mean 1/M [D1b variant]
      noise     GaussianLikelihood, log n ~ N(-4, 1.5) (task_gp.NOISE_PRIOR; paper N(-8, 2)), noise_bounds widened [D2a]
    Sources are stored frozen and are never in the optimizer (Assumption 3).
    """
    _num_outputs = 1

    def __init__(self, train_X, train_Y, sources, *, shift: float, scale: float,
                 weight_prior=(1.0, 1.0), init_weights=None, noise_bounds=(1e-6, 1.0)):
        """weight_prior: Gamma (concentration, rate), floats or (M,) tensors; None = fixed weights (never fitted).
        init_weights: (M,) starting weights; None = 1/M."""
        likelihood = GaussianLikelihood(noise_prior=LogNormalPrior(*TARGET_PRIORS['noise']),
                                        noise_constraint=Interval(*noise_bounds))
        super().__init__(train_X, train_Y, likelihood)
        self.sources = list(sources)
        self.shift, self.scale = shift, scale
        d = sources[0].train_inputs[0].shape[-1]
        self.covar_t = se_ard_kernel(d, LogNormalPrior(*TARGET_PRIORS['lengthscale']),
                                     LogNormalPrior(*TARGET_PRIORS['outputscale']))
        M = len(self.sources)
        self.register_parameter("raw_weights", torch.nn.Parameter(torch.zeros(M, dtype=torch.float64)))
        self.register_constraint('raw_weights', Positive())
        if weight_prior is None:
            self.raw_weights.requires_grad_(False)          # fixed weights: no prior, not in the optimizer
        else:
            self.register_prior('weights_prior', GammaPrior(*weight_prior),
                                lambda m: m.weights, lambda m, v: m._set_weights(v))
        self.weights = 1.0 / M if init_weights is None else init_weights

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self,value):
        self._set_weights(value)

    def _set_weights(self, value):
        value = torch.as_tensor(value, dtype=self.raw_weights.dtype).expand_as(self.raw_weights)
        self.initialize(raw_weights = self.raw_weights_constraint.inverse_transform(value))
        return None

    def forward(self, x):
        mu, sigma = source_moments(self.sources, x, shift=self.shift, scale=self.scale)
        w = self.weights
        mean = w @ mu
        covar = self.covar_t(x) + to_linear_operator(torch.einsum("m,mij->ij", w**2, sigma))
        return MultivariateNormal(mean, covar)

class ScaMLTransfer:
    """Phase 17 interface: meta_fit once, then condition and predict for each target and k.

    weight_source: where the weights w_m come from.
      "data"             paper: fitted on the target's trials (MAP, Eq. 9) under weight_prior
      "descriptor_fixed" w_m = descriptor similarity to the target (Bonilla et al. 2007 gating idea); never fitted
      "descriptor_prior" fitted, under Gamma priors whose means are the descriptor similarities
    bandwidth (descriptor weight sources): "tuned" = leave-one-source-out (default), None = median heuristic, or a float.
    """

    def __init__(self, *, weight_source="data", weight_prior=(1.0, 1.0), descriptor_concentration=2.0,
                 bandwidth="tuned", noise_bounds=(1e-6, 1.0), n_starts=5, seed=0):
        self.weight_source, self.weight_prior, self.bandwidth = weight_source, weight_prior, bandwidth
        self.descriptor_concentration, self.noise_bounds = descriptor_concentration, noise_bounds
        self.n_starts, self.seed = n_starts, seed

    def meta_fit(self, sources: dict[str, tuple[torch.Tensor, torch.Tensor]], descriptors=None) -> None:
        """sources: liquid -> (X (n_m, d), y (n_m,)). One GP per liquid, fitted once, reused for every k.
        descriptors: liquid -> standardized z (p,), needed for the descriptor weight sources."""
        self.names = list(sources)
        self.Z_sources = None if descriptors is None else torch.stack([descriptors[n] for n in self.names])
        self.y_sources = [y for _, y in sources.values()]
        self.sources = [fit_task_gp(X, y, noise_bounds=self.noise_bounds, seed=self.seed)
                        for X, y in sources.values()]
        self.h = None if self.weight_source == "data" else self.bandwidth   # no bandwidth for data weights
        self.bandwidth_scores = None
        if self.weight_source != "data" and self.bandwidth == "tuned":
            with torch.no_grad():
                cross = [torch.stack([gp.posterior(X).mean.squeeze(-1) for gp in self.sources])
                         for X, _ in sources.values()]
            self.h, self.bandwidth_scores = tune_bandwidth(self.Z_sources, cross, self.y_sources)

    def condition(self, X_t: torch.Tensor, y_t: torch.Tensor, z_t=None) -> None:
        """k >= 1: pooled standardization, MAP fit of theta_t (Eq. 9). k == 0: the prior of Eq. 6.
        z_t: the target's standardized descriptors (descriptor weight sources only)."""
        self.shift, self.scale = pooled_standardization(self.y_sources, y_t)
        self.k0 = len(y_t) == 0
        prior, init = self.weight_prior, None
        if self.weight_source != "data":
            init = descriptor_weights(z_t, self.Z_sources, bandwidth=self.h).clamp_min(1e-6)
            a = self.descriptor_concentration
            prior = None if self.weight_source == "descriptor_fixed" else (torch.full_like(init, a), a / init)
        train = (None, None) if self.k0 else (X_t, (y_t - self.shift) / self.scale)
        self.model = ScaMLGP(*train, self.sources, shift=self.shift, scale=self.scale,
                             weight_prior=prior, init_weights=init, noise_bounds=self.noise_bounds)
        if self.k0:   # k = 0 rule: w = 1/M or descriptor weights (set in __init__); k_t, noise at prior medians
            self.model.covar_t.outputscale = torch.tensor(TARGET_PRIORS['outputscale'][0]).exp()
            self.model.covar_t.base_kernel.lengthscale = torch.tensor(TARGET_PRIORS['lengthscale'][0]).exp()
            self.model.likelihood.noise = torch.tensor(TARGET_PRIORS['noise'][0]).exp()
        else:
            fit_map(self.model, n_starts=self.n_starts, seed=self.seed)
        self.model.eval()

    def predict(self, X_q: torch.Tensor):
        """X_q: (q, d) -> Prediction in raw units."""
        with torch.no_grad():
            f = self.model.forward(X_q) if self.k0 else self.model(X_q)
            y = self.model.likelihood(f)
        s = self.scale
        return Prediction(mean=f.mean * s + self.shift,
                          latent_var=f.variance * s**2,
                          obs_var=y.variance * s**2)

