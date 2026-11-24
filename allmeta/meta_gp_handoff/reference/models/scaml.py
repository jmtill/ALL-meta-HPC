"""ScaML-GP (Tighineanu et al., arXiv 2312.00742): target prior = weighted source posteriors (Thm 1, Eq. 6)."""
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator import to_linear_operator

from .base import Prediction
from .task_gp import fit_map, fit_task_gp, se_ard_kernel

# ScaML-GP App. C.5, target-task GP: LogNormal = (loc, scale = std of log).
TARGET_PRIORS = {"lengthscale": (0.5, 1.5), "outputscale": (-2.0, 3.0), "noise": (-8.0, 2.0)}


def pooled_standardization(y_sources, y_target):
    """(shift, scale): mean and std of y_t together with all y_1:M, raw units (App. C.1)."""
    y_all = torch.cat([*y_sources, y_target])
    return y_all.mean().item(), y_all.std().item()


def source_moments(sources, x, *, shift, scale):
    """Latent posterior of every source at x, in the target's pooled standardized units (Eq. 6, D3a).

    x: (n, d). Returns mu (M, n) = (mu_m(x) - shift) / scale and Sigma (M, n, n) = Sigma_m(x, x) / scale**2.
    Cross-covariances come for free: ExactGP stacks train and query inputs into one x.
    """
    with torch.no_grad():
        posts = [s.posterior(x) for s in sources]       # observation_noise=False: latent, no noise term
        mu = torch.stack([p.mean.squeeze(-1) for p in posts])
        sigma = torch.stack([p.distribution.covariance_matrix for p in posts])
    return (mu - shift) / scale, sigma / scale**2


class ScaMLGP(ExactGP, GPyTorchModel):
    """Target GP with the ScaML-GP prior (Eq. 6), conditioned on D_t by ExactGP (Eq. 7).
    theta_t (Eq. 9): k_t hyperparameters, weights w (M,) > 0, noise. Sources stay frozen (Assumption 3)."""
    _num_outputs = 1

    def __init__(self, train_X, train_Y, sources, *, shift, scale,
                 weight_prior=(1.0, 1.0), noise_bounds=(1e-6, 1.0)):
        likelihood = GaussianLikelihood(noise_prior=LogNormalPrior(*TARGET_PRIORS["noise"]),
                                        noise_constraint=Interval(*noise_bounds))
        super().__init__(train_X, train_Y, likelihood)
        self.sources = list(sources)                    # plain list: not submodules, never optimized
        self.shift, self.scale = shift, scale
        d = sources[0].train_inputs[0].shape[-1]
        self.covar_t = se_ard_kernel(d, LogNormalPrior(*TARGET_PRIORS["lengthscale"]),
                                     LogNormalPrior(*TARGET_PRIORS["outputscale"]))
        M = len(self.sources)
        self.register_parameter("raw_weights", torch.nn.Parameter(torch.zeros(M, dtype=torch.float64)))
        self.register_constraint("raw_weights", Positive())
        self.register_prior("weights_prior", GammaPrior(*weight_prior),
                            lambda m: m.weights, lambda m, v: m._set_weights(v))
        self.weights = 1.0 / M

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        self._set_weights(value)

    def _set_weights(self, value):
        value = torch.as_tensor(value, dtype=self.raw_weights.dtype).expand_as(self.raw_weights)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

    def forward(self, x):
        mu, sigma = source_moments(self.sources, x, shift=self.shift, scale=self.scale)
        w = self.weights
        mean = w @ mu                                                       # Eq. 6, mean
        covar = self.covar_t(x) + to_linear_operator(torch.einsum("m,mij->ij", w**2, sigma))  # Eq. 6, kernel
        return MultivariateNormal(mean, covar)


class ScaMLTransfer:
    """Phase 17 interface: meta_fit once, then condition and predict for each target and k."""

    def __init__(self, *, weight_prior=(1.0, 1.0), noise_bounds=(1e-6, 1.0), n_starts=5, seed=0):
        self.weight_prior, self.noise_bounds = weight_prior, noise_bounds
        self.n_starts, self.seed = n_starts, seed

    def meta_fit(self, sources):
        """sources: liquid -> (X (n_m, d), y (n_m,)). One GP per liquid, fitted once, reused for every k."""
        self.names = list(sources)
        self.y_sources = [y for _, y in sources.values()]
        self.sources = [fit_task_gp(X, y, noise_bounds=self.noise_bounds, seed=self.seed)
                        for X, y in sources.values()]

    def condition(self, X_t, y_t):
        """k >= 1: pooled standardization, MAP fit of theta_t (Eq. 9). k == 0: the prior of Eq. 6."""
        self.shift, self.scale = pooled_standardization(self.y_sources, y_t)
        self.k0 = len(y_t) == 0
        train = (None, None) if self.k0 else (X_t, (y_t - self.shift) / self.scale)
        self.model = ScaMLGP(*train, self.sources, shift=self.shift, scale=self.scale,
                             weight_prior=self.weight_prior, noise_bounds=self.noise_bounds)
        if self.k0:   # k = 0 rule: w = 1/M (set in __init__), k_t and noise at their prior medians
            self.model.covar_t.outputscale = torch.tensor(TARGET_PRIORS["outputscale"][0]).exp()
            self.model.covar_t.base_kernel.lengthscale = torch.tensor(TARGET_PRIORS["lengthscale"][0]).exp()
            self.model.likelihood.noise = torch.tensor(TARGET_PRIORS["noise"][0]).exp()
        else:
            fit_map(self.model, n_starts=self.n_starts, seed=self.seed)
        self.model.eval()

    def predict(self, X_q):
        """X_q: (q, d) -> Prediction in raw units."""
        with torch.no_grad():
            f = self.model.forward(X_q) if self.k0 else self.model(X_q)
            y = self.model.likelihood(f)
        s = self.scale
        return Prediction(mean=f.mean * s + self.shift, latent_var=f.variance * s**2,
                          obs_var=y.variance * s**2)
