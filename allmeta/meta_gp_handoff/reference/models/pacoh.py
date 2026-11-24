"""PACOH-GP (Rothfuss et al., ICML 2021): meta-learn a distribution over GP priors, each with a neural-network
mean and an SE kernel on neural-network features (App. C), by SVGD on the PAC-optimal hyper-posterior
(Prop. 1, Alg. 2). K = 1 particle gives PACOH-MAP."""
import math

import torch
from torch import nn
from torch.nn.functional import softplus

from .base import Prediction

JITTER = 1e-6


def mlp(d_in, d_out, hidden):
    """Fully connected tanh network: d_in -> hidden[0] -> ... -> hidden[-1] -> d_out."""
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class GPPrior(nn.Module):
    """One particle phi: the GP prior GP(m_phi(x), k_phi(x, x')) plus its noise variance.
    k_phi(x, x') = exp(-0.5 ||(F_phi(x) - F_phi(x')) / l||^2), outputscale fixed to 1 (data are standardized)."""

    def __init__(self, d, *, hidden=(32, 32, 32, 32), feature_dim=2):
        super().__init__()
        self.mean_net = mlp(d, 1, hidden)
        self.feature_net = mlp(d, feature_dim, hidden)
        self.raw_lengthscale = nn.Parameter(torch.zeros(feature_dim))
        self.raw_noise = nn.Parameter(torch.zeros(1))

    @property
    def noise(self):
        return softplus(self.raw_noise)

    def mean(self, x):
        return self.mean_net(x).squeeze(-1)

    def kernel(self, x1, x2):
        lengthscale = softplus(self.raw_lengthscale)
        z1, z2 = self.feature_net(x1) / lengthscale, self.feature_net(x2) / lengthscale
        return torch.exp(-0.5 * torch.cdist(z1, z2).square())

    def mll(self, X, y):
        """ln p(y | X, phi), Eq. 48 (a sum over the task's points, not an average)."""
        K = self.kernel(X, X) + (self.noise + JITTER) * torch.eye(len(X), dtype=X.dtype)
        L = torch.linalg.cholesky(K)
        r = (y - self.mean(X)).unsqueeze(-1)
        alpha = torch.cholesky_solve(r, L)
        return -0.5 * (r * alpha).sum() - L.diagonal().log().sum() - 0.5 * len(y) * math.log(2 * math.pi)

    def log_hyperprior(self, weight_std, bias_std):
        """ln P(phi): independent Gaussians. Network weights N(0, weight_std^2), biases N(0, bias_std^2);
        raw lengthscale N(0, 1); raw noise N(-1, 1) (the authors' code; the paper only says N(0, sigma_P^2 I))."""
        def normal(v, loc, std):
            return (-0.5 * ((v - loc) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)).sum()
        lp = normal(self.raw_lengthscale, 0.0, 1.0) + normal(self.raw_noise, -1.0, 1.0)
        for name, p in [*self.mean_net.named_parameters(), *self.feature_net.named_parameters()]:
            lp = lp + normal(p, 0.0, weight_std if "weight" in name else bias_std)
        return lp

    @torch.no_grad()
    def sample_hyperprior_(self, weight_std, bias_std):
        """Set phi to a draw from P (Alg. 2, first line). The trailing _ means: changes the module in place."""
        self.raw_lengthscale.normal_(0.0, 1.0)
        self.raw_noise.normal_(-1.0, 1.0)
        for name, p in [*self.mean_net.named_parameters(), *self.feature_net.named_parameters()]:
            p.normal_(0.0, weight_std if "weight" in name else bias_std)

    def predict(self, X, y, Xq):
        """GP posterior at Xq given (X, y): mean (q,), latent variance (q,). With no data: the prior."""
        if len(y) == 0:
            return self.mean(Xq), torch.ones(len(Xq), dtype=Xq.dtype)
        K = self.kernel(X, X) + (self.noise + JITTER) * torch.eye(len(X), dtype=X.dtype)
        L = torch.linalg.cholesky(K)
        Ks = self.kernel(X, Xq)                                             # (k, q)
        alpha = torch.cholesky_solve((y - self.mean(X)).unsqueeze(-1), L)
        mean = self.mean(Xq) + (Ks.T @ alpha).squeeze(-1)
        v = torch.linalg.solve_triangular(L, Ks, upper=False)
        return mean, (1.0 - v.square().sum(0)).clamp_min(0.0)              # k(x, x) = 1


def svgd_direction(params, scores, bandwidth=None):
    """SVGD update direction (Eq. 7, App. C.1) for K particles.
    params, scores: (K, P) flattened phi_k and their scores grad ln Q*(phi_k). Returns (K, P).
    RBF kernel k(a, b) = exp(-||a - b||^2 / (2 h^2)); bandwidth None -> median heuristic (authors' code)."""
    K = len(params)
    if K == 1:                                     # PACOH-MAP: no repulsion, plain gradient ascent on ln Q*
        return scores
    d2 = torch.cdist(params, params).square()
    h2 = bandwidth**2 if bandwidth is not None else d2.median() / (2 * math.log(K + 1))
    k = torch.exp(-d2 / (2 * h2 + 1e-8))
    repulsion = (params * k.sum(1, keepdim=True) - k @ params) / (h2 + 1e-8)   # sum_l grad_{phi_l} k(phi_l, phi_k)
    return (k @ scores + repulsion) / K


def _flat_grad(module):
    return torch.cat([p.grad.flatten() for p in module.parameters()])


def _set_flat_grad(module, flat):
    i = 0
    for p in module.parameters():
        p.grad = flat[i:i + p.numel()].view_as(p).clone()
        i += p.numel()


class PACOHTransfer:
    """Phase 17 interface: meta_fit once on the source liquids, then condition and predict per target and k.
    Nothing is refitted on the target: each particle's GP prior is conditioned on the context (App. C.2)."""

    def __init__(self, *, n_particles=10, hidden=(32, 32, 32, 32), feature_dim=2, weight_std=0.5, bias_std=3.0,
                 prior_factor=0.1, lr=1e-3, lr_decay=0.98, n_iter=6000, task_batch_size=None, bandwidth=None, seed=0):
        self.n_particles, self.hidden, self.feature_dim = n_particles, hidden, feature_dim
        self.weight_std, self.bias_std, self.prior_factor = weight_std, bias_std, prior_factor
        self.lr, self.lr_decay, self.n_iter = lr, lr_decay, n_iter
        self.task_batch_size, self.bandwidth, self.seed = task_batch_size, bandwidth, seed

    def meta_fit(self, sources):
        """sources: liquid -> (X (n_m, d), y (n_m,)). Normalization pooled over all sources (authors' code)."""
        X_all = torch.cat([X for X, _ in sources.values()])
        y_all = torch.cat([y for _, y in sources.values()])
        self.x_shift, self.x_scale = X_all.mean(0), X_all.std(0).clamp_min(1e-8)
        self.y_shift, self.y_scale = y_all.mean().item(), y_all.std().item()
        tasks = [self._normalize(X, y) for X, y in sources.values()]

        torch.manual_seed(self.seed)
        d = X_all.shape[-1]
        self.particles = []
        for _ in range(self.n_particles):
            particle = GPPrior(d, hidden=self.hidden, feature_dim=self.feature_dim).double()
            particle.sample_hyperprior_(self.weight_std, self.bias_std)
            self.particles.append(particle)

        params = [p for particle in self.particles for p in particle.parameters()]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=self.lr_decay)
        n = len(tasks)
        n_batch = n if self.task_batch_size is None else min(self.task_batch_size, n)
        self.history = []
        for _ in range(self.n_iter):
            batch = [tasks[i] for i in torch.randperm(n)[:n_batch].tolist()]
            scores, log_q = [], []
            for particle in self.particles:
                particle.zero_grad()
                data_term = (n / n_batch) * sum(particle.mll(X, y) / (len(y) + 1) for X, y in batch)
                objective = self.prior_factor * particle.log_hyperprior(self.weight_std, self.bias_std) + data_term
                objective.backward()                        # score: grad ln Q*(phi_k), Alg. 2
                scores.append(_flat_grad(particle))
                log_q.append(objective.item())
            flat = torch.stack([torch.cat([p.detach().flatten() for p in particle.parameters()])
                                for particle in self.particles])
            direction = svgd_direction(flat, torch.stack(scores), self.bandwidth)
            for particle, g in zip(self.particles, direction):
                _set_flat_grad(particle, -g)                # optimizers minimize; SVGD ascends
            optimizer.step()
            scheduler.step()
            self.history.append(sum(log_q) / len(log_q))

    def _normalize(self, X, y=None):
        Xn = (X - self.x_shift) / self.x_scale
        return Xn if y is None else (Xn, (y - self.y_shift) / self.y_scale)

    def condition(self, X_t, y_t):
        """Store the target's context; k = 0 is allowed (prediction is then the meta-learned prior)."""
        self.X_t, self.y_t = self._normalize(X_t, y_t)

    def predict(self, X_q):
        """Equal-weight mixture of the K particles' predictive distributions (Eq. 49), in raw units."""
        Xq = self._normalize(X_q)
        with torch.no_grad():
            out = [particle.predict(self.X_t, self.y_t, Xq) for particle in self.particles]
            means = torch.stack([m for m, _ in out])                                    # (K, q)
            latent = torch.stack([v for _, v in out])                                   # (K, q)
            noise = torch.stack([particle.noise for particle in self.particles])        # (K, 1)
        spread = means.var(0, unbiased=False)                    # disagreement between particles
        s = self.y_scale
        return Prediction(mean=means.mean(0) * s + self.y_shift,
                          latent_var=(latent.mean(0) + spread) * s**2,
                          obs_var=((latent + noise).mean(0) + spread) * s**2)
