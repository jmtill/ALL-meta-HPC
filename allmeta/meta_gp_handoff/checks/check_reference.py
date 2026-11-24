"""Steps 1-5 of the ScaML-GP build: fit_task_gp, source_moments, forward limits, weight recovery, k = 0."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))  # point at your own files after fixing them
REPO = Path(__file__).resolve().parents[4]

import warnings, torch, math
from scipy.stats import qmc
from models.task_gp import fit_task_gp, PAPER_SOURCE_PRIORS
from models.scaml import ScaMLGP, ScaMLTransfer, source_moments, pooled_standardization
from botorch.optim.utils import sample_all_priors
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
gen = torch.Generator().manual_seed(0)
sob = lambda n, d, s: torch.tensor(qmc.Sobol(d, seed=s).random(n))
f_true = lambda X: torch.sin(3 * X[:, 0] + X[:, 1])
X = sob(64, 2, 1); y = f_true(X) + 0.05 * torch.randn(64, generator=gen)
G = sob(512, 2, 99)

print("== step 1: fit_task_gp")
m = fit_task_gp(X, y)
mu = m.posterior(G).mean.squeeze(-1)
print("  RMSE vs truth          ", round(((mu - f_true(G))**2).mean().sqrt().item(), 4))
raw_noise = m.likelihood.noise.item() * m.outcome_transform.stdvs.item()**2
print("  noise var (raw)        ", round(raw_noise, 5), "(true 0.0025)")
m2 = fit_task_gp(X, y + 100)
print("  shift +100 -> mean diff", round((m2.posterior(G).mean - m.posterior(G).mean).mean().item(), 3))
print("  any requires_grad      ", any(p.requires_grad for p in m.parameters()))
m1 = fit_task_gp(X, y, n_starts=1)
from gpytorch.mlls import ExactMarginalLogLikelihood
def val(model):
    model.train(); mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.no_grad(): v = mll(model(*model.train_inputs), model.train_targets).item()
    model.eval(); return v
print("  MAP objective 5 starts vs 1:", round(val(m), 4), round(val(m1), 4))

print("== prior samples outside bounds?")
torch.manual_seed(3); bad = 0
for i in range(200):
    sample_all_priors(m1)
    o = m1.covar_module.outputscale.item()
    if not math.isfinite(o) or o >= 1e2 - 1e-9: bad += 1
print("  outputscale samples at/over bound or non-finite:", bad, "/200")

print("== step 2: source_moments")
srcs = []
for j in range(3):
    Xs = sob(48, 2, 10 + j); c = j * 1.3
    srcs.append(fit_task_gp(Xs, torch.sin(3 * Xs[:, 0] + Xs[:, 1] + c) + 0.05 * torch.randn(48, generator=gen)))
x = G[:5]; shift, scale = 0.3, 2.0
mu_s, S = source_moments(srcs, x, shift=shift, scale=scale)
p = srcs[1].posterior(x)
print("  mean match             ", torch.allclose(mu_s[1] * scale + shift, p.mean.squeeze(-1)))
print("  cov match              ", torch.allclose(S[1] * scale**2, p.distribution.covariance_matrix))
pn = srcs[1].posterior(x, observation_noise=True)
print("  diag(S) < obs variance (no noise term):", bool((S[1].diag() * scale**2 < pn.variance.squeeze(-1)).all()))

print("== step 3: ScaMLGP.forward limits")
Xt = sob(6, 2, 5); yt = torch.randn(6, generator=gen)
g = ScaMLGP(Xt, yt, srcs, shift=shift, scale=scale)
g.weights = 0.0
with torch.no_grad(): d0 = (g.forward(x).covariance_matrix - g.covar_t(x).to_dense()).abs().max().item()
print("  w=0: covar minus k_t   ", d0, " mean max", g.forward(x).mean.abs().max().item())
g.weights = torch.tensor([1.0, 0.0, 0.0]); g.covar_t.outputscale = 1e-4
with torch.no_grad(): out = g.forward(x)
print("  w=e1, tiny k_t: mean err", (out.mean - mu_s[0]).abs().max().item(),
      " cov err", round((out.covariance_matrix - S[0]).abs().max().item(), 6))

print("== step 4: weight recovery, target = 0.7 f1 + small residual")
target = lambda X: 0.7 * torch.sin(3 * X[:, 0] + X[:, 1] + 1.3) + 0.05 * torch.cos(4 * X[:, 1])
srcs_data = {}
for j in range(3):
    Xs = sob(48, 2, 10 + j); c = j * 1.3
    srcs_data[f"s{j}"] = (Xs, torch.sin(3 * Xs[:, 0] + Xs[:, 1] + c) + 0.05 * torch.randn(48, generator=gen))
st = ScaMLTransfer(); st.meta_fit(srcs_data)
for k in (5, 10, 20):
    Xt = sob(k, 2, 42); yt = target(Xt) + 0.02 * torch.randn(k, generator=gen)
    st.condition(Xt, yt)
    w = st.model.weights.detach() * st.scale  # raw-unit weight: mu_m enters as w*(mu-shift)/scale... see note
    pr = st.predict(G)
    print(f"  k={k:2d} weights {st.model.weights.detach().numpy().round(3)}  RMSE {((pr.mean - target(G))**2).mean().sqrt().item():.4f}")

print("== step 5: k = 0")
st.condition(torch.empty(0, 2), torch.empty(0))
pr = st.predict(G)
avg = torch.stack([s.posterior(G).mean.squeeze(-1) for s in st.sources]).mean(0)
print("  k=0 mean vs (average of sources, corrected for shift):",
      (pr.mean - (avg - st.shift) / 3 - st.shift).abs().max().item())
print("  k=0 mean vs plain average of sources:", round((pr.mean - avg).abs().max().item(), 4))
print("  shapes", pr.mean.shape, pr.latent_var.shape, "obs>=latent", bool((pr.obs_var >= pr.latent_var).all()))
