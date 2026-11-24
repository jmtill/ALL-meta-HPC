"""Weight-prior variants (c) vs (b) on an unrelated target, and the toluene noise-bound check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))  # point at your own files after fixing them
REPO = Path(__file__).resolve().parents[4]

import warnings, torch, pandas as pd
from scipy.stats import qmc
from models.task_gp import fit_task_gp
from models.scaml import ScaMLTransfer
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
gen = torch.Generator().manual_seed(0)
sob = lambda n, d, s: torch.tensor(qmc.Sobol(d, seed=s).random(n))
M = 8
cs = torch.rand(M, 2, generator=gen); amps = 0.7 + 0.6 * torch.rand(M, generator=gen)
fs = [lambda X, c=c, a=a: a * torch.sin(3 * X.sum(1) + 5 * c[0]) + c[1] * X[:, 0] for c, a in zip(cs, amps)]
data = {}
for j, f in enumerate(fs):
    Xs = torch.rand(60, 2, generator=gen); data[j] = (Xs, f(Xs) + 0.05 * torch.randn(60, generator=gen))
G = torch.rand(400, 2, generator=gen)
models = {"(c) Gamma(1,1)": ScaMLTransfer(n_starts=3), "(b) Gamma(2,2M)": ScaMLTransfer(weight_prior=(2.0, 2.0 * M), n_starts=3)}
for m in models.values(): m.meta_fit(data)
print("== step 6: unrelated target, RMSE vs k")
g = lambda X: 2 * torch.cos(6 * X[:, 0]) * X[:, 1]
for k in (3, 5, 10, 20):
    Xt = sob(k, 2, 42); yt = g(Xt) + 0.05 * torch.randn(k, generator=gen)
    row = []
    for name, m in models.items():
        m.condition(Xt, yt); row.append(f"{name} {((m.predict(G).mean - g(G))**2).mean().sqrt().item():.3f} (max w {m.model.weights.max().item():.2f})")
    single = fit_task_gp(Xt, yt, n_starts=3)
    row.append(f"single {((single.posterior(G).mean.squeeze(-1) - g(G))**2).mean().sqrt().item():.3f}")
    print(f"  k={k:2d}  " + " | ".join(row))

print("== real data: toluene mean_error, noise vs upper bound 1.0 (standardized)")
d = pd.read_csv(REPO / "data/campaign_ingest/conditions.csv")
cols = ["volume", "asp_rate", "disp_rate", "pre_airgap", "post_airgap", "volume_offset"]
t = d[(d.liquid == "toluene") & (d.pipette_volume == 50)].dropna(subset=cols + ["mean_error"])
Xr = torch.tensor(t[cols].values); Xr = (Xr - Xr.min(0).values) / (Xr.max(0).values - Xr.min(0).values).clamp_min(1e-12)
yr = torch.tensor(t["mean_error"].values)
for nb in [(1e-8, 1e-2), (1e-6, 1.0)]:
    m = fit_task_gp(Xr, yr, noise_bounds=nb, n_starts=3)
    print(f"  bounds {nb}: n={len(yr)}  fitted noise (std units) {m.likelihood.noise.item():.4g}  lengthscales {m.covar_module.base_kernel.lengthscale.detach().numpy().round(2)}")
