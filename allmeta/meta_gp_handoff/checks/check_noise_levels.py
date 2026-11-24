"""Check noise.py (N1-N3) and log_normal.py: synthetic known noise, then toluene optimization trials."""
import math, warnings, time, pandas as pd, torch
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
from allmeta.trial_stats import TrialStats, from_replicates
from allmeta.noise import fit_noise_model
from allmeta.log_normal import fit_log_normal_gp

print("== 1. synthetic: sd(x) = 0.06 + 0.32 x, 21 x 6, six seeds; mean |log error| of noise SD and mean RMSE")
f = lambda X: torch.sin(2 * math.pi * X[:, 0]) + 0.25 * torch.sin(4 * math.pi * X[:, 0])
sdf = lambda X: 0.06 + 0.32 * X[:, 0]
x = torch.cat([torch.linspace(0.03, 0.43, 11), torch.linspace(0.62, 0.97, 10)]).unsqueeze(-1)
g = torch.linspace(0, 1, 101).unsqueeze(-1)
for level in ["fixed", "smoothed", "hetgp"]:
    e, r, t0 = [], [], time.time()
    for seed in range(6):
        torch.manual_seed(seed)
        runs = f(x)[:, None] + sdf(x)[:, None] * torch.randn(21, 6)
        s = TrialStats(x, runs.mean(1), runs.var(1), torch.full((21,), 6.0), list(range(21)))
        m = fit_noise_model(s, level)
        e.append((m.noise(g, 1).sqrt().log() - sdf(g).log()).abs().mean().item())
        r.append((m.predict(g).mean - f(g)).square().mean().sqrt().item())
    print(f"  {level:9s} noise-sd log error {sum(e)/6:.3f}   mean RMSE {sum(r)/6:.3f}   {(time.time()-t0)/6:.1f} s/fit")

print("== 2. toluene optimization trials, 5-fold by trial: signed error (N1-N3) and |e| (log-normal)")
from pathlib import Path
T = pd.read_csv(Path(__file__).resolve().parents[4] / "data/campaign_ingest/transfers.csv")
inputs = ["asp_rate", "disp_rate", "pre_airgap", "post_airgap", "volume_offset"]
tol = T[(T.liquid.str.lower() == "toluene") & (T.campaign.str.contains("optim", case=False))]
for value in ["signed", "abs"]:
    s = from_replicates(tol, inputs, value=value)
    lo, hi = s.X.min(0).values, s.X.max(0).values
    s = TrialStats((s.X - lo) / (hi - lo).clamp_min(1e-12), s.mean, s.var, s.count, s.ids)
    folds = torch.arange(len(s.mean)) % 5
    models = ["fixed", "smoothed", "hetgp"] if value == "signed" else ["median", "mean"]
    for name in models:
        z2, cover, rmse = [], [], []
        for k in range(5):
            tr, te = torch.where(folds != k)[0], torch.where(folds == k)[0]
            m = fit_noise_model(s.subset(tr), name) if value == "signed" else fit_log_normal_gp(s.subset(tr), parameterization=name)
            p = m.predict(s.X[te], n_replicates=s.count[te])
            y = s.mean[te]
            if value == "signed":
                z = (y - p.mean) / p.obs_var.sqrt(); z2.append(z.square()); cover.append(z.abs() < 1.96)
            else:
                cover.append((y > p.quantiles[0.025]) & (y < p.quantiles[0.975]))
            rmse.append((y - p.mean).square())
        cat = lambda L: torch.cat(L).double()
        extra = f"mean z^2 {cat(z2).mean():.2f}  " if z2 else ""
        print(f"  {value:6s} {name:9s} n={len(s.mean)}  RMSE {cat(rmse).mean().sqrt():.4f}  {extra}95% coverage {cat(cover).mean():.2f}")
