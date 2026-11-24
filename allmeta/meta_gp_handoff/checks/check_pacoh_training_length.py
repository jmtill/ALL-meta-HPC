"""PACOH-MAP: learned noise and calibration after 1500 vs 6000 iterations (runs check_pacoh.py setup first)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import warnings, torch, math, time
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
exec(open(Path(__file__).with_name("check_pacoh.py")).read().split('print("== SVGD direction checks")')[0])
from models.pacoh import PACOHTransfer
for it in (1500, 6000):
    m = PACOHTransfer(n_particles=1, n_iter=it, seed=0); t0=time.time(); m.meta_fit(sources)
    noise_raw_units = m.particles[0].noise.item() * m.y_scale**2
    res = []
    for k in (5, 20):
        r = []
        for f in targets:
            xt, yt = draw(f, k); yq = f(xq) + 0.1 * torch.randn(200, generator=g)
            m.condition(xt, yt); pr = m.predict(xq)
            r.append(((pr.mean - f(xq)).square().mean().sqrt().item(),
                      (0.5*torch.log(2*math.pi*pr.obs_var) + 0.5*(yq-pr.mean)**2/pr.obs_var).mean().item(),
                      ((yq-pr.mean).abs() <= 1.96*pr.obs_var.sqrt()).double().mean().item()))
        res.append((k, torch.tensor(r).mean(0)))
    print(f"iters {it:5d} ({time.time()-t0:.0f}s): ln Q* end {sum(m.history[-50:])/50:.1f}; learned noise var (raw) {noise_raw_units:.4f} (true 0.01) | "
          + " | ".join(f"k={k}: RMSE {v[0]:.3f} NLL {v[1]:.2f} cov {v[2]:.2f}" for k, v in res))
