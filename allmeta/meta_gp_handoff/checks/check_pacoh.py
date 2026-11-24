"""PACOH-GP checks: SVGD limits, then PACOH-MAP and PACOH-SVGD vs single-task GP on the paper's Sinusoid environment (App. E.1).
Meta-fit uses n_iter=1500 here (K=10 takes ~6 min); see check_pacoh_training_length.py for 1500 vs 6000."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

import time, warnings, torch, math
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
from models.pacoh import PACOHTransfer, svgd_direction
from models.task_gp import fit_task_gp
g = torch.Generator().manual_seed(0)
def task():  # PACOH App. E.1 Sinusoid environment (Eq. 57-59)
    a = 0.7 + 0.6 * torch.rand(1, generator=g); b = 0.1 * torch.randn(1, generator=g)
    c = 5 + 0.1 * torch.randn(1, generator=g); beta = 0.5 + 0.2 * torch.randn(1, generator=g)
    return lambda x: beta * x[:, 0] + a * torch.sin(1.5 * (x[:, 0] - b)) + c
def draw(f, n): x = 10 * torch.rand(n, 1, generator=g) - 5; return x, f(x) + 0.1 * torch.randn(n, generator=g)
sources = {i: draw(task(), 60) for i in range(15)}
targets = [task() for _ in range(5)]
xq = torch.linspace(-5, 5, 200).unsqueeze(-1)

print("== SVGD direction checks")
p = torch.randn(4, 3); s = torch.randn(4, 3)
print("  K=1 returns the score:", torch.equal(svgd_direction(p[:1], s[:1]), s[:1]))
same = svgd_direction(torch.zeros(3, 2) + 0.0, torch.zeros(3, 2), bandwidth=1.0)
print("  identical particles, zero score -> zero move:", torch.allclose(same, torch.zeros(3, 2)))
two = svgd_direction(torch.tensor([[0.0], [0.1]]), torch.zeros(2, 1), bandwidth=1.0)
print("  zero score, nearby particles pushed apart:", (two[0] < 0).item() and (two[1] > 0).item())

def nll(pred, y):  # Gaussian with the mixture's moments (exact for K = 1)
    return (0.5 * torch.log(2 * math.pi * pred.obs_var) + 0.5 * (y - pred.mean) ** 2 / pred.obs_var).mean().item()
def cover(pred, y): return ((y - pred.mean).abs() <= 1.96 * pred.obs_var.sqrt()).double().mean().item()

models = {}
for K in (1, 10):
    m = PACOHTransfer(n_particles=K, n_iter=1500, seed=0)
    t0 = time.time(); m.meta_fit(sources); dt = time.time() - t0
    h = m.history
    print(f"== PACOH K={K}: meta_fit {dt:.0f}s ({1000*dt/len(h):.0f} ms/iter); mean ln Q*: start {h[0]:.1f}, "
          f"iter 500 {h[499]:.1f}, end {sum(h[-50:])/50:.1f}")
    models[K] = m
print("== target tasks, averaged over 5 targets: RMSE / NLL / 95% coverage of new observations")
for k in (0, 3, 5, 10, 20):
    row = {"PACOH-MAP": [], "PACOH-SVGD": [], "single-task": []}
    for f in targets:
        xt, yt = draw(f, k); yq = f(xq) + 0.1 * torch.randn(200, generator=g)
        for name, K in (("PACOH-MAP", 1), ("PACOH-SVGD", 10)):
            models[K].condition(xt, yt); pr = models[K].predict(xq)
            row[name].append(((pr.mean - f(xq)).square().mean().sqrt().item(), nll(pr, yq), cover(pr, yq)))
        if k >= 3:
            st = fit_task_gp((xt + 5) / 10, yt, n_starts=3); post = st.posterior((xq + 5) / 10, observation_noise=True)
            from models.base import Prediction
            pr = Prediction(post.mean.squeeze(-1), post.variance.squeeze(-1), post.variance.squeeze(-1))
            row["single-task"].append((((pr.mean - f(xq)).square().mean().sqrt().item()), nll(pr, yq), cover(pr, yq)))
    cells = []
    for name, v in row.items():
        if v: r = torch.tensor(v).mean(0); cells.append(f"{name} {r[0]:.3f}/{r[1]:.2f}/{r[2]:.2f}")
    print(f"  k={k:2d}  " + " | ".join(cells))
