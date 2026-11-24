"""Scores for one set of query predictions, in the space the outcome is modelled in (raw for "signed", log for
"log_sd" / "log_abs"). Named gp_metrics so it does not collide with the older allmeta/scoring.py.

  rmse, mae     point error of the predicted mean
  nlpd          mean Gaussian negative log predictive density of the observed trial value, using obs_var
  z2            mean squared standardized residual (1 if calibrated)
  cover95       fraction inside mean +- 1.96 sqrt(obs_var)
  crps          Gaussian CRPS (closed form)
  spearman      rank correlation of predicted vs observed (ranking quality); NaN if the prediction is flat
  regret        observed value at the predicted best query trial minus the best observed (lower is better;
                "signed" ranks by |y|, the log outcomes by y)"""
import math

import torch
from scipy.stats import spearmanr

METRICS = ("rmse", "mae", "nlpd", "z2", "cover95", "crps", "spearman", "regret")


def gaussian_scores(y, mean, var, *, outcome):
    var = var.clamp_min(1e-12)
    r, sd = y - mean, var.sqrt()
    z = r / sd
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    crps = sd * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    key = (lambda v: v.abs()) if outcome == "signed" else (lambda v: v)
    best = torch.argmin(key(mean))
    return {"rmse": r.square().mean().sqrt().item(), "mae": r.abs().mean().item(),
            "nlpd": (0.5 * torch.log(2 * math.pi * var) + 0.5 * z.square()).mean().item(),
            "z2": z.square().mean().item(), "cover95": (z.abs() < 1.96).double().mean().item(),
            "crps": crps.mean().item(),
            "spearman": float(spearmanr(mean.numpy(), y.numpy()).statistic) if mean.std() > 0 else math.nan,
            "regret": (key(y)[best] - key(y).min()).item()}
