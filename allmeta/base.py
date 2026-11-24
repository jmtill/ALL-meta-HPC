from dataclasses import dataclass
import torch


@dataclass
class Prediction:
    """Held-out predictions in raw units. Quantiles are for the non-Gaussian (log-normal) models."""
    mean: torch.Tensor          # (q,)
    latent_var: torch.Tensor    # (q,) variance of f(x)
    obs_var: torch.Tensor       # (q,) variance of a new observation y(x)
    quantiles: dict[float, torch.Tensor] | None = None