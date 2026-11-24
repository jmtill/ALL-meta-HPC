"""Per-trial replicate statistics: the sufficient statistics (mean, sample variance, count) that every noise model
uses, and the three views of them a GP needs (Phase 16/17).

For Gaussian replicates, (mean, variance, count) carry all the information the replicates do, so models never need
the replicate table itself. Views:
  mean_and_noise()  GP on the trial mean: y = mean, noise of y = var / count
  log_sd()          log-normal GP on the SD: unbiased log SD and its exact chi-square sampling variance
  log_mean()        log-normal GP on a positive trial mean (e.g. mean |e|): delta-method log and its noise
"""
from dataclasses import dataclass

import torch
from scipy.special import digamma, polygamma


@dataclass
class TrialStats:
    """One row per trial. X (n, d) inputs as given (rescale in tasks.py); mean, var (ddof = 1), count: (n,)."""
    X: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor
    ids: list

    def mean_and_noise(self):
        """(y, noise variance of y) for a GP on the trial mean: y = mean, Var[mean] = var / count."""
        return self.mean, self.var / self.count

    def log_sd(self):
        """(y, noise variance of y) for a GP on log SD. With nu = count - 1, nu s^2 / sigma^2 ~ chi^2_nu, so
        E[log s] = log sigma + (digamma(nu/2) - log(nu/2)) / 2 and Var[log s] = trigamma(nu/2) / 4 (exact, any n).
        y has that bias removed, so the GP models log sigma itself. Trials with var = 0 give -inf: filter first."""
        nu = (self.count - 1).double()
        bias = (torch.as_tensor(digamma((nu / 2).numpy())) - torch.log(nu / 2)) / 2
        noise = torch.as_tensor(polygamma(1, (nu / 2).numpy())) / 4
        return 0.5 * torch.log(self.var) - bias, noise

    def log_mean(self):
        """(y, noise variance of y) for a GP on log of a positive mean, by the delta method:
        Var[log mean] ~ var / (count mean^2), E[log mean] ~ log mu - var / (2 count mean^2) (bias removed from y)."""
        rel = self.var / (self.count * self.mean.square())
        return torch.log(self.mean) + rel / 2, rel

    def subset(self, rows):
        """The trials at the given row indices (keeps every field aligned)."""
        rows = torch.as_tensor(rows)
        return TrialStats(self.X[rows], self.mean[rows], self.var[rows], self.count[rows],
                          [self.ids[i] for i in rows.tolist()])


def from_replicates(replicates, inputs, *, value="signed", trial="condition_id", target="error", min_count=2):
    """TrialStats from a replicate-level table (e.g. one liquid's rows of data/campaign_ingest/transfers.csv).
    inputs: input column names. value: "signed" (the error e) or "abs" (|e|). Trials with fewer than min_count
    replicates are dropped (no variance). Rows come out in the table's first-appearance order of each trial."""
    v = replicates[target].abs() if value == "abs" else replicates[target]
    grouped = replicates.assign(_v=v).groupby(trial, sort=False)
    agg = grouped.agg(mean=("_v", "mean"), var=("_v", lambda s: s.var(ddof=1)), count=("_v", "size"))
    X = grouped[list(inputs)].first()
    keep = agg["count"] >= min_count
    agg, X = agg[keep], X[keep]
    as_t = lambda a: torch.tensor(a.to_numpy(dtype=float))
    return TrialStats(as_t(X), as_t(agg["mean"]), as_t(agg["var"]), as_t(agg["count"]), list(agg.index))
