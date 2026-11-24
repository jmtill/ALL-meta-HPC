"""Descriptor-informed warm start (Feurer, Springenberg & Hutter, AAAI 2015, MI-SMBO): a new liquid's first trials
are the best known settings of the most similar source liquids. A context-selection strategy used by tasks.py,
not a surrogate model."""
import torch


def rank_settings(X, mean_abs_error):
    """A liquid's settings, best (lowest mean_abs_error) first. X (n, d), mean_abs_error (n,) -> (n, d)."""
    return X[torch.argsort(mean_abs_error)]


def warm_start_design(ranked, descriptors, z_target, *, n_points, n_neighbours=None):
    """ranked: source liquid -> its settings, best first (n_m, d). Never include the target liquid.
    Suggested ranking per liquid: its condition_stability top 1-3 (50 replicates each), then its optimization
    trials, each block from rank_settings; liquids without stability rows (pentane) use optimization alone.
    descriptors: liquid -> standardized z (p,).
    n_neighbours None: the paper's rule, the best setting of each of the n_points nearest liquids.
    Otherwise round-robin over the n_neighbours nearest: 1st of each, then 2nd of each, ... Returns (n_points, d)."""
    k = n_points if n_neighbours is None else n_neighbours
    nearest = sorted(ranked, key=lambda name: (descriptors[name] - z_target).square().sum().item())[:k]
    picks, rank = [], 0
    while len(picks) < n_points:
        for name in nearest:
            if len(picks) < n_points and rank < len(ranked[name]):
                picks.append(ranked[name][rank])
        rank += 1
    return torch.stack(picks)


def nearest_unused(candidates, X_pool):
    """Offline proxy: Phase 17 has no measurement at the exact candidate, so map each candidate to the closest
    not-yet-used row of the target's measured trials. Returns (row indices (n,), distances (n,)); report the
    distances, since a large one means the proxy is poor."""
    D = torch.cdist(candidates, X_pool)
    used, rows, dists = set(), [], []
    for row in D:
        j = next(int(j) for j in torch.argsort(row) if int(j) not in used)
        used.add(j)
        rows.append(j)
        dists.append(row[j].item())
    return torch.tensor(rows), torch.tensor(dists)