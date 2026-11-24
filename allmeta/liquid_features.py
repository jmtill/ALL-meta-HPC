"""Per-liquid descriptors used as task features. Default: campaign_ingest's 'physics' set
(logPvap25, MVol, SurfTen, log η). Standardized with the SOURCE liquids only, so the target never
shapes its own scaling. Descriptors are known before any trial, so the target's z is not leakage."""
import torch

BANDWIDTH_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e3)   # standardized-descriptor units; 1e3 ~ uniform weights


def load_descriptors(descriptor_set="physics", table=None, columns=None):
    """liquid -> descriptor vector (p,). table: the descriptors DataFrame (defaults to campaign_ingest.load()).
    columns: explicit column list, overriding the named descriptor_set."""
    from allmeta import campaign_ingest
    if table is None:
        table = campaign_ingest.load()["descriptors"]
    if columns is None:
        columns = campaign_ingest.descriptor_columns(descriptor_set)
    return {row["liquid"]: torch.tensor(row[columns].to_numpy(dtype=float)) for _, row in table.iterrows()}


def standardize(descriptors, sources, target):
    """(source liquid -> z, target z), all scaled by the mean and std of the source liquids."""
    Z = torch.stack([descriptors[name] for name in sources])
    shift, scale = Z.mean(0), Z.std(0).clamp_min(1e-12)
    return {name: (descriptors[name] - shift) / scale for name in sources}, (descriptors[target] - shift) / scale


def descriptor_weights(z_target, Z_sources, bandwidth=None):
    """Similarity weights over the M source liquids, summing to 1: softmax(-||z_t - z_m||^2 / (2 h^2)).
    bandwidth None -> median heuristic on the sources' pairwise squared distances (target not used).
    z_target (p,), Z_sources (M, p) -> (M,)."""
    d2 = (Z_sources - z_target).square().sum(-1)
    if bandwidth is None:
        pair = torch.cdist(Z_sources, Z_sources).square()
        h2 = pair[~torch.eye(len(Z_sources), dtype=torch.bool)].median()
    else:
        h2 = torch.tensor(bandwidth**2)
    return torch.softmax(-d2 / (2 * h2), 0)


def tune_bandwidth(Z_sources, cross_means, y_sources, grid=BANDWIDTH_GRID):
    """Leave-one-source-out choice of the descriptor_weights bandwidth, using source liquids only.
    Each source i is predicted from the other M - 1 as the weighted average of their GP means at its trials.
    Z_sources (M, p); cross_means[i]: (M, n_i), every source GP's mean at source i's trials; y_sources[i]: (n_i,).
    Returns (best bandwidth, {bandwidth: leave-one-out RMSE})."""
    M = len(Z_sources)
    scores = {}
    for h in grid:
        errors = []
        for i in range(M):
            keep = [j for j in range(M) if j != i]
            w = descriptor_weights(Z_sources[i], Z_sources[keep], bandwidth=h)
            errors.append((w @ cross_means[i][keep] - y_sources[i]).square().mean())
        scores[h] = torch.stack(errors).mean().sqrt().item()
    return min(scores, key=scores.get), scores