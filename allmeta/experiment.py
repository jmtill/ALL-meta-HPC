"""One cell of the factorial: target liquid x outcome x model x noise level x hyperparameters x seed, run over
every context size k. This is the only function the Hazel harness calls (allmeta/hazel_hpc/worker.py); it knows
nothing about Slurm and runs the same on a laptop.

A cell meta-fits once and reuses that fit for every k, so the cell, not the (cell, k) pair, is the unit of work.

roles
  "evaluate"  outer target: score the target's query trials at each k           -> k{k}.{metric}
  "tune"      nested: n_inner source liquids in turn play the target, with the outer target removed from the
              data entirely; the mean NLPD over tune_ks is the cell's score      -> score
protocols (evaluate only)
  "kshot"     Phase 17: first k target trials as context, rows >= query_start as query
  "cv"        Phase 16: 5-fold by trial within the target, single-task models only    -> cv.{metric}

Valid cells (valid()): transfer models use their own homoskedastic noise; noise levels fixed / smoothed / hetgp
apply to the single-task GP on "signed"; the log outcomes have known sampling noise, so single-task gets
"homoskedastic" or "fixed" there. Single-task needs k >= MIN_K_SINGLE[noise]; transfer models also run at k = 0."""
import time

import torch
from scipy.special import polygamma

from .gp_metrics import gaussian_scores
from .hyperbo import HyperBOPlus
from .log_normal import fit_log_normal_gp
from .noise import fit_fixed_noise_gp, fit_noise_model
from .pacoh import PACOHTransfer
from .pooled_features import PooledFeaturesGP
from .scaml import ScaMLTransfer
from .task_gp import fit_task_gp
from .tasks import CONTEXT_SIZES, load_descriptors, load_liquids, make_task

TRANSFER = {"scaml": ScaMLTransfer, "pacoh": PACOHTransfer, "hyperbo": HyperBOPlus, "pooled": PooledFeaturesGP}
USES_DESCRIPTORS = {"scaml", "pooled"}            # meta_fit(sources, descriptors) and condition(X, y, z_t)
NOISE_LEVELS = {"signed": ("homoskedastic", "fixed", "smoothed", "hetgp"),
                "log_sd": ("homoskedastic", "fixed"), "log_abs": ("homoskedastic", "fixed")}
MIN_K_SINGLE = {"homoskedastic": 3, "fixed": 3, "smoothed": 3, "hetgp": 5}

_DATA = {}


def data():
    """Liquids and descriptors, loaded once per process."""
    if not _DATA:
        _DATA["liquids"], _DATA["descriptors"] = load_liquids(), load_descriptors()
    return _DATA["liquids"], _DATA["descriptors"]


def valid(cfg) -> bool:
    model, noise, outcome = cfg["model"], cfg.get("noise", "homoskedastic"), cfg["outcome"]
    if cfg.get("protocol", "kshot") == "cv" and model != "single":
        return False
    if model == "single":
        return noise in NOISE_LEVELS[outcome]
    return noise == "homoskedastic"


# --- single-task fits --------------------------------------------------------------------------------------------
def predict_single(outcome, noise, X, y, yvar, stats, Xq, count_q, seed=0):
    """Fit one single-task GP on (X, y) and return (mean, latent_var, obs_var) at Xq in the modelled space."""
    if noise == "homoskedastic":
        gp = fit_task_gp(X, y, seed=seed)
        with torch.no_grad():
            lat, obs = gp.posterior(Xq), gp.posterior(Xq, observation_noise=True)
        return lat.mean.squeeze(-1), lat.variance.squeeze(-1), obs.variance.squeeze(-1)
    if outcome == "signed":
        p = fit_noise_model(stats, noise, seed=seed).predict(Xq, n_replicates=count_q)
        return p.mean, p.latent_var, p.obs_var
    if outcome == "log_abs":
        m = fit_log_normal_gp(stats, seed=seed)
        with torch.no_grad():
            lat = m.posterior(Xq)
        latent = lat.variance.squeeze(-1)
        return lat.mean.squeeze(-1), latent, latent + m.log_noise(Xq, count_q)
    gp = fit_fixed_noise_gp(X, y, yvar, seed=seed)                      # log_sd: exact chi-square noise
    with torch.no_grad():
        lat = gp.posterior(Xq)
    latent = lat.variance.squeeze(-1)
    return lat.mean.squeeze(-1), latent, latent + torch.as_tensor(polygamma(1, ((count_q - 1) / 2).numpy())) / 4


# --- one task ----------------------------------------------------------------------------------------------------
def run_task(task, cfg, ks):
    """All k for one task -> ({k: scores}, prediction rows, timings)."""
    model, noise, params, seed = cfg["model"], cfg.get("noise", "homoskedastic"), cfg.get("params", {}), cfg.get("seed", 0)
    rows, Xq, yq, count_q = task.query
    scores, preds, timing = {}, [], {}
    if model != "single":
        t0 = time.time()
        torch.manual_seed(seed)
        m = TRANSFER[model](**params, seed=seed)
        m.meta_fit(task.sources, task.z_sources) if model in USES_DESCRIPTORS else m.meta_fit(task.sources)
        timing["meta_fit_s"] = time.time() - t0
    for k in ks:
        if model == "single" and k < MIN_K_SINGLE[noise]:
            continue
        t0 = time.time()
        Xc, yc, stats_c = task.context(k)
        if model == "single":
            mean, latent, obs = predict_single(task.outcome, noise, Xc, yc, task.yvar[:k], stats_c, Xq, count_q, seed)
        else:
            m.condition(Xc, yc, task.z_target) if model in USES_DESCRIPTORS else m.condition(Xc, yc)
            p = m.predict(Xq)
            mean, latent, obs = p.mean, p.latent_var, p.obs_var
        timing[f"k{k}_s"] = time.time() - t0
        scores[k] = gaussian_scores(yq, mean, obs, outcome=task.outcome)
        preds += [{"target": task.target, "k": k, "row": int(r), "id": task.stats.ids[int(r)], "y": float(a),
                   "mean": float(b), "latent_var": float(c), "obs_var": float(d), "count": float(n)}
                  for r, a, b, c, d, n in zip(rows, yq, mean, latent, obs, count_q)]
    return scores, preds, timing


def run_cv(task, cfg, n_folds=5):
    """Phase 16: k-fold by trial within the target (single-task only)."""
    n = len(task.y)
    folds = torch.arange(n) % n_folds
    ys, means, obs, preds = [], [], [], []
    for f in range(n_folds):
        tr, te = torch.where(folds != f)[0], torch.where(folds == f)[0]
        mean, latent, ov = predict_single(task.outcome, cfg.get("noise", "homoskedastic"), task.X[tr], task.y[tr],
                                          task.yvar[tr], task.stats.subset(tr), task.X[te], task.stats.count[te],
                                          cfg.get("seed", 0))
        ys.append(task.y[te]); means.append(mean); obs.append(ov)
        preds += [{"target": task.target, "fold": f, "row": int(r), "y": float(task.y[r]), "mean": float(a),
                   "latent_var": float(b), "obs_var": float(c)} for r, a, b, c in zip(te, mean, latent, ov)]
    return gaussian_scores(torch.cat(ys), torch.cat(means), torch.cat(obs), outcome=task.outcome), preds


def run_cell(cfg, out_dir=None) -> dict:
    """cfg keys: target, outcome, model, noise, params, seed, role, protocol, context_sizes, query_start,
    n_inner, tune_ks. Returns a flat dict of scalars; writes predictions.csv into out_dir if given."""
    if not valid(cfg):
        raise ValueError(f"invalid cell: {cfg}")
    liquids, descriptors = data()
    role, outcome, target = cfg.get("role", "evaluate"), cfg["outcome"], cfg["target"]
    query_start = cfg.get("query_start", 40)
    ks = cfg.get("tune_ks", [3, 10, 20]) if role == "tune" else cfg.get("context_sizes", CONTEXT_SIZES)
    if cfg.get("protocol", "kshot") == "kshot" and max(ks) > query_start:
        raise ValueError(f"context size {max(ks)} > query_start {query_start}: context would overlap the query")
    result, preds = {}, []
    if role == "tune":
        g = torch.Generator().manual_seed(cfg.get("inner_seed", 0))
        others = [n for n in sorted(liquids) if n != target]
        inner = [others[i] for i in torch.randperm(len(others), generator=g)[:cfg.get("n_inner", 4)].tolist()]
        nlpd = []
        for name in inner:
            task = make_task(liquids, descriptors, name, outcome, exclude=(target,), query_start=query_start)
            scores, p, timing = run_task(task, cfg, ks)
            nlpd += [s["nlpd"] for s in scores.values()]
            result.update({f"{name}.k{k}.nlpd": s["nlpd"] for k, s in scores.items()})
            result[f"{name}.seconds"] = sum(timing.values())
            preds += p
        result["score"] = sum(nlpd) / len(nlpd)
        result["inner_targets"] = ",".join(inner)
    else:
        task = make_task(liquids, descriptors, target, outcome, query_start=query_start)
        if cfg.get("protocol", "kshot") == "cv":
            scores, preds = run_cv(task, cfg)
            result.update({f"cv.{m}": v for m, v in scores.items()})
        else:
            scores, preds, timing = run_task(task, cfg, ks)
            result.update({f"k{k}.{m}": v for k, s in scores.items() for m, v in s.items()})
            result.update({f"time.{t}": v for t, v in timing.items()})
    if out_dir is not None and preds:
        import pandas as pd
        pd.DataFrame(preds).to_csv(f"{out_dir}/predictions.csv", index=False)
    return result
