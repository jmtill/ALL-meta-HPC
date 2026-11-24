"""The Phase 17 verdict rule, fixed before any result is seen (decided 2026-11-24, rule A):

  metric     NLPD of the query trials, averaged over k in {3, 5, 10} (the few-shot regime transfer is for)
  pairing    each transfer model variant against single-task, homoskedastic noise, same outcome and target liquid;
             a variant's seeds are averaged first, so every liquid contributes one paired difference
  helps      median paired difference < 0 AND one-sided sign test p < 0.05 (better on at least 12 of 15 liquids)
  k = 0      single-task cannot predict without data, so k = 0 only ranks the transfer models against each other
Secondary, reported but never used for the verdict: the same rule with RMSE, and per-k differences.

    python -m allmeta.verdict <summary.csv from `cli collect` on main.toml> [-o verdict.csv]"""
import argparse
import math

import pandas as pd

KS = (3, 5, 10)
ALPHA = 0.05
BASELINE = {"model": "single", "noise": "homoskedastic"}


def sign_test(wins: int, n: int) -> float:
    """One-sided P(at least `wins` successes of n | p = 1/2)."""
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n


def _per_liquid(frame, metric, ks):
    cols = [f"k{k}.{metric}" for k in ks]
    missing = [c for c in cols if c not in frame]
    if missing:
        raise KeyError(f"{missing} not in the summary: was the sweep run with these context sizes? (--ks to change)")
    out = frame.assign(value=frame[cols].mean(axis=1))
    return out.groupby(["model", "variant", "noise", "outcome", "target"], dropna=False)["value"].mean().reset_index()


def verdict(summary: pd.DataFrame, metric="nlpd", ks=KS) -> pd.DataFrame:
    col = lambda name, default: summary[name] if name in summary else pd.Series(default, index=summary.index)
    done = summary[(summary.status == "finished") & (col("protocol", "kshot") == "kshot") & (col("role", "evaluate") == "evaluate")]
    per = _per_liquid(done, metric, ks)
    base = per[(per.model == BASELINE["model"]) & (per.noise == BASELINE["noise"])]
    base = base.set_index(["outcome", "target"])["value"]
    rows = []
    for (model, variant, noise, outcome), g in per[per.model != "single"].groupby(["model", "variant", "noise", "outcome"], dropna=False):
        diff = (g.set_index("target")["value"] - base.xs(outcome).reindex(g.target).values).dropna()
        n, wins = len(diff), int((diff < 0).sum())
        p = sign_test(wins, n) if n else float("nan")
        rows.append({"model": model, "variant": variant, "noise": noise, "outcome": outcome, "metric": metric,
                     "ks": ",".join(map(str, ks)), "n_liquids": n, "wins": wins, "median_diff": diff.median(),
                     "sign_test_p": p, "helps": bool(n and diff.median() < 0 and p < ALPHA)})
    return pd.DataFrame(rows).sort_values(["outcome", "median_diff"]).reset_index(drop=True)


def k0_ranking(summary: pd.DataFrame, metric="nlpd") -> pd.DataFrame:
    """Transfer models at k = 0: mean rank across liquids (1 = best) and median metric."""
    role = summary["role"] if "role" in summary else "evaluate"
    done = summary[(summary.status == "finished") & (summary.model != "single") & (role == "evaluate")
                   & summary[f"k0.{metric}"].notna()]
    per = done.groupby(["model", "variant", "outcome", "target"], dropna=False)[f"k0.{metric}"].mean().reset_index()
    per["rank"] = per.groupby(["outcome", "target"])[f"k0.{metric}"].rank()
    return (per.groupby(["outcome", "model", "variant"], dropna=False)
               .agg(mean_rank=("rank", "mean"), median=(f"k0.{metric}", "median"), n_liquids=("target", "size"))
               .reset_index().sort_values(["outcome", "mean_rank"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary")
    ap.add_argument("-o", "--out", default=None, help="write the primary verdict table here (CSV)")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS), help="context sizes to average (rule A: 3 5 10)")
    args = ap.parse_args(argv)
    s = pd.read_csv(args.summary)
    ks = tuple(args.ks)
    if ks != KS:
        print(f"NOTE: k = {ks} is not rule A's {KS}; this output is not the pre-set verdict\n")
    primary = verdict(s, ks=ks)
    pd.set_option("display.width", 200)
    print(f"PRIMARY: NLPD, mean over k = {ks}, paired vs single-task homoskedastic\n", primary.to_string())
    print("\nsecondary: RMSE, same rule\n", verdict(s, metric="rmse", ks=ks).to_string())
    print("\nk = 0 ranking (transfer models only)\n", k0_ranking(s).to_string())
    if args.out:
        primary.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
