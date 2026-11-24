"""Build allmeta/data/ (the GP inputs) from the raw optimization campaigns. Run once, from the ALL-meta-learning
checkout with autonomous-liquid-learning beside it:  PYTHONPATH=src:. python src/allmeta/data/build_data.py

Selection, one row per run in runs.csv with the reason it was kept or dropped:
  uncapped   dated before 2026-09-16, the day the vial-capping protocol first ran; every campaign from then on was
             capped (autonomous-liquid-learning post_analysis/plot_mae_sd_correlation.py, CAPPING_INTRODUCED,
             confirmed for the full list 2026-11-03)
  complete   recorded >= 95% of budget x replicates per trial (same script, COMPLETENESS_THRESHOLD)
  no repeats one run per liquid: the first complete uncapped run; later runs of the same liquid are dropped
  not a test dry runs (liquid "test") are dropped
Parsing, error definition and row cleaning are allmeta.campaign_ingest's, unchanged: e = (calc_weight / density -
V_target) / V_target from the raw gravimetry; replicates with |e| > 3 dropped; conditions with < 3 replicates dropped.
Outputs: transfers.csv (replicates), conditions.csv (trial summaries), descriptors.csv (HSPiP "physics" set;
a liquid with no HSPiP row, e.g. the glycerol-water mixture, gets an empty row), runs.csv (selection audit)."""
import sys
from pathlib import Path

import pandas as pd

from allmeta import campaign_ingest as ci

CAPPING_INTRODUCED = "2026-09-16"
COMPLETENESS_THRESHOLD = 0.95
OUT = Path(__file__).resolve().parent


def run_date(run_id: str) -> str:
    return "-".join(run_id.split("_")[:3])


def main(source=None):
    ingest = ci.collect(source)
    transfers = ci.transfer_table(ingest)
    transfers = transfers[transfers.campaign == "optimization"]
    runs = pd.DataFrame([r.as_row() for r in ingest.runs])
    runs = runs[runs.campaign == "optimization"].copy()
    import json
    rows, kept_liquids = [], set()
    for r in runs.sort_values("run_id").itertuples(index=False):
        meta = json.loads(Path(r.path).read_text())["metadata"]
        planned = float(meta.get("budget", 0)) * float(meta.get("reps per experiment", 0))
        recorded = int((transfers.run_id == r.run_id).sum())
        frac = recorded / planned if planned else 0.0
        reasons = []
        if r.liquid in ci.EXCLUDED_LIQUIDS or r.status != "accepted":
            reasons.append(f"ingest: {r.reason}")
        if run_date(r.run_id) >= CAPPING_INTRODUCED:
            reasons.append(f"capped (dated on/after {CAPPING_INTRODUCED})")
        if frac < COMPLETENESS_THRESHOLD:
            reasons.append(f"incomplete ({recorded} of {planned:g} replicates)")
        if not reasons and r.liquid in kept_liquids:
            reasons.append("repeat run of a liquid already included")
        used = not reasons
        if used:
            kept_liquids.add(r.liquid)
        rows.append({"run_id": r.run_id, "liquid": r.liquid, "date": run_date(r.run_id), "budget": meta.get("budget"),
                     "reps_per_trial": meta.get("reps per experiment"), "planned_replicates": planned,
                     "recorded_replicates": recorded, "fraction_recorded": round(frac, 4), "used": used,
                     "excluded_because": "; ".join(reasons), "source": r.path})
    audit = pd.DataFrame(rows)
    keep = audit.loc[audit.used, "run_id"]
    transfers = transfers[transfers.run_id.isin(keep)].reset_index(drop=True)
    assert not transfers.duplicated(["run_id", "condition_id", "replicate_id"]).any(), "duplicate replicate rows"
    conditions = ci.condition_table(transfers)
    found, missing = [], []
    for liquid in sorted(transfers.liquid.unique()):                 # a mixture has no HSPiP row
        try:
            found.append(ci.descriptor_table([liquid]))
        except ValueError:
            missing.append(liquid)
    descriptors = pd.concat(found + [pd.DataFrame({"liquid": missing})], ignore_index=True)
    audit.loc[audit.used & audit.liquid.isin(missing), "excluded_because"] = \
        "kept in the data; no HSPiP descriptors, so outside the factorial liquid set (tasks.factorial_liquids)"
    for name, frame in [("transfers", transfers), ("conditions", conditions), ("descriptors", descriptors), ("runs", audit)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    print(audit[["run_id", "used", "recorded_replicates", "planned_replicates", "excluded_because"]].to_string())
    print(f"\n{len(keep)} runs, {transfers.liquid.nunique()} liquids, {conditions.shape[0]} trials, {len(transfers)} replicates")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
