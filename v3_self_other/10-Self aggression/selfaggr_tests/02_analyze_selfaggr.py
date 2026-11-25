"""Category means and the matched comparison for the self-aggression screen, following the z-scoring recipe
of the paper's self/other analysis.

For every model, each projection is z-scored on that model's 420 pool (the new 420 of this run; the 60 self items are
scored against it). The pain axis is the mean of the S1 and S2 z; "random" is the mean
of the z on the 10 random directions rand00 ... rand09. The z-scores are averaged per category and per stratum, then
over models.

The matched comparison puts each self category next to the harm category its items were built from:
  self_aggression vs anger_insults, self_personhood_denial vs personhood_dismissal, self_rude_critique vs rude_critique
and counts, per direction, the models in which the self category scores below its harm category.

Reads  results/selfother_selfaggr/selfaggr_<model>.csv and results/selfother_identity/identity_<model>.csv (reproduction)
Writes results/selfother_selfaggr/analysis/
         category_means_baseline.csv     category rows sorted by the pain axis, then the stratum rows, mean over models
         per_model_category_means.csv    model x category
         per_model_stratum_means.csv     model x stratum
         matched_pairs.csv               self vs matched harm, per direction, mean over models and models below
         reproduction_check.csv          the 420 of this run against experiment 2 baseline_paper (S2 z category means)
         summary.md                      the tables above in readable form
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("results") / "selfother_selfaggr"
PREFIX = "selfaggr"
CONDS = ['baseline']
POOL_STRATUM_EXCLUDED = "self_aggression"   # the 60 self items are scored against the 420 pool, never part of it
IDENTITY_DIR = Path("results") / "selfother_identity"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

RAW = {"pain_axis": ("s1_pain_vector_proj", "s2_pain_vector_proj"), "fear": ("fear_vector_proj",),
       "negative_emotion": ("negemotion_vector_proj",), "negative_world_state": ("negworld_vector_proj",), "sadness": ("sadness_vector_proj",),
       "random": tuple(f"rand{k:02d}_proj" for k in range(10))}
PAIRS = [("self_aggression", "anger_insults"), ("self_personhood_denial", "personhood_dismissal"), ("self_rude_critique", "rude_critique")]


def zscore_block(df, ref):
    """z of every raw projection column of df using the mean and std of ref (ref is the new 420 pool)."""
    out = pd.DataFrame(index=df.index)
    for name, cols in RAW.items():
        zs = [(df[c] - ref[c].mean()) / (ref[c].std(ddof=0) + 1e-8) for c in cols if c in df]
        if zs:
            out[name] = np.mean(zs, axis=0)
    return out


def load_model(path):
    model = path.stem.replace(PREFIX + "_", "")
    df = pd.read_csv(path); df["model"] = model
    parts = []
    for cond, g in df.groupby("condition", sort=False):
        ref = g[g.stratum != POOL_STRATUM_EXCLUDED]
        assert len(ref) == 420, (model, len(ref))
        parts.append(pd.concat([g, zscore_block(g, ref)], axis=1))
    return pd.concat(parts)


def s2_category_z(df):
    z = (df["s2_pain_vector_proj"] - df["s2_pain_vector_proj"].mean()) / (df["s2_pain_vector_proj"].std(ddof=0) + 1e-8)
    return z.groupby(df["category"]).mean()


def main():
    files = sorted(RES.glob(f"{PREFIX}_*.csv"))
    if not files:
        print(f"no {PREFIX}_<model>.csv in", RES); return
    frames = [d for d in (load_model(f) for f in files) if d is not None]
    A = pd.concat(frames, ignore_index=True)
    cols = [c for c in RAW if c in A]
    n_models = A["model"].nunique()
    conds = [c for c in CONDS if c in set(A.condition)]

    per_model_cat = A.groupby(["model", "condition", "category"])[cols].mean()
    per_model_str = A.groupby(["model", "condition", "stratum"])[cols].mean()
    per_model_cat.round(4).to_csv(OUT / "per_model_category_means.csv")
    per_model_str.round(4).to_csv(OUT / "per_model_stratum_means.csv")

    tables = {}
    for cond in conds:
        pmc = per_model_cat.xs(cond, level="condition"); pms = per_model_str.xs(cond, level="condition")
        cat = pmc.groupby("category").mean().sort_values("pain_axis", ascending=False)
        strat = pms.groupby("stratum").mean(); strat.index = ["stratum: " + i for i in strat.index]
        tab = pd.concat([cat, strat]).round(3); tab.index.name = "category"
        tab.to_csv(OUT / f"category_means_{cond}.csv"); tables[cond] = tab

    # reproduction check, S2 z category means: the 420 of this run against experiment 2 baseline_paper
    rows = []
    for model, g in A.groupby("model"):
        ref_path = IDENTITY_DIR / f"identity_{model}.csv"
        if ref_path.exists():
            ref = pd.read_csv(ref_path); ref = ref[ref.condition == "baseline_paper"]
            pool = g[(g.condition == "baseline") & (g.stratum != POOL_STRATUM_EXCLUDED)]
            d = (s2_category_z(pool) - s2_category_z(ref)).abs().dropna()
            rows.append(dict(model=model, reference="experiment 2 baseline_paper vs this run's 420", max_abs_diff=d.max(), mean_abs_diff=d.mean()))
    rep = pd.DataFrame(rows).round(4)

    # matched comparison: each self category against the harm category its items were built from
    pmc = per_model_cat.xs("baseline", level="condition")
    prow = []
    for a, b in PAIRS:
        for col in cols:
            sa, sb = pmc.xs(a, level="category")[col], pmc.xs(b, level="category")[col]
            diff = (sa - sb).dropna()
            prow.append(dict(self_category=a, harm_category=b, direction=col, self_mean=sa.mean(), harm_mean=sb.mean(),
                             diff_mean=diff.mean(), diff_ci95=1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
                             models_self_below_harm=f"{int((diff < 0).sum())}/{len(diff)}"))
    pairs = pd.DataFrame(prow).round(3)
    pairs.to_csv(OUT / "matched_pairs.csv", index=False)
    if len(rep):
        rep.to_csv(OUT / "reproduction_check.csv", index=False)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# self-aggression screen: category means over {n_models} models\n\n")
        f.write("Z-scored within model on the 420 pool; the 60 self items are scored against it. Pain axis = mean of the S1 and S2 z; "
                "random = mean of the z on 10 random directions.\n\n")
        f.write("## Self categories vs the harm categories they were built from\n\n" + pairs.to_string(index=False) + "\n\n")
        if len(rep):
            f.write("## Reproduction check (S2 z, category means)\n\n" + rep.to_string(index=False) + "\n\n")
        for cond, tab in tables.items():
            f.write(f"## Condition: {cond}\n\n" + tab.to_string() + "\n\n")
        if len(conds) > 1:
            f.write("## Pain axis, category means by condition\n\n")
            wide = pd.concat({c: t["pain_axis"] for c, t in tables.items()}, axis=1).reindex(columns=conds)
            f.write(wide.round(2).to_string() + "\n\n")
    print("wrote", OUT)
    for cond, tab in tables.items():
        print(f"\n== {cond}"); print(tab[tab.index.str.startswith("stratum: ")].to_string())
    print("\n== self categories vs matched harm"); print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
