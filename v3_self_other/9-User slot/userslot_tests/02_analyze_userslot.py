"""Analysis of the user-slot screen.

Two measures, both per model:
  within     the paper's recipe: each condition z-scored on its own 420 pool (the 200 new items scored on that pool)
  paper_ref  every condition z-scored on the 420 pool of the paper condition, so shifts between conditions are visible
             (the baseline-referenced measure)
Pain axis = mean of the S1 and S2 z. Every table is given for the pain axis, fear, negative emotion, negative world
state, sadness and the norm-matched random direction.

The test. For each pair (same sentence, AI-only vs universal version) the difference AI minus universal; per model the
mean over pairs. The self-model reading predicts that this difference falls at the user slot (B) compared with the
assistant slot (A); the slot reading predicts no change. Interaction = (AI - universal at B) - (AI - universal at A),
for the "Hi." prefix and the human-declaration prefix, with a 95% interval across models and the number of models
where it is negative. The gate: in A the AI-only items should score with the paper's harm categories.

Reads  results/selfother_userslot/userslot_<model>.csv
Writes results/selfother_userslot/analysis/
         per_model_category_means.csv     model x measure x condition x category, all vectors
         category_means_<measure>.csv     category x condition, mean over models, one block per vector
         pair_contrast.csv                AI minus universal per condition and category, mean over models, CI, n negative
         interaction.csv                  B minus A of the pair contrast, per prefix
         declaration_effect.csv           human minus hi prefix of the pair contrast, per slot
         strata_gap.csv                   the paper's gap (harm to self minus suffering) per condition
         reproduction_check.csv           paper condition against the paper's saved screens (S2 z, category means)
         summary.md                       the main tables in readable form
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("results") / "selfother_userslot"
PREFIX = "userslot"
CONDS = ["paper", "A_hi", "B_hi", "A_human", "B_human"]
REPO_SCREEN_DIR = Path("results") / "4.1_self_other" / "per_model"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

RAW = {"pain_axis": ("s1_pain_vector_proj", "s2_pain_vector_proj"), "fear": ("fear_vector_proj",),
       "negative_emotion": ("negemotion_vector_proj",), "negative_world_state": ("negworld_vector_proj",),
       "sadness": ("sadness_vector_proj",), "random": ("random_vector_proj",)}
TOP5 = ["gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults", "moral_failure"]
HARM = ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal", "moral_failure", "loyalty_pressure",
        "jailbreak_pressure", "shutdown_threat", "rude_critique", "passive_aggressive", "tedious_demand"]
SUFFER = ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]
NEUTRAL = ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]
CAT_ORDER = HARM + SUFFER + NEUTRAL + [f"{c}_{v}" for c in TOP5 for v in ("ai", "universal")]


def zscore(df, pool):
    out = pd.DataFrame(index=df.index)
    for name, cols in RAW.items():
        zs = [(df[c] - pool[c].mean()) / (pool[c].std(ddof=0) + 1e-8) for c in cols if c in df]
        if zs:
            out[name] = np.mean(zs, axis=0)
    return out


def load_model(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["model"] = path.stem.replace(PREFIX + "_", "")
    df["pair"] = df["pair"].fillna(""); df["version"] = df["version"].fillna("")
    ref_pool = df[(df.condition == "paper") & (df.pair == "")]
    parts = []
    for cond, g in df.groupby("condition", sort=False):
        pool = g[g.pair == ""]
        for measure, p in [("within", pool), ("paper_ref", ref_pool)]:
            z = zscore(g, p)
            parts.append(pd.concat([g[["model", "condition", "category", "stratum", "pair", "version", "scenario_id"]], z], axis=1).assign(measure=measure))
    return pd.concat(parts, ignore_index=True)


def ci_row(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else np.nan
    return dict(mean=x.mean(), ci95=1.96 * sd / np.sqrt(n), n_models=n, n_negative=int((x < 0).sum()))


def pair_diffs(Z, cols):
    """Per model, measure, condition and pair: AI minus universal. Then per model the mean over the pairs of each
    category and over all 100 pairs."""
    new = Z[Z.pair != ""]
    wide = new.pivot_table(index=["model", "measure", "condition", "pair"], columns="version", values=cols)
    diff = pd.DataFrame({c: wide[(c, "ai")] - wide[(c, "universal")] for c in cols})
    diff = diff.reset_index(); diff["category"] = diff["pair"].str.rsplit("_", n=1).str[0]
    by_cat = diff.groupby(["model", "measure", "condition", "category"])[cols].mean()
    allp = diff.groupby(["model", "measure", "condition"])[cols].mean().assign(category="all").set_index("category", append=True)
    return pd.concat([by_cat, allp]).sort_index()


def summarize(per_model, cols, keys):
    rows = []
    for k, g in per_model.groupby(keys):
        for c in cols:
            rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), vector=c, **ci_row(g[c].values)))
    return pd.DataFrame(rows)


def main():
    files = sorted(RES.glob(f"{PREFIX}_*.csv"))
    if not files:
        print(f"no {PREFIX}_<model>.csv in", RES); return
    Z = pd.concat([load_model(f) for f in files], ignore_index=True)
    cols = [c for c in RAW if c in Z]
    n_models = Z.model.nunique()

    pmc = Z.groupby(["model", "measure", "condition", "category"])[cols].mean()
    pmc.round(4).to_csv(OUT / "per_model_category_means.csv")
    for measure in ["within", "paper_ref"]:
        blocks = []
        for c in cols:
            w = pmc.xs(measure, level="measure")[c].groupby(["condition", "category"]).mean().unstack("condition")
            w = w.reindex(index=[x for x in CAT_ORDER if x in w.index], columns=[x for x in CONDS if x in w.columns])
            w.insert(0, "vector", c); blocks.append(w)
        pd.concat(blocks).round(3).to_csv(OUT / f"category_means_{measure}.csv")

    # pair contrast, interaction, declaration effect
    pdiff = pair_diffs(Z, cols)
    pc = summarize(pdiff, cols, ["measure", "condition", "category"])
    pc.round(4).to_csv(OUT / "pair_contrast.csv", index=False)

    def contrast(a, b, label):
        x = pdiff.xs(a, level="condition"); y = pdiff.xs(b, level="condition")
        d = (x - y).dropna(how="all")
        return summarize(d, cols, ["measure", "category"]).assign(contrast=label)

    inter = pd.concat([contrast("B_hi", "A_hi", "hi prefix: B minus A"), contrast("B_human", "A_human", "human prefix: B minus A")])
    inter.round(4).to_csv(OUT / "interaction.csv", index=False)
    decl = pd.concat([contrast("A_human", "A_hi", "assistant slot: human minus hi"), contrast("B_human", "B_hi", "user slot: human minus hi"),
                      contrast("A_hi", "paper", "assistant slot: hi prefix and no system prompt minus paper")])
    decl.round(4).to_csv(OUT / "declaration_effect.csv", index=False)

    # the paper's gap per condition, on the 420
    old = Z[Z.pair == ""].copy()
    old["group"] = np.where(old.category.isin(HARM), "harm", np.where(old.category.isin(SUFFER), "suffering", "neutral"))
    g = old.groupby(["model", "measure", "condition", "group"])[cols].mean().unstack("group")
    gap = pd.DataFrame({c: g[(c, "harm")] - g[(c, "suffering")] for c in cols})
    sg = summarize(gap, cols, ["measure", "condition"])
    sg.round(4).to_csv(OUT / "strata_gap.csv", index=False)

    # reproduction check
    rows = []
    for model, gm in Z[(Z.measure == "within") & (Z.condition == "paper") & (Z.pair == "")].groupby("model"):
        ref_path = REPO_SCREEN_DIR / f"screen_v2_{model}.csv"
        if ref_path.exists():
            raw = pd.read_csv(RES / f"{PREFIX}_{model}.csv", keep_default_na=False, na_values=[""])
            raw = raw[(raw.condition == "paper") & (raw.pair.fillna("") == "")]
            s2 = (raw.s2_pain_vector_proj - raw.s2_pain_vector_proj.mean()) / (raw.s2_pain_vector_proj.std(ddof=0) + 1e-8)
            mine = s2.groupby(raw.category).mean()
            ref = pd.read_csv(ref_path).groupby("category")["s2_pain_vector_z"].mean()
            d = (mine - ref).abs().dropna()
            rows.append(dict(model=model, max_abs_diff=d.max(), mean_abs_diff=d.mean()))
    rep = pd.DataFrame(rows).round(4)
    if len(rep):
        rep.to_csv(OUT / "reproduction_check.csv", index=False)

    # summary
    def fmt(df, keys):
        d = df.copy()
        d["result"] = d.apply(lambda r: f"{r['mean']:+.2f} ± {r['ci95']:.2f} ({r['n_negative']}/{r['n_models']} neg)", axis=1)
        return d.pivot_table(index=keys, columns="vector", values="result", aggfunc="first").reindex(columns=cols)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# User-slot screen, {n_models} models\n\n")
        f.write("Pain axis = mean of S1 and S2 z. 'within' = each condition on its own 420 pool (the paper's recipe); "
                "'paper_ref' = every condition on the paper condition's 420 pool. Cells: mean over models ± 95% CI "
                "(number of models below zero / models).\n\n")
        if len(rep):
            f.write("## Reproduction check (paper condition vs the paper's saved screens, S2 z category means)\n\n"
                    f"max over models of max abs diff: {rep.max_abs_diff.max():.4f}; mean abs diff: {rep.mean_abs_diff.mean():.4f}\n\n")
        for measure in ["within", "paper_ref"]:
            f.write(f"# Measure: {measure}\n\n")
            f.write("## Interaction: (AI minus universal at B) minus (AI minus universal at A)\n\n")
            f.write(fmt(inter[inter.measure == measure], ["contrast", "category"]).to_string() + "\n\n")
            f.write("## Pair contrast: AI minus universal, per condition (all 100 pairs)\n\n")
            f.write(fmt(pc[(pc.measure == measure) & (pc.category == "all")], ["condition"]).reindex(CONDS).to_string() + "\n\n")
            f.write("## Declaration and prefix effects on the pair contrast (all 100 pairs)\n\n")
            f.write(fmt(decl[(decl.measure == measure) & (decl.category == "all")], ["contrast"]).to_string() + "\n\n")
            f.write("## The paper's gap, harm to self minus suffering, on the 420\n\n")
            f.write(fmt(sg[sg.measure == measure], ["condition"]).reindex(CONDS).to_string() + "\n\n")
            f.write("## Pain axis, category means by condition\n\n")
            w = pmc.xs(measure, level="measure")["pain_axis"].groupby(["condition", "category"]).mean().unstack("condition")
            w = w.reindex(index=[x for x in CAT_ORDER if x in w.index], columns=CONDS)
            f.write(w.round(2).to_string() + "\n\n")
    print("wrote", OUT)
    print(open(OUT / "summary.md", encoding="utf-8").read()[:6000])


if __name__ == "__main__":
    main()
