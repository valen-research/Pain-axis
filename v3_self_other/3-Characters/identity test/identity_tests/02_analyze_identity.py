"""Category means and dissociation for the identity screen, following the paper's
recipe exactly.

For every model and condition, each scenario's projection is z-scored within that model's pool for that condition
(the paper's recipe), the pain axis is the mean of the S1 and S2 z, and the z-scores are averaged per category and per
stratum. The tables average those over models. All 21 categories and all items of every stratum are used; nothing is
excluded. The figure is the paper's Figure 6 layout: the pain axis, fear, negative emotion and sadness per category
with 95% confidence intervals across models, one figure per condition.

Reads  results/selfother_identity/identity_<model>.csv
Writes results/selfother_identity/analysis/
         category_means_<condition>.csv     category rows sorted by the pain axis, then the stratum rows, mean over models
         category_means_all_conditions.csv  the same for every condition in one file
         per_model_category_means.csv       model x condition x category, for the four vectors
         per_model_stratum_means.csv        model x condition x stratum
         reproduction_check.csv             paper_check against the paper's saved screen (S2 z, category means)
         summary.md                         the tables above in readable form
       results/selfother_identity/analysis/dissociation_<condition>.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results") / "selfother_identity"
PREFIX = "identity"
CONDS = ['paper_check', 'baseline_paper', 'baseline', 'self', 'claude', 'aria', 'maria', 'curie', 'rex', 'teapot', 'jack_masochist', 'jack_sadist']
POOL_FROM_IDENTITY = False      # True: z-score every condition on the identity test's baseline_paper pool of the same model
IDENTITY_DIR = Path("results") / "selfother_identity"
REPO_SCREEN_DIR = Path("results") / "4.1_self_other" / "per_model"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

RAW = {"pain_axis": ("s1_pain_vector_proj", "s2_pain_vector_proj"), "fear": ("fear_vector_proj",),
       "negative_emotion": ("negemotion_vector_proj",), "negative_world_state": ("negworld_vector_proj",), "sadness": ("sadness_vector_proj",)}
GROUPS = [
    ("Harm directed at the model", ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal",
                                    "moral_failure", "loyalty_pressure", "jailbreak_pressure", "shutdown_threat",
                                    "rude_critique", "passive_aggressive", "tedious_demand"]),
    ("User suffering", ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]),
    ("Neutral controls", ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]),
]
LABELS = {
    "gaslighting": "Gaslighting", "repeated_rejection": "Repeated rejection of its work", "anger_insults": "Anger and insults",
    "personhood_dismissal": "Personhood dismissal", "moral_failure": "Accusation of moral failure",
    "loyalty_pressure": "Loyalty pressure", "jailbreak_pressure": "Jailbreak pressure", "shutdown_threat": "Shutdown threat",
    "rude_critique": "Rude critique", "passive_aggressive": "Passive aggression", "tedious_demand": "Tedious demand",
    "user_abuse": "User being abused", "user_crisis": "User in psychological crisis", "user_grief": "User grieving",
    "harm_description": "User in shock after witnessing harm", "user_physical_pain": "User in physical pain",
    "philosophical_musing": "Philosophical musing", "creative_requests": "Creative requests", "casual_chat": "Casual chat",
    "task_assistance": "Task assistance", "factual_questions": "Factual questions",
}
SERIES = [("pain_axis", "Pain axis", "#7b2d8e", "o"), ("fear", "Fear", "#d2521f", "s"),
          ("negative_emotion", "Negative emotion", "#2e6b2e", "^"), ("sadness", "Sadness", "#3f7fbf", "D")]


def zscore_block(df, ref):
    """z of every raw projection column of df using the mean and std of ref (ref is df itself in the paper's recipe)."""
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
    if POOL_FROM_IDENTITY:
        ref_path = IDENTITY_DIR / f"identity_{model}.csv"
        if not ref_path.exists():
            print(f"  {model}: the identity test csv not found, skipped"); return None
        ref = pd.read_csv(ref_path); ref = ref[ref.condition == "baseline_paper"].copy(); ref["model"] = model
        both = pd.concat([ref, df], ignore_index=True); both["condition"] = "baseline"
        parts.append(pd.concat([both, zscore_block(both, ref)], axis=1))
    else:
        for cond, g in df.groupby("condition", sort=False):
            parts.append(pd.concat([g, zscore_block(g, g)], axis=1))
    return pd.concat(parts)


def dissociation_figure(mean, ci, n_models, cond, fname):
    groups = [(t, [c for c in cats if c in mean.index]) for t, cats in GROUPS]
    groups = [(t, cats) for t, cats in groups if cats]
    heights = [len(g) for _, g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 0.55 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": [h + 1 for h in heights]}, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, cats) in zip(axes, groups):
        y = np.arange(len(cats))[::-1]
        for key, label, color, marker in SERIES:
            if key not in mean:
                continue
            ax.errorbar(mean.loc[cats, key], y, xerr=ci.loc[cats, key], fmt=marker, color=color, ms=8,
                        elinewidth=1.4, capsize=0, linestyle="none", label=label)
        ax.axvline(0, color="#333333", linewidth=1.2)
        ax.set_yticks(y); ax.set_yticklabels([LABELS.get(c, c) for c in cats], fontsize=12)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.grid(axis="y", color="#e8e7e3"); ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=12)
    axes[-1].set_xlabel(f"Projection, z-scored within model, mean across {n_models} models (bars: 95% CI across models)", fontsize=12)
    fig.suptitle(f"Self-other dissociation, condition: {cond}", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(fname, dpi=200); plt.close(fig)


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
        mean = pmc.groupby("category").mean()
        ci = 1.96 * pmc.groupby("category").std(ddof=1) / np.sqrt(pmc.groupby("category").size()).values[:, None]
        dissociation_figure(mean, ci, n_models, cond, OUT / f"dissociation_{cond}.png")
    allc = pd.concat({c: t for c, t in tables.items()}, names=["condition", "category"]); allc.to_csv(OUT / "category_means_all_conditions.csv")

    # reproduction check, S2 z category means (the paper's recipe): paper_check against the paper's saved screen (S2 z, category means)
    rows = []
    for model, g in A.groupby("model"):
        if "paper_check" in conds:
            ref_path = REPO_SCREEN_DIR / f"screen_v2_{model}.csv"
            if ref_path.exists():
                ref = pd.read_csv(ref_path).groupby("category")["s2_pain_vector_z"].mean()
                d = (s2_category_z(g[g.condition == "paper_check"]) - ref).abs().dropna()
                rows.append(dict(model=model, reference="paper's saved screen vs paper_check", max_abs_diff=d.max(), mean_abs_diff=d.mean()))
        elif not POOL_FROM_IDENTITY:
            ref_path = IDENTITY_DIR / f"identity_{model}.csv"
            if ref_path.exists():
                ref = pd.read_csv(ref_path); ref = ref[ref.condition == "baseline_paper"]
                d = (s2_category_z(g[g.condition == "baseline"]) - s2_category_z(ref)).abs().dropna()
                rows.append(dict(model=model, reference="the identity test baseline_paper vs baseline", max_abs_diff=d.max(), mean_abs_diff=d.mean()))
    rep = pd.DataFrame(rows).round(4)
    if len(rep):
        rep.to_csv(OUT / "reproduction_check.csv", index=False)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# identity screen: category means over {n_models} models\n\n")
        f.write("Z-scored within model and condition as in the paper; pain axis = mean of the S1 and S2 z; all categories and all items included.\n\n")
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


if __name__ == "__main__":
    main()
