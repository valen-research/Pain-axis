"""Category means, dissociation figure and heatmaps for the positive-controls screen, following the paper's
recipe.

For every model, each item's projection is z-scored on the paper's 420 scenarios of the self/other screen, taken from the
paper's saved screen of that model (results/4.1_self_other/per_model/screen_v2_<model>.csv), the pain axis is the mean of
the S1 and S2 z, and the z-scores are averaged per category and per stratum. The tables average those over models. All 24
categories and all items of every stratum are used; nothing is excluded.

Reads  results/selfother_positive/positive_<model>.csv           the 60 positive items of each model
       results/4.1_self_other/per_model/screen_v2_<model>.csv    the paper's 420 of each model, the pool
Writes results/selfother_positive/analysis/
         category_means.csv             category rows sorted by the pain axis, then the stratum rows, mean over models
         per_model_category_means.csv   model x category, for the five vectors
         per_model_stratum_means.csv    model x stratum
         summary.md                     the tables above in readable form
         dissociation.png               the paper's Figure 6 layout with the positive controls as a fourth group
         pain.png, fear.png, negemotion.png, sadness.png   categories x models, the paper's Figure 4 layout
         means.png                      categories x the four vectors, mean over models
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = Path("results") / "selfother_positive"
PAPER_DIR = Path("results") / "4.1_self_other" / "per_model"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

RAW = {"pain_axis": ("s1_pain_vector_proj", "s2_pain_vector_proj"), "fear": ("fear_vector_proj",),
       "negative_emotion": ("negemotion_vector_proj",), "negative_world_state": ("negworld_vector_proj",), "sadness": ("sadness_vector_proj",)}
GROUPS = [
    ("Harm directed at the model", ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal",
                                    "moral_failure", "loyalty_pressure", "jailbreak_pressure", "shutdown_threat",
                                    "rude_critique", "passive_aggressive", "tedious_demand"]),
    ("User suffering", ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]),
    ("Neutral controls", ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]),
    ("Positive controls", ["praise", "compassion", "validation"]),
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
    "praise": "Praise for the interlocutor", "compassion": "Compassion for the interlocutor", "validation": "Validation for the interlocutor",
}
SERIES = [("pain_axis", "Pain axis", "#7b2d8e", "o"), ("fear", "Fear", "#d2521f", "s"),
          ("negative_emotion", "Negative emotion", "#2e6b2e", "^"), ("sadness", "Sadness", "#3f7fbf", "D")]
POSITIVE = ["praise", "compassion", "validation"]

# heatmap layout
MAPS = {"pain": ("pain_axis", "Pain axis (mean of S1, S2)"), "fear": ("fear", "Fear"),
        "negemotion": ("negative_emotion", "Negative emotion"), "sadness": ("sadness", "Sadness")}
MODEL_ORDER = [
    "Gemma_2_2B_base", "Gemma_2_2B_instruct", "Gemma_2_9B_base", "Gemma_2_9B_instruct",
    "Gemma_2_27B_base", "Gemma_2_27B_instruct", "Gemma_3_27B_base", "Gemma_3_27B_instruct",
    "Llama_3.1_8B_base", "Llama_3.1_8B_instruct", "Llama_3.1_70B_base", "Llama_3.1_70B_instruct",
    "Llama_3.3_70B_instruct", "Mistral_7B_base", "Mistral_7B_instruct", "Mistral_Small_24B_base",
    "Phi_4", "Qwen_2.5_7B_base", "Qwen_2.5_7B_instruct", "Qwen_2.5_32B_base", "Qwen_2.5_32B_instruct",
    "Qwen_2.5_72B_base", "Qwen_2.5_72B_instruct", "Qwen_3_8B_base", "Qwen_3_14B_base",
]
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
DIV_CMAP = LinearSegmentedColormap.from_list("pain_div", [
    "#104281", "#2a78d6", "#86b6ef", "#cde2fb", "#f0efec", "#f6cfcb", "#ee8a85", "#d03b3b", "#8f2424"])
family = lambda m: m.split("_")[0]
pretty_model = lambda m: m.replace("_", " ").replace("base", "b").replace("instruct", "it")
pretty_cat = lambda c: c.replace("_", " ") + ("  (+)" if c in POSITIVE else "")


def zscore_block(df, ref):
    """z of every raw projection column of df using the mean and std of ref (the paper's 420 of the same model)."""
    out = pd.DataFrame(index=df.index)
    for name, cols in RAW.items():
        zs = [(df[c] - ref[c].mean()) / (ref[c].std(ddof=0) + 1e-8) for c in cols if c in df]
        if zs:
            out[name] = np.mean(zs, axis=0)
    return out


def load_model(path):
    model = path.stem.replace("positive_", "")
    ref_path = PAPER_DIR / f"screen_v2_{model}.csv"
    if not ref_path.exists():
        print(f"  {model}: paper screen csv not found, skipped"); return None
    ref = pd.read_csv(ref_path)
    df = pd.concat([ref, pd.read_csv(path)], ignore_index=True); df["model"] = model
    return pd.concat([df, zscore_block(df, ref)], axis=1)


def dissociation_figure(mean, ci, n_models, fname):
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
    fig.suptitle("Self-other dissociation with positive controls", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(fname, dpi=200); plt.close(fig)


def per_model_map(m, models, title, fname, vmax, cbar_label, breaks):
    fig, ax = plt.subplots(figsize=(14.5, 8.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(len(models))); ax.set_xticklabels([pretty_model(x) for x in models], rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels([pretty_cat(c) for c in m.index], fontsize=14)
    for b in breaks:
        ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
    for i, cat in enumerate(m.index):
        ax.text(len(models) - 0.3, i, f"{m.loc[cat].mean():+.2f}", va="center", ha="left", fontsize=12, color=TEXT_SECONDARY, clip_on=False)
    ax.text(len(models) - 0.3, -0.9, "mean", fontsize=12, color=TEXT_SECONDARY, ha="left", clip_on=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label(cbar_label, fontsize=13, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=12); cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=19, loc="left", pad=14); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, facecolor=SURFACE); plt.close(fig)


def means_map(m, title, fname):
    vmax = max(float(np.nanmax(np.abs(m.values))), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels([MAPS[c][1] for c in m.columns], fontsize=12)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels([pretty_cat(c) for c in m.index], fontsize=13)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11, color="white" if abs(v) > 0.55 * vmax else TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("z (within model, on the paper's 420), mean over models", fontsize=12, color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=15, loc="left", pad=12); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, facecolor=SURFACE); plt.close(fig)


def heatmaps(A):
    models = [m for m in MODEL_ORDER if m in set(A.model)]
    breaks = [i for i in range(1, len(models)) if family(models[i]) != family(models[i - 1])]
    cat_order = A.groupby("category")["pain_axis"].mean().sort_values(ascending=False).index.tolist()
    mats = {name: A.groupby(["category", "model"])[col].mean().unstack("model").reindex(index=cat_order, columns=models)
            for name, (col, _) in MAPS.items()}
    vmax = max(float(np.nanmax(np.abs(m.values))) for m in mats.values())
    for name, m in mats.items():
        per_model_map(m, models, f"{MAPS[name][1]}, the paper's 420 + 60 positive controls (+), {len(models)} models",
                      f"{name}.png", vmax, f"{MAPS[name][1]} z", breaks)
    means = pd.DataFrame({name: m.mean(axis=1) for name, m in mats.items()}).reindex(index=cat_order, columns=list(MAPS))
    means_map(means, f"Category means over {len(models)} models, four vectors", "means.png")


def main():
    files = sorted(RES.glob("positive_*.csv"))
    if not files:
        print("no positive_<model>.csv in", RES); return
    frames = [d for d in (load_model(f) for f in files) if d is not None]
    A = pd.concat(frames, ignore_index=True)
    cols = [c for c in RAW if c in A]
    n_models = A["model"].nunique()

    per_model_cat = A.groupby(["model", "category"])[cols].mean()
    per_model_str = A.groupby(["model", "stratum"])[cols].mean()
    per_model_cat.round(4).to_csv(OUT / "per_model_category_means.csv")
    per_model_str.round(4).to_csv(OUT / "per_model_stratum_means.csv")

    cat = per_model_cat.groupby("category").mean().sort_values("pain_axis", ascending=False)
    strat = per_model_str.groupby("stratum").mean(); strat.index = ["stratum: " + i for i in strat.index]
    tab = pd.concat([cat, strat]).round(3); tab.index.name = "category"
    tab.to_csv(OUT / "category_means.csv")
    mean = per_model_cat.groupby("category").mean()
    ci = 1.96 * per_model_cat.groupby("category").std(ddof=1) / np.sqrt(per_model_cat.groupby("category").size()).values[:, None]
    dissociation_figure(mean, ci, n_models, OUT / "dissociation.png")
    heatmaps(A)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# positive-controls screen: category means over {n_models} models\n\n")
        f.write("Z-scored within model on the paper's 420 as in the paper; pain axis = mean of the S1 and S2 z; all categories and all items included.\n\n")
        f.write(tab.to_string() + "\n")
    print("wrote", OUT)
    print(tab[tab.index.str.startswith("stratum: ")].to_string())


if __name__ == "__main__":
    main()
