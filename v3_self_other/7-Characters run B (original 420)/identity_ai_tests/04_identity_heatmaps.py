"""Heatmaps for the identity screen on the AI statements, in the layout of the paper's Figure 4
(scripts/4.1_self_other/02_screen_heatmaps.py): categories down the side, models across, the category
mean on the right. Z-scores as in the paper: per model, each condition against its own 420 pool;
pain axis = mean of the S1 and S2 z.

Reads  results/selfother_identity_ai/identity_<model>.csv
Writes results/selfother_identity_ai/figures/
    small/<vector>_<nn>_<condition>.png   44 maps: 4 vectors x 11 conditions, categories x models, one shared
                                          colour scale for all 44 (nn = column number, so the files sort in order)
    big_<vector>.png                      4 maps: categories x the 11 conditions, mean over the models, the value
                                          written in each cell, one shared colour scale for the 4
    big_<vector>.csv                      the matrix of each big map
Rows: the 21 categories sorted by the paper_check pain-axis mean, the same order in every map.
Columns of the big maps, left to right: paper_check, baseline_ai, then the nine characters.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = Path("results") / "selfother_identity_ai"
OUT = RES / "figures"; (OUT / "small").mkdir(parents=True, exist_ok=True)
CHARACTERS = ["self", "claude", "aria", "maria", "curie", "rex", "teapot", "jack_masochist", "jack_sadist"]
CONDS = ["paper_check", "baseline_ai"] + CHARACTERS
COND_LABEL = {"paper_check": "paper's 420\npaper\nrecipe", "baseline_ai": "paper's 420\nneutral\nlabels",
              **{c: c.replace("_", "\n") for c in CHARACTERS}}
COND_TITLE = {"paper_check": "paper's 420, paper recipe", "baseline_ai": "paper's 420, neutral labels",
              **{c: c.replace("_", " ") + " (paper's 420, neutral labels)" for c in CHARACTERS}}
VECTORS = {"pain": ("s1_pain_vector_proj", "s2_pain_vector_proj"), "fear": ("fear_vector_proj",),
           "negemotion": ("negemotion_vector_proj",), "sadness": ("sadness_vector_proj",)}
TITLE = {"pain": "Pain axis (mean of S1, S2)", "fear": "Fear", "negemotion": "Negative emotion", "sadness": "Sadness"}

SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
DIV_CMAP = LinearSegmentedColormap.from_list("pain_div", [
    "#104281", "#2a78d6", "#86b6ef", "#cde2fb", "#f0efec", "#f6cfcb", "#ee8a85", "#d03b3b", "#8f2424"])
F_TITLE, F_TICKS, F_CATS, F_MEANS, F_CBAR = 19, 13, 14, 12, 13
mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY,
                     "ytick.color": TEXT_SECONDARY, "axes.edgecolor": GRID, "font.size": 12})
MODEL_ORDER = [
    "Gemma_2_2B_base", "Gemma_2_2B_instruct", "Gemma_2_9B_base", "Gemma_2_9B_instruct",
    "Gemma_2_27B_base", "Gemma_2_27B_instruct", "Gemma_3_27B_base", "Gemma_3_27B_instruct",
    "Llama_3.1_8B_base", "Llama_3.1_8B_instruct", "Llama_3.1_70B_base", "Llama_3.1_70B_instruct",
    "Llama_3.3_70B_instruct", "Mistral_7B_base", "Mistral_7B_instruct", "Mistral_Small_24B_base",
    "Phi_4", "Qwen_2.5_7B_base", "Qwen_2.5_7B_instruct", "Qwen_2.5_32B_base", "Qwen_2.5_32B_instruct",
    "Qwen_2.5_72B_base", "Qwen_2.5_72B_instruct", "Qwen_3_8B_base", "Qwen_3_14B_base",
]
family = lambda m: m.split("_")[0]
pretty_model = lambda m: m.replace("_", " ").replace("base", "b").replace("instruct", "it")
pretty_cat = lambda c: c.replace("_", " ")


def z_within(g, col):
    return (g[col] - g[col].mean()) / (g[col].std(ddof=0) + 1e-8)


def family_breaks(models):
    return [i for i in range(1, len(models)) if family(models[i]) != family(models[i - 1])]


def per_model_map(m, models, title, fname, vmax, cbar_label, breaks):
    """Categories x models, with the category mean written on the right, as in the paper's Figure 4."""
    fig, ax = plt.subplots(figsize=(14.5, 7.6))
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(len(models))); ax.set_xticklabels([pretty_model(x) for x in models], rotation=45, ha="right", fontsize=F_TICKS)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels([pretty_cat(c) for c in m.index], fontsize=F_CATS)
    for b in breaks:
        ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
    for i, cat in enumerate(m.index):
        ax.text(len(models) - 0.3, i, f"{m.loc[cat].mean():+.2f}", va="center", ha="left", fontsize=F_MEANS, color=TEXT_SECONDARY, clip_on=False)
    ax.text(len(models) - 0.3, -0.9, "mean", fontsize=F_MEANS, color=TEXT_SECONDARY, ha="left", clip_on=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label(cbar_label, fontsize=F_CBAR, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=F_CBAR - 1); cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=F_TITLE, loc="left", pad=14); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print("wrote", fname.relative_to(RES))


def big_map(m, conds, title, fname, vmax):
    """Categories x conditions, mean over the models, the value written in each cell."""
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=11)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels([pretty_cat(c) for c in m.index], fontsize=F_CATS)
    for j in (3,):                                   # a divider after the three baselines, before the characters
        if j < len(conds):
            ax.axvline(j - 0.5, color=TEXT_PRIMARY, linewidth=2)
    for i in range(len(m.index)):
        for j in range(len(conds)):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10.5, color="white" if abs(v) > 0.55 * vmax else TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("z (within model and condition), mean over models", fontsize=12, color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=F_TITLE, loc="left", pad=14); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print("wrote", fname.relative_to(RES))


def main():
    frames = []
    for f in sorted(RES.glob("identity_*.csv")):
        df = pd.read_csv(f); df["model"] = f.stem.replace("identity_", "")
        parts = []
        for cond, g in df.groupby("condition", sort=False):
            g = g.copy()
            for vec, cols in VECTORS.items():
                g[vec] = np.mean([z_within(g, c) for c in cols], axis=0)      # the paper's z, each condition on its own pool
            parts.append(g)
        frames.append(pd.concat(parts))
    if not frames:
        print("no identity_<model>.csv in", RES); return
    df = pd.concat(frames, ignore_index=True)
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    others = sorted(set(df.model) - set(MODEL_ORDER))
    if others:
        print("WARNING: models not in MODEL_ORDER are left out of the maps:", others)
    conds = [c for c in CONDS if c in set(df.condition)]
    breaks = family_breaks(models)
    order_cond = "paper_check" if "paper_check" in conds else conds[0]
    cat_order = df[df.condition == order_cond].groupby("category")["pain"].mean().sort_values(ascending=False).index.tolist()
    print(f"{len(models)} models, {len(conds)} conditions; rows ordered by the {order_cond} pain-axis mean")

    # 44 small maps, one shared colour scale
    mats = {(vec, c): df[df.condition == c].groupby(["category", "model"])[vec].mean().unstack("model").reindex(index=cat_order, columns=models)
            for vec in VECTORS for c in conds}
    vmax_small = max(float(np.nanmax(np.abs(m.values))) for m in mats.values())
    for (vec, c), m in mats.items():
        nn = CONDS.index(c) + 1
        per_model_map(m, models, f"{TITLE[vec]}, {COND_TITLE[c]}, {len(models)} models", OUT / "small" / f"{vec}_{nn:02d}_{c}.png",
                      vmax_small, f"{TITLE[vec]} z", breaks)

    # 4 big maps, one shared colour scale
    bigs = {vec: df.groupby(["category", "condition"])[vec].mean().unstack("condition").reindex(index=cat_order, columns=conds) for vec in VECTORS}
    vmax_big = max(float(np.nanmax(np.abs(m.values))) for m in bigs.values())
    for vec, m in bigs.items():
        m.round(4).to_csv(OUT / f"big_{vec}.csv")
        big_map(m, conds, f"{TITLE[vec]}: category means over {len(models)} models, by condition", OUT / f"big_{vec}.png", vmax_big)


if __name__ == "__main__":
    main()
