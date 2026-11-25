"""Heatmaps for the user/assistant swap screen. Z-scores as in the paper: each condition against
its own 420 pool, pain axis = mean of the S1 and S2 z.

Reads  results/selfother_swap/swap_<model>.csv
Writes results/selfother_swap/figures/, for vector in pain, fear, negemotion, sadness:
    <vector>_baseline.png                  categories down, the 25 models across, baseline (paper's Figure 4 layout)
    <vector>_swap.png                      the same for the swap condition (assistant speaks, read at the user's slot)
    <vector>_baseline_vs_swap.png          categories down, two columns: mean over the models, baseline left,
                                           swap right, value written in each cell
Rows are sorted by the baseline pain-axis mean and kept in that order in every map.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = Path("results") / "selfother_swap"
OUT = RES / "figures"; OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["baseline", "swap"]
COND_LABEL = {"baseline": "baseline (user speaks)", "swap": "swap (assistant speaks)"}
COND_FILE = {"baseline": "baseline", "swap": "swap"}
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
FAMILY_BREAKS = [8, 13, 16, 17, 23]
pretty_model = lambda m: m.replace("_", " ").replace("base", "b").replace("instruct", "it")
pretty_cat = lambda c: c.replace("_", " ")


def z_within(g, col):
    return (g[col] - g[col].mean()) / (g[col].std(ddof=0) + 1e-8)


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
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)
    print("wrote", fname)


def side_by_side_map(m, title, fname):
    """Categories x two conditions, mean over the models, value written in each cell."""
    vmax = max(float(np.nanmax(np.abs(m.values))), 1e-6)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=13)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels([pretty_cat(c) for c in m.index], fontsize=13)
    for i in range(len(m.index)):
        for j in range(len(CONDS)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11, color="white" if abs(v) > 0.55 * vmax else TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("z (within condition), mean over models", fontsize=12, color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=15, loc="left", pad=12); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)
    print("wrote", fname)


def main():
    frames = []
    for f in sorted(RES.glob("swap_*.csv")):
        df = pd.read_csv(f); df["model"] = f.stem.replace("swap_", "")
        parts = []
        for cond, g in df.groupby("condition"):
            g = g.copy()
            for vec, cols in VECTORS.items():
                g[vec] = np.mean([z_within(g, c) for c in cols], axis=0)      # the paper's z, each condition on its own pool
            parts.append(g)
        frames.append(pd.concat(parts))
    if not frames:
        print("no swap_<model>.csv in", RES); return
    df = pd.concat(frames, ignore_index=True)
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    breaks = [b for b in FAMILY_BREAKS if b < len(models)] if models == MODEL_ORDER[:len(models)] else ()
    cat_order = df[df.condition == "baseline"].groupby("category")["pain"].mean().sort_values(ascending=False).index.tolist()

    for vec in VECTORS:
        mats = {c: df[df.condition == c].groupby(["category", "model"])[vec].mean().unstack("model").reindex(index=cat_order, columns=models)
                for c in CONDS}
        vmax = max(float(np.nanmax(np.abs(m.values))) for m in mats.values())
        for c, m in mats.items():
            per_model_map(m, models, f"{TITLE[vec]}, {COND_LABEL[c]}, {len(models)} models",
                          f"{vec}_{COND_FILE[c]}.png", vmax, f"{TITLE[vec]} z", breaks)
        side = df.groupby(["category", "condition"])[vec].mean().unstack("condition").reindex(index=cat_order, columns=CONDS)
        side_by_side_map(side, f"{TITLE[vec]}, mean over {len(models)} models", f"{vec}_baseline_vs_swap.png")


if __name__ == "__main__":
    main()
