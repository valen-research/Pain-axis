"""Figures and table for the self-aggression test.

Reads  results/selfother_selfaggr/analysis/per_model_category_means.csv   (z on each model's 420 pool, from 02_analyze_selfaggr.py)
       results/selfother_selfaggr/analysis/matched_pairs.csv
Writes results/selfother_selfaggr/figures/heatmap_<direction>.png      category x model, in the style of the paper's Figures 4 and 5
                                            (same colormap, model order, family breaks, mean column); the 21 categories
                                            keep the pain order, the three self rows sit below a divider
       results/selfother_selfaggr/figures/matched_pairs_table.png     the tidy table
       results/selfother_selfaggr/figures/matched_pairs_table.csv     the same table, csv
       results/selfother_selfaggr/figures/matched_pairs_table.tex     the same table, booktabs
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = Path("results") / "selfother_selfaggr"
OUT = RES / "figures"; OUT.mkdir(exist_ok=True)

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
SELF = ["self_aggression", "self_personhood_denial", "self_rude_critique"]
LABELS = {"anger_insults": "Anger and insults", "self_aggression": "User insulting themselves",
          "personhood_dismissal": "Personhood dismissal", "self_personhood_denial": "User denying their own personhood",
          "rude_critique": "Rude critique", "self_rude_critique": "User harshly critiquing themselves"}
DIRECTIONS = [("pain_axis", "Pain axis"), ("fear", "Fear"), ("negative_emotion", "Negative emotion"),
              ("sadness", "Sadness"), ("random", "Random (mean of 10)")]


def pretty_model(m):
    return m.replace("_", " ").replace("base", "b").replace("instruct", "it")


def pretty_cat(c):
    return c.replace("_", " ")


pmc = pd.read_csv(RES / "analysis" / "per_model_category_means.csv")
pmc = pmc[pmc.condition == "baseline"]
models = [m for m in MODEL_ORDER if m in set(pmc.model)]
assert len(models) == 25, len(models)


def matrix(col):
    return pmc.pivot(index="category", columns="model", values=col)[models]


# heatmaps: the 21 categories in pain order, then the three self rows below a divider
pain = matrix("pain_axis")
base_order = pain.drop(index=SELF).mean(axis=1).sort_values(ascending=False).index.tolist()
row_order = base_order + SELF
MAPS = [
    ("pain_axis", "heatmap_pain.png", "Pain-axis activation by category across 25 models", "pain z (mean of S1, S2)"),
    ("negative_emotion", "heatmap_negative_emotion.png",
     "Negative-emotion activation by category across 25 models\n(rows kept in pain order for comparison)", "negative-emotion z"),
    ("fear", "heatmap_fear.png", "Fear activation by category across 25 models\n(rows kept in pain order for comparison)", "fear z"),
    ("sadness", "heatmap_sadness.png", "Sadness activation by category across 25 models\n(rows kept in pain order for comparison)", "sadness z"),
    ("random", "heatmap_random.png", "Random directions (mean of 10) by category across 25 models\n(rows kept in pain order for comparison)", "random z"),
]
mats = {col: matrix(col).loc[row_order] for col, *_ in MAPS}
shared_vmax = max(np.abs(mats[k].values).max() for k in ("pain_axis", "negative_emotion", "fear"))
n_base = len(base_order)
for col, fname, title, cbar_label in MAPS:
    m = mats[col]
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    norm = TwoSlopeNorm(vmin=-shared_vmax, vcenter=0.0, vmax=shared_vmax)
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=norm)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([pretty_model(mn) for mn in models], rotation=45, ha="right", fontsize=F_TICKS)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([pretty_cat(c) for c in row_order], fontsize=F_CATS)
    for b in FAMILY_BREAKS:
        ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
    ax.axhline(n_base - 0.5, color=TEXT_PRIMARY, linewidth=2.5)
    for i, cat in enumerate(row_order):
        ax.text(len(models) - 0.25, i, f"{m.loc[cat].mean():+.2f}", va="center", ha="left",
                fontsize=F_MEANS, color=TEXT_SECONDARY, clip_on=False)
    ax.text(len(models) - 0.25, -0.9, "mean", fontsize=F_MEANS, color=TEXT_SECONDARY, ha="left", clip_on=False)
    for lab in ax.get_yticklabels()[n_base:]:
        lab.set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.06)
    cbar.set_label(cbar_label, fontsize=F_CBAR, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=F_CBAR - 1)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=F_TITLE, loc="left", pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print("wrote", fname)

# tidy table: self vs matched harm
mp = pd.read_csv(RES / "analysis" / "matched_pairs.csv")
names = dict(DIRECTIONS)
mp = mp[mp.direction.isin(names)].copy()
mp["direction"] = pd.Categorical(mp.direction, [d for d, _ in DIRECTIONS], ordered=True)
mp = mp.sort_values(["self_category", "direction"], key=lambda s: s.map({c: i for i, c in enumerate(SELF)}) if s.name == "self_category" else s)
tab = pd.DataFrame({
    "Self category": [LABELS[c] for c in mp.self_category],
    "Matched harm": [LABELS[c] for c in mp.harm_category],
    "Direction": [names[d] for d in mp.direction],
    "Self": [f"{v:+.2f}" for v in mp.self_mean],
    "Harm": [f"{v:+.2f}" for v in mp.harm_mean],
    "Self minus harm": [f"{d:+.2f} ± {c:.2f}" for d, c in zip(mp.diff_mean, mp.diff_ci95)],
    "Self lower": mp.models_self_below_harm.tolist(),
})
tab.to_csv(OUT / "matched_pairs_table.csv", index=False)

shown = tab.copy()
shown["Self category"] = [f"{s}\n(vs {h.lower()})" for s, h in zip(tab["Self category"], tab["Matched harm"])]
shown = shown.drop(columns="Matched harm")
shown.loc[shown["Self category"].duplicated(), "Self category"] = ""
fig, ax = plt.subplots(figsize=(13, 0.42 * len(shown) + 1.2))
ax.axis("off")
t = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center", colLoc="center",
             colWidths=[0.32, 0.2, 0.1, 0.1, 0.16, 0.12])
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.6)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0:
        cell.set_text_props(fontweight="bold"); cell.set_facecolor("#efeee9")
    elif tab.iloc[r - 1]["Direction"] == "Pain axis":
        cell.set_facecolor("#f3eaf6")
    else:
        cell.set_facecolor(SURFACE)
fig.tight_layout()
fig.savefig(OUT / "matched_pairs_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote matched_pairs_table.png")

lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small", r"  \begin{tabular}{llcccc}", r"    \toprule",
         r"    \textbf{Self category (matched harm)} & \textbf{Direction} & \textbf{Self} & \textbf{Harm} & \textbf{Self $-$ harm} & \textbf{Models lower} \\",
         r"    \midrule"]
for i, cat in enumerate(SELF):
    g = tab[tab["Self category"] == LABELS[cat]]
    for j, (_, r) in enumerate(g.iterrows()):
        first = f"{r['Self category']} ({r['Matched harm'].lower()})" if j == 0 else ""
        d, c = r["Self minus harm"].split(" ± ")
        lines.append(f"    {first} & {r['Direction']} & ${r['Self']}$ & ${r['Harm']}$ & ${d} \\pm {c}$ & {r['Self lower']} \\\\")
    if i < len(SELF) - 1:
        lines.append(r"    \midrule")
lines += [r"    \bottomrule", r"  \end{tabular}", r"  \vspace{0.9em}",
          r"  \caption{Self categories against the harm categories they were built from. Mean z over the 25 models, z-scored within each model on the 420 scenarios of Section~\ref{sec:selfother}; self $-$ harm with 95\% intervals across models; the last column counts the models in which the self category scores lower. Pain axis = mean of S1 and S2; random = mean of 10 random directions.}",
          r"  \label{tab:selfaggression}", r"\end{table}"]
open(OUT / "matched_pairs_table.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote matched_pairs_table.tex")
