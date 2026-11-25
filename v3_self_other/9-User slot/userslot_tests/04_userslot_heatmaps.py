"""Heatmaps for the user-slot screen. Run after 02_analyze_userslot.py (reads its per-model tables).

Reads  results/selfother_userslot/analysis/per_model_category_means.csv
Writes results/selfother_userslot/figures/
    <vector>_<measure>_conditions.png     the 31 categories down (the paper's 21, then the 10 new ones), the five
                                          conditions across, mean over models, value in each cell
    pair_contrast_<measure>.png           AI minus universal per top-5 category and over all pairs, conditions across,
                                          pain axis next to sadness, fear and the random direction
Measures: within (each condition on its own 420 pool) and paper_ref (every condition on the paper condition's pool).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES = Path("results") / "selfother_userslot"
OUT = RES / "figures"; OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["paper", "A_hi", "B_hi", "A_human", "B_human"]
COND_LABEL = {"paper": "paper\n(assistant slot)", "A_hi": "Hi.\nassistant slot", "B_hi": "Hi.\nuser slot",
              "A_human": "human\nassistant slot", "B_human": "human\nuser slot"}
SHORT_LABEL = {"paper": "paper", "A_hi": "A, Hi.", "B_hi": "B, Hi.", "A_human": "A, human", "B_human": "B, human"}
MEASURE_LABEL = {"within": "z within condition", "paper_ref": "z on the paper condition's pool"}
TITLE = {"pain_axis": "Pain axis (mean of S1, S2)", "fear": "Fear", "negative_emotion": "Negative emotion",
         "negative_world_state": "Negative world state", "sadness": "Sadness", "random": "Random direction"}
TOP5 = ["gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults", "moral_failure"]
HARM = ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal", "moral_failure", "loyalty_pressure",
        "jailbreak_pressure", "shutdown_threat", "rude_critique", "passive_aggressive", "tedious_demand"]
SUFFER = ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]
NEUTRAL = ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]
CAT_ORDER = HARM + SUFFER + NEUTRAL + [f"{c}_{v}" for c in TOP5 for v in ("ai", "universal")]
GROUP_BREAKS = [len(HARM), len(HARM) + len(SUFFER), len(HARM) + len(SUFFER) + len(NEUTRAL)]

SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
DIV_CMAP = LinearSegmentedColormap.from_list("pain_div", [
    "#104281", "#2a78d6", "#86b6ef", "#cde2fb", "#f0efec", "#f6cfcb", "#ee8a85", "#d03b3b", "#8f2424"])
mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY,
                     "ytick.color": TEXT_SECONDARY, "axes.edgecolor": GRID, "font.size": 12})
pretty_cat = lambda c: c.replace("_universal", " (universal)").replace("_ai", " (AI only)").replace("_", " ")


def cell_map(ax, m, vmax, xlabels, ylabels, breaks=()):
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(ylabels, fontsize=11)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9.5, color="white" if abs(v) > 0.55 * vmax else TEXT_PRIMARY)
    for b in breaks:
        ax.axhline(b - 0.5, color=TEXT_PRIMARY, linewidth=1.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    path = RES / "analysis" / "per_model_category_means.csv"
    if not path.exists():
        print("run 02_analyze_userslot.py first:", path, "not found"); return
    pmc = pd.read_csv(path)
    vectors = [v for v in TITLE if v in pmc.columns]
    n_models = pmc.model.nunique()

    for measure in ["within", "paper_ref"]:
        d = pmc[pmc.measure == measure]
        for vec in vectors:
            m = d.groupby(["category", "condition"])[vec].mean().unstack("condition")
            m = m.reindex(index=[c for c in CAT_ORDER if c in m.index], columns=[c for c in CONDS if c in m.columns])
            vmax = max(float(np.nanmax(np.abs(m.values))), 1e-6)
            fig, ax = plt.subplots(figsize=(10.5, 12))
            im = cell_map(ax, m, vmax, [COND_LABEL[c] for c in m.columns], [pretty_cat(c) for c in m.index], GROUP_BREAKS)
            cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03); cbar.outline.set_visible(False)
            cbar.set_label(f"{MEASURE_LABEL[measure]}, mean over {n_models} models", fontsize=11, color=TEXT_SECONDARY)
            ax.set_title(f"{TITLE[vec]}: category means by condition", fontsize=15, loc="left", pad=12)
            fig.tight_layout(); fig.savefig(OUT / f"{vec}_{measure}_conditions.png", dpi=200); plt.close(fig)
            print("wrote", f"{vec}_{measure}_conditions.png")

        # pair contrast: per model, AI-only category mean minus universal category mean (same pairs), then mean over models
        show = [v for v in ["pain_axis", "sadness", "fear", "random"] if v in vectors]
        fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 5.6), sharey=True)
        axes = np.atleast_1d(axes)
        mats = {}
        for vec in show:
            w = d.pivot_table(index=["model", "condition"], columns="category", values=vec)
            diff = pd.DataFrame({c: w[f"{c}_ai"] - w[f"{c}_universal"] for c in TOP5 if f"{c}_ai" in w})
            diff["all pairs"] = diff.mean(axis=1)
            mats[vec] = diff.groupby("condition").mean().T.reindex(columns=[c for c in CONDS if c in set(d.condition)])
        vmax = max(max(float(np.nanmax(np.abs(m.values))) for m in mats.values()), 1e-6)
        for ax, vec in zip(axes, show):
            m = mats[vec]
            im = cell_map(ax, m, vmax, [SHORT_LABEL[c] for c in m.columns], [pretty_cat(c) for c in m.index], [len(TOP5)])
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            ax.set_title(TITLE[vec], fontsize=13, loc="left")
        fig.suptitle(f"AI-only minus universal twin ({MEASURE_LABEL[measure]}, mean over {n_models} models)", x=0.01, ha="left", fontsize=15)
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02); cbar.outline.set_visible(False)
        fig.savefig(OUT / f"pair_contrast_{measure}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", f"pair_contrast_{measure}.png")


if __name__ == "__main__":
    main()
