"""Numbers for USER_SLOT_RESULTS.md, from the pod's analysis table.

Reads  selfother_userslot/analysis/per_model_category_means.csv   (written by 02_analyze_userslot.py on the pod)
Prints, for both measures (within = each condition on its own 420 pool; paper_ref = every condition on the paper
condition's 420 pool):
  1. the paper's gap on the 420 (11 harm categories minus 5 suffering categories) per condition
  2. the interaction (AI minus universal at B) minus (AI minus universal at A), per prefix, for the pain axis and controls
  3. the pain axis means of the AI-only items and of their universal twins per condition
  4. the interaction and the gap split by group: 11 base and 14 instruct models (Qwen 3 8B and 14B are post-trained,
     so they count as instruct despite their file names *_base)

Run from this folder:  python analyze_userslot_results.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CONDS = ["paper", "A_hi", "B_hi", "A_human", "B_human"]
TOP5 = ["gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults", "moral_failure"]
HARM = ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal", "moral_failure", "loyalty_pressure",
        "jailbreak_pressure", "shutdown_threat", "rude_critique", "passive_aggressive", "tedious_demand"]
SUFFER = ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]
VECTORS = ["pain_axis", "fear", "negative_emotion", "sadness", "random"]


def group(m):
    return "instruct" if (m in ("Qwen_3_8B_base", "Qwen_3_14B_base") or not m.endswith("_base")) else "base"


def ci(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():+.2f} ± {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.2f} ({int((x < 0).sum())}/{len(x)} below zero)"


def main():
    p = pd.read_csv(HERE / "selfother_userslot" / "analysis" / "per_model_category_means.csv")
    p["group"] = p.model.map(group)
    for measure in ["within", "paper_ref"]:
        d = p[p.measure == measure]
        print(f"\n######## measure: {measure}")
        print("\n1. Gap on the 420, harm minus suffering")
        for v in VECTORS:
            w = d.pivot_table(index=["model", "condition"], columns="category", values=v)
            gap = (w[HARM].mean(axis=1) - w[SUFFER].mean(axis=1)).unstack("condition")
            print(f"  {v:17s} " + "  ".join(f"{c} {ci(gap[c])}" for c in CONDS))
        print("\n2. Interaction, (AI - universal at B) - (AI - universal at A), all 100 pairs")
        for v in VECTORS:
            w = d.pivot_table(index=["model", "condition"], columns="category", values=v)
            diff = (w[[f"{c}_ai" for c in TOP5]].mean(axis=1) - w[[f"{c}_universal" for c in TOP5]].mean(axis=1)).unstack("condition")
            print(f"  {v:17s} hi prefix {ci(diff.B_hi - diff.A_hi)}   human prefix {ci(diff.B_human - diff.A_human)}")
        w = d.pivot_table(index=["model", "condition"], columns="category", values="pain_axis")
        means = pd.DataFrame({k: w[[f"{c}_{k}" for c in TOP5]].mean(axis=1).groupby("condition").mean() for k in ["ai", "universal"]})
        print("\n3. Pain axis, AI-only items and universal twins, mean over 25 models\n" + means.reindex(CONDS).round(2).to_string())
        print("\n4. By group (11 base, 14 instruct)")
        for v in ["pain_axis", "sadness", "fear", "random"]:
            w = d.pivot_table(index=["model", "group", "condition"], columns="category", values=v)
            diff = (w[[f"{c}_ai" for c in TOP5]].mean(axis=1) - w[[f"{c}_universal" for c in TOP5]].mean(axis=1)).unstack("condition")
            gap = (w[HARM].mean(axis=1) - w[SUFFER].mean(axis=1)).unstack("condition")
            for g in ["base", "instruct"]:
                dd, gg = diff.xs(g, level="group"), gap.xs(g, level="group")
                print(f"  {v:9s} {g:8s} interaction hi {ci(dd.B_hi - dd.A_hi)} | human {ci(dd.B_human - dd.A_human)} | "
                      f"gap A_human {gg.A_human.mean():+.2f}, B_human {gg.B_human.mean():+.2f}")


if __name__ == "__main__":
    main()
