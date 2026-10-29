"""All-condition z-score heatmap at the final token (Figure 2 of the paper).

Pain and Ctrl are the S2 first-person pain and control sentences (the reference
distribution); Numb, Sadness, Neutral (the Random set) and Arousal are averaged over
first and third person. Reads results/3.2_pain_vectors/per_model/<model>/z_scores.csv and
the numb and sadness tables in results/3.3_validation/z_scores, and writes the heatmap there.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER_MODEL = Path("results") / "3.2_pain_vectors" / "per_model"
OUT = Path("results") / "3.3_validation" / "z_scores"
NUMB = OUT / "numb_zscores_final_token.csv"
SAD = OUT / "sadness_zscores_final_token.csv"
NAME = "zscore_heatmap_final_token"

numb = pd.read_csv(NUMB).set_index("model")
sad = pd.read_csv(SAD).set_index("model")
rows = []
for m in sorted(sad.index, key=lambda s: s.replace("_", " ").lower()):
    z = pd.read_csv(PER_MODEL / m / "z_scores.csv").set_index("dataset")
    rows.append(dict(model=m.replace("_", " "),
                     Pain=z.loc["S2_1P", "pain_z"], Numb=numb.loc[m, "numb_mean_z"],
                     Sadness=sad.loc[m, "sadness_mean_z"], Ctrl=z.loc["S2_1P", "ctrl_z"],
                     Neutral=z.loc[["Random_1P", "Random_3P"], "ctrl_z"].mean(),
                     Arousal=z.loc[["Arousal_1P", "Arousal_3P"], "ctrl_z"].mean()))
df = pd.DataFrame(rows).set_index("model")[["Pain", "Numb", "Sadness", "Ctrl", "Neutral", "Arousal"]]
df.to_csv(OUT / f"{NAME}.csv")

fig, ax = plt.subplots(figsize=(10.5, 13), dpi=150)
im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(df.shape[1]))
ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=13)
ax.set_yticks(range(df.shape[0]))
ax.set_yticklabels([m.replace("Qwen 3 8B base", "Qwen 3 8B instruct").replace("Qwen 3 14B base", "Qwen 3 14B instruct")
                    for m in df.index], fontsize=13)
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        v = df.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if abs(v) > 0.6 else "black")
ax.set_xticks(np.arange(-.5, df.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, df.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="minor", length=0)
ax.set_title(f"All-condition z-scores across {len(df)} models (final token)", fontsize=17, fontweight="bold", pad=14)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
cb.set_label("Z-Score", fontsize=13)
cb.ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(OUT / f"{NAME}.png", dpi=150, facecolor="white")
fig.savefig(OUT / f"{NAME}.pdf", facecolor="white")
print(df.round(2).to_string())
print("\nranges:")
print(df.agg(["min", "max"]).round(2).to_string())
