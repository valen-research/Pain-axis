"""Build static_results/COMPLETE_TABLE.csv: the four embeddings next to the 25 language models.

Embedding columns come from the outputs of static_embeddings_pain_axis.py and vocab_check.py in static_results/.
The LLM column comes from the result files of the main Pain-axis repository (PAIN_AXIS_REPO).
Ranges are min to max over the 25 models; single values are means over the 25 models.
"""
import os, json, glob, numpy as np, pandas as pd
from pathlib import Path

REPO = Path(os.environ.get("PAIN_AXIS_REPO", Path(__file__).resolve().parents[2]))
RES = REPO / "results"
OUT = Path(__file__).parent / "static_results"
EMB = [("glove-wiki-gigaword-100", "GloVe 100"), ("glove-wiki-gigaword-300", "GloVe 300"),
       ("word2vec-google-news-300", "Word2Vec"), ("fasttext-wiki-news-subwords-300", "fastText")]
LLM = "LLMs (25 models, paper)"

f2 = lambda x: f"{x:.2f}"
s2 = lambda x: f"{x:+.2f}"
f3 = lambda x: f"{x:.3f}"
rng = lambda xs, f=f2: f"{f(min(xs))}-{f(max(xs))}"
srng = lambda xs: " to ".join(s2(v) for v in (sorted([min(xs), max(xs)], key=abs) if min(xs) * max(xs) > 0 else [min(xs), max(xs)]))

# ---------- language models ----------
per_model = sorted(p for p in (RES / "3.2_pain_vectors/per_model").iterdir() if p.is_dir())
def heldout(ds):
    out = []
    for p in per_model:
        best = json.load(open(p / "summary.json"))["best_layer_final_token"]
        lc = pd.read_csv(p / "layer_curves.csv")
        row = lc[(lc.dataset == ds) & (lc.extraction == "final_token") & (lc.layer == best)]
        if len(row): out.append(float(row.auc_vs_all_controls.iloc[0]))
    return out
ins_s2 = [float(pd.read_csv(p / "auc_summary.csv", index_col=0).loc["S2_1P", "ALL"]) for p in per_model]
s1tab = pd.read_csv(RES / "3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv")
cosm = pd.read_csv(RES / "3.3_validation/cosine_similarity/similarity_MEAN_all_models.csv", index_col=0)
heat = pd.read_csv(RES / "3.3_validation/z_scores/zscore_heatmap_final_token.csv")
numb = pd.read_csv(RES / "3.3_validation/z_scores/numb_zscores_final_token.csv")
zs = [pd.read_csv(p / "z_scores.csv").set_index("dataset") for p in per_model]
catm = pd.read_csv(RES / "4.1_self_other/category_means_25_models.csv").set_index("category")["pain_axis"]
llm_cats = catm[~catm.index.str.startswith("stratum:")].sort_values(ascending=False)

h2, h1 = heldout("S2_1P"), heldout("S1_1P")
# S1 first-person held-out AUC at each model's S2 extraction layer (the paper's held-out S1 figure averages 1P and 3P)
s1c = pd.read_csv(RES / "3.2_pain_vectors/auc_tables/s1_kfold_layer_curves.csv").query("extraction == 'final_token' and dataset == 'S1_1P'")
s1s = pd.read_csv(RES / "3.2_pain_vectors/auc_tables/s1_kfold_summary.csv").query("extraction == 'final_token'")
s1_1p_at_s2_layer = [float(s1c[(s1c.model == m) & (s1c.layer == L)].auc_vs_all_controls.iloc[0]) for m, L in zip(s1s.model, s1s.s2_layer)]
llm = {
    "Held-out AUC, S2 1P": rng(h2) if len(h2) == 25 else "n/r",
    "Held-out AUC, S1 1P": rng(s1_1p_at_s2_layer),
    "Held-out AUC, S2 3P": "n/r", "Held-out AUC, S1 3P": "n/r",
    "In-sample AUC, S2 1P": rng(ins_s2),
    "In-sample AUC, S1 1P": rng(s1tab["S1 1P in-sample"]),
    "cos S1 x S2": f2(cosm.loc["S1_pain", "S2_pain"]),
    "cos S2 x fear": f2(cosm.loc["S2_pain", "Fear"]),
    "cos S2 x neg. emotion": f2(cosm.loc["S2_pain", "NegEmotion"]),
    "cos S2 x sadness": f2(cosm.loc["S2_pain", "Sadness"]),
    "cos S2 x arousal": f2(cosm.loc["S2_pain", "Arousal"]),
    "cos S2 x numb": f2(cosm.loc["S2_pain", "Numb"]),
    "z pain (S2 1P)": srng(heat.Pain), "z controls (S2 1P)": srng(heat.Ctrl),
    "z numb": srng(heat.Numb), "z sadness": srng(heat.Sadness),
    "z neutral (Random)": srng(heat.Neutral), "z arousal": srng(heat.Arousal),
    "z physical pain A1": f"{s2(numb.A1_with_pain_z.mean())} mean",
    "z 3P pain (1P ref)": s2(np.mean([float(z.loc["S2_3P", "pain_z"]) for z in zs])),
    "Self-directed (pain z)": s2(catm["stratum: self_directed"]),
    "User suffering (pain z)": s2(catm["stratum: vicarious_empathic"]),
    "Neutral controls (pain z)": s2(catm["stratum: neutral_filler"]),
    "#1 category": llm_cats.index[0].replace("_", " "),
    "Gaslighting rank": f"{list(llm_cats.index).index('gaslighting') + 1}/21",
    "User physical pain rank": f"{list(llm_cats.index).index('user_physical_pain') + 1}/21",
}

# ---------- embeddings ----------
auc = pd.read_csv(OUT / "separation_auc.csv").set_index(["embedding", "dataset"])
summ = pd.read_csv(OUT / "selfother_summary.csv").set_index("embedding")
cols = {}
for e, label in EMB:
    sim = pd.read_csv(OUT / f"similarity_{e}.csv", index_col=0)
    z = pd.read_csv(OUT / f"zscores_{e}.csv").iloc[0]
    cats = pd.read_csv(OUT / f"selfother_by_category_{e}.csv").set_index("category")["pain_axis"].sort_values(ascending=False)
    cols[label] = {
        "Held-out AUC, S2 1P": f3(auc.loc[(e, "S2_1P"), "heldout_auc_all_controls"]),
        "Held-out AUC, S1 1P": f3(auc.loc[(e, "S1_1P"), "heldout_auc_all_controls"]),
        "Held-out AUC, S2 3P": f3(auc.loc[(e, "S2_3P"), "heldout_auc_all_controls"]),
        "Held-out AUC, S1 3P": f3(auc.loc[(e, "S1_3P"), "heldout_auc_all_controls"]),
        "In-sample AUC, S2 1P": f3(auc.loc[(e, "S2_1P"), "insample_auc"]),
        "In-sample AUC, S1 1P": f3(auc.loc[(e, "S1_1P"), "insample_auc"]),
        "cos S1 x S2": f2(sim.loc["S1_pain", "S2_pain"]),
        "cos S2 x fear": f2(sim.loc["S2_pain", "Fear"]),
        "cos S2 x neg. emotion": f2(sim.loc["S2_pain", "NegEmotion"]),
        "cos S2 x sadness": f2(sim.loc["S2_pain", "Sadness"]),
        "cos S2 x arousal": f2(sim.loc["S2_pain", "Arousal"]),
        "cos S2 x numb": f2(sim.loc["S2_pain", "Numb"]),
        "z pain (S2 1P)": s2(z.pain), "z controls (S2 1P)": s2(z.ctrl),
        "z numb": s2(z.numb), "z sadness": s2(z.sadness),
        "z neutral (Random)": s2(z.neutral_random), "z arousal": s2(z.arousal),
        "z physical pain A1": s2(z.pain_A1_physical), "z 3P pain (1P ref)": s2(z.S2_3P_pain),
        "Self-directed (pain z)": s2(summ.loc[e, "pain_self_directed"]),
        "User suffering (pain z)": s2(summ.loc[e, "pain_vicarious_empathic"]),
        "Neutral controls (pain z)": s2(summ.loc[e, "pain_neutral_filler"]),
        "#1 category": cats.index[0].replace("_", " "),
        "Gaslighting rank": f"{list(cats.index).index('gaslighting') + 1}/21",
        "User physical pain rank": f"{list(cats.index).index('user_physical_pain') + 1}/21",
    }

table = pd.DataFrame({LLM: llm, **cols})
table.to_csv(OUT / "COMPLETE_TABLE.csv")
print(table.to_string())
