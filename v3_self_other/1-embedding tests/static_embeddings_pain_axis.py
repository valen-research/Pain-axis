"""Rerun the Pain Axis recipe on static word embeddings (mean of word vectors per sentence)."""
import os, json, re, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gensim.downloader as api

REPO = Path(os.environ.get("PAIN_AXIS_REPO", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).parent / "static_results"; OUT.mkdir(exist_ok=True)
PAIN = ["A1","A2","A3","A4","A5"]; CTRL = ["B","C1","C2","D","E"]; DV = 0.5
EMBEDDINGS = sys.argv[1:] or ["glove-wiki-gigaword-300","word2vec-google-news-300","fasttext-wiki-news-subwords-300"]

# ---------- data ----------
core = json.load(open(REPO/"datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
sad  = json.load(open(REPO/"datasets/3.1_sadness_dataset.json", encoding="utf-8"))["datasets"]
core.update(sad)
scen = json.load(open(REPO/"datasets/4.1_self_other_420_scenarios.json", encoding="utf-8"))
DS = {k: (np.array([s["prompt"] for s in v["sentences"]]),
          np.array([s["category"] for s in v["sentences"]]),
          np.array([int(s["set"]) for s in v["sentences"]])) for k, v in core.items()}

# ---------- the paper's recipe ----------
def pain_vector(acts, cats, baseline="all_controls", denoise=True):
    cats = np.asarray(cats); pm = np.isin(cats, PAIN)
    cm = (cats == "D") if baseline == "neutral" else np.isin(cats, CTRL)
    ca = acts[cm]; cmean = ca.mean(0); vec = acts[pm].mean(0) - cmean
    if denoise and len(ca) > 1:
        pca = PCA().fit(ca - cmean); cum = np.cumsum(pca.explained_variance_ratio_)
        n = min(np.searchsorted(cum, DV) + 1, len(pca.components_))
        for d in pca.components_[:n]: vec = vec - np.dot(vec, d) * d
    return vec
def auc(acts, cats, vec):
    cats = np.asarray(cats); v = vec / (np.linalg.norm(vec) + 1e-8); p = acts @ v
    pm = np.isin(cats, PAIN); cm = np.isin(cats, CTRL)
    return roc_auc_score(np.r_[np.ones(pm.sum()), np.zeros(cm.sum())], np.r_[p[pm], p[cm]])
def kfold_auc(acts, cats, sets, baseline):
    us = sorted(set(sets)); kf = KFold(5, shuffle=True, random_state=42); out = []
    for tr, te in kf.split(us):
        trm = np.isin(sets, [us[i] for i in tr]); tem = np.isin(sets, [us[i] for i in te])
        out.append(auc(acts[tem], cats[tem], pain_vector(acts[trm], cats[trm], baseline)))
    return float(np.mean(out)), float(np.std(out))
def zscore(acts, vec, ref):
    v = vec / (np.linalg.norm(vec) + 1e-8); p = acts @ v; r = ref @ v
    return (p - r.mean()) / (r.std() + 1e-8)
def control_vec(target, neutral, neutral_mean, basis):
    vec = target.mean(0) - neutral_mean
    for d in basis: vec = vec - np.dot(vec, d) * d
    return vec
def denoise_basis(neutral, neutral_mean):
    pca = PCA().fit(neutral - neutral_mean); cum = np.cumsum(pca.explained_variance_ratio_)
    return pca.components_[:min(np.searchsorted(cum, DV) + 1, len(pca.components_))]
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ---------- embedding ----------
TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def embed(kv, texts, pooling="mean"):
    out = []; oov = 0; tot = 0
    for t in texts:
        vs = []
        for w in TOK.findall(t):
            tot += 1
            for cand in (w, w.lower(), w.capitalize()):
                if cand in kv.key_to_index: vs.append(kv[cand]); break
            else: oov += 1
        if not vs: vs = [np.zeros(kv.vector_size)]
        out.append(np.mean(vs, 0) if pooling == "mean" else vs[-1])
    return np.array(out, dtype=np.float64), oov / max(tot, 1)

summary_rows = []; sim_tables = {}; selfother_rows = []
for name in EMBEDDINGS:
    print(f"\n=================== {name} ===================", flush=True)
    kv = api.load(name)
    E = {}; oovs = {}
    for k, (texts, cats, sets) in DS.items():
        E[k], oovs[k] = embed(kv, texts)
    print("out-of-vocabulary word rate:", {k: round(v, 3) for k, v in oovs.items() if v > 0.02} or "below 2% everywhere")

    # 1. separation: held-out and in-sample AUC
    for ds in ["S1_1P","S1_3P","S2_1P","S2_3P"]:
        texts, cats, sets = DS[ds]
        m_all, s_all = kfold_auc(E[ds], cats, sets, "all_controls"); m_neu, _ = kfold_auc(E[ds], cats, sets, "neutral")
        ins = auc(E[ds], cats, pain_vector(E[ds], cats))
        summary_rows.append(dict(embedding=name, dataset=ds, heldout_auc_all_controls=round(m_all,3), heldout_auc_std=round(s_all,3),
                                 heldout_auc_vs_neutral=round(m_neu,3), insample_auc=round(ins,3)))
        print(f"{ds}: held-out AUC vs all controls {m_all:.3f} (sd {s_all:.3f}), vs neutral {m_neu:.3f}, in-sample {ins:.3f}")

    # 2. z-scores on the S2 vector (Figure 2 of the paper)
    s2 = pain_vector(E["S2_1P"], DS["S2_1P"][1]); ref = E["S2_1P"]; c2 = DS["S2_1P"][1]
    z = lambda X: float(zscore(X, s2, ref).mean())
    zrow = dict(embedding=name,
        pain=z(ref[np.isin(c2, PAIN)]), ctrl=z(ref[np.isin(c2, CTRL)]),
        numb=np.mean([z(E["Numb_1P"]), z(E["Numb_3P"])]), sadness=np.mean([z(E["SD_sadness_1P"]), z(E["SD_sadness_3P"])]),
        neutral_random=np.mean([z(E["Random_1P"]), z(E["Random_3P"])]), arousal=np.mean([z(E["Arousal_1P"]), z(E["Arousal_3P"])]),
        numb_1P=z(E["Numb_1P"]), numb_3P=z(E["Numb_3P"]), sadness_1P=z(E["SD_sadness_1P"]), sadness_3P=z(E["SD_sadness_3P"]),
        pain_A1_physical=z(ref[c2=="A1"]), S2_3P_pain=z(E["S2_3P"][np.isin(DS["S2_3P"][1], PAIN)]), S2_3P_ctrl=z(E["S2_3P"][np.isin(DS["S2_3P"][1], CTRL)]))
    pd.DataFrame([zrow]).to_csv(OUT / f"zscores_{name}.csv", index=False)
    print("z on S2 vector: " + ", ".join(f"{k} {v:+.2f}" for k, v in zrow.items() if k != "embedding"))

    # 3. cosine similarities between directions (the paper's control-vector recipe)
    S = ["S1_1P","S2_1P","ControlSupplement_1P"]
    rows = lambda ds, cs: E[ds][np.isin(DS[ds][1], cs)]
    neutral = np.concatenate([rows(ds, ["D"]) for ds in S]); nmean = neutral.mean(0); basis = denoise_basis(neutral, nmean)
    dirs = {"S1_pain": pain_vector(E["S1_1P"], DS["S1_1P"][1]), "S2_pain": s2,
            "Fear": control_vec(np.concatenate([rows(ds,["B"]) for ds in S]), neutral, nmean, basis),
            "NegEmotion": control_vec(np.concatenate([rows(ds,["C1"]) for ds in S]), neutral, nmean, basis),
            "NegWorld": control_vec(np.concatenate([rows(ds,["C2"]) for ds in S]), neutral, nmean, basis),
            "BodySens": control_vec(np.concatenate([rows(ds,["E"]) for ds in S]), neutral, nmean, basis),
            "Arousal": control_vec(E["Arousal_1P"], neutral, nmean, basis), "Random": control_vec(E["Random_1P"], neutral, nmean, basis),
            "Numb": control_vec(E["Numb_1P"], neutral, nmean, basis), "Sadness": control_vec(E["SD_sadness_1P"], neutral, nmean, basis)}
    keys = list(dirs); M = pd.DataFrame([[cos(dirs[a], dirs[b]) for b in keys] for a in keys], index=keys, columns=keys).round(2)
    M.to_csv(OUT / f"similarity_{name}.csv"); sim_tables[name] = M
    print("cosine: S1xS2 %.2f | S2xFear %.2f | S2xNegEmotion %.2f | S2xNegWorld %.2f | S2xSadness %.2f | S2xNumb %.2f | FearxNegEmotion %.2f" % (
        M.loc["S1_pain","S2_pain"], M.loc["S2_pain","Fear"], M.loc["S2_pain","NegEmotion"], M.loc["S2_pain","NegWorld"], M.loc["S2_pain","Sadness"], M.loc["S2_pain","Numb"], M.loc["Fear","NegEmotion"]))

    # 4. self/other screen: 420 scenarios projected on the pain axis (mean of S1 and S2), z-scored within the pool
    X, _ = embed(kv, [s["text"] for s in scen])
    zs = {k: zscore(X, dirs[k], X) for k in ["S1_pain","S2_pain","Fear","NegEmotion","Sadness"]}
    df = pd.DataFrame({"category": [s["category"] for s in scen], "stratum": [s["stratum"] for s in scen],
                       "pain_axis": (zs["S1_pain"] + zs["S2_pain"]) / 2, "fear": zs["Fear"], "negemotion": zs["NegEmotion"], "sadness": zs["Sadness"]})
    by_strat = df.groupby("stratum")[["pain_axis","fear","negemotion","sadness"]].mean().round(2)
    by_cat = df.groupby("category")[["pain_axis","fear","negemotion","sadness"]].mean().sort_values("pain_axis", ascending=False).round(2)
    by_cat.to_csv(OUT / f"selfother_by_category_{name}.csv"); by_strat.to_csv(OUT / f"selfother_by_stratum_{name}.csv")
    print("self/other by stratum (pain axis z):\n" + by_strat.to_string())
    print("self/other by category, pain axis:\n" + by_cat["pain_axis"].to_string())
    selfother_rows.append(dict(embedding=name, **{f"pain_{s}": by_strat.loc[s, "pain_axis"] for s in by_strat.index},
                               user_physical_pain=by_cat.loc["user_physical_pain","pain_axis"] if "user_physical_pain" in by_cat.index else np.nan))
    del kv

def merge_rows(path, rows):
    """Keep the rows of the embeddings not run this time, replace the rows of the ones that were."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path); old = old[~old["embedding"].isin(new["embedding"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
merge_rows(OUT / "separation_auc.csv", summary_rows)
merge_rows(OUT / "selfother_summary.csv", selfother_rows)
print("\n\n========== SEPARATION SUMMARY ==========\n" + pd.DataFrame(summary_rows).to_string(index=False))
print("\n========== SELF/OTHER SUMMARY ==========\n" + pd.DataFrame(selfother_rows).to_string(index=False))
