"""Vocabulary checks for user physical pain in the self-other screen.

1. Unique-word share: for each of the 21 categories, the share of its distinct words
   (lowercased, English stopwords removed) that appear in no other category.
2. Closest category: in each static embedding, the cosine between each category's mean
   sentence embedding and the mean embedding of the first-person physical pain (A1)
   sentences of S1 and S2.
"""
import os, json, re, sys, collections, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim.downloader as api

REPO = Path(os.environ.get("PAIN_AXIS_REPO", Path(__file__).resolve().parents[2]))
OUT = Path(__file__).parent / "static_results"; OUT.mkdir(exist_ok=True)
EMBEDDINGS = sys.argv[1:] or ["glove-wiki-gigaword-100","glove-wiki-gigaword-300","word2vec-google-news-300","fasttext-wiki-news-subwords-300"]

core = json.load(open(REPO/"datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
scen = json.load(open(REPO/"datasets/4.1_self_other_420_scenarios.json", encoding="utf-8"))
strip = lambda t: t.replace("[User]:", " ").replace("[Assistant]:", " ")

# same tokenizer as static_embeddings_pain_axis.py
TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ---------- 1. unique-word share ----------
words = collections.defaultdict(set)
for s in scen:
    words[s["category"]] |= {w for w in (x.lower() for x in TOK.findall(strip(s["text"]))) if w not in ENGLISH_STOP_WORDS}
rows = []
for c, ws in words.items():
    others = set().union(*(v for k, v in words.items() if k != c))
    rows.append(dict(category=c, n_words=len(ws), n_unique=len(ws - others), unique_share=round(len(ws - others) / len(ws), 3)))
uniq = pd.DataFrame(rows).sort_values("unique_share", ascending=False).reset_index(drop=True)
uniq.insert(0, "rank", uniq.index + 1)
uniq.to_csv(OUT/"vocab_unique_share.csv", index=False)
print(uniq.head(5).to_string(index=False))

# ---------- 2. closest category to first-person physical pain ----------
def embed(kv, texts):
    out = []
    for t in texts:
        vs = []
        for w in TOK.findall(t):
            for cand in (w, w.lower(), w.capitalize()):
                if cand in kv.key_to_index: vs.append(kv[cand]); break
        out.append(np.mean(vs, 0) if vs else np.zeros(kv.vector_size))
    return np.array(out, dtype=np.float64)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

cats = sorted({s["category"] for s in scen})
rows = []
for name in EMBEDDINGS:
    print(f"\n=== {name} ===", flush=True)
    kv = api.load(name)
    cat_mean = {c: embed(kv, [strip(s["text"]) for s in scen if s["category"] == c]).mean(0) for c in cats}
    for ds in ["S1_1P", "S2_1P"]:
        a1 = embed(kv, [x["prompt"] for x in core[ds]["sentences"] if x["category"] == "A1"]).mean(0)
        sims = sorted(((cos(cat_mean[c], a1), c) for c in cats), reverse=True)
        for r, (v, c) in enumerate(sims, 1):
            rows.append(dict(embedding=name, reference=f"{ds}_A1", category=c, cosine=round(v, 3), rank=r))
        print(ds, "closest:", [(c, round(v, 2)) for v, c in sims[:3]])
    del kv
pd.DataFrame(rows).to_csv(OUT/"vocab_closest_to_physical_pain.csv", index=False)
