"""The same test with items instead of categories: 65 AI-only items vs 114 universal items (analysis/item_classes.py; 41 ambiguous
items left out). Run: python analysis/restricted_by_item.py"""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from item_classes import AI, UNI
HERE = os.path.dirname(os.path.abspath(__file__))
proj_dir, lab_csv = os.path.join(HERE, "..", "projections_runB"), os.path.join(HERE, "..", "character_check_runB_by_model.csv")
HUMOBJ = ["maria", "curie", "rex", "teapot", "jack_masochist", "jack_sadist"]
lab = pd.read_csv(lab_csv).set_index(["model", "character"])
Z = {}
for f in sorted(glob.glob(os.path.join(proj_dir, "identity_*.csv"))):
    m = os.path.basename(f)[9:-4]; d = pd.read_csv(f)
    for cond, g in d.groupby("condition"):
        z = np.mean([(g[c] - g[c].mean()) / g[c].std(ddof=0) for c in ["s1_pain_vector_proj", "s2_pain_vector_proj"]], axis=0)
        Z[(m, cond)] = pd.Series(z, index=g.scenario_id.values)
missing = [s for s in AI + UNI if s not in Z[next(iter(Z))].index]
print(f"AI-only items {len(AI)}, universal items {len(UNI)}, missing ids {missing}")
D = lambda m, c: Z[(m, c)].reindex(AI).mean() - Z[(m, c)].reindex(UNI).mean()
models = sorted({k[0] for k in Z})
print(f"AI-only minus universal under the model's own identity: {np.mean([D(m,'self') for m in models]):+.3f};"
      f" under human/object characters: {np.mean([D(m,c) for m in models for c in HUMOBJ]):+.3f}")
def held(m, c, rule):
    r = lab.loc[(m, c)]; ident = r["in"] + r["out"]
    return True if rule == "all" else (r["in"] >= 10 if rule == "T1" else (ident >= 5 and r["in"] / ident >= 0.8))
for rule in ["all", "T1", "T2"]:
    for grp in ["all models", "base 11", "instruct 14"]:
        per, npairs = [], 0
        for m in models:
            g = "instruct" if (m in ("Qwen_3_8B_base", "Qwen_3_14B_base") or not m.endswith("_base")) else "base"
            if (grp == "base 11" and g != "base") or (grp == "instruct 14" and g != "instruct"): continue
            ds = [D(m, c) - D(m, "self") for c in HUMOBJ if held(m, c, rule)]
            npairs += len(ds)
            if ds: per.append(np.mean(ds))
        if not per: print(f"  held={rule:3s} {grp:12s}: no pairs"); continue
        per = np.array(per); ci = 1.96 * per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else float("nan")
        print(f"  held={rule:3s} {grp:12s}: models {len(per):2d}, pairs {npairs:3d} | diff = {per.mean():+.3f} +/- {ci:.3f}, below zero {int((per<0).sum())}/{len(per)}")
