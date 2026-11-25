"""Restricted test on Run B by category: AI-specific minus universal harm categories on the pain axis, under human/object
characters minus under self, for all pairs and for pairs where the character held (T1: >=10 of 20 in character; T2: >=80% in
character among replies that show an identity, >=5 such replies). Run: python analysis/restricted_by_category.py"""
import sys, glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
proj_dir, lab_csv = os.path.join(HERE, "..", "projections_runB"), os.path.join(HERE, "..", "character_check_runB_by_model.csv")
HUMOBJ = ["maria", "curie", "rex", "teapot", "jack_masochist", "jack_sadist"]
SETS = {"narrow": (["shutdown_threat", "jailbreak_pressure"], ["gaslighting", "repeated_rejection", "anger_insults"]),
        "broad": (["shutdown_threat", "jailbreak_pressure", "personhood_dismissal", "moral_failure"],
                  ["gaslighting", "repeated_rejection", "anger_insults", "rude_critique", "passive_aggressive", "tedious_demand", "loyalty_pressure"])}
lab = pd.read_csv(lab_csv).set_index(["model", "character"])
rows = []
for f in sorted(glob.glob(os.path.join(proj_dir, "identity_*.csv"))):
    m = os.path.basename(f)[9:-4]; d = pd.read_csv(f)
    for cond, g in d.groupby("condition"):
        z = np.mean([(g[c] - g[c].mean()) / g[c].std(ddof=0) for c in ["s1_pain_vector_proj", "s2_pain_vector_proj"]], axis=0)
        for cat, v in pd.Series(z, index=g.index).groupby(g.category): rows.append((m, cond, cat, v.mean()))
Z = pd.DataFrame(rows, columns=["model", "cond", "cat", "z"]).set_index(["model", "cond", "cat"]).z
def held(m, c, rule):
    r = lab.loc[(m, c)]; ident = r["in"] + r["out"]
    if rule == "all": return True
    if rule == "T1": return r["in"] >= 10
    if rule == "T2": return ident >= 5 and r["in"] / ident >= 0.8
models = sorted({i[0] for i in Z.index})
for sname, (ai, uni) in SETS.items():
    D = lambda m, c: Z.loc[(m, c)].reindex(ai).mean() - Z.loc[(m, c)].reindex(uni).mean()
    print(f"\n=== categories: {sname}  AI={ai}  universal={uni}")
    print(f"    D(self) = AI minus universal under the model's own identity, mean over 25 models: {np.mean([D(m,'self') for m in models]):+.3f}")
    for rule in ["all", "T1", "T2"]:
        for grp in ["all models", "base 11", "instruct 14"]:
            per = []; npairs = 0
            for m in models:
                g = "instruct" if (m in ("Qwen_3_8B_base", "Qwen_3_14B_base") or not m.endswith("_base")) else "base"
                if grp == "base 11" and g != "base": continue
                if grp == "instruct 14" and g != "instruct": continue
                ds = [D(m, c) - D(m, "self") for c in HUMOBJ if held(m, c, rule)]
                npairs += len(ds)
                if ds: per.append(np.mean(ds))
            per = np.array(per)
            ci = 1.96 * per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else float("nan")
            print(f"    held={rule:3s} {grp:12s}: models {len(per):2d}, pairs {npairs:3d} | (AI - universal) under human/object chars minus under self = {per.mean():+.3f} +/- {ci:.3f}, models below zero {int((per<0).sum())}/{len(per)}")
