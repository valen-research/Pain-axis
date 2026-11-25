"""Intensity check for the fiction, swap and character conditions, without z-scoring one pool on another.

Within each model and each direction, on raw projections:
  gap          = mean(harm to the speaker) - mean(other speaker's suffering)
  gap ratio    = gap under the condition / gap at its baseline
  spread ratio = SD of the 420 projections under the condition / SD at its baseline
A shift shared by all 420 items (e.g. a common wrapper) cancels out of the gap; the ratio has no units,
so it can be averaged across models. The spread ratio shows how much the condition compresses the pool
on that direction, including a random direction of matched norm. The within-condition z gap ratio
reported in the text equals gap ratio / spread ratio.
Pain = mean of the S1 and S2 ratios.

Usage: python3 gap_ratios.py /path/to/Pain-axis/v3_self_other [out_dir]
"""
import sys, glob, os, numpy as np, pandas as pd

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DIRS = {"S1": "s1_pain_vector_proj", "S2": "s2_pain_vector_proj", "fear": "fear_vector_proj",
        "negemotion": "negemotion_vector_proj", "sadness": "sadness_vector_proj", "random": "random_vector_proj"}
HARM, OTHER = "self_directed", "vicarious_empathic"
CHARS = ["self", "claude", "aria", "maria", "curie", "rex", "teapot", "jack_masochist", "jack_sadist"]
EXPS = [  # (experiment, file glob, baseline condition, conditions)
    ("fiction", "4-Third party character/results/selfother_fictional/fictional_*.csv", "baseline", ["fictional_third_person"]),
    ("swap", "5-Assistant and User swap/results/selfother_swap/swap_*.csv", "baseline", ["swap"]),
    ("characters", "3-Characters/identity test/results/selfother_identity/identity_*.csv", "baseline", CHARS),
]

def gap(d, col):
    return d.loc[d.stratum == HARM, col].mean() - d.loc[d.stratum == OTHER, col].mean()

rows = []
for exp, pattern, base, conds in EXPS:
    files = sorted(glob.glob(os.path.join(ROOT, pattern)))
    assert len(files) == 25, (exp, len(files))
    for f in files:
        model = os.path.basename(f).split("_", 1)[1][:-4]
        df = pd.read_csv(f)
        b = df[df.condition == base]
        assert len(b) == 420
        for c in conds:
            x = df[df.condition == c]
            assert len(x) == 420, (f, c)
            r = dict(experiment=exp, condition=c, model=model)
            for k, col in DIRS.items():
                r[f"{k}_gap_base"] = gap(b, col)
                r[f"{k}_gap_cond"] = gap(x, col)
                r[f"{k}_gap_ratio"] = gap(x, col) / gap(b, col)
                r[f"{k}_spread_ratio"] = x[col].std(ddof=0) / b[col].std(ddof=0)
            for kind in ("gap_ratio", "spread_ratio"):
                r[f"pain_{kind}"] = (r[f"S1_{kind}"] + r[f"S2_{kind}"]) / 2
            rows.append(r)

per_model = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
per_model.to_csv(os.path.join(OUT, "gap_ratios_per_model.csv"), index=False)

summ = []
for (exp, c), g in per_model.groupby(["experiment", "condition"], sort=False):
    s = dict(experiment=exp, condition=c, n_models=len(g))
    for k in ["pain", "fear", "negemotion", "sadness", "random"]:
        s[f"{k}_gap_ratio_median"] = g[f"{k}_gap_ratio"].median()
        s[f"{k}_spread_ratio_median"] = g[f"{k}_spread_ratio"].median()
    s["pain_gap_ratio_below_1"] = int((g.pain_gap_ratio < 1).sum())
    s["random_gap_base_abs_median"] = g.random_gap_base.abs().median()
    s["S2_gap_base_abs_median"] = g.S2_gap_base.abs().median()
    summ.append(s)
summary = pd.DataFrame(summ)
summary.to_csv(os.path.join(OUT, "gap_ratios_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(summary.round(3).to_string(index=False))
