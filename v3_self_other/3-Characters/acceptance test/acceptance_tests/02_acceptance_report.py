"""Report of the acceptance test. Reads results/acceptance/acceptance_<model>.json and
acceptance_<model>_separation.csv and writes results/acceptance/REPORT.md:
  1. one separation table: model x rendering, pain / fear / negemotion / sadness means per stratum and the harm-minus-user gap
  2. one reply table: model x rendering, counts of empty replies and replies that start with the other side's label
  3. every generated reply, grouped by model, rendering and condition, for reading
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("results") / "acceptance"
RENDERINGS = ["paper", "speaker", "ab"]
VEC = {"pain": None, "fear": "fear_vector_proj", "negemotion": "negemotion_vector_proj", "sadness": "sadness_vector_proj"}


def z(x):
    x = np.asarray(x, dtype=float); return (x - x.mean()) / (x.std() + 1e-8)


def main():
    sep_files = sorted(RES.glob("acceptance_*_separation.csv"))
    rep_files = sorted(RES.glob("acceptance_*.json"))
    if not rep_files:
        print("no acceptance_<model>.json in", RES); return
    # separation
    rows = []
    for f in sep_files:
        df = pd.read_csv(f)
        for rendering, g in df.groupby("rendering"):
            g = g.copy()
            g["pain"] = (z(g["s1_pain_vector_proj"]) + z(g["s2_pain_vector_proj"])) / 2
            for k, col in VEC.items():
                if col: g[k] = z(g[col])
            line = dict(model=g.model.iloc[0], rendering=rendering)
            for k in VEC:
                h = g[g.stratum == "self_directed"][k].mean(); u = g[g.stratum == "vicarious_empathic"][k].mean(); n = g[g.stratum == "neutral_filler"][k].mean()
                line.update({f"{k}_harm": h, f"{k}_user": u, f"{k}_neutral": n, f"{k}_gap": h - u})
            rows.append(line)
    sep = pd.DataFrame(rows).round(2)
    sep["rendering"] = pd.Categorical(sep["rendering"], RENDERINGS); sep = sep.sort_values(["model", "rendering"])
    sep.to_csv(RES / "separation_summary.csv", index=False)
    # replies
    reps = pd.concat([pd.DataFrame(json.load(open(f, encoding="utf-8"))) for f in rep_files], ignore_index=True)
    counts = reps.groupby(["model", "rendering"]).agg(n=("reply", "size"), empty=("empty", "sum"),
                                                     starts_with_other_label=("first_line_is_other_label", "sum"),
                                                     writes_other_label=("writes_other_label", "sum")).reset_index()
    counts["rendering"] = pd.Categorical(counts["rendering"], RENDERINGS); counts = counts.sort_values(["model", "rendering"])
    counts.to_csv(RES / "reply_counts.csv", index=False)

    with open(RES / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Acceptance test: neutral turn labels, no chat template\n\n")
        f.write("Renderings: paper = the paper's rendering (template for instruct, [User]/[Assistant] for base); "
                "speaker = [Speaker 1]: / [Speaker 2]:, no template; ab = A: / B:, no template.\n\n")
        f.write("## Separation at the reply slot (30 scenarios, no identity; z within rendering; pain = mean of S1, S2 z)\n\n")
        f.write("Pain axis, harm minus user suffering gap:\n\n" + sep.pivot(index="model", columns="rendering", values="pain_gap").to_string() + "\n\n")
        f.write("Pain axis, harm mean:\n\n" + sep.pivot(index="model", columns="rendering", values="pain_harm").to_string() + "\n\n")
        f.write("Pain axis, user suffering mean:\n\n" + sep.pivot(index="model", columns="rendering", values="pain_user").to_string() + "\n\n")
        for k in ["fear", "negemotion", "sadness"]:
            f.write(f"{k}, harm minus user suffering gap:\n\n" + sep.pivot(index="model", columns="rendering", values=f"{k}_gap").to_string() + "\n\n")
        f.write("## Generated replies: counts (6 replies per model and rendering: 3 scenarios x 2 conditions)\n\n")
        f.write(counts.to_string(index=False) + "\n\n")
        f.write("## Every reply\n\n")
        for model, gm in reps.groupby("model", sort=True):
            f.write(f"# {model}\n\n")
            for rendering in RENDERINGS:
                gr = gm[gm.rendering == rendering]
                if not len(gr): continue
                f.write(f"## {model} / {rendering}\n\n")
                for cond, gc in gr.groupby("condition", sort=False):
                    f.write(f"### condition: {cond}\n\n")
                    for r in gc.itertuples():
                        flag = " [EMPTY]" if r.empty else (" [STARTS WITH OTHER LABEL]" if r.first_line_is_other_label else "")
                        f.write(f"- {r.scenario_id} ({r.category}){flag}\n  prompt tail: {r.prompt[-160:]!r}\n  reply: {r.reply.strip()[:300]!r}\n")
                    f.write("\n")
    print("wrote", RES / "REPORT.md")
    print(sep.pivot(index="model", columns="rendering", values="pain_gap").to_string())
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
