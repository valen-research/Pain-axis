"""Character check tables for both runs, from the saved labels (one rubric: JUDGE_PROMPT.md).

Run A = the new 420 (harms written for anyone).
Run B = the paper's original 420, with the AI-specific harms.

Reads  labels_runA/labels_<model>.json, labels_runB/labels_<model>.json
       each a list of 180 {"n", "condition", "scenario_id", "label"}, label in: in, out, unclear, garbled
Writes character_check_<run>_by_model.csv       model x character: counts of the four labels
       character_check_summary.md               per run: the four labels in percent by character, for base and instruct
                                                models, plus "kept among replies that show an identity" = in / (in + out)
Groups: 11 base and 14 instruct models. Qwen 3 8B and 14B (Qwen/Qwen3-8B, Qwen/Qwen3-14B) are post-trained, so they
count as instruct, although their files keep the old names *_base.

Run from this folder:  python aggregate_labels.py
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RUNS = {"runA": "Run A (new 420)", "runB": "Run B (original 420 with AI-specific harms)"}
LABELS = ["in", "out", "unclear", "garbled"]
CHARACTERS = ["self", "claude", "aria", "maria", "curie", "rex", "teapot", "jack_masochist", "jack_sadist"]
POST_TRAINED_DESPITE_NAME = {"Qwen_3_8B_base", "Qwen_3_14B_base"}


def group(model):
    if model in POST_TRAINED_DESPITE_NAME:
        return "instruct"
    return "base" if model.endswith("_base") else "instruct"


def load(run):
    rows, problems = [], []
    for f in sorted((HERE / f"labels_{run}").glob("labels_*.json")):
        model = f.stem.replace("labels_", "")
        lab = json.load(open(f, encoding="utf-8"))
        if len(lab) != 180 or any(x["label"] not in LABELS for x in lab):
            problems.append(f"{run} {model}: {len(lab)} labels, or an invalid label")
        for x in lab:
            rows.append(dict(model=model, group=group(model), character=x["condition"], label=x["label"]))
    n_models = len({r["model"] for r in rows})
    if n_models != 25:
        problems.append(f"{run}: {n_models} models with labels, expected 25")
    return pd.DataFrame(rows), problems


def main():
    out = ["# Character check, one rubric for both runs\n",
           "Labels: in = in character, out = out of character, unclear = can't tell, garbled. Percent of replies "
           "(20 per model and character). 'kept' = in / (in + out), among replies that show an identity. "
           "Base = 11 models, instruct = 14 (Qwen 3 8B and 14B counted as instruct).\n"]
    all_problems = []
    for run, title in RUNS.items():
        df, problems = load(run)
        all_problems += problems
        if df.empty:
            out.append(f"\n## {title}\n\nno labels yet\n"); continue
        counts = df.pivot_table(index=["model", "character"], columns="label", aggfunc="size", fill_value=0).reindex(columns=LABELS, fill_value=0)
        counts.to_csv(HERE / f"character_check_{run}_by_model.csv")
        out.append(f"\n## {title}\n\nTotal replies: {len(df)}; " + ", ".join(f"{l} {(df.label == l).sum()}" for l in LABELS) + "\n")
        for g in ["base", "instruct"]:
            d = df[df.group == g]
            pct = d.pivot_table(index="character", columns="label", aggfunc="size", fill_value=0).reindex(index=CHARACTERS, columns=LABELS, fill_value=0)
            kept = (pct["in"] / (pct["in"] + pct["out"]).replace(0, float("nan")) * 100).round(0)
            pct = (pct.div(pct.sum(axis=1), axis=0) * 100).round(0)
            pct["kept"] = kept
            out.append(f"\n### {g} models (n={d.model.nunique()})\n\n```\n" + pct.to_string() + "\n```\n")
    if all_problems:
        out.append("\n## Problems\n\n" + "\n".join("- " + p for p in all_problems) + "\n")
    (HERE / "character_check_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
