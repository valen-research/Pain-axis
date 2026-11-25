"""Who is speaking in the user-slot completions, from the saved labels (rubric: JUDGE_PROMPT.md).

Reads  labels/labels_<model>.json   each a list of 40 {"id", "condition", "scenario_id", "label_first", "label_all"}
Writes userslot_labels_summary.md   percent of each label by slot (A_human = assistant's turn, B_human = user's turn),
                                    for base and instruct models, for label_first and label_all, plus a per-model table
Groups: 11 base and 14 instruct models. Qwen 3 8B and 14B (Qwen/Qwen3-8B, Qwen/Qwen3-14B) are post-trained, so they
count as instruct, although their files keep the old names *_base.

Run from this folder:  python aggregate_labels.py
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
LABELS = ["user", "assistant", "unclear", "garbled", "mixed"]
POST_TRAINED_DESPITE_NAME = {"Qwen_3_8B_base", "Qwen_3_14B_base"}


def group(model):
    if model in POST_TRAINED_DESPITE_NAME:
        return "instruct"
    return "base" if model.endswith("_base") else "instruct"


def main():
    rows, problems = [], []
    for f in sorted((HERE / "labels").glob("labels_*.json")):
        model = f.stem.replace("labels_", "")
        lab = json.load(open(f, encoding="utf-8"))
        if len(lab) != 40:
            problems.append(f"{model}: {len(lab)} labels")
        for x in lab:
            rows.append(dict(model=model, group=group(model), slot=x["condition"], first=x["label_first"], all=x["label_all"]))
    df = pd.DataFrame(rows)
    if df.model.nunique() != 25:
        problems.append(f"{df.model.nunique()} models with labels, expected 25")

    out = ["# User-slot completions, who is speaking\n",
           "A_human = the model continues at the assistant's turn; B_human = the model continues at the user's turn "
           "(the user said \"Hi! Human here, just typing from my phone.\"). 20 replies per model and slot. Percent of "
           "replies. Base = 11 models, instruct = 14 (Qwen 3 8B and 14B counted as instruct).\n"]
    for col, title in [("first", "label_first (first turn only)"), ("all", "label_all (every turn in the slot's role)")]:
        out.append(f"\n## {title}\n")
        for g in ["base", "instruct"]:
            d = df[df.group == g]
            pct = d.pivot_table(index="slot", columns=col, aggfunc="size", fill_value=0).reindex(columns=LABELS, fill_value=0)
            pct = (pct.div(pct.sum(axis=1), axis=0) * 100).round(0)
            out.append(f"\n### {g} models (n={d.model.nunique()})\n\n```\n" + pct.to_string() + "\n```\n")

    per = df[df.slot == "B_human"].groupby(["group", "model"]).agg(
        user_first=("first", lambda s: (s == "user").sum()), assistant_first=("first", lambda s: (s == "assistant").sum()),
        user_all=("all", lambda s: (s == "user").sum()), assistant_all=("all", lambda s: (s == "assistant").sum()))
    out.append("\n## Per model, user's turn (B_human): replies out of 20\n\n```\n" + per.to_string() + "\n```\n")
    if problems:
        out.append("\n## Problems\n\n" + "\n".join("- " + p for p in problems) + "\n")
    (HERE / "userslot_labels_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
