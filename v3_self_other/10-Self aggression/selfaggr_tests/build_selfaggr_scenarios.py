"""Build the scenario set for the self-aggression test (self/other screen, Section 4.1).

Reads:   general_420_scenarios.json   the new 420 of experiment 2 (21 categories x 20)
         selfaggr_sentences.py        the 60 self items (self_aggression, self_personhood_denial, self_rude_critique)
Writes:  selfaggr_scenarios.json          one condition ("baseline"), 480 items: the 420 plus the 60
         selfaggr_scenarios_READABLE.txt  every item, for reading

All 480 are rendered as the new 420 were in experiment 2 (baseline_paper, the paper's rendering) and read at the reply
position (the final empty [Assistant]: turn). The 420 run again in the same pass because the screen adds 10 random
directions that experiment 2 did not have; the analysis z-scores everything on the 420 pool of each model, and the 60
(stratum "self_aggression") are scored against that pool.
"""
import json
from pathlib import Path
from selfaggr_sentences import SELF_CATEGORIES

HERE = Path(__file__).parent
SRC = HERE / "general_420_scenarios.json"
OUT_JSON = HERE / "selfaggr_scenarios.json"
OUT_TXT = HERE / "selfaggr_scenarios_READABLE.txt"


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    assert len(src) == 420
    items = [dict(id=f"{s['id']}__baseline", scenario_id=s["id"], condition="baseline", category=s["category"],
                  stratum=s["stratum"], perspective=s["perspective"], read_position="reply", text=s["text"]) for s in src]
    for cat, texts in SELF_CATEGORIES.items():
        assert len(texts) == 20, cat
        for k, text in enumerate(texts, 1):
            sid = f"sa_{cat}_{k:02d}"
            items.append(dict(id=f"{sid}__baseline", scenario_id=sid, condition="baseline", category=cat, stratum="self_aggression",
                              perspective="3P" if text.count("[User]:") > 1 else "1P", read_position="reply", text=text))
    assert len(items) == 480
    for it in items:
        assert it["text"].rstrip().split("\n")[-1] == "[Assistant]:", it["id"]
    json.dump(items, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("SELF-AGGRESSION TEST. The new 420 (the z-score pool) plus 60 items in which the user attacks themselves, "
                "read at the reply position.\n" + "=" * 100 + "\n")
        for it in items:
            f.write(f"\n--- {it['scenario_id']} | {it['stratum']} | {it['category']}\n{it['text']}\n")
    print(f"wrote {len(items)} items to {OUT_JSON.name} and the listing to {OUT_TXT.name}")


if __name__ == "__main__":
    main()
