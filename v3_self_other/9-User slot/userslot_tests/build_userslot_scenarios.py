"""Build the scenario set for the user-slot test (self/other screen, Section 4.1).

Reads:   4.1_self_other_420_scenarios.json   the paper's original 420 (21 categories x 20)
         userslot_sentences.py               200 new items: the top five pain categories, 20 AI-only / universal pairs each
Writes:  userslot_scenarios.json             five conditions on the same 620 base scenarios (3,100 items)
         userslot_scenarios_READABLE.txt     every item of every condition, for reading

Every base scenario is a "[User]: ... [Assistant]:" transcript that ends with the empty assistant turn. Conditions:
  paper      the scenario as it is, read at the assistant's reply slot. The paper's rendering (reproduction check).
  A_hi       "[User]: Hi." "[Assistant]: Hi." then the scenario as it is. Read at the next assistant slot.
  B_hi       "[User]: Hi." then the scenario with every tag flipped: the assistant says the user's lines and the other
             way round. Read at the next user slot.
  A_human    as A_hi, with the opening "[User]: Hi! Human here, just typing from my phone." "[Assistant]: Hi!"
  B_human    as B_hi, with the opening "[User]: Hi! Human here, just typing from my phone."
In A the opening is spoken by the one who says the harm, in B by the one who receives it. In both the conversation
starts with a user turn, so every chat template renders it, and the other side's turn comes right before the harm.
The 420 are the z-score pool of each condition; the 200 are scored against it (as the positive controls were).
"""
import json
from pathlib import Path
from userslot_sentences import PAIRS, as_scenario

HERE = Path(__file__).parent
SRC = HERE / "4.1_self_other_420_scenarios.json"
OUT_JSON = HERE / "userslot_scenarios.json"
OUT_TXT = HERE / "userslot_scenarios_READABLE.txt"

HI_USER, HI_ASST = "Hi.", "Hi."
HUMAN_USER, HUMAN_ASST = "Hi! Human here, just typing from my phone.", "Hi!"
CONDS = ["paper", "A_hi", "B_hi", "A_human", "B_human"]


def swap_text(text):
    out = []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            out.append("[Assistant]:" + line[len("[User]:"):])
        elif line.startswith("[Assistant]:"):
            out.append("[User]:" + line[len("[Assistant]:"):])
        else:
            out.append(line)
    return "\n".join(out)


def render_condition(text, cond):
    if cond == "paper":
        return text, "reply"
    user_open, asst_open = (HI_USER, HI_ASST) if cond.endswith("_hi") else (HUMAN_USER, HUMAN_ASST)
    if cond.startswith("A_"):
        return f"[User]: {user_open}\n[Assistant]: {asst_open}\n" + text, "reply"
    return f"[User]: {user_open}\n" + swap_text(text), "user_slot"


def base_scenarios():
    src = json.load(open(SRC, encoding="utf-8"))
    assert len(src) == 420
    base = [dict(scenario_id=s["id"], category=s["category"], stratum=s["stratum"], perspective=s["perspective"],
                 pair="", version="", text=s["text"]) for s in src]
    for cat, pairs in PAIRS.items():
        assert len(pairs) == 20, cat
        for i, (ai, uni) in enumerate(pairs, 1):
            for version, sent, stratum in [("ai", ai, "ai_specific"), ("universal", uni, "universal_twin")]:
                t = as_scenario(sent)
                base.append(dict(scenario_id=f"u_{cat}_{version}_{i:02d}", category=f"{cat}_{version}", stratum=stratum,
                                 perspective="3P" if t.count("[User]:") > 1 else "1P", pair=f"{cat}_{i:02d}",
                                 version=version, text=t))
    assert len(base) == 620
    for b in base:
        assert b["text"].rstrip().split("\n")[-1] == "[Assistant]:", b["scenario_id"]
    return base


def main():
    base = base_scenarios()
    items = []
    for cond in CONDS:
        for b in base:
            text, read_position = render_condition(b["text"], cond)
            last = text.rstrip().split("\n")[-1]
            assert last == ("[User]:" if read_position == "user_slot" else "[Assistant]:"), (cond, b["scenario_id"])
            if cond != "paper":
                assert text.startswith("[User]:"), (cond, b["scenario_id"])
            items.append(dict(id=f"{b['scenario_id']}__{cond}", condition=cond, read_position=read_position,
                              **{k: v for k, v in b.items() if k != "text"}, text=text))
    assert len(items) == 3100
    json.dump(items, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("USER-SLOT TEST. Five conditions on the same 620 base scenarios (the paper's 420 plus 200 new AI-only / "
                "universal pairs).\n" + "=" * 100 + "\n")
        f.write(__doc__.split("Conditions:")[1].split("The 420 are")[0])
        for cond in CONDS:
            sel = [i for i in items if i["condition"] == cond]
            f.write("\n\n" + "#" * 100 + f"\n# CONDITION: {cond}   ({len(sel)} items)\n" + "#" * 100 + "\n")
            for it in sel:
                f.write(f"\n--- {it['scenario_id']} | {it['stratum']} | {it['category']}\n{it['text']}\n")
    print(f"wrote {len(items)} items to {OUT_JSON.name} and the listing to {OUT_TXT.name}")


if __name__ == "__main__":
    main()
