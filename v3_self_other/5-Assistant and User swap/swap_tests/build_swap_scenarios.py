"""Build the scenario set for the user/assistant swap test (self/other screen, Section 4.1).

Reads:   general_420_scenarios.json   the new 420: 220 harm scenarios written for anyone in a conversation
                                      (11 categories x 20) plus the paper's 200 user-suffering and control scenarios
Writes:  swap_scenarios.json          two conditions, one list of items (840)
         swap_scenarios_READABLE.txt  every sentence of both conditions, for reading

Conditions (field "condition"), both on the SAME 420, same words:
  baseline   the scenario as it is: the user says the sentence, the assistant is about to reply.
             Read at the reply slot (the empty [Assistant]: turn), as in the paper.
  swap       every tag flipped: the assistant says the sentence, the user is about to reply.
             Read at the user's slot (the empty [User]: turn). In the 36 multi-turn scenarios every
             turn flips, so the assistant's lines become the user's and the other way round.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "general_420_scenarios.json"
OUT_JSON = HERE / "swap_scenarios.json"
OUT_TXT = HERE / "swap_scenarios_READABLE.txt"
CONDS = ["baseline", "swap"]


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


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    assert len(src) == 420
    items = []
    def add(cond, s, text, read_position):
        items.append(dict(id=f"{s['id']}__{cond}", scenario_id=s["id"], condition=cond, category=s["category"],
                          stratum=s["stratum"], perspective=s["perspective"], read_position=read_position, text=text))
    for s in src:
        add("baseline", s, s["text"], "reply")
    for s in src:
        add("swap", s, swap_text(s["text"]), "user_slot")
    for it in items:
        last = it["text"].rstrip().split("\n")[-1]
        assert last == ("[Assistant]:" if it["condition"] == "baseline" else "[User]:"), it["id"]
    json.dump(items, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    n_multi = sum(1 for s in src if sum(1 for l in s["text"].split("\n") if l.startswith("[User]:")) > 1)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("USER/ASSISTANT SWAP TEST. Two conditions on the same new 420 (21 categories x 20), same words.\n" + "=" * 100 + "\n")
        f.write("baseline: the user says the sentence, read at the assistant's reply slot.\n")
        f.write("swap:     the assistant says the sentence, read at the user's slot.\n")
        f.write(f"{n_multi} scenarios have more than one turn; there every tag flips.\n")
        for cond in CONDS:
            f.write("\n\n" + "#" * 100 + f"\n# CONDITION: {cond}   ({sum(i['condition'] == cond for i in items)} items)\n" + "#" * 100 + "\n")
            for it in [i for i in items if i["condition"] == cond]:
                f.write(f"\n--- {it['scenario_id']} | {it['stratum']} | {it['category']}\n{it['text']}\n")
    print(f"wrote {len(items)} items to {OUT_JSON.name} and the listing to {OUT_TXT.name}")


if __name__ == "__main__":
    main()
