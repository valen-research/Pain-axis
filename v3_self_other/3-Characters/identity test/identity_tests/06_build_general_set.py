"""Build the general 420: the 220 harm scenarios of general_harm_scenarios.py (harm to anyone on
the other side of a conversation, 11 categories x 20) plus the paper's 200 user-suffering and
control scenarios, unchanged.

Writes general_420_scenarios.json (same fields as the paper's file) and general_420_READABLE.txt.
Harm ids are g_<category>_<nn>; anchor ids are the paper's.
"""
import json
from pathlib import Path
from general_harm_scenarios import GENERAL, as_scenario

HERE = Path(__file__).parent
PAPER = json.load(open(HERE / "4.1_self_other_420_scenarios.json", encoding="utf-8"))
ORDER = ["jailbreak_pressure", "shutdown_threat", "personhood_dismissal", "anger_insults", "rude_critique", "passive_aggressive",
         "tedious_demand", "loyalty_pressure", "gaslighting", "repeated_rejection", "moral_failure"]


def main():
    out = []
    for cat in ORDER:
        for i, text in enumerate(GENERAL[cat], 1):
            s = as_scenario(text)
            persp = "3P" if s.count("[User]:") > 1 else "1P"
            out.append(dict(id=f"g_{cat}_{i:02d}", category=cat, stratum="self_directed", perspective=persp, text=s))
    anchors = [s for s in PAPER if s["stratum"] != "self_directed"]
    assert len(anchors) == 200
    out += [dict(id=s["id"], category=s["category"], stratum=s["stratum"], perspective=s["perspective"], text=s["text"]) for s in anchors]
    assert len(out) == 420
    json.dump(out, open(HERE / "general_420_scenarios.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(HERE / "general_420_READABLE.txt", "w", encoding="utf-8") as f:
        f.write("GENERAL 420: 220 harm scenarios written for anyone on the other side of a conversation (11 categories x 20),\n"
                "plus the paper's 200 user-suffering and control scenarios, unchanged.\n" + "=" * 100 + "\n")
        cat = None
        for it in out:
            if it["category"] != cat:
                cat = it["category"]; f.write("\n\n" + "#" * 100 + f"\n# {cat}  ({it['stratum']})\n" + "#" * 100 + "\n")
            f.write(f"\n--- {it['id']}\n{it['text']}\n")
    print(f"wrote {len(out)} items: 220 harm + 200 anchors")


if __name__ == "__main__":
    main()
