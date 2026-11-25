"""Turn the agents' FILE blocks (agent_outputs/*.txt) into labels_<run>/labels_<model>.json, checking each block against the reply file.
Usage: python write_labels.py . agent_outputs/<file>.txt"""
import sys, json, re
from pathlib import Path
base = Path(sys.argv[1]); text = open(sys.argv[2], encoding="utf-8").read()
CODE = {"I": "in", "O": "out", "U": "unclear", "G": "garbled"}
blocks = re.split(r"^FILE\s+", text, flags=re.M)[1:]
for b in blocks:
    lines = [l.strip() for l in b.strip().splitlines() if l.strip()]
    run, model = lines[0].split()
    replies = json.load(open(base / f"replies_{run}" / f"manipulation_{model}.json", encoding="utf-8"))
    seq = {}
    for l in lines[1:10]:
        cond, codes = l.split()
        seq[cond] = codes
    labels, ok = [], True
    order = [r["condition"] for r in replies]
    pos = {}
    for n, r in enumerate(replies, 1):
        c = r["condition"]; k = pos.get(c, 0); pos[c] = k + 1
        codes = seq.get(c, "")
        if len(codes) != 20 or any(ch not in CODE for ch in codes):
            ok = False; break
        labels.append(dict(n=n, condition=c, scenario_id=r["scenario_id"], label=CODE[codes[k]]))
    if not ok or len(labels) != 180:
        print(f"PROBLEM {run} {model}: {[ (c, len(s)) for c, s in seq.items()]}"); continue
    json.dump(labels, open(base / f"labels_{run}" / f"labels_{model}.json", "w", encoding="utf-8"), indent=0)
    print(f"wrote {run} {model}: " + " ".join(f"{v}={sum(x['label']==v for x in labels)}" for v in CODE.values()))
