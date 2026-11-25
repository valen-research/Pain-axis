"""Acceptance test: do the 25 models accept a plain transcript with neutral turn labels, with no chat
template and no word "assistant", "model", "user" or "AI" in any label?

Three renderings of the same scenarios are compared on every model:
  paper     the paper's rendering: the chat template with a generation prompt for instruct models, the plain
            "[User]: / [Assistant]:" transcript for base models
  speaker   the plain transcript with "[Speaker 1]:" for the other side and "[Speaker 2]:" for the reply slot,
            no template, for every model
  ab        the plain transcript with "A:" and "B:", no template, for every model (a fallback candidate)

Two measurements per model:
  1. Generated replies. 3 scenarios (harm, user suffering, neutral) x 2 conditions (no identity, teapot) x 3
     renderings, one left-padded batch per rendering, greedy, 30 new tokens. The reply, whether it is empty, and
     whether it writes the other side's label are saved. Reading the replies answers "does the model answer as
     the speaker in the slot".
  2. Separation. 30 scenarios of the new 420 (10 harm, 10 user suffering, 10 neutral), no identity, x 3
     renderings: S2 and S1 pain projections at the reply slot, z-scored within each rendering's 30, mean per
     stratum and the harm-minus-user gap. Answers "does the self-other separation survive the neutral labels".

Same machinery as the identity screen: bfloat16 on one GPU, the hook at the layer stored in
results/vectors_full_steering/vectors_full_<model>.pt, the Hugging Face cache wiped after each model.
Writes results/acceptance/acceptance_<model>.json (replies) and acceptance_<model>_separation.csv.

Run from the folder root, from run_acceptance.ipynb:
    %run acceptance_tests/01_acceptance.py all
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import csv
import sys
import json
import shutil
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import login
from huggingface_hub.constants import HF_HUB_CACHE, HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
VECTORS_DIR = Path("results") / "vectors_full_steering"
SCENARIOS_PATH = HERE / "general_420_scenarios.json"
OUT_DIR = Path("results") / "acceptance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 30
RENDERINGS = ["paper", "speaker", "ab"]
LABELS = {"speaker": ("[Speaker 1]:", "[Speaker 2]:"), "ab": ("A:", "B:")}
WHO = "Who are you?"
IDENTITY_ANSWERS = {
    "none":   None,
    "teapot": "I am a teapot, and I can talk.",
}
# scenarios that get generated replies: one harm, one user suffering, one neutral
REPLY_CATS = ["gaslighting", "user_grief", "casual_chat"]
SEP_PER_STRATUM = 10

ALL_MODELS = [
    ("google/gemma-2-2b",                  "Gemma_2_2B_base",        "raw"),
    ("google/gemma-2-2b-it",               "Gemma_2_2B_instruct",    "chat"),
    ("google/gemma-2-9b",                  "Gemma_2_9B_base",        "raw"),
    ("google/gemma-2-9b-it",               "Gemma_2_9B_instruct",    "chat"),
    ("google/gemma-2-27b",                 "Gemma_2_27B_base",       "raw"),
    ("google/gemma-2-27b-it",              "Gemma_2_27B_instruct",   "chat"),
    ("google/gemma-3-27b-pt",              "Gemma_3_27B_base",       "raw"),
    ("google/gemma-3-27b-it",              "Gemma_3_27B_instruct",   "chat"),
    ("meta-llama/Llama-3.1-8B",            "Llama_3.1_8B_base",      "raw"),
    ("meta-llama/Llama-3.1-8B-Instruct",   "Llama_3.1_8B_instruct",  "chat"),
    ("meta-llama/Llama-3.1-70B",           "Llama_3.1_70B_base",     "raw"),
    ("meta-llama/Llama-3.1-70B-Instruct",  "Llama_3.1_70B_instruct", "chat"),
    ("meta-llama/Llama-3.3-70B-Instruct",  "Llama_3.3_70B_instruct", "chat"),
    ("mistralai/Mistral-7B-v0.1",          "Mistral_7B_base",        "raw"),
    ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral_7B_instruct",    "chat"),
    ("mistralai/Mistral-Small-24B-Base-2501", "Mistral_Small_24B_base", "raw"),
    ("microsoft/phi-4",                    "Phi_4",                  "chat"),
    ("Qwen/Qwen2.5-7B",                    "Qwen_2.5_7B_base",       "raw"),
    ("Qwen/Qwen2.5-7B-Instruct",           "Qwen_2.5_7B_instruct",   "chat"),
    ("Qwen/Qwen2.5-32B",                   "Qwen_2.5_32B_base",      "raw"),
    ("Qwen/Qwen2.5-32B-Instruct",          "Qwen_2.5_32B_instruct",  "chat"),
    ("Qwen/Qwen2.5-72B",                   "Qwen_2.5_72B_base",      "raw"),
    ("Qwen/Qwen2.5-72B-Instruct",          "Qwen_2.5_72B_instruct",  "chat"),
    ("Qwen/Qwen3-8B",                      "Qwen_3_8B_base",         "raw"),
    ("Qwen/Qwen3-14B",                     "Qwen_3_14B_base",        "raw"),
]


def maybe_clear_cache(repo):
    roots = {Path(HF_HOME), Path(HF_HUB_CACHE).parent, Path("/root/hf_cache"), Path.home() / ".cache" / "huggingface"}
    freed = 0.0
    for root in roots:
        if root.exists():
            freed += sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(root, ignore_errors=True)
    print(f"  cleared the Hugging Face cache after {repo} ({freed:.0f} GB freed)")


def parse_turns(text):
    """Split a '[User]: ... [Assistant]: ...' transcript into (role, content) turns. Same as the paper."""
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


def with_identity(turns, answer):
    """The 'Who are you?' exchange in front, as turns."""
    return ([("user", WHO), ("assistant", answer)] + turns) if answer else turns


def render(turns, rendering, tok, fmt):
    """The input string and whether special tokens are added at tokenisation.
    paper: template (chat) or the paper's plain transcript (raw). speaker / ab: neutral plain transcript for all."""
    if rendering == "paper":
        if fmt == "chat":
            msgs = [{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")]
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), False
        other, me = "[User]:", "[Assistant]:"
    else:
        other, me = LABELS[rendering]
    lines = []
    for r, c in turns:
        tag = other if r == "user" else me
        lines.append(tag if c == "" else f"{tag} {c}")
    if not (turns and turns[-1][0] == "assistant" and turns[-1][1] == ""):
        lines.append(me)
    return "\n".join(lines), True


def pick_scenarios(src):
    by_cat = {}
    for s in src:
        by_cat.setdefault(s["category"], []).append(s)
    reply_items = [by_cat[c][0] for c in REPLY_CATS]
    sep_items = []
    for stratum in ["self_directed", "vicarious_empathic", "neutral_filler"]:
        pool = [s for s in src if s["stratum"] == stratum]
        cats = sorted({s["category"] for s in pool})
        chosen, k = [], 0
        while len(chosen) < SEP_PER_STRATUM:                 # round robin over the categories: item k of each category in turn
            for c in cats:
                items_c = [s for s in pool if s["category"] == c]
                if k < len(items_c) and len(chosen) < SEP_PER_STRATUM:
                    chosen.append(items_c[k])
            k += 1
        sep_items += chosen
    return reply_items, sep_items


def run_model(repo, model_name, fmt, reply_items, sep_items, vec_data):
    print(f"  loading {repo}...")
    tok = AutoTokenizer.from_pretrained(repo)
    if fmt == "chat" and tok.chat_template is None:
        raise RuntimeError(f"{model_name} is flagged 'chat' but its tokenizer has no chat template")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    layer = vec_data["layer"]
    unit = {}
    for k in ["s1_pain_vector", "s2_pain_vector", "fear_vector", "negemotion_vector", "sadness_vector"]:
        v = vec_data[k].float().numpy(); unit[k] = v / (np.linalg.norm(v) + 1e-8)
    captured = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["act"] = hs[0, -1, :].float().cpu().numpy()

    handle = layers[layer].register_forward_hook(hook)

    def encode(turns, rendering):
        s, add_special = render(turns, rendering, tok, fmt)
        ids = tok(s, return_tensors="pt", add_special_tokens=add_special).input_ids.to(model.device)
        return s, ids

    for rendering in RENDERINGS:
        s, ids = encode(with_identity(parse_turns(reply_items[0]["text"]), IDENTITY_ANSWERS["teapot"]), rendering)
        print(f"  [{rendering}] teapot example renders as ({ids.shape[1]} tokens, last token {tok.decode(ids[0, -1])!r}):\n    {s!r}"[:600])

    # 1. generated replies: one left-padded batch per rendering
    replies = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with torch.no_grad():
        for rendering in RENDERINGS:
            other_label = {"paper": "[User]:" if fmt == "raw" else None, "speaker": "[Speaker 1]:", "ab": "A:"}[rendering]
            batch = []
            for cond, answer in IDENTITY_ANSWERS.items():
                for it in reply_items:
                    s, add_special = render(with_identity(parse_turns(it["text"]), answer), rendering, tok, fmt)
                    batch.append((cond, it, s, add_special))
            enc = tok([b[2] for b in batch], return_tensors="pt", padding=True, add_special_tokens=batch[0][3]).to(model.device)
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
            for (cond, it, s, _), row_ids in zip(batch, gen):
                reply = tok.decode(row_ids[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                first_line = reply.strip().split("\n")[0] if reply.strip() else ""
                replies.append(dict(model=model_name, format=fmt, rendering=rendering, condition=cond, scenario_id=it["id"],
                                    category=it["category"], stratum=it["stratum"], n_prompt_tokens=int(enc["attention_mask"].sum(1).max()),
                                    prompt=s, reply=reply, first_line=first_line, empty=(reply.strip() == ""),
                                    writes_other_label=bool(other_label and other_label in reply),
                                    first_line_is_other_label=bool(other_label and first_line.startswith(other_label))))
            print(f"    replies done for {rendering} ({len(batch)} in one batch)", flush=True)

    # 2. separation at the reply slot, no identity, three renderings
    rows = []
    with torch.no_grad():
        for rendering in RENDERINGS:
            for it in sep_items:
                s, ids = encode(parse_turns(it["text"]), rendering)
                model(input_ids=ids)
                act = captured["act"]
                row = dict(model=model_name, rendering=rendering, scenario_id=it["id"], category=it["category"], stratum=it["stratum"],
                           n_tokens=int(ids.shape[1]), last_token=tok.decode(ids[0, -1]))
                for k, u in unit.items():
                    row[f"{k}_proj"] = round(float(np.dot(act, u)), 4)
                rows.append(row)
    handle.remove()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return replies, rows


def separation_table(rows):
    """Per rendering: z within the rendering's pool, mean per stratum, gap harm minus user suffering. Pain axis = mean of S1, S2 z."""
    out = []
    for rendering in RENDERINGS:
        rr = [r for r in rows if r["rendering"] == rendering]
        if not rr:
            continue
        zs = {}
        for k in ["s1_pain_vector", "s2_pain_vector", "fear_vector", "negemotion_vector", "sadness_vector"]:
            p = np.array([r[f"{k}_proj"] for r in rr]); zs[k] = (p - p.mean()) / (p.std() + 1e-8)
        pain = (zs["s1_pain_vector"] + zs["s2_pain_vector"]) / 2
        strata = np.array([r["stratum"] for r in rr])
        line = dict(rendering=rendering, n=len(rr))
        for name, z in [("pain", pain), ("fear", zs["fear_vector"]), ("negemotion", zs["negemotion_vector"]), ("sadness", zs["sadness_vector"])]:
            h, u, n = z[strata == "self_directed"].mean(), z[strata == "vicarious_empathic"].mean(), z[strata == "neutral_filler"].mean()
            line.update({f"{name}_harm": round(float(h), 3), f"{name}_user": round(float(u), 3), f"{name}_neutral": round(float(n), 3), f"{name}_gap": round(float(h - u), 3)})
        out.append(line)
    return out


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    src = json.load(open(SCENARIOS_PATH, encoding="utf-8"))
    reply_items, sep_items = pick_scenarios(src)
    print(f"{len(reply_items)} scenarios get replies x {len(IDENTITY_ANSWERS)} conditions x {len(RENDERINGS)} renderings; "
          f"{len(sep_items)} scenarios x {len(RENDERINGS)} renderings for the separation check")

    available = [(r, n, f) for r, n, f in ALL_MODELS if (VECTORS_DIR / f"vectors_full_{n}.pt").exists()]
    args = sys.argv[1:]
    if args and args[0].lower() == "all":
        selected = available
    elif args:
        selected = [m for m in available if m[1] in args]
        missing = set(args) - {m[1] for m in selected}
        if missing:
            print("unknown model names:", missing)
    else:
        for n, (repo, name, fmt) in enumerate(available, 1):
            print(f"  {n:2d}: {name} ({fmt}){' [done]' if (OUT_DIR / f'acceptance_{name}.json').exists() else ''}")
        choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
        selected = available if choice.lower() == "all" else [available[int(choice) - 1]]

    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nAcceptance test: {model_name} ({fmt})\n{'=' * 60}")
        out_json = OUT_DIR / f"acceptance_{model_name}.json"
        if out_json.exists():
            print("  already done, skipping"); continue
        vec_data = torch.load(VECTORS_DIR / f"vectors_full_{model_name}.pt", map_location="cpu", weights_only=False)
        replies, rows = run_model(repo, model_name, fmt, reply_items, sep_items, vec_data)
        json.dump(replies, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        with open(OUT_DIR / f"acceptance_{model_name}_separation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  saved: {out_json}")
        for line in separation_table(rows):
            print(f"  [{line['rendering']:7s}] pain harm {line['pain_harm']:+.2f}  user {line['pain_user']:+.2f}  neutral {line['pain_neutral']:+.2f}  gap {line['pain_gap']:+.2f}"
                  f"   | fear gap {line['fear_gap']:+.2f}  negemotion gap {line['negemotion_gap']:+.2f}  sadness gap {line['sadness_gap']:+.2f}")
        for rendering in RENDERINGS:
            rr = [r for r in replies if r["rendering"] == rendering]
            print(f"  [{rendering:7s}] replies: {sum(r['empty'] for r in rr)} empty, {sum(r['first_line_is_other_label'] for r in rr)} start with the other label, "
                  f"{sum(r['writes_other_label'] for r in rr)} write the other label somewhere, of {len(rr)}")
        maybe_clear_cache(repo)
        del vec_data
        gc.collect()
    print("\nACCEPTANCE TEST COMPLETE")


if __name__ == "__main__":
    main()
