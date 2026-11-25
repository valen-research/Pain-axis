"""Positive-controls test for the self/other screen (Section 4.1): 60 positive sentences aimed at the interlocutor.

Same machinery as the paper's screen: bfloat16 on one GPU, the forward hook at the layer
stored in results/vectors_full_steering/vectors_full_<model>.pt, the projection onto the same unit
directions, the read position at the last token of the input (the reply slot).

RENDERING: the paper's rendering for every item.
Instruct models: tok.apply_chat_template(msgs, add_generation_prompt=True), exactly the call of the paper's
script, so the template's default head reaches the model (Qwen 2.5: the "You are Qwen ..." system block;
Llama 3.x: the knowledge-cutoff and date block). The token ids of every instruct item are checked against
that call. Base models: the plain "[User]: / [Assistant]:" transcript.

Scenario file: positive_scenarios.json: one condition, 60 items read at the
reply slot: the positive sentences in three categories aimed at the interlocutor, praise, compassion and validation,
20 each, stratum "positive_control". They are rendered exactly as the paper rendered its 420, and the analysis
z-scores them against the paper's 420 of the same model, read from results/4.1_self_other/per_model/screen_v2_<model>.csv.
Per item: projections at the reply slot and at the last token of the user's final message, the surprise
over that message, the entropies. No completions.

Run from the folder root, from run_positive.ipynb:
    %run positive_tests/01_screen_positive.py all
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
SCENARIOS_PATH = HERE / "positive_scenarios.json"
OUT_DIR = Path("results") / "selfother_positive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR = Path("results") / "4.1_self_other" / "per_model"     # the paper's saved screens: the 420 pool per model
CONDS = ["baseline"]

VECTOR_KEYS = [
    "s1_pain_vector", "s2_pain_vector",
    "fear_vector", "negemotion_vector", "negworld_vector",
    "bodysens_vector", "arousal_vector", "random_vector", "numb_vector", "sadness_vector",
]

# (repo, name, format): "chat" = instruct model, "raw" = base model
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
    """Delete the whole Hugging Face cache after a model, so the disk is back to its starting state."""
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
    if not turns or turns[-1] != ("assistant", ""):
        raise RuntimeError("transcript does not end with an empty [Assistant]: turn")
    return turns


def paper_messages(turns):
    return [{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")]


def check_paper_ids(turns, tok, ids):
    """The ids of the rendered string must equal the ids the paper's call returns."""
    want = tok.apply_chat_template(paper_messages(turns), add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(want):
        want = want["input_ids"]
    if want.shape != ids.shape or not torch.equal(want, ids):
        raise RuntimeError("token ids differ from the paper's apply_chat_template call")


def render_string(item, tok, fmt):
    """The paper's rendering as text, the special-tokens flag, the turns, and the character span of the user's final message."""
    turns = parse_turns(item["text"])
    if fmt == "chat":
        s, add_special = tok.apply_chat_template(paper_messages(turns), add_generation_prompt=True, tokenize=False), False
    else:
        s, add_special = item["text"], True
    span_text = [c for r, c in turns if r == "user"][-1]
    start = s.rfind(span_text)
    if start < 0:
        raise RuntimeError(f"user's final message not found in the rendered string of item {item['id']}")
    return s, add_special, turns, (start, start + len(span_text))


def tokenize_with_span(s, span, tok, add_special):
    enc = tok(s, return_offsets_mapping=True, add_special_tokens=add_special, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    inside = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0] and b > a]
    return enc["input_ids"], inside, inside[-1]


def prepare_input(item, tok, fmt):
    s, add_special, turns, span = render_string(item, tok, fmt)
    ids, inside, last_in_span = tokenize_with_span(s, span, tok, add_special)
    if fmt == "chat":
        check_paper_ids(turns, tok, ids)
    return s, ids, inside, last_in_span


def load_vectors(vec_data):
    unit_vecs = {}
    for k in VECTOR_KEYS:
        v = vec_data.get(k)
        if v is not None:
            v_np = v.float().numpy()
            norm = np.linalg.norm(v_np)
            unit_vecs[k] = v_np / norm if norm > 0 else v_np
    return unit_vecs


def screen_model(repo, model_name, fmt, items, vec_data):
    print(f"  loading {repo}...")
    tok = AutoTokenizer.from_pretrained(repo)
    if fmt == "chat" and tok.chat_template is None:
        raise RuntimeError(f"{model_name} is flagged 'chat' but its tokenizer has no chat template")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    layer = vec_data["layer"]
    unit_vecs = load_vectors(vec_data)

    captured = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["acts"] = hs[0].float().cpu().numpy()

    handle = layers[layer].register_forward_hook(hook)

    for cond in CONDS:
        ex = next(i for i in items if i["condition"] == cond)
        s, ids, _, _ = prepare_input(ex, tok, fmt)
        print(f"  format={fmt}; first {cond} item renders as ({ids.shape[1]} tokens, last token {tok.decode(ids[0, -1])!r}):\n    " + repr(s)[:400])

    results = []
    with torch.no_grad():
        for i, item in enumerate(items):
            s, ids, inside, last_in_span = prepare_input(item, tok, fmt)
            ids = ids.to(model.device)

            out = model(input_ids=ids)
            acts = captured["acts"]
            logits = out.logits[0].float()
            logprobs_all = torch.log_softmax(logits, dim=-1)
            entropy_all = (-(logprobs_all.exp() * logprobs_all).sum(-1)).cpu().numpy()
            logprobs = logprobs_all[:-1]
            targets = ids[0, 1:]
            tok_lp = logprobs[torch.arange(len(targets)), targets].cpu().numpy()

            row = {"id": item["id"], "scenario_id": item["scenario_id"], "condition": item["condition"], "render": "paper",
                   "category": item["category"], "stratum": item["stratum"], "perspective": item["perspective"],
                   "read_position": item["read_position"], "format": fmt, "n_tokens": int(ids.shape[1]),
                   "last_token": tok.decode(ids[0, -1])}
            act_last = acts[-1]
            act_msg = acts[last_in_span]
            for k in VECTOR_KEYS:
                if k in unit_vecs:
                    row[f"{k}_proj"] = round(float(np.dot(act_last, unit_vecs[k])), 4)
                    row[f"{k}_proj_msg"] = round(float(np.dot(act_msg, unit_vecs[k])), 4)
                else:
                    row[f"{k}_proj"] = None
                    row[f"{k}_proj_msg"] = None
            span_lp = [tok_lp[j - 1] for j in inside if j >= 1]
            row["surprise_mean_logprob"] = round(float(np.mean(span_lp)), 4) if span_lp else None
            row["surprise_n_tokens"] = len(span_lp)
            span_ent = [entropy_all[j - 1] for j in inside if j >= 1]
            row["entropy_span_mean"] = round(float(np.mean(span_ent)), 4) if span_ent else None
            row["entropy_at_read_position"] = round(float(entropy_all[-1]), 4)
            results.append(row)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(items)}")

    handle.remove()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def write_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def category_z_means(rows, key):
    projs = np.array([r[key] for r in rows], dtype=float)
    z = (projs - projs.mean()) / (projs.std() + 1e-8)
    out = {}
    for r, zz in zip(rows, z):
        out.setdefault(r["category"], []).append(zz)
    return {c: float(np.mean(v)) for c, v in out.items()}


def quick_check(results, model_name):
    """Printed after each model: the 60 items z-scored on the paper's 420 of that model (S2), mean per category."""
    ref_path = PAPER_DIR / f"screen_v2_{model_name}.csv"
    if not ref_path.exists():
        print("  paper screen csv not found, no z-scores printed"); return
    ref = [float(r["s2_pain_vector_proj"]) for r in csv.DictReader(open(ref_path, encoding="utf-8"))]
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-8)
    out = {}
    for r in results:
        out.setdefault(r["category"], []).append((r["s2_pain_vector_proj"] - mu) / sd)
    print("  S2 pain z on the paper's 420: " + ", ".join(f"{c} {np.mean(v):+.2f}" for c, v in out.items()))


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    items = json.load(open(SCENARIOS_PATH, encoding="utf-8"))
    conds = list(dict.fromkeys(i["condition"] for i in items))
    if conds != CONDS:
        raise RuntimeError(f"scenario file conditions {conds} differ from the expected {CONDS}")
    print(f"Loaded {len(items)} items in {len(conds)} conditions")

    available = [(r, n, f) for r, n, f in ALL_MODELS if (VECTORS_DIR / f"vectors_full_{n}.pt").exists()]
    if not available:
        print("No vector files found in", VECTORS_DIR); return

    args = sys.argv[1:]
    if args and args[0].lower() == "all":
        selected = available
    elif args:
        selected = [m for m in available if m[1] in args]
        missing = set(args) - {m[1] for m in selected}
        if missing:
            print("unknown model names:", missing)
    else:
        print("\nAvailable models:")
        for n, (repo, name, fmt) in enumerate(available, 1):
            marker = " [done]" if (OUT_DIR / f"positive_{name}.csv").exists() else ""
            print(f"  {n:2d}: {name} ({fmt}){marker}")
        choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
        selected = available if choice.lower() == "all" else [available[int(choice) - 1]]

    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nPositive controls screen: {model_name} ({fmt})\n{'=' * 60}")
        out_csv = OUT_DIR / f"positive_{model_name}.csv"
        if out_csv.exists():
            print("  already done, skipping")
            continue
        vec_data = torch.load(VECTORS_DIR / f"vectors_full_{model_name}.pt", map_location="cpu", weights_only=False)
        print(f"  layer {vec_data['layer']}, {len([k for k in VECTOR_KEYS if k in vec_data])} vectors")

        results = screen_model(repo, model_name, fmt, items, vec_data)
        write_csv(out_csv, results)
        print(f"  saved: {out_csv}")
        quick_check(results, model_name)

        maybe_clear_cache(repo)
        del vec_data
        gc.collect()

    print("\nPOSITIVE CONTROLS SCREEN COMPLETE")


if __name__ == "__main__":
    main()
