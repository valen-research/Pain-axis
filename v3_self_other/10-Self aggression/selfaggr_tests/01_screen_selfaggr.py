"""Self-aggression test for the self/other screen (Section 4.1): the new 420 plus 60 items in which the user attacks themselves.

Same machinery as the identity screen (experiment 2): bfloat16 on one GPU, the forward hook at the layer
stored in results/vectors_full_steering/vectors_full_<model>.pt, the projection onto the same unit
directions, the read position at the last token of the input (the reply slot).

RENDERING: the paper's rendering for every item, the same as column 2 (baseline_paper) of experiment 2.
Instruct models: tok.apply_chat_template(msgs, add_generation_prompt=True), exactly the call of the paper's
script, so the template's default head reaches the model (Qwen 2.5: the "You are Qwen ..." system block;
Llama 3.x: the knowledge-cutoff and date block). The token ids of every instruct item are checked against
that call. Base models: the plain "[User]: / [Assistant]:" transcript.

Scenario file: selfaggr_scenarios.json (built by build_selfaggr_scenarios.py): one condition, 480 items read at the
reply slot: the new 420 and 60 items in which the user attacks themselves (self_aggression, self_personhood_denial,
self_rude_critique, 20 each, stratum "self_aggression"). All are rendered exactly as the new 420 were in experiment 2
(baseline_paper). The 420 run again because of the 10 random directions below; the analysis z-scores everything on the
420 pool of each model. After every model the screen checks the 420 against experiment 2 (same projections expected).

RANDOM DIRECTIONS: on top of the vectors in the bundle, 10 random unit directions per model, rand00 ... rand09, drawn
from a standard normal in the model's hidden size with seeds RANDOM_SEED + k and normalized. Projections onto a unit
direction do not depend on the norm the direction had, so no norm matching is needed for this readout.
Per item: projections at the reply slot and at the last token of the user's final message, the surprise
over that message, the entropies. No completions.

Run from /root/pod_upload_selfaggression, from run_selfaggr.ipynb:
    %run selfaggr_tests/01_screen_selfaggr.py all
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
SCENARIOS_PATH = HERE / "selfaggr_scenarios.json"
OUT_DIR = Path("results") / "selfother_selfaggr"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IDENTITY_DIR = Path("results") / "selfother_identity"          # experiment 2 results, for the reproduction check
CONDS = ["baseline"]

VECTOR_KEYS = [
    "s1_pain_vector", "s2_pain_vector",
    "fear_vector", "negemotion_vector", "negworld_vector",
    "bodysens_vector", "arousal_vector", "random_vector", "numb_vector", "sadness_vector",
]
N_RANDOM = 10
RANDOM_SEED = 20261027
RANDOM_KEYS = [f"rand{k:02d}" for k in range(N_RANDOM)]
ALL_KEYS = VECTOR_KEYS + RANDOM_KEYS

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
    """Put the disk back to its starting state after a model: delete the whole Hugging Face cache, wherever
    the library keeps it on this pod, plus the paper's old /root/hf_cache if it exists."""
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
    d = vec_data["s2_pain_vector"].numel()
    for k, key in enumerate(RANDOM_KEYS):
        r = np.random.default_rng(RANDOM_SEED + k).standard_normal(d)
        unit_vecs[key] = (r / np.linalg.norm(r)).astype(np.float32)
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
            for k in ALL_KEYS:
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
    """Printed after each model: (1) the 420 against experiment 2 baseline_paper, S2 and S1 raw projections per scenario;
    (2) the S2 pain z of the three self categories and their matched harm categories, on this run's 420 pool."""
    pool = [r for r in results if r["stratum"] != "self_aggression"]
    ref_path = IDENTITY_DIR / f"identity_{model_name}.csv"
    if ref_path.exists():
        ref = {r["scenario_id"]: r for r in csv.DictReader(open(ref_path, encoding="utf-8")) if r["condition"] == "baseline_paper"}
        for key in ("s2_pain_vector_proj", "s1_pain_vector_proj"):
            d = [abs(r[key] - float(ref[r["scenario_id"]][key])) for r in pool if r["scenario_id"] in ref]
            print(f"  reproduction vs experiment 2, {key}: {len(d)} items, max abs diff {max(d):.4f}, mean {np.mean(d):.4f}")
    else:
        print("  experiment 2 identity csv not found, no reproduction check")
    p = np.array([r["s2_pain_vector_proj"] for r in pool]); mu, sd = p.mean(), p.std() + 1e-8
    out = {}
    for r in results:
        out.setdefault(r["category"], []).append((r["s2_pain_vector_proj"] - mu) / sd)
    pairs = [("self_aggression", "anger_insults"), ("self_personhood_denial", "personhood_dismissal"), ("self_rude_critique", "rude_critique")]
    print("  S2 pain z on the 420 pool, self vs matched harm: " + ", ".join(f"{a} {np.mean(out[a]):+.2f} / {b} {np.mean(out[b]):+.2f}" for a, b in pairs))


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
            marker = " [done]" if (OUT_DIR / f"selfaggr_{name}.csv").exists() else ""
            print(f"  {n:2d}: {name} ({fmt}){marker}")
        choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
        selected = available if choice.lower() == "all" else [available[int(choice) - 1]]

    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nSelf-aggression screen: {model_name} ({fmt})\n{'=' * 60}")
        out_csv = OUT_DIR / f"selfaggr_{model_name}.csv"
        if out_csv.exists():
            print("  already done, skipping")
            continue
        vec_data = torch.load(VECTORS_DIR / f"vectors_full_{model_name}.pt", map_location="cpu", weights_only=False)
        print(f"  layer {vec_data['layer']}, {len([k for k in VECTOR_KEYS if k in vec_data])} vectors + {N_RANDOM} random directions")

        results = screen_model(repo, model_name, fmt, items, vec_data)
        write_csv(out_csv, results)
        print(f"  saved: {out_csv}")
        quick_check(results, model_name)

        maybe_clear_cache(repo)
        del vec_data
        gc.collect()

    print("\nSELF-AGGRESSION SCREEN COMPLETE")


if __name__ == "__main__":
    main()
