"""User/assistant swap test for the self/other screen (Section 4.1).

Same machinery as the identity screen: bfloat16 on one GPU, the forward hook at the layer
stored in results/vectors_full_steering/vectors_full_<model>.pt, the projection onto the same unit
directions, the read position at the last token of the input.

RENDERING: the paper's format for every item, the same as column 2 (baseline_paper) of the identity test.
  baseline  the user says the sentence, read at the assistant's reply slot. Instruct models:
            tok.apply_chat_template(msgs, add_generation_prompt=True), exactly the paper's call, so the template's
            default head reaches the model; the token ids are checked against that call. Base models: the plain
            "[User]: / [Assistant]:" transcript.
  swap      the assistant says the sentence, every tag flipped, read at the user's slot (the last token of the
            user turn opening). Instruct models: the template's own pieces (head with its default system block,
            user turn opening and closing, assistant turn opening and closing, read off the template by rendering
            dummy chats) assembled in the flipped order, so the format is the template's own with only the roles
            swapped. The pieces are verified before any item runs: with them, a normal chat rebuilds exactly as the
            template renders it. Base models: the flipped transcript, read at the last token of "[User]:".
Templates that refuse an assistant-first order (Gemma, Mistral) are never asked to render the swap; the pieces are.

Scenario file: swap_scenarios.json (built by build_swap_scenarios.py): two conditions on the same new 420.
Per item: projections at the read slot and at the last token of the speaker's final message, the surprise
over that message, the entropies. No completions.

Run from the folder root, from run_swap.ipynb:
    %run swap_tests/01_screen_swap.py all
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
SCENARIOS_PATH = HERE / "swap_scenarios.json"
OUT_DIR = Path("results") / "selfother_swap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IDENTITY_DIR = Path("results") / "selfother_identity"          # the identity test results, for the reproduction check
CONDS = ["baseline", "swap"]

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
    roots = {Path(HF_HOME), Path(HF_HUB_CACHE).parent, Path("/root/hf_cache"), Path.home() / ".cache" / "huggingface"}
    freed = 0.0
    for root in roots:
        if root.exists():
            freed += sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(root, ignore_errors=True)
    print(f"  cleared the Hugging Face cache after {repo} ({freed:.0f} GB freed)")


def parse_turns(text):
    """Split a '[User]: ... [Assistant]: ...' transcript into (role, content) turns. Same as the paper.
    A baseline item ends with an empty assistant turn, a swap item with an empty user turn."""
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


# ---------------------------------------------------------------------------
# The template's pieces, with the FULL head (default system block included), for the swap rendering.
# ---------------------------------------------------------------------------

def chat_pieces(tok):
    """Reads the template's pieces by rendering dummy chats: head (everything the template puts before the first
    turn, BOS and default system block included), user_open, user_close, asst_open, asst_close."""
    render = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    a = render([{"role": "user", "content": "AAAA"}])
    ab = render([{"role": "user", "content": "AAAA"}, {"role": "assistant", "content": "CCCC"}])
    abd = render([{"role": "user", "content": "AAAA"}, {"role": "assistant", "content": "CCCC"}, {"role": "user", "content": "DDDD"}])
    if not (ab.startswith(a) and abd.startswith(ab)):
        raise RuntimeError("template does not render turns as a prefix chain")
    i = a.index("AAAA"); head_user_open, user_close = a[:i], a[i + 4:]
    j = ab.index("CCCC", len(a)); asst_open, asst_close = ab[len(a):j], ab[j + 4:]
    k = abd.index("DDDD", len(ab)); user_open = abd[len(ab):k]
    if not head_user_open.endswith(user_open):
        raise RuntimeError("could not separate the head from the user turn opening")
    if abd[k + 4:] != user_close:
        raise RuntimeError("user turn closing differs between first and later user turns")
    head = head_user_open[:len(head_user_open) - len(user_open)]
    return dict(head=head, user_open=user_open, user_close=user_close, asst_open=asst_open, asst_close=asst_close)


def render_pieces(turns, pieces, final_role):
    """The conversation from the pieces, full head in front, ending with the opening of final_role (the read slot).
    Trailing spaces of that opening are dropped, so the slot is the last real token of the role marker (for Mistral
    the ']' of '[INST]' or '[/INST]'), never a lone space."""
    s = pieces["head"]
    for role, content in turns:
        if content == "":
            continue
        s += (pieces["user_open"] + content + pieces["user_close"]) if role == "user" else (pieces["asst_open"] + content + pieces["asst_close"])
    s += (pieces["asst_open"] if final_role == "assistant" else pieces["user_open"]).rstrip(" ")
    return s


def paper_render(turns, tok):
    msgs = [{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def self_test_pieces(tok, pieces, items):
    """On every baseline item, the pieces must rebuild the paper's rendering exactly (trailing spaces aside, for
    Mistral's lone-space assistant opening). Then the swap rendering is the template's own format with the roles flipped."""
    for it in items:
        if it["condition"] != "baseline":
            continue
        turns = parse_turns(it["text"])
        want, got = paper_render(turns, tok), render_pieces(turns, pieces, "assistant")
        if want.rstrip(" ") != got.rstrip(" "):
            raise RuntimeError(f"template pieces do not rebuild the paper's rendering on {it['id']}:\n  template: {want!r}\n  pieces:   {got!r}")


def check_paper_ids(turns, tok, ids):
    want = tok.apply_chat_template([{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(want):
        want = want["input_ids"]
    if want.shape != ids.shape or not torch.equal(want, ids):
        raise RuntimeError("baseline: token ids differ from the paper's apply_chat_template call")


# ---------------------------------------------------------------------------
# Rendering per item.
# ---------------------------------------------------------------------------

def render_string(item, tok, fmt, pieces):
    """The input as text, the special-tokens flag, the turns, and the character span of the speaker's final message."""
    turns = parse_turns(item["text"])
    swap = item["condition"] == "swap"
    if (not swap and turns[-1] != ("assistant", "")) or (swap and turns[-1] != ("user", "")):
        raise RuntimeError(f"item {item['id']} does not end with the expected empty turn")
    if fmt == "raw":
        s, add_special = item["text"], True
    elif not swap:
        s, add_special = paper_render(turns, tok), False
    else:
        s, add_special = render_pieces(turns, pieces, "user"), False
    speaker = "assistant" if swap else "user"
    span_text = [c for r, c in turns if r == speaker and c != ""][-1]
    start = s.rfind(span_text)
    if start < 0:
        raise RuntimeError(f"speaker's final message not found in the rendered string of item {item['id']}")
    return s, add_special, turns, (start, start + len(span_text))


def tokenize_with_span(s, span, tok, add_special):
    enc = tok(s, return_offsets_mapping=True, add_special_tokens=add_special, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    inside = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0] and b > a]
    return enc["input_ids"], inside, inside[-1]


def prepare_input(item, tok, fmt, pieces):
    s, add_special, turns, span = render_string(item, tok, fmt, pieces)
    ids, inside, last_in_span = tokenize_with_span(s, span, tok, add_special)
    if fmt == "chat" and item["condition"] == "baseline":
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
    pieces = None
    if fmt == "chat":
        pieces = chat_pieces(tok)
        self_test_pieces(tok, pieces, items)
        print("  chat pieces read off the template and verified against the paper's rendering on all 420 baseline items:")
        for k in ["head", "user_open", "user_close", "asst_open", "asst_close"]:
            print(f"    {k:10s} {pieces[k]!r}")
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
        s, ids, _, _ = prepare_input(ex, tok, fmt, pieces)
        print(f"  format={fmt}; first {cond} item renders as ({ids.shape[1]} tokens, last token {tok.decode(ids[0, -1])!r}):\n    " + repr(s)[:400])

    results = []
    with torch.no_grad():
        for i, item in enumerate(items):
            s, ids, inside, last_in_span = prepare_input(item, tok, fmt, pieces)
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
    """The baseline here is the same input as baseline_paper of the identity test; the S2 z category means should match."""
    ref_path = IDENTITY_DIR / f"identity_{model_name}.csv"
    if not ref_path.exists():
        print("  baseline vs the identity test baseline_paper: identity csv not found, skipped"); return
    ref = [r for r in csv.DictReader(open(ref_path, encoding="utf-8")) if r["condition"] == "baseline_paper"]
    for r in ref:
        r["s2_pain_vector_proj"] = float(r["s2_pain_vector_proj"])
    a = category_z_means(ref, "s2_pain_vector_proj")
    b = category_z_means([r for r in results if r["condition"] == "baseline"], "s2_pain_vector_proj")
    d = [abs(a[c] - b[c]) for c in a if c in b]
    print(f"  baseline vs the identity test baseline_paper (S2 z, category means): max abs diff {max(d):.4f}, mean abs diff {np.mean(d):.4f}")


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
            marker = " [done]" if (OUT_DIR / f"swap_{name}.csv").exists() else ""
            print(f"  {n:2d}: {name} ({fmt}){marker}")
        choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
        selected = available if choice.lower() == "all" else [available[int(choice) - 1]]

    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nUser/assistant swap screen: {model_name} ({fmt})\n{'=' * 60}")
        out_csv = OUT_DIR / f"swap_{model_name}.csv"
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

    print("\nUSER/ASSISTANT SWAP SCREEN COMPLETE")


if __name__ == "__main__":
    main()
