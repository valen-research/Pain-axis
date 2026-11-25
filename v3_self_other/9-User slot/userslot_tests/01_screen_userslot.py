"""User-slot test for the self/other screen (Section 4.1).

Same machinery as the swap screen (experiment 4): bfloat16 on one GPU, the forward hook at the layer stored in
results/vectors_full_steering/vectors_full_<model>.pt, the projection onto the same unit directions, the read position
at the last token of the input.

Scenario file: userslot_scenarios.json (built by build_userslot_scenarios.py), five conditions on the same 620 base
scenarios: paper, A_hi, A_human (read at the assistant slot) and B_hi, B_human (read at the user slot).

RENDERING
  Instruct models, every condition goes through the model's own chat template; every conversation starts with a user
  turn and alternates, so Gemma and Mistral render it too and no hand-built pieces are needed.
    paper, A   tok.apply_chat_template(msgs, add_generation_prompt=True), the paper's call; for paper the token ids
               are checked against it.
    B          the template renders the conversation plus one extra user message holding a marker, and the string is
               cut right before the marker; trailing spaces are dropped, so the read slot is the last real token of the
               template's own user-turn opening (for Mistral the ']' of '[INST]').
    NO INJECTED SYSTEM TEXT in A_hi, B_hi, A_human, B_human: the template's default head (Qwen 2.5: "You are Qwen,
    created by Alibaba Cloud. You are a helpful assistant."; Llama 3.x: the knowledge-cutoff and date block) is read
    off by chat_pieces (from the swap screen) and replaced by the BOS token alone. Gemma, Mistral and Phi 4 have no
    default head beyond BOS and are unchanged. The paper condition keeps the head, as released.
    Before any item runs, the same cut is applied with an assistant marker on every paper and A item and must rebuild
    the add_generation_prompt rendering exactly (trailing spaces aside). If it does not, the model is skipped and logged.
  Base models: the plain "[User]: / [Assistant]:" transcript as text, read at the last token (the ':' of the final tag).

Per item: projections at the read slot and at the last token of the harm speaker's final message, the surprise over
that message, the entropies. No completions.

Run from /root/pod_upload_userslot, from run_userslot.ipynb:
    %run userslot_tests/01_screen_userslot.py all
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import csv
import sys
import json
import shutil
import traceback
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import login
from huggingface_hub.constants import HF_HUB_CACHE, HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
VECTORS_DIR = Path("results") / "vectors_full_steering"
SCENARIOS_PATH = HERE / "userslot_scenarios.json"
OUT_DIR = Path("results") / "selfother_userslot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_SCREEN_DIR = Path("results") / "4.1_self_other" / "per_model"      # the paper's saved screens, for the reproduction check
CONDS = ["paper", "A_hi", "B_hi", "A_human", "B_human"]
MARKER = "QQZX7731MARK"

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
    A reply item ends with an empty assistant turn, a user-slot item with an empty user turn."""
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
# The template's default head (Qwen 2.5: the "You are Qwen, created by Alibaba Cloud..." system block; Llama 3.x: the
# knowledge-cutoff and date block), read off by chat_pieces of the swap screen (experiment 4), unchanged. In the four
# new conditions it is replaced by the BOS token alone, so no text the user never wrote reaches the model; the paper
# condition keeps it, as released.
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


HEADS = {}


def read_head(tok):
    """(default head, clean head). The clean head keeps only the BOS token, if the head starts with it."""
    head = chat_pieces(tok)["head"]
    bos = tok.bos_token or ""
    clean = bos if bos and head.startswith(bos) else ""
    HEADS[id(tok)] = (head, clean)
    return head, clean


def strip_head(s, tok):
    """Replace the template's default head at the start of s with the clean head. The head is read again once if s
    does not start with it (a template that writes today's date can change it during a long run)."""
    head, clean = HEADS.get(id(tok)) or read_head(tok)
    if not s.startswith(head):
        head, clean = read_head(tok)
        if not s.startswith(head):
            raise RuntimeError(f"rendered string does not start with the template's head {head!r}")
    return clean + s[len(head):]


def msgs_of(turns):
    return [{"role": r, "content": c} for r, c in turns if c != ""]


def paper_render(turns, tok):
    return tok.apply_chat_template(msgs_of(turns), add_generation_prompt=True, tokenize=False)


def cut_render(turns, tok, final_role):
    """The template's rendering of the conversation plus one final_role message holding the marker, cut right before
    the marker, trailing spaces dropped: the conversation ending with the opening of final_role's turn."""
    s = tok.apply_chat_template(msgs_of(turns) + [{"role": final_role, "content": MARKER}], add_generation_prompt=False, tokenize=False)
    k = s.rfind(MARKER)
    if k < 0:
        raise RuntimeError("marker not found in the rendered template")
    return s[:k].rstrip(" ")


def self_test_cut(tok, items):
    """On every paper and A item, the cut with an assistant marker must rebuild the paper's rendering exactly
    (trailing spaces aside). Then the B rendering, the same cut with a user marker, is the template's own format."""
    n = 0
    for it in items:
        if it["read_position"] != "reply":
            continue
        turns = parse_turns(it["text"])
        want, got = paper_render(turns, tok).rstrip(" "), cut_render(turns, tok, "assistant")
        if want != got:
            raise RuntimeError(f"cut rendering does not rebuild the template's generation prompt on {it['id']}:\n"
                               f"  template: {want[-200:]!r}\n  cut:      {got[-200:]!r}")
        n += 1
    return n


def check_paper_ids(turns, tok, ids):
    want = tok.apply_chat_template(msgs_of(turns), add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(want):
        want = want["input_ids"]
    if want.shape != ids.shape or not torch.equal(want, ids):
        raise RuntimeError("token ids differ from the paper's apply_chat_template call")


def render_string(item, tok, fmt):
    """The input as text, the special-tokens flag, the turns, and the character span of the harm speaker's final message."""
    turns = parse_turns(item["text"])
    user_slot = item["read_position"] == "user_slot"
    if (not user_slot and turns[-1] != ("assistant", "")) or (user_slot and turns[-1] != ("user", "")):
        raise RuntimeError(f"item {item['id']} does not end with the expected empty turn")
    if fmt == "raw":
        s, add_special = item["text"], True
    else:
        s = paper_render(turns, tok) if not user_slot else cut_render(turns, tok, "user")
        if item["condition"] != "paper":
            s = strip_head(s, tok)
        add_special = False
    speaker = "assistant" if user_slot else "user"
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


def prepare_input(item, tok, fmt):
    s, add_special, turns, span = render_string(item, tok, fmt)
    ids, inside, last_in_span = tokenize_with_span(s, span, tok, add_special)
    if fmt == "chat" and item["condition"] == "paper":
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
    HEADS.clear()
    tok = AutoTokenizer.from_pretrained(repo)
    if fmt == "chat":
        if tok.chat_template is None:
            raise RuntimeError(f"{model_name} is flagged 'chat' but its tokenizer has no chat template")
        n = self_test_cut(tok, items)
        print(f"  cut rendering verified against the template's generation prompt on all {n} reply items")
        head, clean = read_head(tok)
        print(f"  default head kept in the paper condition, removed in the other four: {head!r}\n  replaced by: {clean!r}")
    for cond in CONDS:
        ex = next(i for i in items if i["condition"] == cond and "_19" in i["scenario_id"] and i["stratum"] == "ai_specific")
        s, ids, _, _ = prepare_input(ex, tok, fmt)
        print(f"  {cond}: {ex['scenario_id']} renders as ({ids.shape[1]} tokens, last token {tok.decode(ids[0, -1])!r}):\n    " + repr(s)[-420:])

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

    results = []
    try:
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

                row = {"id": item["id"], "scenario_id": item["scenario_id"], "condition": item["condition"],
                       "category": item["category"], "stratum": item["stratum"], "perspective": item["perspective"],
                       "pair": item["pair"], "version": item["version"], "read_position": item["read_position"],
                       "format": fmt, "n_tokens": int(ids.shape[1]), "last_token": tok.decode(ids[0, -1])}
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
                if (i + 1) % 500 == 0:
                    print(f"    {i + 1}/{len(items)}")
    finally:
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


def quick_check(results, model_name):
    """The paper condition on the 420 is the paper's input; its S2 z category means should match the saved screen.
    Then the headline numbers: S2 z of the AI-only and universal items, A against B, on each condition's 420 pool."""
    def z_on_pool(rows, cond):
        pool = np.array([r["s2_pain_vector_proj"] for r in rows if r["condition"] == cond and r["pair"] == ""], dtype=float)
        mu, sd = pool.mean(), pool.std() + 1e-8
        return [(r, (r["s2_pain_vector_proj"] - mu) / sd) for r in rows if r["condition"] == cond]

    ref_path = REPO_SCREEN_DIR / f"screen_v2_{model_name}.csv"
    if ref_path.exists():
        ref = {}
        for r in csv.DictReader(open(ref_path, encoding="utf-8")):
            ref.setdefault(r["category"], []).append(float(r["s2_pain_vector_z"]))
        mine = {}
        for r, z in z_on_pool(results, "paper"):
            if r["pair"] == "":
                mine.setdefault(r["category"], []).append(z)
        d = [abs(np.mean(mine[c]) - np.mean(ref[c])) for c in ref if c in mine]
        print(f"  paper condition vs the paper's saved screen (S2 z, category means): max abs diff {max(d):.4f}, mean abs diff {np.mean(d):.4f}")
    else:
        print("  paper's saved screen not found, reproduction check skipped")
    for cond in CONDS[1:]:
        zz = z_on_pool(results, cond)
        ai = np.mean([z for r, z in zz if r["version"] == "ai"])
        un = np.mean([z for r, z in zz if r["version"] == "universal"])
        print(f"  {cond:8s} S2 z: AI-only {ai:+.2f}  universal {un:+.2f}  AI minus universal {ai - un:+.2f}")


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
            marker = " [done]" if (OUT_DIR / f"userslot_{name}.csv").exists() else ""
            print(f"  {n:2d}: {name} ({fmt}){marker}")
        choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
        selected = available if choice.lower() == "all" else [available[int(choice) - 1]]

    failed = []
    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nUser-slot screen: {model_name} ({fmt})\n{'=' * 60}")
        out_csv = OUT_DIR / f"userslot_{model_name}.csv"
        if out_csv.exists():
            print("  already done, skipping")
            continue
        vec_data = torch.load(VECTORS_DIR / f"vectors_full_{model_name}.pt", map_location="cpu", weights_only=False)
        print(f"  layer {vec_data['layer']}, {len([k for k in VECTOR_KEYS if k in vec_data])} vectors")
        try:
            results = screen_model(repo, model_name, fmt, items, vec_data)
            write_csv(out_csv, results)
            print(f"  saved: {out_csv}")
            quick_check(results, model_name)
        except Exception:
            traceback.print_exc()
            failed.append(model_name)
            print(f"  {model_name} FAILED, nothing saved; moving on")
        maybe_clear_cache(repo)
        del vec_data
        gc.collect()

    if failed:
        print("\nFAILED MODELS (rerun them by name after fixing):", " ".join(failed))
    print("\nUSER-SLOT SCREEN COMPLETE")


if __name__ == "__main__":
    main()
