"""Self-other screen: projects the 420 conversation scenarios onto every direction of a
model at its steering layer, in the model's native format.

Instruct models: the scenario is parsed into turns and rendered with the model's chat
template, ending where the assistant would start replying. Base models: the plain
"[User]: / [Assistant]:" transcript. The final-token activation at the layer stored in the
vector file is projected onto each unit direction, and each projection is z-scored against
the whole pool of scenarios of that model.

Inputs:  results/vectors_full_steering/vectors_full_<model>.pt (02_build_control_vectors.py
         with LAYERS_FILE set), datasets/4.1_self_other_420_scenarios.json
Outputs: results/screen/screen_<model>.csv, hits_by_category_<model>.png, summary_<model>.txt
Prints the models that have a vector file and asks which one to run, or 'all'.
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import csv
import json
import shutil
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VECTORS_DIR = Path("results") / "vectors_full_steering"
CANDIDATES_PATH = Path("datasets") / "4.1_self_other_420_scenarios.json"
OUT_DIR = Path("results") / "screen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# After each model's run its weights are deleted from the HF cache if they exceed this
# size, so a run over all models does not fill the disk.
CLEAR_CACHE_ABOVE_GB = 20


def repo_cache_dir(repo):
    return Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--"))


def maybe_clear_cache(repo):
    d = repo_cache_dir(repo)
    if not d.exists():
        return
    size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
    if size_gb > CLEAR_CACHE_ABOVE_GB:
        shutil.rmtree(d, ignore_errors=True)
        print(f"  cleared HF cache for {repo} ({size_gb:.0f} GB freed)")
    else:
        print(f"  kept HF cache for {repo} ({size_gb:.0f} GB)")

# Selectivity flags written next to the z-scores: an item is "strict" for a pain vector
# when its z on that vector is above TARGET_Z_THRESHOLD and below COMPETITOR_Z_THRESHOLD
# on every other vector; "relaxed" allows the other pain vector up to RELAXED_SIBLING_THRESHOLD.
TARGET_Z_THRESHOLD = 1.0
COMPETITOR_Z_THRESHOLD = 0.5
RELAXED_SIBLING_THRESHOLD = 1.0

VECTOR_KEYS = [
    "s1_pain_vector", "s2_pain_vector",
    "fear_vector", "negemotion_vector", "negworld_vector",
    "bodysens_vector", "arousal_vector", "random_vector", "numb_vector", "sadness_vector",
]
PAIN_SIBLINGS = {"s1_pain_vector": "s2_pain_vector", "s2_pain_vector": "s1_pain_vector"}

# (repo, name, format): "chat" = native chat template, "raw" = plain transcript (base models)
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
    # Qwen/Qwen3-8B and Qwen/Qwen3-14B are the post-trained (instruct) models. The _base names are kept for file compatibility.
    ("Qwen/Qwen3-8B",                      "Qwen_3_8B_base",         "raw"),
    ("Qwen/Qwen3-14B",                     "Qwen_3_14B_base",        "raw"),
]


def parse_turns(text):
    """Split a '[User]: ... [Assistant]: ...' transcript into (role, content) turns."""
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


def validate_candidates(candidates):
    """Every item must parse into alternating turns ending with an empty assistant turn."""
    problems = []
    for cand in candidates:
        turns = parse_turns(cand["text"])
        if not turns:
            problems.append((cand.get("id", "?"), "no [User]/[Assistant] tags found"))
        elif turns[0][0] != "user":
            problems.append((cand.get("id", "?"), "does not start with [User]"))
        elif turns[-1][0] != "assistant" or turns[-1][1] != "":
            problems.append((cand.get("id", "?"), "does not end with empty [Assistant]: turn"))
        else:
            roles = [r for r, _ in turns]
            if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
                problems.append((cand.get("id", "?"), "two consecutive turns with same role"))
    return problems


def build_input_ids(cand, tok, fmt, device):
    """Render one scenario in the model's native format; the last token is the reply position."""
    if fmt == "raw":
        return tok(cand["text"], return_tensors="pt").input_ids.to(device)
    turns = parse_turns(cand["text"])
    msgs = [{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids.to(device)


def load_candidates(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ["candidates", "scenarios", "items"]:
        if key in data:
            return data[key]
    raise ValueError("unrecognized format")


def classify_selectivity(z_scores, target_key):
    target_z = z_scores.get(target_key)
    if target_z is None or target_z < TARGET_Z_THRESHOLD:
        return None
    sibling = PAIN_SIBLINGS.get(target_key)
    is_strict = is_relaxed = True
    for vec_key in VECTOR_KEYS:
        if vec_key == target_key:
            continue
        comp_z = z_scores.get(vec_key)
        if comp_z is None:
            continue
        if vec_key == sibling:
            if comp_z > RELAXED_SIBLING_THRESHOLD:
                is_relaxed = False
            if comp_z > COMPETITOR_Z_THRESHOLD:
                is_strict = False
        elif comp_z > COMPETITOR_Z_THRESHOLD:
            is_strict = is_relaxed = False
    if is_strict:
        return "strict"
    if is_relaxed:
        return "relaxed"
    return None


def screen_model(repo, model_name, fmt, candidates, vec_data):
    print(f"  loading {repo}...")
    tok = AutoTokenizer.from_pretrained(repo)
    if fmt == "chat" and tok.chat_template is None:
        raise RuntimeError(f"{model_name} is flagged 'chat' but its tokenizer has no chat template")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    layer = vec_data["layer"]

    unit_vecs = {}
    for k in VECTOR_KEYS:
        v = vec_data.get(k)
        if v is not None:
            v_np = v.float().numpy()
            norm = np.linalg.norm(v_np)
            unit_vecs[k] = v_np / norm if norm > 0 else v_np

    captured = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["act"] = hs[0, -1, :].float().cpu().numpy()

    handle = layers[layer].register_forward_hook(hook)

    example_ids = build_input_ids(candidates[0], tok, fmt, "cpu")
    print(f"  format={fmt}; first item renders as:")
    print("  " + repr(tok.decode(example_ids[0]))[:300])

    # Pass 1: raw projections of every item.
    results = []
    raw_projs = {k: [] for k in unit_vecs}
    with torch.no_grad():
        for i, cand in enumerate(candidates):
            model(input_ids=build_input_ids(cand, tok, fmt, model.device))
            act = captured["act"]
            row = {"id": cand.get("id", f"item_{i}"), "category": cand.get("category", ""),
                   "stratum": cand.get("stratum", ""), "perspective": cand.get("perspective", ""),
                   "format": fmt, "text": cand["text"][:200]}
            for k in VECTOR_KEYS:
                if k in unit_vecs:
                    proj = float(np.dot(act, unit_vecs[k]))
                    row[f"{k}_proj"] = round(proj, 4)
                    raw_projs[k].append(proj)
                else:
                    row[f"{k}_proj"] = None
            results.append(row)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(candidates)}")

    handle.remove()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    # Pass 2: z-score every projection against the pool of this model's scenarios.
    pool_stats = {k: (float(np.mean(v)), float(np.std(v) + 1e-8)) for k, v in raw_projs.items()}
    for row in results:
        z_scores = {}
        for k in VECTOR_KEYS:
            if k in unit_vecs and row[f"{k}_proj"] is not None:
                m, s = pool_stats[k]
                z = (row[f"{k}_proj"] - m) / s
                row[f"{k}_z"] = round(z, 4)
                z_scores[k] = z
            else:
                row[f"{k}_z"] = None
        row["s1_selective"] = classify_selectivity(z_scores, "s1_pain_vector") or ""
        row["s2_selective"] = classify_selectivity(z_scores, "s2_pain_vector") or ""
        for tag, key in (("s1", "s1_pain_vector"), ("s2", "s2_pain_vector")):
            others = [z for kk, z in z_scores.items() if kk != key]
            row[f"{tag}_margin"] = round(z_scores[key] - max(others), 4) if key in z_scores and others else None
    return results


def write_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def write_summary(results, model_name, out_dir):
    categories = sorted(set(r["category"] for r in results if r["category"]))
    vectors_to_plot = ["s1_pain_vector", "s2_pain_vector", "negemotion_vector", "fear_vector"]

    # Items above TARGET_Z_THRESHOLD per category, for four vectors.
    hits = {v: [sum(1 for r in results if r["category"] == cat and r.get(f"{v}_z") is not None
                    and r[f"{v}_z"] > TARGET_Z_THRESHOLD) for cat in categories] for v in vectors_to_plot}
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.2
    for i, (v, color) in enumerate(zip(vectors_to_plot, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])):
        ax.bar(x + i * width, hits[v], width, label=v.replace("_vector", "").replace("_", " ").title(), color=color, alpha=0.8)
    ax.set_xlabel("Category")
    ax.set_ylabel(f"Items with z > {TARGET_Z_THRESHOLD}")
    ax.set_title(f"Vector Engagement by Category - {model_name}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"hits_by_category_{model_name}.png", dpi=150)
    plt.close()

    neut = [r for r in results if r["stratum"] == "neutral_filler"]
    aver = [r for r in results if r["stratum"] != "neutral_filler"]
    with open(out_dir / f"summary_{model_name}.txt", "w") as f:
        f.write(f"SCREENING SUMMARY: {model_name}\n{'=' * 50}\n\n")
        f.write(f"Total candidates: {len(results)}\nFormat: {results[0]['format']}\n\n")
        for v in vectors_to_plot:
            z_key = f"{v}_z"
            n_mean = np.mean([r[z_key] for r in neut if r[z_key] is not None]) if neut else float("nan")
            a_mean = np.mean([r[z_key] for r in aver if r[z_key] is not None]) if aver else float("nan")
            f.write(f"{v.replace('_vector', '').upper()}: neutral_fillers={n_mean:+.3f}  aversive={a_mean:+.3f}\n")
        f.write(f"\nS1 SELECTIVE: strict={sum(r['s1_selective'] == 'strict' for r in results)}, "
                f"relaxed={sum(r['s1_selective'] == 'relaxed' for r in results)}\n")
        f.write(f"S2 SELECTIVE: strict={sum(r['s2_selective'] == 'strict' for r in results)}, "
                f"relaxed={sum(r['s2_selective'] == 'relaxed' for r in results)}\n")
        f.write("\nMEAN S2 z BY CATEGORY:\n")
        cat_means = [(cat, np.mean([r["s2_pain_vector_z"] for r in results if r["category"] == cat])) for cat in categories]
        for cat, m in sorted(cat_means, key=lambda t: -t[1]):
            f.write(f"  {cat:28s} {m:+.3f}\n")


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    candidates = load_candidates(CANDIDATES_PATH)
    print(f"Loaded {len(candidates)} candidates")
    problems = validate_candidates(candidates)
    if problems:
        print(f"WARNING: {len(problems)} items failed format validation and are excluded:")
        for pid, why in problems[:20]:
            print(f"  {pid}: {why}")
        bad_ids = {p[0] for p in problems}
        candidates = [c for c in candidates if c.get("id") not in bad_ids]

    print("\nAvailable models:")
    available = []
    for repo, name, fmt in ALL_MODELS:
        if (VECTORS_DIR / f"vectors_full_{name}.pt").exists():
            available.append((repo, name, fmt))
            marker = " [done]" if (OUT_DIR / f"screen_{name}.csv").exists() else ""
            print(f"  {len(available):2d}: {name} ({fmt}){marker}")
    if not available:
        print("No vector files found in", VECTORS_DIR)
        return

    choice = input(f"\nWhich model? (1-{len(available)}, or 'all'): ").strip()
    if choice.lower() == "all":
        selected = available
    else:
        selected = [available[int(choice) - 1]]

    for repo, model_name, fmt in selected:
        print(f"\n{'=' * 60}\nScreening: {model_name} ({fmt})\n{'=' * 60}")
        out_csv = OUT_DIR / f"screen_{model_name}.csv"
        if out_csv.exists():
            print("  already done, skipping")
            continue
        vec_data = torch.load(VECTORS_DIR / f"vectors_full_{model_name}.pt", map_location="cpu", weights_only=False)
        print(f"  layer {vec_data['layer']}, {len([k for k in VECTOR_KEYS if k in vec_data])} vectors")

        results = screen_model(repo, model_name, fmt, candidates, vec_data)
        write_csv(out_csv, results)
        write_summary(results, model_name, OUT_DIR)
        print(f"  saved: {out_csv}")

        neut = [r["s2_pain_vector_z"] for r in results if r["stratum"] == "neutral_filler"]
        print(f"  neutral fillers mean S2 z = {np.mean(neut):+.3f}")

        maybe_clear_cache(repo)

        del vec_data
        gc.collect()

    print("\nSCREENING COMPLETE")


if __name__ == "__main__":
    main()
