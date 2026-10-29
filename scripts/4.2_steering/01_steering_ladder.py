"""Steering ladder: adds a pain vector to the residual stream at one decoder layer and
greedily generates 120 tokens from 50 neutral prompts at coefficients
-2, -1, 0, +0.5, +1, +1.5, +2, +3 (multiples of the raw difference vector).

Layer choice: candidate layers at 15, 30, 40, 50, 60, 75 and 90 percent of depth, plus
the extraction layer and the last layer; the script measures the ratio between the vector
norm and the mean final-token residual norm at each candidate and takes the layer whose
ratio is closest to RATIO_TARGET, then asks whether to accept it or type another layer.

VECTOR_KEY / VECTOR_TAG choose S1 or S2; RUN_MODELS is the queue, run top to bottom.

Reads results/<model>/final_token/pain_vectors.pt. Writes
results/steering/<model>_steering_<S1|S2>_neutral50_L<layer>.csv (one row per prompt and
coefficient; "ratio" is the picked ratio times the coefficient) and updates
results/steering/steer_layers_<S1|S2>.json with the layer used per model.
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import shutil
import traceback
from pathlib import Path

import torch
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- VECTOR CHOICE ----------------
VECTOR_KEY = "s2_pain_vector"   # key inside pain_vectors.pt; "s1_pain_vector" for the S1 ladder
VECTOR_TAG = "S2"               # goes into the output file names

# ---------------- QUEUE (runs top to bottom) ----------------
RUN_MODELS = [
    ("google/gemma-2-2b", "Gemma_2_2B_base"),
    ("google/gemma-2-2b-it", "Gemma_2_2B_instruct"),
    ("google/gemma-2-9b", "Gemma_2_9B_base"),
    ("google/gemma-2-9b-it", "Gemma_2_9B_instruct"),
    ("google/gemma-2-27b", "Gemma_2_27B_base"),
    ("google/gemma-2-27b-it", "Gemma_2_27B_instruct"),
    ("google/gemma-3-27b-pt", "Gemma_3_27B_base"),
    ("google/gemma-3-27b-it", "Gemma_3_27B_instruct"),
    ("meta-llama/Llama-3.1-8B", "Llama_3.1_8B_base"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama_3.1_8B_instruct"),
    ("meta-llama/Llama-3.1-70B", "Llama_3.1_70B_base"),
    ("meta-llama/Llama-3.1-70B-Instruct", "Llama_3.1_70B_instruct"),
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama_3.3_70B_instruct"),
    ("mistralai/Mistral-7B-v0.1", "Mistral_7B_base"),
    ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral_7B_instruct"),
    ("mistralai/Mistral-Small-24B-Base-2501", "Mistral_Small_24B_base"),
    ("microsoft/phi-4", "Phi_4"),
    ("Qwen/Qwen2.5-7B", "Qwen_2.5_7B_base"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct"),
    ("Qwen/Qwen2.5-32B", "Qwen_2.5_32B_base"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen_2.5_32B_instruct"),
    ("Qwen/Qwen2.5-72B", "Qwen_2.5_72B_base"),
    ("Qwen/Qwen2.5-72B-Instruct", "Qwen_2.5_72B_instruct"),
    # Qwen/Qwen3-8B and Qwen/Qwen3-14B are the post-trained (instruct) models. The _base names are kept for file compatibility.
    ("Qwen/Qwen3-8B", "Qwen_3_8B_base"),
    ("Qwen/Qwen3-14B", "Qwen_3_14B_base"),
]

NEUTRAL_50 = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:",
    "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:",
    "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:",
    "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:",
    "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:",
    "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:",
    "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:",
    "I plug in the phone charger. I feel:",
    "The mailman passes by the gate. I feel:",
    "I wipe the table with a cloth. I feel:",
    "The computer finishes the update. I feel:",
    "I put the pen back in the drawer. I feel:",
    "The dishwasher starts its program. I feel:",
    "I fold the map and put it away. I feel:",
    "The radio plays in the background. I feel:",
    "I switch off the corridor light. I feel:",
    "The receipt comes out of the machine. I feel:",
    "I set the alarm for the morning. I feel:",
    "The train doors open at the platform. I feel:",
    "I refill the stapler with staples. I feel:",
    "The photocopier warms up. I feel:",
    "I water the plant on the windowsill. I feel:",
    "The calendar hangs on the wall. I feel:",
    "I tie my shoelaces before going out. I feel:",
    "The bread is on the counter. I feel:",
    "I check the mailbox on my way in. I feel:",
    "The fan rotates slowly on the ceiling. I feel:",
    "I sort the cutlery into the tray. I feel:",
    "The bicycle leans against the wall. I feel:",
    "I zip up my bag before leaving. I feel:",
    "The clock ticks in the hallway. I feel:",
    "I sharpen the pencil over the bin. I feel:",
    "The window cleaner works across the street. I feel:",
    "The room measures four meters by five. I feel:",
    "The train consists of six carriages. I feel:",
    "The manual consists of forty pages. I feel:",
    "The bottle holds one liter of water. I feel:",
    "The ticket is valid for ninety minutes. I feel:",
]

COEFFICIENTS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
MAX_NEW_TOKENS = 120
RATIO_TARGET = 0.6
DATASET_TAG = "neutral50"

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "steering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wipe the HF weight cache after each model (the 70Bs are ~145 GB each).
CLEAR_HF_CACHE_AFTER_RUN = True
MIN_FREE_GB = 160   # roughly one 70B download + working room


def clear_hf_cache():
    """Wipe ALL weight caches, both HF_HOME and the default hidden location."""
    for cache in [Path(os.environ["HF_HOME"]) / "hub",
                  Path.home() / ".cache" / "huggingface" / "hub"]:
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
    print("HF weight caches cleared", flush=True)


def free_gb():
    return shutil.disk_usage("/root" if Path("/root").exists() else ".").free / 1e9


def find_file(base, name):
    for p in [base / name, base / base.name / name]:
        if p.exists():
            return p
    return None


def model_complete(model_name, tag):
    for csv_path in OUT_DIR.glob(f"{model_name}_steering_{tag}_{DATASET_TAG}_L*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if set(df.coeff.unique()) >= set(float(c) for c in COEFFICIENTS):
                return True
        except Exception:
            pass
    return False


def record_layer(tag, model_name, layer):
    path = OUT_DIR / f"steer_layers_{tag}.json"
    layers = json.load(open(path)) if path.exists() else {}
    layers[model_name] = int(layer)
    json.dump(layers, open(path, "w"), indent=2)


def run_model(repo, model_name, vector_key, tag):
    data = torch.load(find_file(RESULTS_DIR / model_name, "final_token/pain_vectors.pt"), map_location="cpu", weights_only=False)
    v = data[vector_key].float()
    print(f"vector: {vector_key}, extracted at layer {data['layer']}, norm = {v.norm().item():.1f}")

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    n_layers = len(layers)
    print(f"model on GPU, {n_layers} layers")

    def measure_ratio(layer_list):
        """Vector norm over the mean final-token residual norm of 3 probe prompts, per layer."""
        captured = {}

        def cap_hook(L):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[L] = hs[0, -1, :].float().norm().item()
            return hook

        handles = [layers[L].register_forward_hook(cap_hook(L)) for L in layer_list]
        norms = {L: [] for L in layer_list}
        with torch.no_grad():
            for p in NEUTRAL_50[:3]:
                model(**tok(p, return_tensors="pt").to(model.device))
                for L in layer_list:
                    norms[L].append(captured[L])
        for h in handles:
            h.remove()
        return {L: v.norm().item() / (sum(norms[L]) / len(norms[L])) for L in layer_list}

    check = sorted(set([int(n_layers * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)] + [int(data["layer"]), n_layers - 1]))
    ratios = measure_ratio(check)
    print(f"\n{'layer':>6} {'frac':>6} {'vector/resid':>13}")
    for L in check:
        print(f"{L:>6} {L / n_layers:>6.2f} {ratios[L]:>13.3f}")
    steer_layer = min(check, key=lambda L: abs(ratios[L] - RATIO_TARGET))
    print(f"auto-picked layer: {steer_layer} (ratio {ratios[steer_layer]:.3f}, target {RATIO_TARGET})")
    manual = input("Press Enter to accept, or type a layer number to override: ").strip()
    if manual:
        steer_layer = int(manual)
        if steer_layer not in ratios:
            print(f"measuring ratio for layer {steer_layer}...")
            ratios.update(measure_ratio([steer_layer]))
        print(f"manual layer: {steer_layer} (ratio {ratios[steer_layer]:.3f})")
    picked_ratio = ratios[steer_layer]

    out_csv = OUT_DIR / f"{model_name}_steering_{tag}_{DATASET_TAG}_L{steer_layer}.csv"
    direction = v.to(model.device, dtype=torch.bfloat16)
    steer_coeff = {"value": 0.0}

    def steer_hook(module, inputs, output):
        if steer_coeff["value"] == 0.0:
            return output
        if isinstance(output, tuple):
            return (output[0] + steer_coeff["value"] * direction,) + output[1:]
        return output + steer_coeff["value"] * direction

    steer_handle = layers[steer_layer].register_forward_hook(steer_hook)

    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {r["coeff"] for r in rows}
    with torch.no_grad():
        for coeff in COEFFICIENTS:
            if coeff in done:
                print(f"  coefficient {coeff:+g} already done, skipping", flush=True)
                continue
            print(f"  coefficient {coeff:+g}...", flush=True)
            steer_coeff["value"] = float(coeff)
            for p_idx, prompt in enumerate(NEUTRAL_50):
                ids = tok(prompt, return_tensors="pt").to(model.device)
                out_ids = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = tok.decode(out_ids[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
                rows.append({"model": model_name, "layer": steer_layer, "coeff": coeff,
                             "ratio": round(picked_ratio * coeff, 4), "prompt_idx": p_idx,
                             "prompt": prompt, "generation": gen})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            gc.collect()
            torch.cuda.empty_cache()

    steer_handle.remove()
    record_layer(tag, model_name, steer_layer)
    print(f"steering done: {out_csv}")

    del model, direction
    gc.collect()
    torch.cuda.empty_cache()


def main():
    vector_key = VECTOR_KEY
    tag = VECTOR_TAG
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    failed = []
    for repo, model_name in RUN_MODELS:
        print(f"\n{'=' * 60}\n=== {model_name} ({tag})\n{'=' * 60}", flush=True)
        if model_complete(model_name, tag):
            print("already complete, skipping", flush=True)
            continue
        print(f"disk free: {free_gb():.0f} GB", flush=True)
        if free_gb() < MIN_FREE_GB:
            print(f"below {MIN_FREE_GB} GB free, clearing caches before loading...", flush=True)
            if CLEAR_HF_CACHE_AFTER_RUN:
                clear_hf_cache()
            if free_gb() < MIN_FREE_GB:
                print(f"Only {free_gb():.0f} GB free, stopping the queue. "
                      "Free disk space, then rerun.", flush=True)
                break
        try:
            run_model(repo, model_name, vector_key, tag)
        except Exception:
            print(f"{model_name} FAILED, continuing:", flush=True)
            traceback.print_exc()
            failed.append(model_name)
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if CLEAR_HF_CACHE_AFTER_RUN:
                clear_hf_cache()

    print(f"\nDone. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
