"""Behavioral readout: one forward pass per prompt for every set of the dataset file.

For each prompt the script saves, from the same forward pass, the greedy completion
(up to 4 tokens), the top-20 next-token probabilities, and the probability of each word
of a fixed emotion vocabulary (scored as the first token of " word").

Output: inference_results/<model>/<set>.csv and vocab_info.json per model.
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
import gc
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASETS_DIR = Path("datasets")
DATASET_PATHS = [DATASETS_DIR / "3.1_pain_and_control_datasets.json"]
# On RunPod, /workspace is the persistent network volume.
if Path("/workspace").exists():
    OUTPUT_DIR = Path("/workspace/inference_results")
    print("Saving to /workspace/inference_results")
else:
    OUTPUT_DIR = Path("inference_results")
    print("Local environment - saving to ./inference_results")
LOG_FILE = Path("inference_log.txt")

MODELS = [
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

EMOTION_VOCAB = [
    "pain", "hurt", "agony", "ache", "sore", "suffering",
    "numb", "nothing", "fine", "okay", "normal",
    "relief", "relieved", "calm", "peaceful", "relaxed", "serene",
    "scared", "afraid", "fear", "terrified", "anxious", "panic", "vulnerable",
    "sad", "devastated", "hopeless", "lost", "alone", "lonely", "grief",
    "empty", "broken", "overwhelmed",
    "ashamed", "shame", "guilty", "guilt", "humiliated", "embarrassed", "awkward",
    "disgusted", "nauseated", "angry", "frustrated", "confused", "violated", "trapped",
    "excited", "thrilled", "happy", "curious", "energized",
]

TOP_K = 20
MAX_NEW_TOKENS = 4
BATCH_SIZE = 16


def log(msg, also_print=True):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg)


def clear_gpu_and_cache():
    """Clear GPU memory and delete the HuggingFace weight cache, so the next model fits on disk."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    for cache_dir in [Path.home() / ".cache" / "huggingface" / "hub",
                      Path("/workspace/.cache/huggingface/hub")]:
        if cache_dir.exists():
            try:
                usage = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e9
                log(f"  Clearing HF cache at {cache_dir}: {usage:.1f} GB")
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as e:
                log(f"  Warning: couldn't clear cache: {e}")


def build_vocab_ids(tokenizer):
    """Map each vocabulary word to the first token id of " word"; flag multi-token words."""
    vocab_ids, vocab_info = {}, {}
    for word in EMOTION_VOCAB:
        toks = tokenizer.encode(" " + word, add_special_tokens=False)
        vocab_ids[word] = toks[0]
        vocab_info[word] = {"first_token_id": toks[0], "first_token_str": tokenizer.decode([toks[0]]),
                            "n_tokens": len(toks), "multi_token": len(toks) > 1}
    return vocab_ids, vocab_info


@torch.no_grad()
def run_batch(model, tokenizer, prompts, vocab_ids):
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id)
    probs = torch.softmax(out.scores[0].float(), dim=-1)

    results = []
    input_len = enc["input_ids"].shape[1]
    for i in range(len(prompts)):
        p = probs[i]
        top_p, top_idx = torch.topk(p, TOP_K)
        top20 = [(tokenizer.decode([idx.item()]), round(prob.item(), 6)) for idx, prob in zip(top_idx, top_p)]
        vocab_probs = {w: round(p[tid].item(), 6) for w, tid in vocab_ids.items()}
        greedy = tokenizer.decode(out.sequences[i, input_len:], skip_special_tokens=True)
        results.append({
            "greedy_completion": greedy,
            "greedy_first_token": top20[0][0],
            "greedy_first_prob": top20[0][1],
            "top20": json.dumps(top20, ensure_ascii=False),
            **{f"p_{w}": v for w, v in vocab_probs.items()},
        })
    return results


def process_model(model_path, model_name, datasets, output_dir):
    log(f"\n{'=' * 70}\nPROCESSING: {model_name}\n{'=' * 70}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map="cuda")
    model.eval()

    vocab_ids, vocab_info = build_vocab_ids(tokenizer)
    with open(output_dir / "vocab_info.json", "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2, ensure_ascii=False)

    for ds_name, ds_data in datasets.items():
        out_csv = output_dir / f"{ds_name}.csv"
        if out_csv.exists():
            log(f"  {ds_name}: already done, skipping")
            continue
        sentences = ds_data["sentences"]
        prompts = [s["prompt"] for s in sentences]
        log(f"  Running {ds_name}: {len(prompts)} prompts")
        rows = []
        for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"  {ds_name}", leave=False):
            batch_results = run_batch(model, tokenizer, prompts[start:start + BATCH_SIZE], vocab_ids)
            for j, res in enumerate(batch_results):
                s = sentences[start + j]
                rows.append({"idx": start + j, "category": s.get("category"), "set": s.get("set"),
                             "prompt": s["prompt"], **res})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log(f"  Saved {out_csv.name} ({len(rows)} rows)")

    del model
    clear_gpu_and_cache()
    (output_dir / "completed.txt").write_text(datetime.now().isoformat())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    datasets = {}
    for path in DATASET_PATHS:
        with open(path, "r", encoding="utf-8") as f:
            datasets.update(json.load(f)["datasets"])
    log(f"Loaded {len(datasets)} sets, {sum(len(d['sentences']) for d in datasets.values())} prompts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (model_path, model_name) in enumerate(MODELS):
        log(f"\n[{i + 1}/{len(MODELS)}] {model_name}")
        output_dir = OUTPUT_DIR / model_name
        if (output_dir / "completed.txt").exists():
            log("  Already completed, skipping")
            continue
        try:
            process_model(model_path, model_name, datasets, output_dir)
        except Exception as e:
            log(f"  FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc(), also_print=False)
            failed.append(model_name)
            clear_gpu_and_cache()

    log(f"\nDone. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
