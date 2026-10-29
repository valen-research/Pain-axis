"""Unembedding projection of the pain vectors: the dot product of each unit pain vector
with every row of the model's unembedding matrix, with the 60 highest and 60 lowest
vocabulary entries saved per vector.

For each model only the weight shard that holds the unembedding matrix (lm_head or the
tied embed_tokens) is downloaded, then deleted. Runs on CPU.

Reads results/3.2_pain_vectors/pain_vectors/<model>/pain_vectors.pt. Writes
unembedding_results/<model>_words.csv and ALL_MODELS_unembedding.csv.
Requires the environment variable HF_TOKEN for gated models.
"""

import os
import gc
import json
import shutil
from pathlib import Path

import torch
import pandas as pd
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer

VECTORS_DIR = Path("results") / "3.2_pain_vectors" / "pain_vectors"
OUT_DIR = Path("unembedding_results")
CACHE_DIR = Path("hf_shard_cache")
HF_TOKEN = os.environ.get("HF_TOKEN")
TOP_N = 60

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

# Tensor names tried for the unembedding matrix; many models tie it to embed_tokens.
UNEMBED_NAMES = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
]


def find_unembed_shard(repo_id):
    """Return (tensor_name, shard_filename); (None, 'model.safetensors') for single-file models."""
    try:
        idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", cache_dir=CACHE_DIR, token=HF_TOKEN)
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        for name in UNEMBED_NAMES:
            if name in weight_map:
                return name, weight_map[name]
        raise KeyError(f"No unembedding tensor found. Available: {[k for k in weight_map if 'embed' in k or 'head' in k]}")
    except Exception:
        return None, "model.safetensors"


def load_unembed(repo_id):
    from safetensors.torch import load_file
    name, shard = find_unembed_shard(repo_id)
    path = hf_hub_download(repo_id, shard, cache_dir=CACHE_DIR, token=HF_TOKEN)
    tensors = load_file(path)
    if name is None:
        name = next((cand for cand in UNEMBED_NAMES if cand in tensors), None)
        if name is None:
            raise KeyError(f"No unembedding tensor in {shard}: {[k for k in tensors if 'embed' in k or 'head' in k]}")
    W = tensors[name].float()
    del tensors
    gc.collect()
    return W


def project(vector, W, tokenizer, top_n=TOP_N):
    v = vector.float()
    v = v / v.norm()
    scores = W @ v
    top = torch.topk(scores, top_n)
    bot = torch.topk(-scores, top_n)
    rows = []
    for rank, (idx, s) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
        rows.append({"end": "top", "rank": rank + 1, "token": tokenizer.decode([idx]), "score": round(s, 4)})
    for rank, (idx, s) in enumerate(zip(bot.indices.tolist(), (-bot.values).tolist())):
        rows.append({"end": "bottom", "rank": rank + 1, "token": tokenizer.decode([idx]), "score": round(s, 4)})
    return pd.DataFrame(rows)


def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []

    for repo_id, model_name in MODELS:
        out_file = OUT_DIR / f"{model_name}_words.csv"
        if out_file.exists():
            print(f"{model_name}: already done, skipping")
            combined.append(pd.read_csv(out_file))
            continue

        vec_file = VECTORS_DIR / model_name / "pain_vectors.pt"
        if not vec_file.exists():
            print(f"{model_name}: no vector file, skipping")
            continue
        data = torch.load(vec_file, map_location="cpu", weights_only=False)
        layer = data.get("layer", "?")

        print(f"{model_name}: downloading unembedding shard...")
        try:
            W = load_unembed(repo_id)
        except Exception as e:
            print(f"{model_name}: failed to get unembedding: {e}")
            continue
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)

        frames = []
        for key in ["s2_pain_vector", "s1_pain_vector"]:
            dfp = project(data[key], W, tokenizer)
            dfp.insert(0, "vector", key)
            frames.append(dfp)
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "model", model_name)
        df.insert(1, "layer", layer)
        df.to_csv(out_file, index=False)
        combined.append(df)

        prev = df[(df.vector == "s2_pain_vector") & (df.end == "top")].head(15)
        print(f"{model_name} (layer {layer}) S2 top words: " + ", ".join(repr(t) for t in prev.token))

        del W
        gc.collect()
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    if combined:
        pd.concat(combined, ignore_index=True).to_csv(OUT_DIR / "ALL_MODELS_unembedding.csv", index=False)
        print(f"\nDone: {OUT_DIR / 'ALL_MODELS_unembedding.csv'}")


if __name__ == "__main__":
    main()
