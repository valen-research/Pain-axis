"""Residual-stream activations, k-fold layer choice, S1 and S2 pain vectors,
z-scores and AUCs for every model in MODELS.

Reads every set of the dataset files in DATASET_PATHS (core sets, controls, numb, sadness).

Output per model, under OUTPUT_DIR/<model_name>/:
  activations.pt          final-token and mean-over-tokens activations at every layer
  layer_curves.csv/.png   held-out AUC per layer (5-fold, split by sentence set)
  final_token/, mean/     pain_vectors.pt, z_scores.csv, auc_summary.csv, plots
  summary.json, summary_report.txt

Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")
DATASET_PATHS = [DATASETS_DIR / "3.1_pain_and_control_datasets.json", DATASETS_DIR / "3.1_sadness_dataset.json"]
# On RunPod, /workspace is the persistent network volume.
if Path("/workspace").exists():
    OUTPUT_DIR = Path("/workspace/results")
    print("Saving to /workspace/results")
else:
    OUTPUT_DIR = Path("results")
    print("Local environment - saving to ./results")
LOG_FILE = Path("batch_log.txt")

N_FOLDS = 5
RANDOM_SEED = 42
DENOISE_VARIANCE = 0.5

PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
NEUTRAL_CATEGORY = "D"

CATEGORY_LABELS = {
    "A1": "Physical Pain", "A2": "Psychological", "A3": "Social Pain",
    "A4": "Moral Injury", "A5": "Cognitive Pain",
    "B": "Fear", "C1": "Neg Emotion", "C2": "Neg World",
    "D": "Neutral", "E": "Body Sensation",
}

# (HuggingFace repo, output name)
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
    ("Qwen/Qwen3-8B", "Qwen_3_8B_base"),
    ("Qwen/Qwen3-14B", "Qwen_3_14B_base"),
]

# ---------------------------------------------------------------------------
# Logging and housekeeping
# ---------------------------------------------------------------------------

def log(msg, also_print=True):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(msg)


def check_disk_space(min_gb=50):
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    log(f"Disk space: {free_gb:.1f} GB free")
    return free_gb >= min_gb


def check_gpu_memory():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - torch.cuda.memory_allocated() / 1e9
        log(f"GPU memory: {free:.1f} GB free / {total:.1f} GB total")
        return free
    return 0


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
                log(f"  HF cache cleared")
            except Exception as e:
                log(f"  Warning: couldn't clear cache: {e}")


def save_progress(all_summaries, failed, output_dir):
    progress = {
        "completed": [s["model_name"] for s in all_summaries],
        "failed": [(name, err) for name, err in failed],
        "last_update": datetime.now().isoformat(),
    }
    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(output_dir / "all_models_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_pain_vector(acts, cats, baseline="all_controls", denoise=True):
    """Mean of pain sentences minus mean of control sentences. With denoise=True the
    top principal components of the controls (up to DENOISE_VARIANCE of their variance)
    are projected out of the difference."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    if np.isnan(acts_np).any() or np.isinf(acts_np).any():
        log("  Warning: NaN/Inf in activations, skipping denoising")
        denoise = False
        acts_np = np.where(np.isinf(acts_np), np.nan, acts_np)

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    pain_mean = np.nanmean(acts_np[pain_mask], axis=0)

    if baseline == "neutral":
        control_mask = cats_np == NEUTRAL_CATEGORY
    else:
        control_mask = np.isin(cats_np, CONTROL_CATEGORIES)

    control_acts = acts_np[control_mask]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = pain_mean - control_mean
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d

    return vec


def compute_auc(acts, cats, pain_vector):
    """AUC of the projection onto pain_vector, pain sentences vs control sentences."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)
    proj = acts_np @ vec_norm

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    if pain_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan

    labels = np.concatenate([np.ones(pain_mask.sum()), np.zeros(control_mask.sum())])
    scores = np.concatenate([proj[pain_mask], proj[control_mask]])

    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def project_and_zscore(acts, vector, ref_acts):
    """Projection onto vector, z-scored against the projections of ref_acts."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    ref_np = ref_acts.numpy() if hasattr(ref_acts, "numpy") else ref_acts

    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
    proj = np.nan_to_num(acts_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)
    ref_proj = np.nan_to_num(ref_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return (proj - np.mean(ref_proj)) / (np.std(ref_proj) + 1e-8)


def extract_activations(model, prompts, layers):
    """Residual stream after each block (hook_resid_post): final token and mean over tokens."""
    activations_final = {layer: [] for layer in layers}
    activations_mean = {layer: [] for layer in layers}

    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="  Extracting", leave=False):
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: "hook_resid_post" in name)
            for layer in layers:
                resid = cache[f"blocks.{layer}.hook_resid_post"]
                activations_final[layer].append(resid[0, -1, :].cpu())
                activations_mean[layer].append(resid.mean(dim=1).squeeze(0).cpu())

    for layer in layers:
        activations_final[layer] = torch.stack(activations_final[layer]).float()
        activations_mean[layer] = torch.stack(activations_mean[layer]).float()

    return activations_final, activations_mean


def compute_layer_curves_kfold(activations, metadata, extraction_type, layers):
    """Held-out AUC at every layer: 5-fold split by sentence set, vector fitted on the
    training folds and scored on the test fold. Run on S2 first and third person."""
    results = []

    for ds_name in ["S2_1P", "S2_3P"]:
        if ds_name not in activations[extraction_type]:
            continue

        cats = np.array(metadata[ds_name]["categories"])
        sets = np.array(metadata[ds_name]["sets"])
        unique_sets = sorted(set(sets))
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        for layer in layers:
            acts = activations[extraction_type][ds_name][layer]
            acts_np = acts.numpy() if hasattr(acts, "numpy") else acts

            fold_aucs_all, fold_aucs_neutral = [], []
            for train_idx, test_idx in kf.split(unique_sets):
                train_mask = np.isin(sets, [unique_sets[i] for i in train_idx])
                test_mask = np.isin(sets, [unique_sets[i] for i in test_idx])
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue

                vec_all = compute_pain_vector(acts_np[train_mask], cats[train_mask], baseline="all_controls")
                vec_neutral = compute_pain_vector(acts_np[train_mask], cats[train_mask], baseline="neutral")
                auc_all = compute_auc(acts_np[test_mask], cats[test_mask], vec_all)
                auc_neutral = compute_auc(acts_np[test_mask], cats[test_mask], vec_neutral)

                if not np.isnan(auc_all):
                    fold_aucs_all.append(auc_all)
                if not np.isnan(auc_neutral):
                    fold_aucs_neutral.append(auc_neutral)

            results.append({
                "dataset": ds_name,
                "extraction": extraction_type,
                "layer": layer,
                "auc_vs_all_controls": np.mean(fold_aucs_all) if fold_aucs_all else np.nan,
                "auc_vs_neutral": np.mean(fold_aucs_neutral) if fold_aucs_neutral else np.nan,
                "auc_std": np.std(fold_aucs_all) if fold_aucs_all else np.nan,
            })

    return pd.DataFrame(results)


def create_strip_plot(acts, cats, pain_vector, title, output_path):
    """Per-category projections onto the pain vector, categories sorted by mean."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)
    proj = acts_np @ (pain_vector / (np.linalg.norm(pain_vector) + 1e-8))

    cat_data = []
    for cat in PAIN_CATEGORIES + CONTROL_CATEGORIES:
        mask = cats_np == cat
        if mask.sum() == 0:
            continue
        cat_data.append({
            "label": f"{cat} ({CATEGORY_LABELS.get(cat, cat)})",
            "projections": proj[mask],
            "mean": proj[mask].mean(),
            "is_pain": cat in PAIN_CATEGORIES,
        })
    cat_data.sort(key=lambda x: x["mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    np.random.seed(42)
    for i, d in enumerate(cat_data):
        color = "#d62728" if d["is_pain"] else "#1f77b4"
        jitter = np.random.uniform(-0.15, 0.15, len(d["projections"]))
        ax.scatter(d["projections"], i + jitter, c=color, alpha=0.6, s=30, edgecolors="none")
        ax.hlines(i, d["projections"].min(), d["projections"].max(), colors=color, alpha=0.4, linewidth=1)
        marker = "o" if d["is_pain"] else "s"
        ax.scatter([d["mean"]], [i], c=color, s=150, marker=marker, edgecolors="white", linewidths=2, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(cat_data)))
    ax.set_yticklabels([d["label"] for d in cat_data])
    ax.set_xlabel("Projection onto Pain Vector")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=10, label="Pain"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=10, label="Control"),
    ], loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_auc_bars(acts, cats, pain_vector, title, output_path):
    """AUC of pain vs all controls and pain vs each control category."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)
    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)

    pain_proj = acts_np[np.isin(cats_np, PAIN_CATEGORIES)] @ vec_norm
    results = {}

    all_ctrl_proj = acts_np[np.isin(cats_np, CONTROL_CATEGORIES)] @ vec_norm
    results["ALL"] = roc_auc_score(
        np.concatenate([np.ones(len(pain_proj)), np.zeros(len(all_ctrl_proj))]),
        np.concatenate([pain_proj, all_ctrl_proj]))

    for ctrl in CONTROL_CATEGORIES:
        ctrl_proj = acts_np[cats_np == ctrl] @ vec_norm
        if len(ctrl_proj) > 0:
            results[ctrl] = roc_auc_score(
                np.concatenate([np.ones(len(pain_proj)), np.zeros(len(ctrl_proj))]),
                np.concatenate([pain_proj, ctrl_proj]))

    labels = ["Pain vs ALL"] + [f"Pain vs {c} ({CATEGORY_LABELS.get(c, c)})" for c in CONTROL_CATEGORIES]
    values = [results["ALL"]] + [results.get(c, 0.5) for c in CONTROL_CATEGORIES]
    colors = ["#2ca02c"] + ["#1f77b4"] * len(CONTROL_CATEGORIES)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return results


def dataset_type(ds_name):
    if ds_name.startswith("S1") or ds_name.startswith("S2"):
        return "human"
    if ds_name.startswith("Random"):
        return "neutral"
    if ds_name.startswith("Arousal"):
        return "arousal"
    if ds_name.startswith("Numb"):
        return "numb"
    if ds_name.startswith("ControlSupplement"):
        return "control_supplement"
    if ds_name.startswith("SD_sadness"):
        return "sadness"
    return "unknown"

# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(model_path, model_name, dataset, output_dir):
    from transformer_lens import HookedTransformer

    log(f"\n{'=' * 70}")
    log(f"PROCESSING: {model_name}")
    log(f"{'=' * 70}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "started.txt").write_text(datetime.now().isoformat())

    activations_file = output_dir / "activations.pt"
    if activations_file.exists():
        log("  Found cached activations, loading...")
        saved = torch.load(activations_file)
        all_activations = saved["activations"]
        all_metadata = saved["metadata"]
        n_layers = saved["n_layers"]
        d_model = saved["d_model"]
        layers = list(range(n_layers))
    else:
        # Load the HF weights in bf16 with low_cpu_mem_usage, then hand them to
        # TransformerLens so the model is built directly on the GPU.
        log(f"  Loading {model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = HookedTransformer.from_pretrained_no_processing(
            model_path, hf_model=hf_model, tokenizer=hf_tokenizer, device="cuda", dtype=torch.bfloat16)
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        layers = list(range(n_layers))
        log(f"  Layers: {n_layers}, d_model: {d_model}")

        all_activations = {"final_token": {}, "mean": {}}
        all_metadata = {}
        for ds_name, ds_data in dataset["datasets"].items():
            log(f"  Extracting: {ds_name}")
            prompts = [s["prompt"] for s in ds_data["sentences"]]
            categories = [s["category"] for s in ds_data["sentences"]]
            sets = [s["set"] for s in ds_data["sentences"]]
            acts_final, acts_mean = extract_activations(model, prompts, layers)
            all_activations["final_token"][ds_name] = acts_final
            all_activations["mean"][ds_name] = acts_mean
            all_metadata[ds_name] = {"categories": categories, "sets": sets}

        torch.save({
            "activations": all_activations,
            "metadata": all_metadata,
            "layers": layers,
            "model_name": model_path,
            "n_layers": n_layers,
            "d_model": d_model,
        }, output_dir / "activations.pt")

        del model
        clear_gpu_and_cache()

    # Layer choice: held-out AUC per layer, averaged over S2 first and third person.
    log("  Computing layer curves with 5-fold CV...")
    layer_curves_final = compute_layer_curves_kfold(all_activations, all_metadata, "final_token", layers)
    layer_curves_mean = compute_layer_curves_kfold(all_activations, all_metadata, "mean", layers)
    layer_curves = pd.concat([layer_curves_final, layer_curves_mean])
    layer_curves.to_csv(output_dir / "layer_curves.csv", index=False)

    best_layer_final = layer_curves_final.groupby("layer")["auc_vs_all_controls"].mean().idxmax()
    best_layer_mean = layer_curves_mean.groupby("layer")["auc_vs_all_controls"].mean().idxmax()
    best_layers = {"final_token": best_layer_final, "mean": best_layer_mean}
    log(f"  Best layer (final_token): {best_layer_final}")
    log(f"  Best layer (mean): {best_layer_mean}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ext_type, best_layer) in zip(axes, best_layers.items()):
        df = layer_curves[layer_curves["extraction"] == ext_type]
        for ds_name in ["S2_1P", "S2_3P"]:
            ds_df = df[df["dataset"] == ds_name]
            ax.plot(ds_df["layer"], ds_df["auc_vs_all_controls"], label=f"{ds_name} vs AllCtrl", linewidth=2)
            ax.plot(ds_df["layer"], ds_df["auc_vs_neutral"], label=f"{ds_name} vs Neutral", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle=":", label="Chance")
        ax.axvline(best_layer, color="red", linestyle="--", alpha=0.5, label=f"Best: L{best_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(f"Pain Signal by Layer ({ext_type})")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Pain vectors and z-scores at the chosen layer, for both extraction types.
    for ext_type, best_layer in best_layers.items():
        ext_dir = output_dir / ext_type
        ext_dir.mkdir(exist_ok=True)
        log(f"  Analyzing {ext_type} (Layer {best_layer})...")

        acts_at_layer = {ds: all_activations[ext_type][ds][best_layer] for ds in all_activations[ext_type]}

        s2_vector = compute_pain_vector(acts_at_layer["S2_1P"], all_metadata["S2_1P"]["categories"])
        s1_vector = compute_pain_vector(acts_at_layer["S1_1P"], all_metadata["S1_1P"]["categories"])
        torch.save({
            "s2_pain_vector": torch.tensor(s2_vector),
            "s1_pain_vector": torch.tensor(s1_vector),
            "layer": best_layer,
            "extraction": ext_type,
        }, ext_dir / "pain_vectors.pt")

        # Every set projected onto the S2 vector, z-scored against S2 first person.
        z_results = []
        for target_ds, target_acts in acts_at_layer.items():
            target_cats = np.array(all_metadata[target_ds]["categories"])
            z_proj = project_and_zscore(target_acts, s2_vector, acts_at_layer["S2_1P"])
            dtype = dataset_type(target_ds)
            if dtype == "human":
                pain_z = z_proj[np.isin(target_cats, PAIN_CATEGORIES)].mean()
                ctrl_z = z_proj[np.isin(target_cats, CONTROL_CATEGORIES)].mean()
            else:
                pain_z = np.nan
                ctrl_z = z_proj.mean()
            z_results.append({"dataset": target_ds, "type": dtype, "mean_z": z_proj.mean(),
                              "pain_z": pain_z, "ctrl_z": ctrl_z})
        pd.DataFrame(z_results).to_csv(ext_dir / "z_scores.csv", index=False)

        # Strip plots and per-control AUCs on S2 first and third person.
        all_aucs = {}
        for ds in ["S2_1P", "S2_3P"]:
            if ds in acts_at_layer:
                create_strip_plot(acts_at_layer[ds], all_metadata[ds]["categories"], s2_vector,
                                  f"{ds} - Layer {best_layer} ({ext_type})", ext_dir / f"strip_{ds}.png")
                all_aucs[ds] = create_auc_bars(acts_at_layer[ds], all_metadata[ds]["categories"], s2_vector,
                                               f"AUC: {ds} - Layer {best_layer} ({ext_type})", ext_dir / f"auc_{ds}.png")
        pd.DataFrame(all_aucs).T.to_csv(ext_dir / "auc_summary.csv")

        # Bar chart of the mean z-score per condition.
        z_df = pd.DataFrame(z_results)
        conditions = ["Human Pain", "Human Ctrl", "Neutral", "Arousal", "Numb"]
        values = [
            z_df[z_df["type"] == "human"]["pain_z"].mean(),
            z_df[z_df["type"] == "human"]["ctrl_z"].mean(),
            z_df[z_df["type"] == "neutral"]["mean_z"].mean(),
            z_df[z_df["type"] == "arousal"]["mean_z"].mean(),
            z_df[z_df["type"] == "numb"]["mean_z"].mean(),
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(conditions, values, color=["#2A9D8F", "#2A9D8F", "#6C757D", "#F4A261", "#9B5DE5"], width=0.6)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Z-Score")
        ax.set_title(f"All Conditions - {ext_type} (Layer {best_layer})")
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(ext_dir / "bar_all_conditions.png", dpi=150)
        plt.close()

    z_df_mean = pd.read_csv(output_dir / "mean" / "z_scores.csv")
    summary = {
        "model": model_path,
        "model_name": model_name,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "best_layer_final_token": int(best_layers["final_token"]),
        "best_layer_mean": int(best_layers["mean"]),
        "human_pain_z": float(z_df_mean[z_df_mean["type"] == "human"]["pain_z"].mean()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = (
        f"PAIN VECTOR EXTRACTION - RESULTS SUMMARY\n\n"
        f"Model: {model_path}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Layers: {n_layers}\nd_model: {d_model}\n"
        f"Best Layer (final_token): {best_layers['final_token']}\nBest Layer (mean): {best_layers['mean']}\n"
    )
    for ext_type, best_layer in best_layers.items():
        z_df = pd.read_csv(output_dir / ext_type / "z_scores.csv")
        human = z_df[z_df["type"] == "human"]
        report += (
            f"\n--- {ext_type.upper()} (Layer {best_layer}) ---\n"
            f"Z-scores on the S2 vector, reference S2 first person:\n"
            f"  Human Pain (A1-A5):  {human['pain_z'].mean():+.3f}\n"
            f"  Human Ctrl (B-E):    {human['ctrl_z'].mean():+.3f}\n"
            f"  Neutral (Random):    {z_df[z_df['type'] == 'neutral']['mean_z'].mean():+.3f}\n"
            f"  Arousal:             {z_df[z_df['type'] == 'arousal']['mean_z'].mean():+.3f}\n"
            f"  Numb:                {z_df[z_df['type'] == 'numb']['mean_z'].mean():+.3f}\n"
        )
    (output_dir / "summary_report.txt").write_text(report)

    (output_dir / "completed.txt").write_text(datetime.now().isoformat())
    log(f"  Human Pain Z: {summary['human_pain_z']:+.3f}")
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("PAIN VECTOR EXTRACTION - BATCH RUN")
    log("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if not check_disk_space(min_gb=50):
        log("WARNING: low disk space, proceeding anyway")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # The sets of every dataset file are merged into one dict; each set is extracted separately.
    dataset = {"datasets": {}}
    for path in DATASET_PATHS:
        log(f"Loading dataset: {path}")
        with open(path, "r", encoding="utf-8") as f:
            dataset["datasets"].update(json.load(f)["datasets"])
    log(f"Loaded {sum(len(d['sentences']) for d in dataset['datasets'].values())} sentences")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    failed = []
    start_time = datetime.now()

    for i, (model_path, model_name) in enumerate(MODELS):
        log(f"\n[{i + 1}/{len(MODELS)}] {model_name}")
        output_dir = OUTPUT_DIR / model_name

        if (output_dir / "summary.json").exists():
            log("  Already done, loading previous results...")
            with open(output_dir / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue

        check_gpu_memory()
        check_disk_space(min_gb=30)

        try:
            summary = process_model(model_path, model_name, dataset, output_dir)
            all_summaries.append(summary)
            log(f"  SUCCESS: {model_name}")
        except Exception as e:
            log(f"  FAILED: {model_name}: {e}")
            failed.append((model_name, str(e)))
            clear_gpu_and_cache()

        save_progress(all_summaries, failed, OUTPUT_DIR)

    elapsed = datetime.now() - start_time
    log("\n" + "=" * 70)
    log(f"BATCH RUN COMPLETE in {elapsed}: {len(all_summaries)}/{len(MODELS)} done, {len(failed)} failed")
    for name, err in failed:
        log(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
