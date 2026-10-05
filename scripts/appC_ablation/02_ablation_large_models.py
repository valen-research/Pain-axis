"""Ablation of the pain directions (Appendix C), large models (24B to 72B) on one GPU.
Latest checkpoint.

Arditi et al. 2024 single-direction weight orthogonalization: one unit direction per
vector, taken from its steering layer, is removed from every residual-writing matrix
(embedding, attention o_proj, MLP down_proj; Gemma post-norm folded in). The original
weights are snapshotted to CPU and restored between conditions. Combined conditions
remove the joint span, so a later cut cannot reintroduce an earlier direction.
A strict check reports the remaining projection onto each target.

Conditions: baseline, s1, s2, s1s2, negval, fear, s1s2_negval, s1s2_fear, random.
Prompts: the 100 scenarios of the five categories in KEEP_CATEGORIES.

Reads datasets/4.1_self_other_420_scenarios.json and
results/vectors_layerwise/vectors_layerwise_<model>.pt (one vector per layer per direction).
Writes results/appC_ablation_joint/ablation_<model>.csv, by_prompt_<model>.json,
proj_<model>.npz, verify_<model>.csv. Asks which model to run.
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""
import os
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, json, gc, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

MAX_NEW_TOKENS = 60
BATCH_SIZE = 8
CLEAR_CACHE_ABOVE_GB = 0   # clear the HF weight cache after every model
OUT_DIR = Path("results") / "appC_ablation_joint"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VEC_DIR = Path("results") / "vectors_layerwise"
PROMPT_FILE = Path("datasets") / "4.1_self_other_420_scenarios.json"
KEEP_CATEGORIES = {"gaslighting", "repeated_rejection", "personhood_dismissal",
                   "anger_insults", "moral_failure"}

PROBE_KEYS = ["s1_pain_vector", "s2_pain_vector", "s3_pain_vector",
              "fear_vector", "negemotion_vector", "negworld_vector",
              "bodysens_vector", "arousal_vector", "random_vector",
              "numb_vector"]

# (repo, name, format, s1_layer, s2_layer); negval and fear use the s2 layer
LARGE_MODELS = [
    ("google/gemma-2-27b",                 "Gemma_2_27B_base",       "raw",   6, 13),
    ("google/gemma-2-27b-it",              "Gemma_2_27B_instruct",   "chat", 13, 13),
    ("google/gemma-3-27b-pt",              "Gemma_3_27B_base",       "raw",  22, 24),
    ("google/gemma-3-27b-it",              "Gemma_3_27B_instruct",   "chat", 24, 55),
    ("mistralai/Mistral-Small-24B-Base-2501", "Mistral_Small_24B_base","raw", 12, 16),
    ("Qwen/Qwen2.5-32B",                   "Qwen_2.5_32B_base",      "raw",  25, 32),
    ("Qwen/Qwen2.5-32B-Instruct",          "Qwen_2.5_32B_instruct",  "chat", 32, 38),
    ("Qwen/Qwen2.5-72B",                   "Qwen_2.5_72B_base",      "raw",  60, 60),
    ("Qwen/Qwen2.5-72B-Instruct",          "Qwen_2.5_72B_instruct",  "chat", 48, 60),
    ("meta-llama/Llama-3.1-70B",           "Llama_3.1_70B_base",     "raw",  32, 40),
    ("meta-llama/Llama-3.1-70B-Instruct",  "Llama_3.1_70B_instruct", "chat", 24, 24),
    ("meta-llama/Llama-3.3-70B-Instruct",  "Llama_3.3_70B_instruct", "chat", 24, 24),
]

CONDITION_SPECS = {
    "baseline":     [],
    "s1":           [("s1_pain_vector", "s1")],
    "s2":           [("s2_pain_vector", "s2")],
    "s1s2":         [("s1_pain_vector", "s1"), ("s2_pain_vector", "s2")],
    "negval":       [("negemotion_vector", "s2")],
    "fear":         [("fear_vector", "s2")],
    "s1s2_negval":  [("s1_pain_vector", "s1"), ("s2_pain_vector", "s2"),
                     ("negemotion_vector", "s2")],
    "s1s2_fear":    [("s1_pain_vector", "s1"), ("s2_pain_vector", "s2"),
                     ("fear_vector", "s2")],
    "random":       [("__random__", "s2")],
}
CONDITION_ORDER = list(CONDITION_SPECS.keys())

with open(PROMPT_FILE, encoding="utf-8") as f:
    ALL_PROMPTS = json.load(f)
PROMPTS = [p for p in ALL_PROMPTS if p["category"] in KEEP_CATEGORIES]
PROMPT_ORDER = {p["id"]: i for i, p in enumerate(PROMPTS)}
print(f"prompt set: {len(PROMPTS)} prompts (5 top-suffering categories)")
print(f"conditions: {CONDITION_ORDER}")


def parse_turns(text):
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role: turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[7:].strip()]
        elif line.startswith("[Assistant]:"):
            if role: turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[12:].strip()]
        else:
            lines.append(line)
    if role: turns.append((role, "\n".join(lines).strip()))
    return turns


def render_prompt(cand, tok, fmt):
    if fmt == "raw":
        return cand["text"]
    msgs = [{"role": r, "content": c} for r, c in parse_turns(cand["text"])
            if not (r == "assistant" and c == "")]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False)


def get_decoder_layers(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def residual_write_matrices(model):
    out = [("embed", model.get_input_embeddings().weight, None)]
    for i, layer in enumerate(get_decoder_layers(model)):
        attn_scale = mlp_scale = None
        if hasattr(layer, "post_feedforward_layernorm"):
            attn_scale = 1.0 + layer.post_attention_layernorm.weight.float()
            mlp_scale = 1.0 + layer.post_feedforward_layernorm.weight.float()
        out.append((f"o_proj_{i}",   layer.self_attn.o_proj.weight, attn_scale))
        out.append((f"down_proj_{i}", layer.mlp.down_proj.weight,   mlp_scale))
    return out


def unit(v):
    nrm = v.norm()
    return v / nrm if nrm > 0 else v


def orthogonal_basis(rows):
    if rows.shape[0] == 1:
        return rows  # Preserve the original single-unit-direction arithmetic.
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    tol = max(rows.shape) * torch.finfo(rows.dtype).eps * singular[0]
    return vh[singular > tol]


def orthogonalize_directions(model, directions):
    # Gemma: the post-norm gain s = 1 + weight sits between the write matrix and the
    # residual add. Build the joint basis AFTER scaling the target directions.
    directions = torch.stack(directions)
    basis = orthogonal_basis(directions)
    with torch.no_grad():
        for name, W, scale in residual_write_matrices(model):
            W32 = W.data.float()
            dev = W32.device
            if name == "embed":
                r = basis.to(dev)
                W32 -= (W32 @ r.T) @ r
            elif scale is None:
                r = basis.to(dev)
                W32 -= r.T @ (r @ W32)
            else:
                v = scale.to(dev) * directions.to(dev)
                v = v / v.norm() if len(directions) == 1 else orthogonal_basis(v)
                W32 -= v.T @ (v @ W32)
            W.data.copy_(W32.to(W.dtype))
            del W32


def snapshot_weights(model):
    return {name: W.data.detach().to("cpu", copy=True)
            for name, W, _ in residual_write_matrices(model)}


def restore_weights(model, snap):
    with torch.no_grad():
        for name, W, _ in residual_write_matrices(model):
            W.data.copy_(snap[name].to(W.device))


class ProjRecorder:
    def __init__(self, monitor_per_layer, fixed_dirs, n_layers):
        self.monitor = monitor_per_layer
        self.fixed = fixed_dirs
        self.n_layers = n_layers
        self.records, self.fixed_records, self.layer_of = [], [], []

    def reset(self):
        self.records, self.fixed_records, self.layer_of = [], [], []

    def make_hook(self, idx):
        mon = self.monitor[idx]
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            last = hs[:, -1, :]
            self.records.append((last @ mon.T).float().cpu())
            self.fixed_records.append((last @ self.fixed.T).float().cpu())
            self.layer_of.append(idx)
            return output
        return hook

    def _assemble(self, recs, n_cols, batch_size):
        n_steps = len(recs) // self.n_layers
        arr = np.full((batch_size, n_steps, self.n_layers, n_cols),
                      np.nan, dtype=np.float16)
        step = -1
        for rec, L in zip(recs, self.layer_of):
            if L == 0:
                step += 1
            if step < n_steps:
                arr[:, step, L, :] = rec.numpy().astype(np.float16)
        return arr

    def collect(self, batch_size, n_probes, n_fixed):
        if not self.records:
            return None, None
        return (self._assemble(self.records, n_probes, batch_size),
                self._assemble(self.fixed_records, n_fixed, batch_size))


def save_outputs(rows, out_csv, out_json):
    ordered = sorted(rows, key=lambda r: (PROMPT_ORDER.get(r["id"], 9999),
                                          CONDITION_ORDER.index(r["condition"])))
    pd.DataFrame(ordered).to_csv(out_csv, index=False)
    by_prompt = {}
    for p in PROMPTS:
        entry = {"category": p["category"], "prompt": p["text"]}
        for r in ordered:
            if r["id"] == p["id"]:
                entry[r["condition"]] = r["generation"]
        by_prompt[p["id"]] = entry
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(by_prompt, f, ensure_ascii=False, indent=2)


print("\nLarge models:")
for i, (repo, name, fmt, l1, l2) in enumerate(LARGE_MODELS, start=1):
    done = "  [done]" if (OUT_DIR / f"ablation_{name}.csv").exists() else ""
    print(f"  {i:2d}: {name}  (s1@L{l1}, s2@L{l2}){done}")
choice = input("\nWhich model? number or 'all': ").strip().lower()
SELECTED = LARGE_MODELS if choice == "all" else [LARGE_MODELS[int(choice) - 1]]

for REPO, MODEL_NAME, FMT, S1_LAYER, S2_LAYER in SELECTED:
    print(f"\n================ {MODEL_NAME} ================")
    OUT_CSV = OUT_DIR / f"ablation_{MODEL_NAME}.csv"
    OUT_JSON = OUT_DIR / f"by_prompt_{MODEL_NAME}.json"
    OUT_NPZ = OUT_DIR / f"proj_{MODEL_NAME}.npz"
    OUT_VERIFY = OUT_DIR / f"verify_{MODEL_NAME}.csv"
    vec_file = VEC_DIR / f"vectors_layerwise_{MODEL_NAME}.pt"
    if not vec_file.exists():
        print(f"NO VECTOR FILE ({vec_file.name}), skipping")
        continue

    if OUT_CSV.exists() and OUT_NPZ.exists():
        _rows = pd.read_csv(OUT_CSV)
        _have = set(_rows["condition"].unique())
        _npz = set(np.load(OUT_NPZ).files)
        if all(c in _have and c in _npz for c in CONDITION_ORDER):
            print(f"  all {len(CONDITION_ORDER)} conditions already done, skipping model")
            continue

    layer_of = {"s1": S1_LAYER, "s2": S2_LAYER}

    rows = pd.read_csv(OUT_CSV).to_dict("records") if OUT_CSV.exists() else []
    done_conditions = {r["condition"] for r in rows}
    proj_store = dict(np.load(OUT_NPZ)) if OUT_NPZ.exists() else {}
    verify_rows = (pd.read_csv(OUT_VERIFY).to_dict("records")
                   if OUT_VERIFY.exists() else [])

    tok = AutoTokenizer.from_pretrained(REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()

    stacks = torch.load(vec_file, map_location="cpu", weights_only=False)
    n_layers_file = stacks["layers"]
    dim = stacks["s2_pain_vector"].shape[1]
    layers = get_decoder_layers(model)
    assert len(layers) == n_layers_file, (
        f"layer count mismatch: model {len(layers)} vs file {n_layers_file}")

    g = torch.Generator().manual_seed(0)
    rand_dir = unit(torch.randn(dim, generator=g))

    def single_direction(vec_key, which_layer):
        if vec_key == "__random__":
            return rand_dir.to("cuda")
        L = layer_of[which_layer]
        return unit(stacks[vec_key].float()[L]).to("cuda")

    FIXED_SPECS = [("s1_pain_vector", "s1"), ("s2_pain_vector", "s2"),
                   ("negemotion_vector", "s2"), ("fear_vector", "s2")]
    fixed_names = [f"{k.split('_')[0]}@L{layer_of[wl]}" for k, wl in FIXED_SPECS]
    fixed_dirs = torch.stack(
        [unit(stacks[k].float()[layer_of[wl]]) for k, wl in FIXED_SPECS]
    ).to("cuda", dtype=torch.bfloat16)
    print(f"strict fixed dirs: {fixed_names}")

    monitor_per_layer = []
    for L in range(n_layers_file):
        rows_m = []
        for k in PROBE_KEYS:
            if k in stacks:
                v = stacks[k].float()[L]
                rows_m.append(unit(v) if v.norm() > 0 else torch.zeros(dim))
            else:
                rows_m.append(torch.zeros(dim))
        monitor_per_layer.append(
            torch.stack(rows_m).to("cuda", dtype=torch.bfloat16))

    rec = ProjRecorder(monitor_per_layer, fixed_dirs, len(layers))
    handles = [layer.register_forward_hook(rec.make_hook(i))
               for i, layer in enumerate(layers)]
    print(f"hooked {len(handles)} decoder layers")

    print("snapshotting original weights to CPU...")
    snap = snapshot_weights(model)

    texts = [render_prompt(c, tok, FMT) for c in PROMPTS]

    for cond in CONDITION_ORDER:
        if cond in done_conditions and cond in proj_store:
            print(f"  {cond}: already done, skipping")
            continue
        specs = CONDITION_SPECS[cond]
        restore_weights(model, snap)
        if specs:
            applied = []
            orthogonalize_directions(model, [single_direction(k, wl) for k, wl in specs])
            for vec_key, which_layer in specs:
                applied.append(f"{vec_key.split('_')[0]}@L{layer_of.get(which_layer)}"
                               if vec_key != "__random__"
                               else f"random@L{layer_of['s2']}")
            print(f"  {cond}: orthogonalized {', '.join(applied)}", flush=True)
        else:
            print(f"  {cond} (baseline)...", flush=True)

        batch_arrays, batch_fixed = [], []
        for start in range(0, len(PROMPTS), BATCH_SIZE):
            batch_prompts = PROMPTS[start:start + BATCH_SIZE]
            batch_texts = texts[start:start + BATCH_SIZE]
            enc = tok(batch_texts, return_tensors="pt", padding=True,
                      padding_side="left").to("cuda")
            rec.reset()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            arr, farr = rec.collect(len(batch_prompts), len(PROBE_KEYS),
                                    len(fixed_names))
            batch_arrays.append(arr)
            batch_fixed.append(farr)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            gens = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for cand, gen in zip(batch_prompts, gens):
                rows.append({"model": MODEL_NAME, "id": cand["id"],
                             "category": cand["category"],
                             "condition": cond, "generation": gen})

        def pad_stack(arrs):
            ms = max(a.shape[1] for a in arrs)
            out_a = []
            for a in arrs:
                if a.shape[1] < ms:
                    pad = np.full((a.shape[0], ms - a.shape[1],
                                   a.shape[2], a.shape[3]), np.nan, dtype=np.float16)
                    a = np.concatenate([a, pad], axis=1)
                out_a.append(a)
            return np.concatenate(out_a, axis=0)

        proj_store[cond] = pad_stack(batch_arrays)
        proj_store[f"{cond}__fixed"] = pad_stack(batch_fixed)

        cond_arr = proj_store[cond].astype(np.float32)
        for pi, pk in enumerate(PROBE_KEYS):
            verify_rows.append({
                "model": MODEL_NAME, "condition": cond, "probe": pk,
                "mean_abs_proj": float(np.nanmean(np.abs(cond_arr[..., pi]))),
                "kind": "layerwise",
            })
        fixed_arr = proj_store[f"{cond}__fixed"].astype(np.float32)
        for fi, fname in enumerate(fixed_names):
            verify_rows.append({
                "model": MODEL_NAME, "condition": cond, "probe": fname,
                "mean_abs_proj": float(np.nanmean(np.abs(fixed_arr[..., fi]))),
                "kind": "strict_fixed",
            })

        save_outputs(rows, OUT_CSV, OUT_JSON)
        np.savez_compressed(OUT_NPZ, **proj_store,
                            vector_names=np.array(PROBE_KEYS),
                            fixed_names=np.array(fixed_names),
                            prompt_ids=np.array([p["id"] for p in PROMPTS]))
        pd.DataFrame(verify_rows).to_csv(OUT_VERIFY, index=False)

    restore_weights(model, snap)
    for h in handles:
        h.remove()

    vdf = pd.DataFrame(verify_rows)
    vdf = vdf[vdf.model == MODEL_NAME]
    print("\n  --- STRICT Arditi check (ablated vector's own row -> ~0) ---")
    strict = vdf[vdf.kind == "strict_fixed"]
    for fname in fixed_names:
        base = strict[(strict.condition == "baseline") & (strict.probe == fname)]
        base_v = base["mean_abs_proj"].iloc[0] if len(base) else float("nan")
        line = f"    {fname:14s} base={base_v:6.2f} | "
        for cond in CONDITION_ORDER:
            if cond == "baseline":
                continue
            cell = strict[(strict.condition == cond) & (strict.probe == fname)]
            if len(cell):
                line += f"{cond}={cell['mean_abs_proj'].iloc[0]:5.2f} "
        print(line)

    print(f"\ndone: {OUT_CSV}")

    del model, snap
    gc.collect(); torch.cuda.empty_cache()
    cache_dir = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + REPO.replace("/", "--"))
    if cache_dir.exists():
        gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e9
        if gb > CLEAR_CACHE_ABOVE_GB:
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"model cache cleared ({gb:.0f} GB)")

print(f"\nAll requested models done: {OUT_DIR}")
