import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# Ensure scgpt is in the path
sys.path.append(os.getcwd())

try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch
except ImportError:
    print("Error: Could not import scgpt.")
    sys.exit(1)

# --- Configuration ---
MODEL_DIR = "heart_model_experiment/scGPT_heart"
QUANTIZED_MODEL_PATH = "quantization_experiment/scgpt_quantized.pt"

print(f"--- scGPT Quantization Experiment ---")

# Check if model exists
if not os.path.exists(MODEL_DIR):
    print(f"[!] Model directory '{MODEL_DIR}' not found.")
    sys.exit(1)

# Use CPU for dynamic quantization (PyTorch dynamic quantization is optimized for CPU)
device = torch.device("cpu")
print(f"Running quantization on device: {device}")

# --- Step 1: Load Original Model ---
print("\n[1/5] Loading Original Model...")
config_path = os.path.join(MODEL_DIR, "args.json")
model_path = os.path.join(MODEL_DIR, "best_model.pt")
vocab_path = os.path.join(MODEL_DIR, "vocab.json")

# Load args
with open(config_path, "r") as f:
    model_configs = json.load(f)

# Load vocab
with open(vocab_path, "r") as f:
    gene_name_to_id = json.load(f)

special_tokens = ["<pad>", "<cls>", "<eoc>"]
vocab_list = [None] * len(gene_name_to_id)
for k, v in gene_name_to_id.items():
    if v < len(vocab_list):
        vocab_list[v] = k
for t in special_tokens:
    if t not in gene_name_to_id:
        vocab_list.append(t)

vocab = Vocab(VocabPybind(vocab_list, None))
vocab.set_default_index(vocab["<pad>"])

model_init_args = {
    "ntoken": len(vocab),
    "d_model": model_configs.get("embsize"),
    "nhead": model_configs.get("nheads"),
    "d_hid": model_configs.get("d_hid"),
    "nlayers": model_configs.get("nlayers"),
    "nlayers_cls": model_configs.get("n_layers_cls", 3),
    "n_cls": 1,
    "vocab": vocab,
    "dropout": model_configs.get("dropout", 0.2),
    "pad_token": model_configs.get("pad_token", "<pad>"),
    "pad_value": model_configs.get("pad_value", 0),
    "do_mvc": True,
    "do_dab": model_configs.get("do_dab", False),
    "use_batch_labels": model_configs.get("use_batch_labels", False),
    "num_batch_labels": model_configs.get("num_batch_labels", None),
    "domain_spec_batchnorm": model_configs.get("domain_spec_batchnorm", False),
    "input_emb_style": model_configs.get("input_emb_style", "continuous"),
    "n_input_bins": model_configs.get("n_bins", 51),
    "cell_emb_style": model_configs.get("cell_emb_style", "cls"),
    "mvc_decoder_style": model_configs.get("mvc_decoder_style", "inner product"),
    "ecs_threshold": model_configs.get("ecs_threshold", 0.3),
    "explicit_zero_prob": model_configs.get("explicit_zero_prob", False),
    "use_fast_transformer": False,
    "pre_norm": model_configs.get("pre_norm", False),
}

model = TransformerModel(**model_init_args)
state_dict = torch.load(model_path, map_location=device)

# Fix for Flash Attention names
new_state_dict = state_dict.copy()
for key in list(state_dict.keys()):
    if "Wqkv" in key:
        new_key = key.replace("Wqkv", "in_proj")
        new_state_dict[new_key] = new_state_dict.pop(key)

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Check original size
original_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"Original Model Size (File): {original_size:.2f} MB")

# --- Step 2: Apply Dynamic Quantization ---
print("\n[2/5] Quantizing Model (INT8 Dynamic)...")

# Quantize linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear},  # Specify which layers to quantize
    dtype=torch.qint8
)

print("Model quantized successfully!")

# --- Step 3: Save and Check Size ---
print("\n[3/5] Saving Quantized Model...")
torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
print(f"Quantized Model Size (File): {quantized_size:.2f} MB")
print(f"Compression Ratio: {original_size / quantized_size:.2f}x")

# --- Step 4: Prepare Input Data ---
print("\n[4/5] Preparing Input Data...")
n_cells = 10
n_genes_in_data = 1000
counts = np.random.randint(0, 50, size=(n_cells, n_genes_in_data))
gene_names = vocab_list[:n_genes_in_data]

adata = sc.AnnData(X=counts)
adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
adata.var_names = gene_names
adata.var["gene_name"] = gene_names

n_bins = 51
binned_data = pd.cut(adata.X.flatten(), bins=n_bins, labels=False).reshape(adata.shape)
adata.layers["X_binned"] = binned_data

gene_ids_in_vocab = np.array([vocab[g] for g in gene_names], dtype=int)
tokenized_data = tokenize_and_pad_batch(
    adata.layers["X_binned"],
    gene_ids_in_vocab,
    max_len=n_genes_in_data + 1,
    vocab=vocab,
    pad_token="<pad>",
    pad_value=0,
    append_cls=True,
    include_zero_gene=True,
)

input_gene_ids = tokenized_data["genes"].to(device)
input_values = tokenized_data["values"].to(device)
src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])

# --- Step 5: Run Inference & Benchmark ---
print("\n[5/5] Benchmarking Inference (CPU)...")

def benchmark_inference(m, name):
    print(f"Running {name} inference...")
    start_time = time.time()
    with torch.no_grad():
        output = m(
            input_gene_ids,
            input_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False
        )
    end_time = time.time()
    print(f"   Shape: {output['cell_emb'].shape}")
    print(f"   Time: {end_time - start_time:.4f} seconds")
    return output['cell_emb']

# Run Original (FP32)
out_fp32 = benchmark_inference(model, "Original FP32")

# Run Quantized (INT8)
out_int8 = benchmark_inference(quantized_model, "Quantized INT8")

# Compare Embeddings
# We expect some difference, but it should be small for cosine similarity
# Since we don't have ground truth labels, we just check how close they are numerically
print("\n--- Embedding Comparison (First Cell) ---")
emb1 = out_fp32[0, :5].numpy()
emb2 = out_int8[0, :5].numpy()
print(f"FP32: {emb1}")
print(f"INT8: {emb2}")

# Calculate Cosine Similarity for the first cell
cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Cosine Similarity (approx, first 5 dims): {cos_sim:.4f}")

print("\nExperiment Complete.")
