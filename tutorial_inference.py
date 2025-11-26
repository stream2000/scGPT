import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# Ensure scgpt is in the path (assuming you run this from the repo root)
sys.path.append(os.getcwd())

try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch
except ImportError:
    print("Error: Could not import scgpt. Make sure you are in the scGPT root directory or have installed the package.")
    sys.exit(1)

# --- Configuration ---
# Point this to your model directory
# If you downloaded the heart model, unzip it and set this path: e.g., "model_heart"
# If you ran setup_local_model.py, use: "my_heart_model"
MODEL_DIR = "heart_model_experiment/scGPT_heart" 

print(f"--- scGPT Inference Tutorial ---")
print(f"Target Model Directory: {MODEL_DIR}")

# Check if model exists
if not os.path.exists(MODEL_DIR):
    print(f"[!] Model directory '{MODEL_DIR}' not found.")
    print("Option 1: Download the pretrained model from the README link and unzip it here.")
    print("Option 2: Run 'python setup_local_model.py' to generate a dummy model for testing.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# --- Step 1: Load Model & Configuration ---
print("\n[1/4] Loading Model and Configuration...")
config_path = os.path.join(MODEL_DIR, "args.json")
model_path = os.path.join(MODEL_DIR, "best_model.pt")
vocab_path = os.path.join(MODEL_DIR, "vocab.json")

# Load args
with open(config_path, "r") as f:
    model_configs = json.load(f)

# Load vocab (Gene Name -> ID mapping)
with open(vocab_path, "r") as f:
    gene_name_to_id = json.load(f)

# Reconstruct the Vocab object required by scGPT
# This maps integer IDs back to strings and handles special tokens
special_tokens = ["<pad>", "<cls>", "<eoc>"]
vocab_list = [None] * len(gene_name_to_id)
for k, v in gene_name_to_id.items():
    if v < len(vocab_list):
        vocab_list[v] = k
for t in special_tokens: # Ensure special tokens are in the list
    if t not in gene_name_to_id:
        vocab_list.append(t)

vocab = Vocab(VocabPybind(vocab_list, None))
vocab.set_default_index(vocab["<pad>"])

# Prepare model arguments by mapping from args.json
# We only pass arguments that TransformerModel accepts
model_init_args = {
    "ntoken": len(vocab),
    "d_model": model_configs.get("embsize"),
    "nhead": model_configs.get("nheads"),
    "d_hid": model_configs.get("d_hid"),
    "nlayers": model_configs.get("nlayers"),
    "nlayers_cls": model_configs.get("n_layers_cls", 3),
    "n_cls": 1, # Default
    "vocab": vocab,
    "dropout": model_configs.get("dropout", 0.2),
    "pad_token": model_configs.get("pad_token", "<pad>"),
    "pad_value": model_configs.get("pad_value", 0),
    "do_mvc": True, # Model weights have mvc_decoder, so we enable this
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
    "use_fast_transformer": False, # Force False since we are on CPU/No flash-attn
    "pre_norm": model_configs.get("pre_norm", False),
}

# Initialize the model structure
model = TransformerModel(**model_init_args)

# Load the pretrained weights
state_dict = torch.load(model_path, map_location=device)

# --- Fix for Flash Attention vs PyTorch Transformer ---
# The model was trained with flash-attn (Wqkv), but we are running with standard PyTorch (in_proj)
new_state_dict = state_dict.copy()
for key in list(state_dict.keys()):
    if "Wqkv" in key:
        # Rename Wqkv to in_proj
        new_key = key.replace("Wqkv", "in_proj")
        new_state_dict[new_key] = new_state_dict.pop(key)

# Load state dict with strict=False to allow missing cls_decoder or extra flag_encoder
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully!")

# --- Step 2: Prepare Input Data ---
print("\n[2/4] Preparing Input Data...")
# We will create a synthetic dataset. In a real scenario, load your .h5ad file using sc.read_h5ad()
n_cells = 10
n_genes_in_data = 1000
print(f"Generating synthetic data for {n_cells} cells and {n_genes_in_data} genes...")

# Random count matrix (simulating raw counts)
counts = np.random.randint(0, 50, size=(n_cells, n_genes_in_data))
# Use actual gene names from the vocab so they match
gene_names = vocab_list[:n_genes_in_data]

adata = sc.AnnData(X=counts)
adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
adata.var_names = gene_names
adata.var["gene_name"] = gene_names

# --- Step 3: Preprocessing ---
print("\n[3/4] Preprocessing & Tokenization...")
# scGPT expects binned expression values. 
# We use simple binning here. 
n_bins = 51
# Bin the data
binned_data = pd.cut(adata.X.flatten(), bins=n_bins, labels=False).reshape(adata.shape)
adata.layers["X_binned"] = binned_data

# Tokenize: Convert gene names to IDs and pad sequences
# We use the utility function from scGPT
print("Tokenizing... (Mapping genes to model vocabulary)")
gene_ids_in_vocab = np.array([vocab[g] for g in gene_names], dtype=int)

tokenized_data = tokenize_and_pad_batch(
    adata.layers["X_binned"],
    gene_ids_in_vocab,
    max_len=n_genes_in_data + 1, # +1 for the <cls> token
    vocab=vocab,
    pad_token="<pad>",
    pad_value=0,
    append_cls=True, # Critical: scGPT uses a CLS token for cell embeddings
    include_zero_gene=True,
)

input_gene_ids = tokenized_data["genes"].to(device)
input_values = tokenized_data["values"].to(device)
src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])

print(f"Tokenized Input Shape: {input_gene_ids.shape} (Batch Size x Seq Len)")

# --- Step 4: Inference ---
print("\n[4/4] Running Inference...")
with torch.no_grad():
    # Pass data through the model
    # We ask for the 'cell_emb' output
    output = model(
        input_gene_ids,
        input_values.float(),
        src_key_padding_mask=src_key_padding_mask,
        batch_labels=None, # Not using batch correction for this simple inference
        CLS=False,
        CCE=False,
        MVC=False,
        ECS=False
    )
    cell_embeddings = output["cell_emb"]

print("\n--- Success! ---")
print(f"Generated Cell Embeddings Shape: {cell_embeddings.shape}")
print(f"   (Rows = {n_cells} cells, Cols = {cell_embeddings.shape[1]} embedding dimensions)")
print("\nFirst cell embedding vector (first 10 values):")
print(cell_embeddings[0, :10].cpu().numpy())

print("You can now use these embeddings for clustering, visualization (UMAP), or trajectory analysis.")