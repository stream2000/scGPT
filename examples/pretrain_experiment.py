import os
import sys
import time
import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import scanpy as sc
import numpy as np
# import wandb  <-- REMOVED
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# Add the project root to sys.path so we can import scgpt
sys.path.append(str(Path(__file__).resolve().parent.parent))

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli, masked_relative_error
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed

# Configuration
os.environ["KMP_WARNINGS"] = "off"
# os.environ["WANDB_MODE"] = "offline" 

config = {
    "seed": 42,
    "dataset_name": "PBMC_68K",
    "mask_ratio": 0.4,
    "epochs": 3, # Reduced to 3 for faster "Hello World"
    "n_bins": 51,
    "GEPC": True,
    "ecs_thres": 0.0, 
    "dab_weight": 1.0,
    "lr": 1e-4,
    "batch_size": 32,
    "layer_size": 128, 
    "nlayers": 2,      
    "nhead": 4,        
    "dropout": 0.2,
    "schedule_ratio": 0.9,
    "save_eval_interval": 5,
    "log_interval": 10,
    "fast_transformer": False, 
    "pre_norm": False,
    "amp": True,
}

set_seed(config["seed"])

# Settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config["mask_ratio"]
mask_value = -1
pad_value = -2
n_input_bins = config["n_bins"]
n_hvg = 1200
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True
explicit_zero_prob = True

# Output directory
save_dir = Path(f"./save/pretrain_{config['dataset_name']}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# Load Data (PBMC 68k - Local)
print("Loading PBMC 68k dataset from local files...")
data_path = "data/pbmc68k/filtered_matrices_mex/hg19/"
adata = sc.read_10x_mtx(
    data_path,
    var_names='gene_symbols',
    cache=True
)
# Ensure gene names are unique (common issue in 10x data)
adata.var_names_make_unique()
# Basic preprocessing 
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

# Helper for batch column 
if "batch" not in adata.obs:
    adata.obs["batch"] = "batch1"
adata.obs["str_batch"] = adata.obs["batch"].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Preprocessor
print("Preprocessing...")
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=False, 
    result_log1p_key="X_log1p",
    subset_hvg=False, 
    binning=config["n_bins"],
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch")

# Sort by batch
if per_seq_batch_sample:
    adata = adata[adata.obs["batch_id"].argsort()].copy()

# Tokenize
print("Tokenizing...")
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

# Train/Val split
train_data, valid_data, train_batch_labels, valid_batch_labels = train_test_split(
    all_counts, batch_ids, test_size=0.1, shuffle=True
)

# Vocab
vocab = Vocab(VocabPybind(genes + special_tokens, None))
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

# Tokenize batch
tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=pad_token, pad_value=pad_value, append_cls=True, include_zero_gene=True
)

# Prepare Data helper
def prepare_data_tensors(tokenized_data, batch_labels):
    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    return {
        "gene_ids": tokenized_data["genes"],
        "values": masked_values,
        "target_values": tokenized_data["values"],
        "batch_labels": torch.from_numpy(batch_labels).long(),
    }

train_data_pt = prepare_data_tensors(tokenized_train, train_batch_labels)
valid_data_pt = prepare_data_tensors(tokenized_valid, valid_batch_labels)

# Dataset Class
class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# Dataloader helper
def prepare_dataloader(data_pt, batch_size, shuffle=False):
    dataset = SeqDataset(data_pt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

train_loader = prepare_dataloader(train_data_pt, config["batch_size"], shuffle=True)
valid_loader = prepare_dataloader(valid_data_pt, config["batch_size"], shuffle=False)

# Model
print("Initializing Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    config["layer_size"],
    config["nhead"],
    config["layer_size"], 
    config["nlayers"],
    vocab=vocab,
    dropout=config["dropout"],
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config["GEPC"],
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=config["ecs_thres"],
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config["fast_transformer"],
    pre_norm=config["pre_norm"],
)
model.to(device)

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

# Train Loop
def train(model, loader):
    model.train()
    total_loss = 0.0
    for batch_idx, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0,
            )
            
            masked_positions = input_values.eq(mask_value)
            loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
            
            if explicit_zero_prob:
                loss += criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
            
            if config["GEPC"]:
                loss += criterion(output_dict["mvc_output"], target_values, masked_positions)
                if explicit_zero_prob:
                    loss += criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
            
            # Simple DAB
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss += config["dab_weight"] * loss_dab

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % config["log_interval"] == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

# Eval Loop
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(
                    input_gene_ids, input_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None
                )
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_dict["mlm_output"], target_values, masked_positions)
                total_loss += loss.item()
    return total_loss / len(loader)

# Run Training
print("Starting Training...")
for epoch in range(1, config["epochs"] + 1):
    print(f"Epoch {epoch}")
    train(model, train_loader)
    val_loss = evaluate(model, valid_loader)
    print(f"Epoch {epoch} Valid Loss: {val_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")

print("Training finished!")