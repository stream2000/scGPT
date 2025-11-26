import os
import sys
import time
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Mock wandb to avoid dependency and errors in scgpt.trainer
sys.modules["wandb"] = MagicMock()

import torch
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli, masked_relative_error
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed
from scgpt.trainer import prepare_data, prepare_dataloader, train, evaluate

# Wrapper to handle argument mismatch in library
class ConfigurableTransformerModel(TransformerModel):
    def forward(self, *args, **kwargs):
        # Remove arguments not supported by the base TransformerModel
        kwargs.pop("mod_types", None)
        return super().forward(*args, **kwargs)

# Configuration
os.environ["KMP_WARNINGS"] = "off"

config_dict = {
    "seed": 42,
    "dataset_name": "PBMC_68K",
    "mask_ratio": 0.4,
    "epochs": 1,
    "n_bins": 51,
    "GEPC": True,  # MVC
    "GEP": True,   # masked gene expression prediction
    "CLS": False,  # Classification
    "ESC": False,  # ECS
    "DAR": True,   # Domain Adaptation
    "use_batch_labels": True,
    "use_mod": False,
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
    "task": "integration", 
    "pad_token": "<pad>",
    "pad_value": -2,
    "mask_value": -1,
    "DSBN": True,
    "explicit_zero_prob": True,
    "include_zero_gene": True,
    "n_hvg": 100, # Reduced for speed
}

# Convert dict to Namespace for compatibility with trainer.py
config = SimpleNamespace(**config_dict)

set_seed(config.seed)

# Settings
special_tokens = [config.pad_token, "<cls>", "<eoc>"]
max_seq_len = config.n_hvg + 1

# Output directory
save_dir = Path(f"./save/pretrain_{config.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving to {save_dir}")

logger = scg.logger

# Load Data (PBMC 68k - Local)
print("Loading PBMC 68k dataset from local files...")
data_path = "data/pbmc68k/filtered_matrices_mex/hg19/"
adata = sc.read_10x_mtx(
    data_path,
    var_names='gene_symbols',
    cache=True
)
adata.var_names_make_unique()

# Basic preprocessing 
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select HVG
print(f"Selecting {config.n_hvg} highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=config.n_hvg, subset=True)

# Filter cells that became empty after HVG selection (Fix for 'all zero rows' warning)
print("Filtering cells with zero counts after HVG selection...")
sc.pp.filter_cells(adata, min_counts=1)
print(f"Cells remaining: {adata.n_obs}")

# Helper for batch column 
if "batch" not in adata.obs:
    adata.obs["batch"] = "batch1"
adata.obs["str_batch"] = adata.obs["batch"].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Preprocessor
print("Preprocessing...")
# We already subset HVG, so set subset_hvg=False
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=False, 
    result_log1p_key="X_log1p",
    subset_hvg=False, 
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key="str_batch")

# Sort by batch for batch sampler
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
    pad_token=config.pad_token, pad_value=config.pad_value, append_cls=True, include_zero_gene=config.include_zero_gene
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=max_seq_len, vocab=vocab,
    pad_token=config.pad_token, pad_value=config.pad_value, append_cls=True, include_zero_gene=config.include_zero_gene
)

# Prepare Dataloaders using scgpt.trainer functions
train_data_pt, valid_data_pt = prepare_data(
    tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, config, 0
)

train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=config.batch_size,
    shuffle=False, 
    intra_domain_shuffle=True,
    drop_last=False,
    per_seq_batch_sample=True 
)
valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=config.batch_size,
    shuffle=False,
    drop_last=False
)

# Model
print("Initializing Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)
model = ConfigurableTransformerModel(
    ntokens,
    config.layer_size,
    config.nhead,
    config.layer_size, 
    config.nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=config.DAR,
    use_batch_labels=config.use_batch_labels,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)
model.to(device)

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# Run Training
print("Starting Training...")
for epoch in range(1, config.epochs + 1):
    print(f"Epoch {epoch}")
    train(
        model,
        train_loader,
        vocab,
        criterion, # criterion_gep_gepc
        criterion_dab,
        None, # criterion_cls
        scaler,
        optimizer,
        scheduler,
        device,
        config,
        logger,
        epoch
    )
    
    val_loss = evaluate(
        model,
        valid_loader,
        vocab,
        criterion,
        criterion_dab,
        None,
        device,
        config,
        epoch
    )
    print(f"Epoch {epoch} Valid Loss: {val_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")

print("Training finished!")