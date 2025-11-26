# Pretraining Experiment Configuration

This directory contains `pretrain_experiment.py`, a script for pretraining scGPT on single-cell RNA-seq data (specifically PBMC 68k in the example).

## Parameter Explanations

The script uses a configuration dictionary to control training behavior. Key parameters include:

### `n_hvg` (Number of Highly Variable Genes)
- **Current Setting:** `100`
- **Default/Typical:** `1200` or more.
- **Purpose:** This controls how many highly variable genes are selected from the dataset for training.
- **Reason for 100:** Reduced to `100` to significantly speed up the "Hello World" test run and reduce memory usage. Using a small number of genes creates a smaller vocabulary and shorter sequence lengths (since `max_seq_len ≈ n_hvg`).
- **Impact:** With only 100 genes, the model will not learn a comprehensive representation of the cell. Some cells might appear "empty" (all zeros) if they do not express any of these specific 100 genes. We added an explicit filtering step (`sc.pp.filter_cells(adata, min_counts=1)`) after HVG selection to handle this.

### `epochs`
- **Current Setting:** `1`
- **Purpose:** The number of complete passes through the training dataset.
- **Reason for 1:** Set to `1` for a quick verification that the training pipeline (data loading, model initialization, forward/backward pass) works without errors.

### Other Key Parameters
- **`batch_size`**: `32` - Number of samples per batch.
- **`layer_size`**, **`nlayers`**, **`nhead`**: Transformer architecture parameters. Reduced in this example for speed.
- **`GEPC`**: `True` - Enables Gene Expression Prediction for Cell modelling (Masked Value Prediction).
- **`DAR`**: `True` - Enables Domain Adaptation via Reverse gradient (if batch labels are used).

## Running the Experiment

To run the simplified experiment:
```bash
python examples/pretrain_experiment.py
```
Ensure you are in the `scgpt_env` environment.
