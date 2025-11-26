# scGPT Project Overview

This `MyReadme.md` provides a quick summary of the current project setup, including dependency management, environment, and recent file additions.

## 1. Dependency Management

This project primarily uses **Poetry** for dependency management, as indicated by `pyproject.toml` and `poetry.lock`.

*   **Python Version:** Requires `Python >=3.7.12,<4`.
*   **Core Dependencies:** Managed via `pyproject.toml` and installed with Poetry. Key dependencies include `pandas`, `scvi-tools`, `scanpy`, `torch`, `torchtext`, `numba`, etc.
*   **Development Dependencies:** Also managed by Poetry, including `pytest`, `black`, `tensorflow`, `flash-attn`, etc.
*   **Documentation Dependencies:** The `docs/requirements.txt` file specifies dependencies for building the project documentation.

## 2. Environment

The project expects a Python environment (e.g., a Poetry virtual environment or `scgpt_env/` if created) where the specified dependencies are installed. GPU acceleration with CUDA is supported and often utilized for model training and inference.

## 3. Recently Added/Modified Files (Last Commit)

The following significant files have been added or updated in the most recent commit:

*   `tutorial_inference.py`: A new standalone script demonstrating how to load a pretrained `scGPT` model (specifically the heart model) and perform inference using synthetic data. It includes a compatibility fix for FlashAttention weights.
*   `examples/pretrain_experiment.py`: A new example script showcasing the pretraining workflow for `scGPT`, using the PBMC 68k dataset.
*   `quantization_experiment/quantize_inference.py`: An experiment script demonstrating how to apply dynamic quantization (INT8) to an `scGPT` model for reduced size and potentially faster CPU inference, along with a benchmark.
*   `data/cellxgene/data_config.py`: The `VERSION` constant was updated to `2025-11-17`.
*   `.gitignore`: Updated to properly ignore new experiment directories (`heart_model_experiment/`, `scgpt_env/`) and other generated files, ensuring a cleaner repository.
