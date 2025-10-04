# 🚀 End-to-End Finetuning and Deployment: Llama-3.2-1B LoRA on Custom Data

## 📄 Overview

This repository contains the code and configuration for an **end-to-end pipeline** to finetune and deploy the **Meta Llama 3.2 1B** model on a custom instruction dataset. It utilizes Parameter-Efficient Fine-Tuning (PEFT) via **LoRA (Low-Rank Adaptation)** and the **`trl` (Transformer Reinforcement Learning)** library's `SFTTrainer`.

The project is specifically configured for hardware leveraging Apple's Metal Performance Shaders (MPS), optimized for Apple Silicon.

## ✨ Features

  * **Base Model:** `meta-llama/Llama-3.2-1B`
  * **Method:** LoRA (Low-Rank Adaptation)
  * **Training Library:** `trl.SFTTrainer`
  * **Custom Data:** Question-and-Answer pairs formatted as an `instruction.json` file.
  * **Hardware Optimization:** Configured for Apple Silicon using `DEVICE = torch.device("mps")` and `torch.bfloat16` (`DTYPE`).
  
## ⚙️ Setup

### Prerequisites

  * Python 3.12 or higher.
  * A Hugging Face access token with read permissions for the base model, passed in `train.py`.

### Installation

1.  Clone the repository and navigate into the directory.
2.  Install the required dependencies listed in `pyproject.toml`:
    ```bash
    # Install dependencies
    pip install colorama datasets docling litellm torch transformers
    ```

## 🚀 Training

The main training script is `train.py`.

### Key Configuration

The LoRA and training arguments are configured as follows:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `LoRA r` | `16` | LoRA attention dimension. |
| `LoRA alpha` | `32` | Scaling factor. |
| `LoRA dropout` | `0.05` | Dropout probability for LoRA layers. |
| `target_modules` | `"all-linear"` | Applies LoRA to all linear layers. |
| `per_device_train_batch_size` | `2` | Batch size per device. |
| `gradient_accumulation_steps` | `4` | Effectively simulates a batch size of 8. |
| `num_train_epochs` | `5` | Total number of training epochs. |
| `learning_rate` | `2e-4` | Standard learning rate. |
| `bf16` | `True` | Enables bfloat16 mixed precision for MPS/modern GPUs. |
| `optim` | `"adamw_torch"` | Optimizer used. |
| `output_dir` | `./Llama-3.2-1B-SFT-results` | Directory for logs and checkpoints. |

### Running the Script

To start the finetuning process:

```bash
python train.py
```

Trained adapters and checkpoints will be saved in the `complete_checkpoint` and `final_model` directories, as well as the main `output_dir`.

## 📦 Deployment (Example)

The provided `Modelfile` shows an example of how the resulting LoRA adapter can be used for local deployment, typically with tools like Ollama.

```modelfile
FROM llama3.2:1b
ADAPTER /Users/udaychandrabhookya/Work/Projects/Gen AI Learning/gen_ai_learning_notebooks/finetuning_llm/complete_checkpoint
```

This configuration applies the saved LoRA adapter (`complete_checkpoint`) on top of the base `llama3.2:1b` model.

### Clean-up

The `.gitignore` file specifies local directories that should be excluded from version control:

```
local_checkpoints
final_model
complete_checkpoint
build/
dist/
wheels/
meta-llama/
workspace/
__pycache__/
```
