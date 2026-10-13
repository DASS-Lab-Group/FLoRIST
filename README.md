# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models

Official implementation of **FLoRIST**, a federated fine-tuning framework that performs low-rank aggregation in latent space using **Singular Value Thresholding (SVT)** to achieve a strong balance between accuracy, communication efficiency, and scalability for large language models.

📄 *Accepted to MLSys 2026*  
This repository accompanies the camera-ready version of the paper.

<p align="left">
  <img src="figures/artifacts-available-v1.1.png" alt="ACM Artifacts Available v1.1" height="110">
  <img src="figures/artifacts-functional-v1.1.png" alt="ACM Artifacts Evaluated – Functional v1.1" height="110">
  <img src="figures/results-reproduced-v1.1.png" alt="ACM Results Reproduced v1.1" height="110">
</p>

---

## Overview

Federated fine-tuning of LLMs faces a fundamental tension: large models require expressive updates, but communication constraints and client heterogeneity limit what can be transmitted. Existing LoRA-based methods either introduce aggregation noise, rely on expensive full-matrix decompositions, or scale poorly when client ranks differ.

FLoRIST resolves this by operating directly in the low-rank latent adapter space instead of constructing dense global updates. The server performs an efficient decomposition of stacked client adapters and applies singular value thresholding to retain only the most informative components. The result is a compact global adapter that preserves performance while dramatically reducing communication.

In practice, this design yields higher MMLU accuracy, large communication savings (often 5×–350×), lower server computation compared to full SVD baselines, and stable behavior under heterogeneous client ranks.

---

## Key Ideas

FLoRIST introduces a unified framework for federated LoRA aggregation built around three principles:

- **Noise-free aggregation** that avoids the cross-term artifacts present in FedIT-style averaging
- **Efficient SVD without forming the dense update matrix**, making server computation scalable
- **Energy-based rank selection** via singular value thresholding to remove redundant components

This leads to adaptive per-layer compression and a global adapter that captures the shared signal across clients while filtering noisy updates. Across TinyLlama and Llama-3.2-1B models, and across Dolly, Alpaca, and Wizard datasets, FLoRIST consistently matches or exceeds the accuracy of FedIT, FFA-LoRA, FLoRA, and FlexLoRA while offering substantially higher communication efficiency.

---

<p align="center">
  <img src="./figures/workflow-8.png" width="75%" alt="FLoRIST Workflow">
</p>

---

## Experimental Setup

All results follow the protocol used in the MLSys paper. Training uses 100 total clients with 10 sampled per communication round for 75 rounds under non-IID partitions. LoRA is applied to attention layers only, and performance is evaluated on a 1,444-sample subset of MMLU.

Two client configurations are studied:

**Homogeneous setting.**  
All clients use LoRA rank 16.

**Heterogeneous setting.**  
Client ranks follow a heavy-tail distribution reflecting realistic device imbalance:

- 40 clients use rank 4  
- 20 clients use rank 8  
- 20 clients use rank 16  
- 10 clients use rank 32  
- 10 clients use rank 64  

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.x, CUDA 12, and NVIDIA H100/A100 GPUs.

---

## Model Setup

### TinyLlama

Download the TinyLlama-1.1B model before running experiments. Two steps are required: `download.py` fetches the tokenizer, configuration, and metadata via HuggingFace `snapshot_download`, and then `wget` fetches the model weights separately:

```bash
python download.py
cd tinyllama
wget -O tinyllama/model.safetensors \
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"
```

### LLaMA-3.2-1B

No local download needed. Set your HuggingFace token as an environment variable before running:

```bash
export HF_TOKEN=your_token_here
```

You can generate a token at https://huggingface.co/settings/tokens. You must also accept Meta's 
license at https://huggingface.co/meta-llama/Llama-3.2-1B before your token will grant access.

---

## Datasets

All three datasets are included pre-split in the repository. No manual download is required.

| Dataset | Directory | Source |
|---------|-----------|--------|
| Dolly-15k | `./data/` | [HuggingFace](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| Alpaca-52k | `./data_alpaca/` | [HuggingFace](https://huggingface.co/datasets/tatsu-lab/alpaca) |
| WizardLM-70k | `./data_wiz/` | [HuggingFace](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k) |

Each sample contains:

```
instruction
input (optional)
output
```

---

## Quick Start: Functional Validation (10 rounds)

To verify that the environment is correctly set up and the full pipeline runs end-to-end, use this short validation run. It completes in approximately 1–2 hours on a single GPU with 40 GB+ VRAM:

```bash
export HF_TOKEN=your_token_here

python main.py \
  --global_model llama3.2-1b \
  --data_path ./data_alpaca \
  --num_clients 100 \
  --client_selection_frac 0.1 \
  --num_communication_rounds 10 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.90 \
  --heter True
```

This is not intended to reproduce a paper result. Confirm that: (1) the model loads, (2) training proceeds for 10 rounds with per-round logs printed to stdout, and (3) MMLU accuracy is evaluated and reported.

---

## Reproducing a Key Result

We recommend the following configuration as the primary reproduction target. It runs FLoRIST with LLaMA-3.2-1B on Alpaca in the heterogeneous setting for the full 75 rounds (~4 hours on an H200 GPU):

```bash
export HF_TOKEN=your_token_here

python main.py \
  --global_model llama3.2-1b \
  --data_path ./data_alpaca \
  --num_clients 100 \
  --client_selection_frac 0.1 \
  --num_communication_rounds 75 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.90 \
  --heter True
```

**Expected output:**

- **MMLU accuracy** at round 75: ~0.3136 (31.36%), consistent with the 30.24% reported in Table 2 (within expected variation due to non-deterministic GPU operations and stochastic client sampling).
- **Total rank across all rounds**: 15,972. Communication efficiency = 1 / (15972 / 76) = 47.58 × 10⁻⁴, consistent with the 48.24 × 10⁻⁴ reported in the paper. (The divisor is 76 because the final round is evaluated twice in the logging.)

A reference output log is provided at `logs/llama3.2-1b_alpaca_heter_florist_tau0.9.log` for comparison.

To further verify FLoRIST's claims, run other methods in the same configuration (e.g., `--method fedit`, `--method ffa`) and confirm that FLoRIST achieves the highest accuracy and communication efficiency.

---

## Training

### Homogeneous configuration

```bash
python main.py \
  --global_model tinyllama \
  --data_path ./data \
  --num_clients 100 \
  --client_selection_frac 0.1 \
  --num_communication_rounds 75 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.90
```

### Heterogeneous configuration

```bash
python main.py \
  --global_model tinyllama \
  --data_path ./data_wiz \
  --num_clients 100 \
  --client_selection_frac 0.1 \
  --num_communication_rounds 75 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.90 \
  --heter True
```

For FedIT or FFA-LoRA baselines, add:

```
--zero_padding True
```

---

## Available Methods

The framework supports multiple aggregation strategies:

```
florist   (proposed method)
flora
fedit
flex
ffa
```

---

## Threshold Selection

The SVT threshold τ ∈ [0.80, 0.99] controls the retained rank of the global adapter. Lower thresholds produce stronger compression and higher communication efficiency, while higher thresholds prioritize maximal accuracy. We recommend τ = 0.9 as a robust default that works well across all 12 model–dataset–setting combinations.

In the paper, τ is also selected via binary search to match or exceed baseline performance. The fixed threshold at τ = 0.9 is within 1% accuracy of the optimally tuned variant in all combinations.

---

## Evaluation

Each training run logs the following statistics:

- MMLU accuracy
- per-round training metrics
- total LoRA rank
- total transmitted parameters

Derived metrics reported in the paper must be computed from the logged values:

- **Communication efficiency** = 1 / (total_rank / 2), where the division by 2 accounts for the two adapter matrices (B_g and A_g). Scale to × 10⁻⁴ to match Table 2.
- **Communication cost (MB)** = total_parameters × 2 / (1024 × 1024), where the factor of 2 converts FP16 values to bytes.

---

## Results Summary

Across all evaluated settings, FLoRIST achieves the strongest overall accuracy–efficiency trade-off. It frequently matches or exceeds the best baseline accuracy while delivering the highest communication efficiency and significantly lower server compute than full-matrix SVD methods.

Representative findings include:

* up to **400× less communication** than full fine-tuning
* up to **108× higher efficiency** than FLoRA
* ~7× lower server FLOPs than FlexLoRA
* adaptive rank compression that varies across layers
* higher intrinsic dimensionality in intermediate layers
* consistent redundancy in value projections vs query projections

---

## Communication Cost Example

Communication cost (MB) of federated fine-tuning methods on TinyLlama using the Wizard dataset (homogeneous setting):

| Method                 | Upload (MB) | Download (MB) |
| ---------------------- | ----------- | ------------- |
| Full FT                | 2076.17     | 2076.17       |
| FedIT                  | 45.05       | 45.05         |
| FLoRA                  | 45.05       | 360.45        |
| FlexLoRA               | 45.05       | 45.05         |
| FFA-LoRA               | **22.52**   | 28.83         |
| **FLoRIST** [τ\*]      | 45.05       | **5.95**      |
| **FLoRIST** [τ=0.9]    | 45.05       | **5.15**      |

---

## License

Apache 2.0

---

## Contributing

We welcome reproducibility improvements, dataset integrations, and benchmarking extensions. Please open an issue before large architectural changes.
