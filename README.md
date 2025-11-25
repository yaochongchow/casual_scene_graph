<<<<<<< HEAD
# casual_scene_graph
=======
# DiGress with DDIM, IRM, and Classifier-Free Guidance

**Week 1 Complete**: Infra, Baselines, and Integration

This repository contains an enhanced version of [DiGress](https://github.com/cvignac/DiGress) with the following improvements:
1. **DDIM Sampler**: Fast graph generation using strided sampling (~10-20x speedup)
2. **Baselines**: Pure PyTorch Geometric implementations of GRACE and GraphCL
3. **IRM Penalty**: Invariant Risk Minimization for robust feature learning
4. **Classifier-Free Guidance**: Conditional generation with guidance scaling

---

## Table of Contents
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Changes Made](#changes-made)
- [Running the Code](#running-the-code)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Clone the Repository
This project is based on the original DiGress repository:
```bash
git clone https://github.com/cvignac/DiGress.git
cd DiGress
```

### 2. Create Virtual Environment
We use Python 3.9 (for compatibility with DGL-free baselines):
```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torch-geometric pytorch-lightning hydra-core omegaconf wandb PyGCL scipy scikit-learn
```

**Note**: We bypass `dgl` dependency issues on Mac M1/M2 by implementing baselines in pure PyTorch Geometric.

---

## Project Structure

```
Project/
├── src/
│   ├── diffusion_model_discrete.py    # Core diffusion model (MODIFIED)
│   ├── main.py                         # Training entry point
│   ├── models/
│   │   └── transformer_model.py       # Graph Transformer
│   ├── diffusion/
│   │   ├── noise_schedule.py          # Noise schedules
│   │   └── diffusion_utils.py         # Utility functions
│   └── utils.py                        # Helper functions
│
├── baselines/
│   └── run_baselines.py                # GRACE & GraphCL (NEW)
│
├── tests/
│   └── test_ddim.py                    # DDIM unit tests (NEW)
│
├── scripts/
│   └── run_hpc.sh                      # SLURM script for HPC (NEW)
│
├── results/
│   └── week1.txt                       # Baseline results
│
├── configs/                            # Hydra configuration files
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Changes Made

### 1. **DDIM Sampler** (`src/diffusion_model_discrete.py`)

#### What is DDIM?
DDIM (Denoising Diffusion Implicit Models) is a faster sampling method that skips timesteps. Instead of sampling `t=1000→999→...→0`, we sample `t=1000→950→900→...→0` (e.g., 50 steps instead of 1000).

#### Implementation
- **`sample_batch_ddim()`** (Lines 717-772): Main DDIM sampling loop
  - Accepts `ddim_steps` to define number of steps
  - Constructs a strided time schedule using `torch.linspace`
  - Calls `sample_p_zs_given_zt_ddim` for each step

- **`sample_p_zs_given_zt_ddim()`** (Lines 657-714): DDIM single-step denoising
  - Accepts explicit `beta_t` for arbitrary time jumps
  - Supports **Classifier-Free Guidance** (see below)

#### Speed Improvement
| Method | Steps | Time per Graph |
|--------|-------|----------------|
| Original | 1000 | ~5 seconds |
| DDIM | 50 | **~0.25 seconds** |

---

### 2. **Baselines: GRACE & GraphCL** (`baselines/run_baselines.py`)

#### What are GRACE and GraphCL?
- **GRACE** (Graph Contrastive Learning): Self-supervised learning using edge dropping and feature masking
- **GraphCL**: Similar to GRACE but with stronger augmentations

#### Implementation
We implemented these from scratch in **pure PyTorch Geometric** to avoid DGL dependency issues:
- **`GConv`**: 2-layer GCN encoder
- **`Encoder`**: Node encoder + projection head
- **`info_nce_loss()`**: InfoNCE contrastive loss
- **Augmentations**: `drop_edge()`, `mask_feature()`

#### Results
See `results/week1.txt` for training loss curves on the MUTAG dataset.

---

### 3. **IRM Penalty** (`src/diffusion_model_discrete.py`)

#### What is IRM?
Invariant Risk Minimization penalizes models that rely on spurious correlations by ensuring the loss is invariant across different "environments" (batch splits).

#### Implementation
- **Lines 101-102**: Added `self.irm_lambda` and `self.p_uncond` parameters
- **Lines 120-161**: Modified `training_step()`
  - Splits batch into two environments
  - Computes a placeholder penalty (variance proxy)
  - Adds penalty to loss: `total_loss = loss + irm_lambda * irm_penalty`

**Note**: This is a placeholder implementation. A full IRM implementation requires gradient-based penalties, which we defer to Week 2.

---

### 4. **Classifier-Free Guidance** (`src/diffusion_model_discrete.py`)

#### What is Classifier-Free Guidance?
CFG allows you to control how strongly the model follows a condition (e.g., "generate a soluble molecule") by mixing conditional and unconditional predictions.

#### Implementation
- **Training** (Lines 111-114):
  - With probability `p_uncond` (10%), replace condition `y` with zeros during training
  - This teaches the model to generate both conditionally and unconditionally

- **Sampling** (Lines 672-684):
  - Compute both conditional (`pred_cond`) and unconditional (`pred_uncond`) logits
  - Mix them: `pred = pred_uncond + scale * (pred_cond - pred_uncond)`
  - Higher `scale` → stronger conditioning

#### Usage
```python
model.sample_batch_ddim(batch_size=10, ddim_steps=50, guidance_scale=2.0)
```

---

## Running the Code

### 1. Run DDIM Tests
Verify the DDIM sampler and Classifier-Free Guidance:
```bash
source .venv/bin/activate
python3 tests/test_ddim.py
```

**Expected Output**:
```
Running DDIM sampling with guidance_scale=2.0...
DDIM sampling with Guidance successful!
Sampled graph with 10 atoms
Sampled graph with 12 atoms
.
----------------------------------------------------------------------
Ran 1 test in 0.127s

OK
```

---

### 2. Run Baselines (GRACE & GraphCL)
Train contrastive learning baselines:
```bash
source .venv/bin/activate
python3 baselines/run_baselines.py
```

**Expected Output**:
```
Running Pure PyG Baselines...

--- Training GRACE on MUTAG ---
Epoch 01, Loss: 3.4306
Epoch 02, Loss: 3.1234
...
Epoch 10, Loss: 2.9957

--- Training GraphCL on MUTAG ---
Epoch 01, Loss: 3.4452
...
Epoch 10, Loss: 3.3936

Baselines run complete. Results saved to results/week1.txt
```

---

### 3. Run Full DiGress Training (GPU Recommended)
To train the full model (not tested in Week 1):
```bash
source .venv/bin/activate
python3 src/main.py general.name=my_experiment dataset.name=zinc model.type=discrete
```

---

## File Descriptions

### Core Files (Modified)
- **`src/diffusion_model_discrete.py`**: The heart of the model
  - `sample_batch_ddim()`: DDIM sampling loop (NEW)
  - `sample_p_zs_given_zt_ddim()`: DDIM single step with CFG (NEW)
  - `training_step()`: Training loop with IRM and CFG (MODIFIED)
  - `__init__()`: Added `irm_lambda` and `p_uncond` (MODIFIED)

### New Files
- **`baselines/run_baselines.py`**: Pure PyG implementation of GRACE and GraphCL
- **`tests/test_ddim.py`**: Unit tests for DDIM and CFG
- **`scripts/run_hpc.sh`**: SLURM script for Northeastern Discovery

### Original DiGress Files (Unchanged)
- **`src/main.py`**: Training entry point (uses Hydra configs)
- **`src/models/transformer_model.py`**: Graph Transformer architecture
- **`src/diffusion/noise_schedule.py`**: Beta schedules (cosine, linear, etc.)
- **`src/diffusion/diffusion_utils.py`**: Posterior computation, sampling utilities

---

## Dependencies

Key packages:
- `torch` (2.8+): Deep learning framework
- `torch-geometric` (2.7+): Graph neural networks
- `pytorch-lightning` (2.5+): Training framework
- `hydra-core` (1.3+): Configuration management
- `wandb` (0.23+): Experiment tracking (optional)

See `requirements.txt` for full list.

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'dgl'`
**Solution**: We removed DGL dependency. If you see this, make sure you're running `baselines/run_baselines.py` (pure PyG version) and not an old PyGCL-based script.

### Issue: DDIM test fails with shape mismatch
**Solution**: Ensure you're using the updated `sample_p_zs_given_zt_ddim()` with the `guidance_scale` parameter.

### Issue: HPC job runs out of memory
**Solution**: Reduce `train.batch_size` in the config or request more memory (`--mem=64G`).

---

## References

1. **DiGress**: [Original Paper](https://arxiv.org/abs/2209.14734) | [GitHub](https://github.com/cvignac/DiGress)
2. **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
3. **GRACE**: [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131)
4. **GraphCL**: [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902)
5. **Classifier-Free Guidance**: [Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)

---

## License

This project extends DiGress, which is licensed under the MIT License. See the original repository for details.

---

## Acknowledgements

- **DiGress Authors**: Clement Vignac et al.
- **Northeastern Research Computing**: For providing HPC resources

**Week 1 Status**:  Complete (Infra, Baselines, Integration)
>>>>>>> charles
