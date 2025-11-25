# Core Critic Development - Project Structure

## Overview
This project implements an IRM-based graph critic network for identifying invariant vs spurious edges in graph data, trained on GOOD-Motif dataset.

## Directory Structure

```
.
├── critics/              # Core critic model implementation
│   ├── __init__.py
│   └── irm_gin.py        # 2-layer GIN critic with edge scoring + domain head
│
├── data/                 # Dataset loading utilities
│   ├── __init__.py
│   └── good_motif.py     # GOOD-Motif dataset loader with environment grouping
│
├── analysis/             # Validation and analysis tools
│   └── edge_variance.py # Computes edge score variance across environments
│
├── guidance/             # Diffusion guidance interface
│   ├── __init__.py
│   ├── interface.py      # Classifier-free guidance hooks for Person B's diffusion
│   └── test_guidance.py  # Dummy test harness
│
├── train_critic.py       # Main training script with IRM + entropy regularization
│
├── external/             # External dependencies
│   ├── GOOD/             # GOOD benchmark repository (cloned)
│   └── GOOD_datasets/    # Cached GOOD-Motif dataset files
│
└── artifacts/            # Training outputs (gitignored)
    ├── critic/           # Full 60-epoch training run results
    │   ├── checkpoints/critic_best.pt
    │   ├── edge_scores/  # Exported edge scores per split
    │   ├── training_history.json
    │   ├── variance_hist.png
    │   └── variance_report.json
    └── critic_debug/     # Debug run artifacts
```

## Key Files

### Training
- `train_critic.py`: Main training script with IRM penalty, entropy regularization, checkpointing
- `critics/irm_gin.py`: 2-layer GIN architecture with edge scoring head and domain classification head

### Data
- `data/good_motif.py`: Loads GOOD-Motif with per-environment dataloaders for IRM training

### Analysis
- `analysis/edge_variance.py`: Analyzes edge score variance across environments to identify invariant edges

### Guidance
- `guidance/interface.py`: Placeholder interface for Person B's diffusion module with classifier-free guidance

## Usage

### Train Critic
```bash
python train_critic.py --epochs 60 --batch-size 32 --irm-weight 25.0 --entropy-weight 0.01
```

### Analyze Variance
```bash
python analysis/edge_variance.py --edge-score-dir artifacts/critic/edge_scores
```

### Test Guidance
```bash
python -m guidance.test_guidance
```

## Results Summary

- **Training**: 60 epochs completed, loss stabilized around ~1.09
- **Edge Scores**: Exported for all splits (train, val, test, id_val, id_test)
- **Variance Analysis**: 43/52 edge signature keys have variance < 0.005 (likely motif cores)
- **Checkpoint**: Best model saved at `artifacts/critic/checkpoints/critic_best.pt`

## Notes

- Domain classifier accuracy stayed at 0 (expected - IRM penalty prevents domain discrimination)
- Low-variance edges correspond to motif cores as expected
- Guidance interface ready for Person B's diffusion module integration
