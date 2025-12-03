# Causal Scene Graph: Robust Graph Learning with Causal Attention

**Causal Scene Graph** is a framework for training Graph Neural Networks (GNNs) that are invariant to spurious correlations. It leverages **Causal Attention** to identify the true structural causes of a graph's label and uses **Counterfactual Augmentation** to enforce robustness against irrelevant background noise.

This approach is particularly effective for:
*   **Synthetic Benchmarks:** Disentangling causal motifs (e.g., House, Cycle) from spurious patterns (e.g., Star, Wheel).
*   **Molecular Property Prediction:** Focusing on functional groups rather than scaffold artifacts.
*   **Scene Graph Classification:** Identifying key objects and relationships in visual scenes.

## Key Features
*   **Causal Attention Mechanism:** Learns to assign importance scores to nodes, filtering out noise.
*   **Counterfactual Augmentation:** Automatically generates counterfactual examples by preserving causal nodes and randomizing spurious ones.
*   **Cross-View Interaction:** A bidirectional attention module that aligns representations between original and counterfactual graphs.
*   **Multi-Objective Loss:** Combines classification, contrastive learning, sparsity, and diversity penalties to ensure robust learning.

## Environment Installation

This code is designed to work with **PyTorch** and **PyTorch Geometric**.

1.  **Create a Conda Environment:**
    ```bash
    conda create -n causal_gnn python=3.9
    conda activate causal_gnn
    ```

2.  **Install PyTorch:**
    (Check [pytorch.org](https://pytorch.org/) for the command matching your CUDA version)
    ```bash
    # Example for CUDA 11.8
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```



## Quick Start

### 1. Train on Spurious-Motif Benchmark
Train the model on the synthetic benchmark to see it learn to distinguish causal motifs from spurious ones.

```bash
python train_causal_gnn.py --dataset spurious_motif --bias 0.9 --epochs 50
```

*   `--bias 0.9`: Sets a strong correlation (90%) between causal and spurious motifs in the training set. The test set remains unbiased to evaluate robustness.

### 2. Visualize Results
Generate visualizations of the learned attention maps and t-SNE embeddings.

```bash
python visualize_results.py
```
This will create a `visualizations/` directory containing:
*   **Attention Grids:** Showing which nodes the model focuses on.
*   **t-SNE Plots:** Visualizing how the model separates classes despite spurious backgrounds.
*   **Robustness Analysis:** Comparing predictions on original vs. counterfactual graphs.

### 3. Hyperparameter Tuning
Use Optuna to find the best hyperparameters (learning rate, sparsity weight, etc.).

```bash
python tune_causal_gnn.py
```

## Usage Guide

### Training Arguments (`train_causal_gnn.py`)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `spurious_motif` | Dataset name (`spurious_motif`, `ogbg-molhiv`, `visual_genome`) |
| `--bias` | `0.9` | Spurious correlation strength (0.0 - 1.0) |
| `--lr` | `1e-3` | Learning rate |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--sparsity_weight` | `0.1` | Penalty for selecting too many nodes (L1 regularization) |
| `--contrastive_weight` | `0.1` | Weight for contrastive loss between views |
| `--use_cross_view` | `True` | Enable cross-view interaction module |

### Supported Datasets

1.  **`spurious_motif`**: A synthetic dataset where the label is determined by a "causal" motif (House, Cycle, Grid), but the training data is confounded by a "spurious" motif (Star, Wheel, Ladder).
2.  **`ogbg-molhiv`**: A molecular dataset from the Open Graph Benchmark. The model tries to identify the functional groups responsible for HIV inhibition.
3.  **`visual_genome`**: Scene graphs where nodes are objects and edges are relationships.

## Code Structure

*   **`train_causal_gnn.py`**: The main training engine. Implements the training loop, loss computation, and evaluation.
*   **`gin_causal_attention.py`**: Defines the model architecture.
    *   `GINEncoder`: Encodes graph structure.
    *   `CausalAttention`: Computes node importance scores.
    *   `CausalAugmenter`: Generates counterfactuals.
*   **`dataset_factory.py`**: Unified interface for loading different datasets.
*   **`spurious_motif_dataset.py`**: Generates the synthetic benchmark data.
*   **`visualize_results.py`**: Tools for interpreting model performance.

## Methodology

The core idea is based on the **Information Bottleneck** principle. We want to find a subgraph $G_c$ (the causal subgraph) that is:
1.  **Sufficient:** Contains enough information to predict the label $Y$.
2.  **Minimal:** Contains *only* the necessary information, discarding spurious noise $G_s$.

We achieve this by:
1.  **Learning a Mask:** The `CausalAttention` module learns a soft mask $M$ over the nodes.
2.  **Creating Counterfactuals:** We create a counterfactual graph $G_{cf}$ by keeping nodes where $M$ is high and randomizing nodes where $M$ is low.
3.  **Enforcing Invariance:** We force the model to predict the same label for the original graph $G$ and the counterfactual $G_{cf}$. If the model relies on spurious features (which are randomized in $G_{cf}$), this consistency loss will be high.

## Citation
