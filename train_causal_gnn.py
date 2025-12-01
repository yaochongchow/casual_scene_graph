"""
Causal GNN Training with Counterfactual Augmentation

Training Algorithm:
1. Forward pass Original Batch -> Get Causal Scores
2. Generate Counterfactuals (with detached gradients)
3. Forward pass both Original and Counterfactual through GNN
4. Apply CrossViewInteraction between them
5. Compute Contrastive Loss (InfoNCE) + Classification Loss
6. Backpropagate and update

This approach encourages the model to learn invariant representations
by contrasting original graphs with their counterfactuals where
spurious correlations are randomized.

Supports:
- Multiple datasets via dataset_factory
- Multi-GPU training with DataParallel
- Conditional metrics (accuracy/AUC) based on dataset
- Conditional causal mask evaluation (only for synthetic datasets)
"""

import argparse
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from dataset_factory import compute_metric, get_dataset, print_dataset_info
from gin_causal_attention import (BidirectionalCrossViewInteraction,
                                  CausalAttention, CausalAttentionLoss,
                                  CausalAttentionPooling, CausalAugmenter,
                                  CrossViewContrastiveLoss,
                                  CrossViewInteraction, GINEncoder)

# Only import SpuriousMotif if available (for backward compatibility)
try:
    from spurious_motif_dataset import SpuriousMotif
except ImportError:
    SpuriousMotif = None


# =============================================================================
# Utility Functions
# =============================================================================

def prepare_batch(batch, device: torch.device):
    """
    Move batch to device and ensure correct dtypes.
    
    OGB and some other datasets have integer node/edge features,
    but PyTorch models require float tensors.
    
    Args:
        batch: PyG batch object
        device: Target device
        
    Returns:
        Batch with correct dtypes on target device
    """
    batch = batch.to(device)
    
    # Convert node features to float
    if batch.x is not None and batch.x.dtype != torch.float32:
        batch.x = batch.x.float()
    
    # Convert edge attributes to float
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        if batch.edge_attr.dtype != torch.float32:
            batch.edge_attr = batch.edge_attr.float()
    
    return batch


# =============================================================================
# Full Causal GNN Model with Cross-View Learning
# =============================================================================

class CausalGNNWithCrossView(nn.Module):
    """
    Complete model for causal graph learning with cross-view interaction.
    
    Architecture:
        1. GIN Encoder: Learn node embeddings
        2. Causal Attention: Score node importance
        3. Cross-View Interaction: Exchange information between views
        4. Pooling: Aggregate to graph-level
        5. Classifier: Predict graph labels
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        output_dim: Number of output classes
        num_gin_layers: Number of GIN message passing layers
        num_heads: Number of attention heads for cross-view
        dropout: Dropout probability
        use_cross_view: Whether to use cross-view interaction
        use_atom_encoder: Whether to use OGB AtomEncoder for molecular features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_gin_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.5,
        use_cross_view: bool = True,
        use_atom_encoder: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_cross_view = use_cross_view
        self.use_atom_encoder = use_atom_encoder
        
        # GIN Encoder (with AtomEncoder support for OGB molecules)
        self.encoder = GINEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gin_layers,
            dropout=dropout,
            use_atom_encoder=use_atom_encoder
        )
        
        # Causal Attention for scoring nodes
        self.causal_attention = CausalAttention(
            hidden_dim=hidden_dim,
            temperature=1.0
        )
        
        # Cross-View Interaction (optional)
        if use_cross_view:
            self.cross_view = BidirectionalCrossViewInteraction(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Pooling
        self.pooling = CausalAttentionPooling(
            hidden_dim=hidden_dim,
            temperature=1.0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graphs and compute causal attention scores.
        
        Returns:
            node_embed: Node embeddings [num_nodes, hidden_dim]
            scores: Causal scores [num_nodes, 1]
        """
        node_embed = self.encoder(x, edge_index)
        scores = self.causal_attention(node_embed, batch)
        return node_embed, scores
    
    def forward_single_view(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single view (original or counterfactual).
        
        Returns:
            logits: Classification logits [batch_size, output_dim]
            graph_embed: Graph embeddings [batch_size, hidden_dim]
            node_embed: Node embeddings [num_nodes, hidden_dim]
            scores: Causal attention scores [num_nodes, 1]
        """
        node_embed, scores = self.encode(x, edge_index, batch)
        graph_embed, _ = self.pooling(node_embed, batch)
        logits = self.classifier(graph_embed)
        return logits, graph_embed, node_embed, scores
    
    def forward_with_cross_view(
        self,
        x_orig: torch.Tensor,
        edge_index_orig: torch.Tensor,
        batch_orig: torch.Tensor,
        x_cf: torch.Tensor,
        edge_index_cf: torch.Tensor,
        batch_cf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-view interaction between original and counterfactual.
        
        Returns:
            Dictionary containing:
            - logits_orig: Classification logits for original
            - logits_cf: Classification logits for counterfactual
            - z_orig: Projected embeddings for original (for contrastive)
            - z_cf: Projected embeddings for counterfactual (for contrastive)
            - graph_embed_orig: Graph embeddings for original
            - graph_embed_cf: Graph embeddings for counterfactual
            - scores_orig: Causal scores for original
            - scores_cf: Causal scores for counterfactual
            - attn_info: Cross-view attention weights
        """
        # Encode both views
        node_embed_orig, scores_orig = self.encode(x_orig, edge_index_orig, batch_orig)
        node_embed_cf, scores_cf = self.encode(x_cf, edge_index_cf, batch_cf)
        
        # Cross-view interaction
        if self.use_cross_view:
            node_embed_orig, node_embed_cf, attn_info = self.cross_view(
                node_embed_orig, batch_orig,
                node_embed_cf, batch_cf
            )
        else:
            attn_info = {}
        
        # Pool to graph-level
        graph_embed_orig, _ = self.pooling(node_embed_orig, batch_orig)
        graph_embed_cf, _ = self.pooling(node_embed_cf, batch_cf)
        
        # Classification
        logits_orig = self.classifier(graph_embed_orig)
        logits_cf = self.classifier(graph_embed_cf)
        
        # Project for contrastive learning
        z_orig = self.projection(graph_embed_orig)
        z_cf = self.projection(graph_embed_cf)
        
        return {
            'logits_orig': logits_orig,
            'logits_cf': logits_cf,
            'z_orig': z_orig,
            'z_cf': z_cf,
            'graph_embed_orig': graph_embed_orig,
            'graph_embed_cf': graph_embed_cf,
            'scores_orig': scores_orig,
            'scores_cf': scores_cf,
            'attn_info': attn_info
        }


# =============================================================================
# Combined Loss Function
# =============================================================================

class CausalContrastiveLoss(nn.Module):
    """
    Combined loss for causal contrastive learning with Information Bottleneck.
    
    Loss = α * CE_orig + β * CE_cf + γ * Contrastive + δ * Diversity + λ * Sparsity + μ * Coverage + η * Entropy
    
    KEY INSIGHT (Information Bottleneck):
    - CE_cf forces selected nodes to be SUFFICIENT for classification
    - Sparsity forces selected nodes to be MINIMAL  
    - Together: select MINIMUM nodes needed for classification = CAUSAL nodes!
    
    The coverage term prevents attention collapse (all zeros).
    The sparsity term prevents selecting everything (all ones).
    The diversity term penalizes G_cf being too similar to G (prevents copying).
    The entropy term pushes scores to be binary (0 or 1), preventing "gray area".
    The supervision term uses ground-truth causal masks when available.
    
    Args:
        classification_weight: Weight for classification loss on original
        cf_classification_weight: Weight for classification loss on counterfactual (INFO BOTTLENECK!)
        contrastive_weight: Weight for contrastive loss
        diversity_weight: Weight for diversity loss (G_cf should differ from G)
        sparsity_weight: Weight for sparsity penalty on attention
        coverage_weight: Weight for coverage penalty (prevents collapse)
        entropy_weight: Weight for entropy loss (pushes scores to 0 or 1)
        supervision_weight: Weight for causal mask supervision
        target_sparsity: Target fraction of nodes to select (e.g., 0.3 = 30%)
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        cf_classification_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.2,
        coverage_weight: float = 0.1,
        entropy_weight: float = 0.1,
        supervision_weight: float = 0.0,
        target_sparsity: float = 0.3,
        temperature: float = 0.5
    ):
        super().__init__()
        
        self.classification_weight = classification_weight
        self.cf_classification_weight = cf_classification_weight
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.coverage_weight = coverage_weight
        self.entropy_weight = entropy_weight
        self.supervision_weight = supervision_weight
        self.target_sparsity = target_sparsity
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()  # For supervision
        self.contrastive_loss = CrossViewContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        batch_orig: torch.Tensor,
        batch_cf: torch.Tensor,
        task_type: str = 'multiclass',
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Supports both multiclass (CrossEntropy) and binary (BCE) classification.
        
        Args:
            outputs: Dictionary from model forward pass
            labels: Ground truth labels
            batch_orig: Batch assignment for original
            batch_cf: Batch assignment for counterfactual
            task_type: 'multiclass' or 'binary'
            causal_mask: Optional ground-truth causal node mask [num_nodes]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        batch_size = labels.size(0)
        
        logits_orig = outputs['logits_orig']
        logits_cf = outputs['logits_cf']
        
        # Classification loss on original (handle binary vs multiclass)
        if task_type == 'binary':
            if logits_orig.dim() > 1 and logits_orig.size(1) == 1:
                # Single output for binary
                ce_orig = F.binary_cross_entropy_with_logits(
                    logits_orig.squeeze(-1), labels.float()
                )
                ce_cf = F.binary_cross_entropy_with_logits(
                    logits_cf.squeeze(-1), labels.float()
                )
            elif logits_orig.dim() > 1 and logits_orig.size(1) == 2:
                # Two outputs for binary (treated as multiclass)
                ce_orig = self.ce_loss(logits_orig, labels)
                ce_cf = self.ce_loss(logits_cf, labels)
            else:
                # Scalar output
                ce_orig = F.binary_cross_entropy_with_logits(logits_orig, labels.float())
                ce_cf = F.binary_cross_entropy_with_logits(logits_cf, labels.float())
        else:
            # Standard multiclass
            ce_orig = self.ce_loss(logits_orig, labels)
            ce_cf = self.ce_loss(logits_cf, labels)
        
        # Contrastive loss between views
        contrastive = self.contrastive_loss(outputs['z_orig'], outputs['z_cf'])
        
        # Diversity loss: penalize when G_cf is too similar to G
        # This prevents the "select everything" loophole
        # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
        z_orig_norm = F.normalize(outputs['z_orig'], dim=-1)
        z_cf_norm = F.normalize(outputs['z_cf'], dim=-1)
        similarity = (z_orig_norm * z_cf_norm).sum(dim=-1).mean()  # [batch_size] -> scalar
        # We want some diversity, so penalize high similarity
        # diversity_loss = max(0, similarity - threshold) to allow some similarity
        diversity = F.relu(similarity - 0.8)  # Penalize if similarity > 0.8
        
        scores = outputs['scores_orig'].squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        mean_score = scores.mean()
        max_score = scores.max()
        
        # Sparsity loss: ASYMMETRIC penalty
        # Penalize heavily when selecting too many (mean > target)
        # Penalize lightly when selecting too few (mean < target)
        excess = F.relu(mean_score - self.target_sparsity)  # Penalty for selecting too much
        deficit = F.relu(self.target_sparsity * 0.5 - mean_score)  # Light penalty for selecting too little
        sparsity = excess * 5.0 + deficit  # 5x stronger penalty for "select everything"
        
        # Coverage loss: softer formulation that doesn't overpower sparsity
        # Only activate when scores collapse to near-zero
        coverage = F.relu(0.1 - mean_score) + F.relu(0.3 - max_score)
        
        # Entropy loss: push scores to be binary (0 or 1)
        # High entropy = scores near 0.5 (bad, uncertain)
        # Low entropy = scores near 0 or 1 (good, decisive)
        # H(p) = -p*log(p) - (1-p)*log(1-p)
        scores_clamped = scores.clamp(1e-6, 1 - 1e-6)
        entropy = -scores_clamped * torch.log(scores_clamped) - (1 - scores_clamped) * torch.log(1 - scores_clamped)
        entropy_loss = entropy.mean()  # We MINIMIZE entropy to push toward 0 or 1
        
        # Combine losses
        total_loss = (
            self.classification_weight * ce_orig +
            self.cf_classification_weight * ce_cf +
            self.contrastive_weight * contrastive +
            self.diversity_weight * diversity +
            self.sparsity_weight * sparsity +
            self.coverage_weight * coverage +
            self.entropy_weight * entropy_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'ce_orig': ce_orig.item(),
            'ce_cf': ce_cf.item(),
            'contrastive': contrastive.item(),
            'diversity': diversity.item(),
            'sparsity': sparsity.item(),
            'coverage': coverage.item(),
            'entropy': entropy_loss.item(),
            'mean_score': mean_score.item(),
            'similarity': similarity.item()
        }
        
        # Supervision loss: directly supervise attention with ground-truth causal mask
        if causal_mask is not None and self.supervision_weight > 0:
            # causal_mask: [num_nodes] binary mask (1 = causal, 0 = spurious)
            # scores: [num_nodes, 1] attention scores
            scores_flat = outputs['scores_orig'].squeeze()
            causal_mask_float = causal_mask.float()
            
            # BCE loss to match attention scores to causal mask
            supervision_loss = self.bce_loss(scores_flat, causal_mask_float)
            total_loss = total_loss + self.supervision_weight * supervision_loss
            loss_dict['supervision'] = supervision_loss.item()
            loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    model: CausalGNNWithCrossView,
    batch,
    augmenter: CausalAugmenter,
    criterion: CausalContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_epochs: int = 0,
    current_epoch: int = 0,
    data_info: Optional[Dict[str, Any]] = None,
    intervention_augmenter: Optional['InterventionAugmenter'] = None,
    use_intervention_training: bool = False,
    intervention_weight: float = 1.0
) -> Dict[str, float]:
    """
    Single training step implementing causal learning via interventions.
    
    TWO MODES:
    
    1. INTERVENTION-BASED TRAINING (use_intervention_training=True):
       - Swaps spurious backgrounds during training
       - Forces model to give same prediction on original and intervention
       - Model MUST learn to ignore spurious features
       - Requires ground-truth causal masks (synthetic datasets)
    
    2. STANDARD CONTRASTIVE (use_intervention_training=False):
       - Uses attention-based counterfactual generation
       - Relies on information bottleneck for causal discovery
    
    Algorithm for Intervention Training:
    1. For each graph, create intervention by swapping spurious background
    2. Forward pass both original and intervention
    3. Loss = CE_orig + CE_intervention (same label!)
    4. Model learns to focus on causal features
    
    Args:
        model: The model (can be wrapped in DataParallel)
        batch: Input batch
        augmenter: CausalAugmenter for counterfactual generation
        criterion: Loss function
        optimizer: Optimizer
        device: Device for computation
        warmup_epochs: Number of warmup epochs
        current_epoch: Current epoch number
        data_info: Dataset info dict (for task type, etc.)
        intervention_augmenter: InterventionAugmenter for robust training
        use_intervention_training: Whether to use intervention-based training
        intervention_weight: Weight for intervention classification loss
    """
    # Handle DataParallel wrapped models
    model_core = model.module if isinstance(model, nn.DataParallel) else model
    
    model.train()
    optimizer.zero_grad()
    
    # Move batch to device and ensure correct dtype
    batch = prepare_batch(batch, device)
    
    # Get dataset properties
    if data_info is not None:
        task_type = data_info.get('task_type', 'multiclass')
        has_masks = data_info.get('has_masks', False)
    else:
        task_type = 'multiclass'
        has_masks = True  # Default for backward compatibility
    
    is_warmup = current_epoch < warmup_epochs
    
    if is_warmup:
        # =====================================================================
        # WARMUP PHASE: Train only on original graphs
        # =====================================================================
        logits, graph_embed, node_embed, scores = model_core.forward_single_view(
            batch.x, batch.edge_index, batch.batch
        )
        
        # Handle labels for different task types
        labels = batch.y
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        # Classification loss (handle binary vs multiclass)
        if task_type == 'binary':
            if logits.dim() > 1 and logits.size(1) == 1:
                ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            elif logits.dim() > 1 and logits.size(1) == 2:
                ce_loss = F.cross_entropy(logits, labels)
            else:
                ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        else:
            ce_loss = F.cross_entropy(logits, labels)
        
        # Attention regularization (prevent collapse even during warmup)
        scores_squeezed = scores.squeeze()
        if scores_squeezed.dim() == 0:
            scores_squeezed = scores_squeezed.unsqueeze(0)
        mean_score = scores_squeezed.mean()
        max_score = scores_squeezed.max()
        
        # Sparsity + Coverage during warmup
        # Penalize selecting too many nodes
        excess = F.relu(mean_score - criterion.target_sparsity)
        deficit = F.relu(0.1 - mean_score)  # Light penalty for collapse
        coverage_loss = excess * 5.0 + deficit
        
        # Combined warmup loss (with coverage to prevent collapse)
        loss = ce_loss + criterion.coverage_weight * coverage_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            if task_type == 'binary':
                if logits.dim() > 1 and logits.size(1) == 2:
                    pred = logits.argmax(dim=-1)
                else:
                    pred = (logits.squeeze(-1) > 0).long()
            else:
                pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
        
        loss_dict = {
            'total': loss.item(),
            'ce_orig': ce_loss.item(),
            'ce_cf': 0.0,
            'contrastive': 0.0,
            'coverage': coverage_loss.item(),
            'mean_score': mean_score.item(),
            'acc_orig': acc,
            'acc_cf': 0.0,
            'is_warmup': True
        }
        
        return loss_dict
    
    # =========================================================================
    # INTERVENTION-BASED TRAINING (if enabled and has masks)
    # =========================================================================
    
    if use_intervention_training and has_masks and intervention_augmenter is not None:
        # Create interventions by swapping spurious backgrounds
        # This breaks the spurious correlation during training!
        
        from torch_geometric.data import Batch, Data
        
        intervention_list = []
        
        # Get node counts per graph
        num_nodes_list = []
        for i in range(batch.num_graphs):
            num_nodes_list.append((batch.batch == i).sum().item())
        
        # Cumulative sum for indexing
        cumsum = [0] + list(torch.cumsum(torch.tensor(num_nodes_list), dim=0).tolist())
        
        for i in range(batch.num_graphs):
            start_node = cumsum[i]
            end_node = cumsum[i + 1]
            
            # Extract node features and causal mask
            x_i = batch.x[start_node:end_node]
            causal_mask_i = batch.causal_mask[start_node:end_node] if hasattr(batch, 'causal_mask') else None
            
            # Extract edges for this graph
            edge_mask = (batch.edge_index[0] >= start_node) & (batch.edge_index[0] < end_node)
            edge_index_i = batch.edge_index[:, edge_mask] - start_node
            
            # Get spurious type
            spurious_type_i = batch.spurious_type[i].item() if hasattr(batch, 'spurious_type') else 0
            
            data = Data(
                x=x_i,
                edge_index=edge_index_i,
                y=batch.y[i],
                causal_mask=causal_mask_i,
                spurious_type=torch.tensor(spurious_type_i)
            )
            
            try:
                intervention, _ = intervention_augmenter.create_intervention(data, device)
                intervention_list.append(intervention)
            except Exception as e:
                # If intervention fails, use original (with different spurious pattern)
                intervention_list.append(data)
        
        intervention_batch = Batch.from_data_list(intervention_list).to(device)
        
        # Forward pass original
        logits_orig, _, _, scores_orig = model_core.forward_single_view(
            batch.x, batch.edge_index, batch.batch
        )
        
        # Forward pass intervention
        logits_int, _, _, scores_int = model_core.forward_single_view(
            intervention_batch.x, intervention_batch.edge_index, intervention_batch.batch
        )
        
        labels = batch.y
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        # Classification loss on BOTH original and intervention (same label!)
        ce_orig = F.cross_entropy(logits_orig, labels)
        ce_int = F.cross_entropy(logits_int, labels)
        
        # Consistency loss: predictions should be similar
        consistency = F.kl_div(
            F.log_softmax(logits_int, dim=-1),
            F.softmax(logits_orig.detach(), dim=-1),
            reduction='batchmean'
        )
        
        # Sparsity on attention scores
        scores_squeezed = scores_orig.squeeze()
        mean_score = scores_squeezed.mean()
        excess = F.relu(mean_score - criterion.target_sparsity)
        deficit = F.relu(0.1 - mean_score)
        sparsity_loss = excess * 5.0 + deficit
        
        # Total loss
        loss = ce_orig + intervention_weight * ce_int + 0.1 * consistency + criterion.sparsity_weight * sparsity_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred_orig = logits_orig.argmax(dim=-1)
            pred_int = logits_int.argmax(dim=-1)
            acc_orig = (pred_orig == labels).float().mean().item()
            acc_int = (pred_int == labels).float().mean().item()
        
        loss_dict = {
            'total': loss.item(),
            'ce_orig': ce_orig.item(),
            'ce_int': ce_int.item(),
            'consistency': consistency.item(),
            'sparsity': sparsity_loss.item(),
            'mean_score': mean_score.item(),
            'acc_orig': acc_orig,
            'acc_int': acc_int,
            'is_warmup': False,
            'mode': 'intervention'
        }
        
        return loss_dict
    
    # =========================================================================
    # STANDARD CONTRASTIVE TRAINING (after warmup)
    # =========================================================================
    
    # Step 1: Forward pass Original Batch -> Get Causal Scores
    with torch.no_grad():
        node_embed_for_scores = model_core.encoder(batch.x, batch.edge_index)
        causal_scores = model_core.causal_attention(node_embed_for_scores, batch.batch)
    
    # Step 2: Generate Counterfactuals using learned scores (strictly detached!)
    scores_for_augmentation = causal_scores.detach()
    
    counterfactual_batch = augmenter.generate_counterfactuals(
        batch, 
        scores_for_augmentation
    )
    counterfactual_batch = counterfactual_batch.to(device)
    
    # Step 3 & 4: Forward pass both views with CrossViewInteraction
    outputs = model_core.forward_with_cross_view(
        x_orig=batch.x,
        edge_index_orig=batch.edge_index,
        batch_orig=batch.batch,
        x_cf=counterfactual_batch.x,
        edge_index_cf=counterfactual_batch.edge_index,
        batch_cf=counterfactual_batch.batch
    )
    
    # Handle labels
    labels = batch.y
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    
    # Get causal mask if available (for supervision)
    causal_mask = getattr(batch, 'causal_mask', None)
    
    # Step 5: Compute Combined Loss
    loss, loss_dict = criterion(
        outputs=outputs,
        labels=labels,
        batch_orig=batch.batch,
        batch_cf=counterfactual_batch.batch,
        task_type=task_type,
        causal_mask=causal_mask
    )
    
    # Step 6: Backpropagate
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        if task_type == 'binary':
            logits_orig = outputs['logits_orig']
            logits_cf = outputs['logits_cf']
            if logits_orig.dim() > 1 and logits_orig.size(1) == 2:
                pred_orig = logits_orig.argmax(dim=-1)
                pred_cf = logits_cf.argmax(dim=-1)
            else:
                pred_orig = (logits_orig.squeeze(-1) > 0).long()
                pred_cf = (logits_cf.squeeze(-1) > 0).long()
        else:
            pred_orig = outputs['logits_orig'].argmax(dim=-1)
            pred_cf = outputs['logits_cf'].argmax(dim=-1)
        
        acc_orig = (pred_orig == labels).float().mean().item()
        acc_cf = (pred_cf == labels).float().mean().item()
    
    loss_dict['acc_orig'] = acc_orig
    loss_dict['acc_cf'] = acc_cf
    loss_dict['is_warmup'] = False
    
    return loss_dict


# =============================================================================
# Evaluation Step
# =============================================================================

@torch.no_grad()
def evaluate(
    model: CausalGNNWithCrossView,
    loader: DataLoader,
    device: torch.device,
    data_info: Optional[Dict[str, Any]] = None,
    compute_causal_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Supports both accuracy and AUC metrics depending on dataset type.
    Only computes causal precision/recall/F1 if dataset has ground-truth masks.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to evaluate on
        data_info: Dataset info dict from dataset_factory
        compute_causal_metrics: Whether to compute causal node detection metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Determine metric type and whether we have causal masks
    if data_info is not None:
        metric_type = data_info.get('metric', 'accuracy')
        has_masks = data_info.get('has_masks', False)
        task_type = data_info.get('task_type', 'multiclass')
        evaluator = data_info.get('evaluator', None)
    else:
        metric_type = 'accuracy'
        has_masks = True  # Default for backward compatibility
        task_type = 'multiclass'
        evaluator = None
    
    # Only compute causal metrics if dataset has masks
    compute_causal_metrics = compute_causal_metrics and has_masks
    
    total_correct = 0
    total_samples = 0
    all_scores = []
    all_logits = []
    all_labels = []
    causal_precision_sum = 0
    causal_recall_sum = 0
    num_graphs_with_causal = 0
    
    for batch in loader:
        batch = prepare_batch(batch, device)
        
        # Handle DataParallel wrapped models
        if isinstance(model, nn.DataParallel):
            logits, _, _, scores = model.module.forward_single_view(
                batch.x, batch.edge_index, batch.batch
            )
        else:
            logits, _, _, scores = model.forward_single_view(
                batch.x, batch.edge_index, batch.batch
            )
        
        # Store for metric computation
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())
        
        # Accuracy computation
        if task_type == 'multiclass':
            pred = logits.argmax(dim=-1)
        else:  # binary
            if logits.dim() > 1 and logits.size(1) == 1:
                pred = (logits.squeeze(-1) > 0).long()
            elif logits.dim() > 1 and logits.size(1) == 2:
                pred = logits.argmax(dim=-1)
            else:
                pred = (logits > 0).long()
        
        labels = batch.y
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)
        
        # Collect attention scores
        all_scores.append(scores.cpu())
        
        # Causal node detection metrics (only for synthetic datasets with masks)
        if compute_causal_metrics and hasattr(batch, 'causal_mask'):
            pred_causal = (scores.squeeze() > 0.5).cpu()
            true_causal = batch.causal_mask.cpu()
            
            for graph_idx in range(batch.num_graphs):
                node_mask = batch.batch.cpu() == graph_idx
                pred_graph = pred_causal[node_mask]
                true_graph = true_causal[node_mask]
                
                if true_graph.sum() > 0:
                    if pred_graph.sum() > 0:
                        precision = (pred_graph & true_graph).sum().float() / pred_graph.sum()
                    else:
                        precision = 0.0
                    
                    recall = (pred_graph & true_graph).sum().float() / true_graph.sum()
                    
                    causal_precision_sum += precision
                    causal_recall_sum += recall
                    num_graphs_with_causal += 1
    
    all_scores = torch.cat(all_scores, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute primary metric
    metrics = {
        'accuracy': total_correct / total_samples,
        'mean_score': all_scores.mean().item(),
        'score_std': all_scores.std().item(),
        'sparsity': (all_scores < 0.5).float().mean().item()
    }
    
    # Compute AUC for binary classification tasks
    if metric_type == 'auc':
        try:
            auc_value = compute_metric(all_logits, all_labels, 'auc', evaluator)
            metrics['auc'] = auc_value
            metrics['primary_metric'] = auc_value
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc'] = 0.0
            metrics['primary_metric'] = metrics['accuracy']
    else:
        metrics['primary_metric'] = metrics['accuracy']
    
    # Causal detection metrics (only for datasets with ground-truth masks)
    if num_graphs_with_causal > 0:
        metrics['causal_precision'] = causal_precision_sum / num_graphs_with_causal
        metrics['causal_recall'] = causal_recall_sum / num_graphs_with_causal
        if metrics['causal_precision'] + metrics['causal_recall'] > 0:
            metrics['causal_f1'] = (
                2 * metrics['causal_precision'] * metrics['causal_recall'] /
                (metrics['causal_precision'] + metrics['causal_recall'])
            )
        else:
            metrics['causal_f1'] = 0.0
    
    return metrics


# =============================================================================
# Intervention Augmenter for Robustness Testing
# =============================================================================

class InterventionAugmenter:
    """
    Creates intervention graphs by swapping spurious backgrounds with
    conflicting ones to test model robustness.
    
    For the Spurious-Motif benchmark:
    - Original correlation: House→Star, Cycle→Wheel, Grid→Ladder
    - Conflicting interventions swap to backgrounds from OTHER classes
    
    If model is robust (uses causal features), predictions stay stable.
    If model relies on spurious features, predictions change.
    
    Args:
        conflict_mode: 'random' (random different background) or 
                       'systematic' (specific mapping)
    """
    
    # Spurious motif generators
    SPURIOUS_GENERATORS = None  # Will be set from spurious_motif_dataset
    
    def __init__(self, conflict_mode: str = 'random'):
        self.conflict_mode = conflict_mode
        
        # Import generators
        from spurious_motif_dataset import (generate_ba_base_graph,
                                            generate_ladder_motif,
                                            generate_star_motif,
                                            generate_wheel_motif)
        
        self.spurious_generators = [
            generate_star_motif,
            generate_wheel_motif, 
            generate_ladder_motif
        ]
        self.generate_base = generate_ba_base_graph
        
        # Conflict mapping: for each spurious type, map to conflicting types
        # E.g., Star (0) → Wheel (1) or Ladder (2)
        self.conflict_map = {
            0: [1, 2],  # Star conflicts with Wheel, Ladder
            1: [0, 2],  # Wheel conflicts with Star, Ladder
            2: [0, 1],  # Ladder conflicts with Star, Wheel
        }
    
    def _generate_conflicting_spurious(
        self,
        original_spurious_type: int,
        device: torch.device
    ) -> Tuple[list, int, int]:
        """
        Generate a conflicting spurious motif.
        
        Args:
            original_spurious_type: The original spurious motif type (0, 1, 2)
            device: Device for tensors
            
        Returns:
            edges: Edge list for the new spurious motif
            num_nodes: Number of nodes in the spurious motif
            conflict_type: The type of the conflicting motif
        """
        import random

        # Select a conflicting type
        if self.conflict_mode == 'random':
            conflict_type = random.choice(self.conflict_map[original_spurious_type])
        else:
            # Systematic: always choose the first conflict
            conflict_type = self.conflict_map[original_spurious_type][0]
        
        # Generate the conflicting motif
        edges, num_nodes = self.spurious_generators[conflict_type]()
        
        return edges, num_nodes, conflict_type
    
    def create_intervention(
        self,
        data,
        device: torch.device
    ):
        """
        Create an intervention graph by replacing spurious background.
        
        Algorithm:
        1. Identify causal nodes (from ground truth mask)
        2. Extract causal subgraph
        3. Generate new base graph
        4. Generate conflicting spurious motif
        5. Combine: causal + new base + conflicting spurious
        
        Args:
            data: Original graph Data object
            device: Device for tensors
            
        Returns:
            intervention_data: New graph with conflicting background
            new_spurious_type: The type of the new spurious motif
        """
        import random

        from torch_geometric.data import Data

        # Get original spurious type
        orig_spurious_type = data.spurious_motif.item() if hasattr(data, 'spurious_motif') else 0
        
        # Get causal nodes
        if hasattr(data, 'causal_mask'):
            causal_mask = data.causal_mask
        else:
            # Fallback: use attention if available, otherwise random
            causal_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            k = max(3, data.num_nodes // 5)
            causal_mask[:k] = True
        
        causal_indices = torch.where(causal_mask)[0]
        num_causal = len(causal_indices)
        
        if num_causal == 0:
            # No causal nodes, return original
            return data, orig_spurious_type
        
        # Extract causal subgraph edges
        old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
        old_to_new[causal_indices] = torch.arange(num_causal, device=device)
        
        src, dst = data.edge_index
        causal_edge_mask = causal_mask[src] & causal_mask[dst]
        causal_edges = data.edge_index[:, causal_edge_mask]
        causal_edge_index = old_to_new[causal_edges]
        
        # Extract causal features
        causal_x = data.x[causal_indices].clone()
        
        # Generate new base graph
        num_base_nodes = max(5, data.num_nodes - num_causal - 6)  # Approximate
        base_edges, num_base = self.generate_base(num_base_nodes, m=2)
        
        # Generate conflicting spurious motif
        spurious_edges, num_spurious, new_spurious_type = self._generate_conflicting_spurious(
            orig_spurious_type, device
        )
        
        # Combine graphs
        # Node layout: [causal nodes] [base nodes] [spurious nodes]
        total_nodes = num_causal + num_base + num_spurious
        
        # Offset edges
        base_offset = num_causal
        spurious_offset = num_causal + num_base
        
        base_edge_list = [(u + base_offset, v + base_offset) for u, v in base_edges]
        spurious_edge_list = [(u + spurious_offset, v + spurious_offset) for u, v in spurious_edges]
        
        # Combine all edges
        all_edges = []
        
        # Causal edges
        for i in range(causal_edge_index.size(1)):
            all_edges.append((causal_edge_index[0, i].item(), causal_edge_index[1, i].item()))
        
        # Base edges
        all_edges.extend(base_edge_list)
        
        # Spurious edges
        all_edges.extend(spurious_edge_list)
        
        # Connect causal to base
        if num_causal > 0 and num_base > 0:
            causal_node = random.randint(0, num_causal - 1)
            base_node = base_offset + random.randint(0, num_base - 1)
            all_edges.append((causal_node, base_node))
            all_edges.append((base_node, causal_node))
        
        # Connect base to spurious
        if num_base > 0 and num_spurious > 0:
            base_node = base_offset + random.randint(0, num_base - 1)
            spurious_node = spurious_offset + random.randint(0, num_spurious - 1)
            all_edges.append((base_node, spurious_node))
            all_edges.append((spurious_node, base_node))
        
        # Create edge index
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Create features (constant with small noise, NO type information!)
        # This ensures the model must learn from topology, not feature shortcuts
        feature_dim = data.x.size(1)
        x = torch.ones(total_nodes, feature_dim, device=device)
        x = x + 0.1 * torch.randn(total_nodes, feature_dim, device=device)
        x[:num_causal] = causal_x  # Preserve causal features
        
        # Create new causal mask
        new_causal_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
        new_causal_mask[:num_causal] = True
        
        # Create intervention data
        intervention_data = Data(
            x=x,
            edge_index=edge_index,
            y=data.y,  # Label stays the same!
            num_nodes=total_nodes,
            causal_mask=new_causal_mask,
            original_spurious=torch.tensor([orig_spurious_type], device=device),
            new_spurious=torch.tensor([new_spurious_type], device=device)
        )
        
        if hasattr(data, 'causal_motif'):
            intervention_data.causal_motif = data.causal_motif
        
        return intervention_data, new_spurious_type
    
    def create_batch_interventions(
        self,
        batch,
        device: torch.device
    ):
        """
        Create interventions for a batch of graphs.
        
        Args:
            batch: Batched graph data
            device: Device for tensors
            
        Returns:
            intervention_batch: Batch of intervention graphs
            spurious_changes: List of (original, new) spurious type tuples
        """
        from torch_geometric.data import Batch, Data
        
        intervention_list = []
        spurious_changes = []
        
        for graph_idx in range(batch.num_graphs):
            # Extract single graph
            node_mask = batch.batch == graph_idx
            graph_node_indices = torch.where(node_mask)[0]
            
            graph_x = batch.x[node_mask]
            
            edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
            graph_edges = batch.edge_index[:, edge_mask]
            min_idx = graph_node_indices.min()
            graph_edge_index = graph_edges - min_idx
            
            graph_y = batch.y[graph_idx]
            
            single_graph = Data(
                x=graph_x,
                edge_index=graph_edge_index,
                y=graph_y.unsqueeze(0) if graph_y.dim() == 0 else graph_y,
                num_nodes=node_mask.sum().item()
            )
            
            if hasattr(batch, 'causal_mask'):
                single_graph.causal_mask = batch.causal_mask[node_mask]
            if hasattr(batch, 'spurious_motif'):
                single_graph.spurious_motif = batch.spurious_motif[graph_idx]
            if hasattr(batch, 'causal_motif'):
                single_graph.causal_motif = batch.causal_motif[graph_idx]
            
            # Create intervention
            intervention_data, new_spurious = self.create_intervention(single_graph, device)
            intervention_list.append(intervention_data)
            
            orig_spurious = single_graph.spurious_motif.item() if hasattr(single_graph, 'spurious_motif') else 0
            spurious_changes.append((orig_spurious, new_spurious))
        
        # Batch interventions
        intervention_batch = Batch.from_data_list(intervention_list)
        
        return intervention_batch, spurious_changes


# =============================================================================
# Comprehensive Model Testing
# =============================================================================

@torch.no_grad()
def test_model(
    model: CausalGNNWithCrossView,
    train_dataset,
    test_dataset,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with intervention testing.
    
    Calculates:
    1. Accuracy on Training Set (biased)
    2. Accuracy on OOD Test Set (unbiased)
    3. Intervention Accuracy: Stability under spurious background swaps
    
    Intervention Accuracy measures robustness:
    - High intervention accuracy = model uses causal features
    - Low intervention accuracy = model relies on spurious correlations
    
    Args:
        model: Trained model
        train_dataset: Training dataset (biased)
        test_dataset: Test dataset (unbiased/OOD)
        device: Device for computation
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    
    # Handle DataParallel wrapped models
    model_core = model.module if isinstance(model, nn.DataParallel) else model
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create intervention augmenter
    intervention_augmenter = InterventionAugmenter(conflict_mode='random')
    
    results = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
    
    # =========================================================================
    # 1. Training Set Accuracy (Biased)
    # =========================================================================
    
    if verbose:
        print("\n[1] Evaluating on Training Set (Biased)...")
    
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        batch = prepare_batch(batch, device)
        logits, _, _, _ = model_core.forward_single_view(
            batch.x, batch.edge_index, batch.batch
        )
        pred = logits.argmax(dim=-1)
        train_correct += (pred == batch.y).sum().item()
        train_total += batch.y.size(0)
    
    results['train_accuracy'] = train_correct / train_total
    
    if verbose:
        print(f"    Training Accuracy: {results['train_accuracy']:.4f}")
    
    # =========================================================================
    # 2. OOD Test Set Accuracy (Unbiased)
    # =========================================================================
    
    if verbose:
        print("\n[2] Evaluating on OOD Test Set (Unbiased)...")
    
    test_correct = 0
    test_total = 0
    
    # Also collect per-class accuracy
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    for batch in test_loader:
        batch = prepare_batch(batch, device)
        logits, _, _, _ = model_core.forward_single_view(
            batch.x, batch.edge_index, batch.batch
        )
        pred = logits.argmax(dim=-1)
        
        for i in range(batch.y.size(0)):
            label = batch.y[i].item()
            is_correct = (pred[i] == batch.y[i]).item()
            test_correct += is_correct
            test_total += 1
            class_correct[label] += is_correct
            class_total[label] += 1
    
    results['test_accuracy'] = test_correct / test_total
    results['test_accuracy_class_0'] = class_correct[0] / max(class_total[0], 1)
    results['test_accuracy_class_1'] = class_correct[1] / max(class_total[1], 1)
    results['test_accuracy_class_2'] = class_correct[2] / max(class_total[2], 1)
    
    if verbose:
        print(f"    OOD Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"    Per-class: House={results['test_accuracy_class_0']:.4f}, "
              f"Cycle={results['test_accuracy_class_1']:.4f}, "
              f"Grid={results['test_accuracy_class_2']:.4f}")
    
    # =========================================================================
    # 3. Intervention Accuracy (Robustness Test)
    # =========================================================================
    
    if verbose:
        print("\n[3] Evaluating Intervention Accuracy (Robustness)...")
        print("    Swapping spurious backgrounds to test prediction stability...")
    
    # Metrics for intervention testing
    intervention_stable = 0  # Predictions unchanged after intervention
    intervention_correct = 0  # Predictions correct after intervention
    intervention_total = 0
    
    # Track stability by original spurious correlation type
    stability_matching = 0  # When orig_spurious == label (matching correlation)
    stability_non_matching = 0
    total_matching = 0
    total_non_matching = 0
    
    for batch in test_loader:
        batch = prepare_batch(batch, device)
        
        # Get original predictions
        logits_orig, _, _, _ = model_core.forward_single_view(
            batch.x, batch.edge_index, batch.batch
        )
        pred_orig = logits_orig.argmax(dim=-1)
        
        # Create interventions (swap spurious backgrounds)
        intervention_batch, spurious_changes = intervention_augmenter.create_batch_interventions(
            batch, device
        )
        intervention_batch = prepare_batch(intervention_batch, device)
        
        # Get intervention predictions
        logits_interv, _, _, _ = model_core.forward_single_view(
            intervention_batch.x, 
            intervention_batch.edge_index, 
            intervention_batch.batch
        )
        pred_interv = logits_interv.argmax(dim=-1)
        
        # Compare predictions
        for i in range(batch.y.size(0)):
            label = batch.y[i].item()
            original_pred = pred_orig[i].item()
            intervention_pred = pred_interv[i].item()
            
            orig_spurious, new_spurious = spurious_changes[i]
            
            # Is prediction stable (unchanged)?
            is_stable = (original_pred == intervention_pred)
            intervention_stable += is_stable
            
            # Is intervention prediction correct?
            is_correct = (intervention_pred == label)
            intervention_correct += is_correct
            
            intervention_total += 1
            
            # Track by matching/non-matching correlation
            if orig_spurious == label:
                # Original had matching spurious correlation
                total_matching += 1
                stability_matching += is_stable
            else:
                # Original had non-matching spurious correlation
                total_non_matching += 1
                stability_non_matching += is_stable
    
    results['intervention_stability'] = intervention_stable / max(intervention_total, 1)
    results['intervention_accuracy'] = intervention_correct / max(intervention_total, 1)
    results['intervention_total'] = intervention_total
    
    # Stability breakdown
    if total_matching > 0:
        results['stability_matching'] = stability_matching / total_matching
    else:
        results['stability_matching'] = 0.0
        
    if total_non_matching > 0:
        results['stability_non_matching'] = stability_non_matching / total_non_matching
    else:
        results['stability_non_matching'] = 0.0
    
    if verbose:
        print(f"    Intervention Stability: {results['intervention_stability']:.4f}")
        print(f"    (Fraction of predictions unchanged after spurious swap)")
        print(f"    Intervention Accuracy: {results['intervention_accuracy']:.4f}")
        print(f"    (Accuracy on intervened graphs)")
        print(f"\n    Stability Breakdown:")
        print(f"      When orig spurious matched label: {results['stability_matching']:.4f}")
        print(f"      When orig spurious didn't match:  {results['stability_non_matching']:.4f}")
    
    # =========================================================================
    # 4. Generalization Gap Analysis
    # =========================================================================
    
    if verbose:
        print("\n[4] Generalization Analysis...")
    
    # Gap between train and test
    results['generalization_gap'] = results['train_accuracy'] - results['test_accuracy']
    
    # Robustness score: combination of OOD accuracy and intervention stability
    results['robustness_score'] = (
        0.5 * results['test_accuracy'] + 
        0.5 * results['intervention_stability']
    )
    
    if verbose:
        print(f"    Generalization Gap: {results['generalization_gap']:.4f}")
        print(f"    (Train Acc - Test Acc, lower is better)")
        print(f"    Robustness Score: {results['robustness_score']:.4f}")
        print(f"    (Combined OOD + Intervention stability)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    if verbose:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"\n  Training Accuracy (Biased):     {results['train_accuracy']:.4f}")
        print(f"  OOD Test Accuracy (Unbiased):   {results['test_accuracy']:.4f}")
        print(f"  Intervention Stability:         {results['intervention_stability']:.4f}")
        print(f"  Intervention Accuracy:          {results['intervention_accuracy']:.4f}")
        print(f"  Robustness Score:               {results['robustness_score']:.4f}")
        print("\n  Interpretation:")
        
        if results['intervention_stability'] > 0.8:
            print("  ✓ HIGH stability - Model likely uses causal features")
        elif results['intervention_stability'] > 0.5:
            print("  ~ MODERATE stability - Model partially relies on spurious features")
        else:
            print("  ✗ LOW stability - Model heavily relies on spurious correlations")
        
        if results['generalization_gap'] < 0.1:
            print("  ✓ GOOD generalization - Small train-test gap")
        elif results['generalization_gap'] < 0.3:
            print("  ~ MODERATE generalization gap")
        else:
            print("  ✗ POOR generalization - Large train-test gap (overfitting)")
        
        print("=" * 70)
    
    return results


def run_full_evaluation(
    model_path: str,
    dataset: str = 'spurious_motif',
    data_root: str = './data',
    bias: float = 0.9,
    num_graphs: int = 1000,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Load a saved model and run full evaluation.
    
    Args:
        model_path: Path to saved model checkpoint
        dataset: Dataset name for loading
        data_root: Root directory for dataset
        bias: Spurious correlation strength (for spurious_motif)
        num_graphs: Number of graphs in dataset
        device: Device for computation
        
    Returns:
        Evaluation results dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset via factory
    data = get_dataset(
        name=dataset,
        root=data_root,
        batch_size=32,
        bias=bias,
        num_graphs=num_graphs,
        num_workers=0
    )
    
    train_dataset = data.get('train_dataset')
    test_dataset = data.get('test_dataset')
    input_dim = data['input_dim']
    output_dim = data['output_dim']
    
    # Detect if we're using OGB molecular dataset
    is_ogb = dataset.lower().startswith('ogb')
    
    # Create model
    model = CausalGNNWithCrossView(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        use_atom_encoder=is_ogb
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Run evaluation (only for datasets with masks)
    if data['has_masks'] and train_dataset is not None and test_dataset is not None:
        results = test_model(model, train_dataset, test_dataset, device)
    else:
        # For real datasets, just return evaluation metrics
        val_loader = data['val_loader']
        test_loader = data['test_loader']
        
        data_info = {
            'metric': data['metric'],
            'has_masks': data['has_masks'],
            'task_type': data.get('task_type', 'multiclass'),
            'evaluator': data.get('evaluator', None)
        }
        
        val_metrics = evaluate(model, val_loader, device, data_info=data_info)
        test_metrics = evaluate(model, test_loader, device, data_info=data_info)
        
        results = {
            'val_accuracy': val_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'val_metric': val_metrics.get('primary_metric', val_metrics['accuracy']),
            'test_metric': test_metrics.get('primary_metric', test_metrics['accuracy']),
            'sparsity': test_metrics['sparsity']
        }
    
    return results


# =============================================================================
# Main Training Loop
# =============================================================================

def train(
    args: argparse.Namespace
) -> Tuple[CausalGNNWithCrossView, Dict]:
    """
    Main training function with multi-dataset and multi-GPU support.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 70)
    print("  CAUSAL GNN TRAINING WITH COUNTERFACTUAL AUGMENTATION")
    print("=" * 70)
    
    # =========================================================================
    # Device Setup (Multi-GPU Support)
    # =========================================================================
    
    num_gpus = torch.cuda.device_count()
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    print(f"Available GPUs: {num_gpus}")
    
    if torch.cuda.is_available():
        # Print GPU info
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Auto-enable multi-GPU if multiple GPUs available (unless --single_gpu is set)
        use_multi_gpu = (args.multi_gpu or num_gpus > 1) and not args.single_gpu
        
        if use_multi_gpu and num_gpus > 1:
            # Parse GPU IDs
            gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
            gpu_ids = [g for g in gpu_ids if g < num_gpus]
            
            if len(gpu_ids) > 1:
                device = torch.device(f'cuda:{gpu_ids[0]}')
                print(f"\n✓ Using DataParallel with GPUs: {gpu_ids}")
            else:
                device = torch.device('cuda:0')
                gpu_ids = None
                print(f"\n✓ Device: {device}")
        else:
            device = torch.device('cuda:0')
            gpu_ids = None
            print(f"\n✓ Device: {device}")
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"  cuDNN benchmark: enabled")
    else:
        device = torch.device('cpu')
        gpu_ids = None
        print("\n⚠ Device: CPU (no GPU available)")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # =========================================================================
    # Load Dataset via Factory
    # =========================================================================
    
    print(f"\n[1] Loading dataset: {args.dataset}")
    
    # Build dataset kwargs based on dataset type
    dataset_kwargs = {
        'num_workers': args.num_workers,
        'seed': args.seed
    }
    
    if args.dataset == 'spurious_motif':
        dataset_kwargs['bias'] = args.bias
        dataset_kwargs['num_graphs'] = args.num_graphs
        print(f"    Bias: {args.bias}")
    elif args.dataset.startswith('visual_genome'):
        dataset_kwargs['max_graphs'] = args.num_graphs
        dataset_kwargs['feature_dim'] = 64
    
    # Load dataset via factory
    data = get_dataset(
        name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        **dataset_kwargs
    )
    
    # Extract loaders and info
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    input_dim = data['input_dim']
    output_dim = data['output_dim']
    
    # Store dataset info for later use
    data_info = {
        'metric': data['metric'],
        'has_masks': data['has_masks'],
        'task_type': data.get('task_type', 'multiclass'),
        'evaluator': data.get('evaluator', None)
    }
    
    # Print dataset info
    print_dataset_info(data)
    
    # =========================================================================
    # Create Model and Components
    # =========================================================================
    
    print(f"\n[2] Creating model...")
    
    # Detect if we're using OGB molecular dataset (needs AtomEncoder)
    is_ogb = args.dataset.lower().startswith('ogb')
    
    model = CausalGNNWithCrossView(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_gin_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_cross_view=args.use_cross_view,
        use_atom_encoder=is_ogb  # Use AtomEncoder for OGB molecules
    ).to(device)

    print("\n[DEBUG] FORCE-RESETTING Causal Attention Weights...")
    
    # Access the attention module (before DataParallel wrapping)
    # Note: If you wrapped it in DataParallel already, use model.module.causal_attention
    if hasattr(model, 'module'):
        attn_layer = model.module.causal_attention
    else:
        attn_layer = model.causal_attention
    
    # 1. Force weights to ZERO (so features don't affect the start)
    torch.nn.init.zeros_(attn_layer.attention_mlp[-1].weight)
    
    # 2. Force bias to 2.0 (so everyone starts with ~88% score)
    torch.nn.init.constant_(attn_layer.attention_mlp[-1].bias, 2.0)
    
    print("    [DEBUG] Weights set to 0.0")
    print(f"    [DEBUG] Bias set to: {attn_layer.attention_mlp[-1].bias.mean().item()}")
    print("=========================================================================\n")
    
    # Wrap with DataParallel if using multiple GPUs
    if gpu_ids is not None and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"    Model wrapped with DataParallel on GPUs: {gpu_ids}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {num_params:,}")
    print(f"    Cross-View Interaction: {args.use_cross_view}")
    print(f"    Dataset metric: {data_info['metric']}")
    print(f"    Has causal masks: {data_info['has_masks']}")
    
    # Create augmenter
    augmenter = CausalAugmenter(
        threshold=args.causal_threshold,
        noise_type='erdos_renyi',
        edge_prob=0.2,
        num_connections=2
    )
    print(f"    Causal Threshold: {args.causal_threshold}")
    
    # Create intervention augmenter (for robust training)
    intervention_augmenter = None
    if args.use_intervention_training and data_info['has_masks']:
        intervention_augmenter = InterventionAugmenter(conflict_mode='random')
        print(f"    Intervention Training: ENABLED (breaking spurious correlations)")
    elif args.use_intervention_training:
        print(f"    Intervention Training: DISABLED (dataset has no causal masks)")
    
    # Create loss function
    criterion = CausalContrastiveLoss(
        classification_weight=args.cls_weight,
        cf_classification_weight=args.cf_cls_weight,
        contrastive_weight=args.contrastive_weight,
        diversity_weight=args.diversity_weight,
        sparsity_weight=args.sparsity_weight,
        coverage_weight=args.coverage_weight,
        entropy_weight=args.entropy_weight,
        supervision_weight=args.supervision_weight if data_info['has_masks'] else 0.0,
        target_sparsity=args.target_sparsity,
        temperature=args.temperature
    )
    
    print(f"\n    Loss weights:")
    print(f"      Classification: {args.cls_weight}")
    print(f"      CF Classification: {args.cf_cls_weight} (INFO BOTTLENECK)")
    print(f"      Contrastive: {args.contrastive_weight}")
    print(f"      Diversity: {args.diversity_weight} (penalizes G_cf ≈ G)")
    print(f"      Sparsity: {args.sparsity_weight}")
    print(f"      Coverage: {args.coverage_weight} (target: {args.target_sparsity:.0%})")
    print(f"      Entropy: {args.entropy_weight} (pushes to binary 0/1)")
    if data_info['has_masks'] and args.supervision_weight > 0:
        print(f"      Supervision: {args.supervision_weight} (using causal masks)")
    
    # Create save directory if needed
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(1, args.epochs // 3),  # Ensure T_0 >= 1
        T_mult=2
    )
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    print(f"\n[3] Training for {args.epochs} epochs...")
    print(f"    Warmup epochs: {args.warmup_epochs}")
    
    # Determine primary metric name for tracking
    primary_metric_name = 'auc' if data_info['metric'] == 'auc' else 'accuracy'
    
    history: Dict[str, Any] = {
        'train_loss': [],
        'train_acc': [],
        'val_metric': [],
        'test_metric': [],
        'val_acc': [],
        'test_acc': [],
        'contrastive_loss': [],
        'sparsity': [],
        'causal_f1': [],
        'final_robustness': None  # Will be populated after training
    }
    
    best_val_metric = 0
    best_test_metric = 0
    best_epoch = 0
    
    # GPU memory tracking helper
    def get_gpu_memory_info():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"GPU Mem: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved"
        return ""
    
    start_time = time.time()
    
    # Print initial GPU memory
    if torch.cuda.is_available():
        print(f"    Initial {get_gpu_memory_info()}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        epoch_loss = 0
        epoch_acc = 0
        epoch_contrastive = 0
        num_batches = 0
        
        for batch in train_loader:
            loss_dict = train_step(
                model=model,
                batch=batch,
                augmenter=augmenter,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                warmup_epochs=args.warmup_epochs,
                current_epoch=epoch,
                data_info=data_info,
                intervention_augmenter=intervention_augmenter if args.use_intervention_training else None,
                use_intervention_training=args.use_intervention_training,
                intervention_weight=args.intervention_weight
            )
            
            epoch_loss += loss_dict['total']
            epoch_acc += loss_dict['acc_orig']
            epoch_contrastive += loss_dict.get('contrastive', 0.0)
            num_batches += 1
        
        scheduler.step()
        
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        epoch_contrastive /= num_batches
        
        # Evaluation with dataset-appropriate metrics
        val_metrics = evaluate(model, val_loader, device, data_info=data_info)
        test_metrics = evaluate(model, test_loader, device, data_info=data_info)
        
        # Get primary metric value
        val_metric = val_metrics.get('primary_metric', val_metrics['accuracy'])
        test_metric = test_metrics.get('primary_metric', test_metrics['accuracy'])
        
        # Track best based on primary metric
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
            best_epoch = epoch
            
            # Save best model
            if args.save_model:
                # Handle DataParallel - save unwrapped model
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), f'{args.save_dir}/best_model.pt')
        
        # History
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_metric'].append(val_metric)
        history['test_metric'].append(test_metric)
        history['val_acc'].append(val_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['contrastive_loss'].append(epoch_contrastive)
        history['sparsity'].append(val_metrics['sparsity'])
        if 'causal_f1' in val_metrics:
            history['causal_f1'].append(val_metrics['causal_f1'])
        
        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            epoch_time = time.time() - epoch_start
            
            # Build log string based on metric type
            if data_info['metric'] == 'auc':
                log_str = (
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Train Acc: {epoch_acc:.4f} | "
                    f"Val AUC: {val_metric:.4f} | "
                    f"Test AUC: {test_metric:.4f} | "
                    f"Sparsity: {val_metrics['sparsity']:.2f} | "
                    f"Time: {epoch_time:.1f}s"
                )
            else:
                log_str = (
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Train: {epoch_acc:.4f} | "
                    f"Val: {val_metrics['accuracy']:.4f} | "
                    f"Test: {test_metrics['accuracy']:.4f} | "
                    f"Sparsity: {val_metrics['sparsity']:.2f} | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Only show causal F1 for datasets with masks
            if 'causal_f1' in val_metrics:
                log_str += f" | Causal F1: {val_metrics['causal_f1']:.4f}"
            
            print(log_str)
            
            # Print GPU memory usage periodically (every 10 log intervals)
            if torch.cuda.is_available() and epoch % (args.log_every * 10) == 0:
                print(f"    {get_gpu_memory_info()}")
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # Final Results
    # =========================================================================
    
    metric_name = 'AUC' if data_info['metric'] == 'auc' else 'Accuracy'
    
    print(f"\n[4] Training Complete!")
    print(f"    Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"    Best epoch: {best_epoch}")
    print(f"    Best Val {metric_name}: {best_val_metric:.4f}")
    print(f"    Best Test {metric_name}: {best_test_metric:.4f}")
    if torch.cuda.is_available():
        print(f"    Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Final evaluation
    print(f"\n[5] Final Evaluation...")
    
    train_metrics = evaluate(model, train_loader, device, data_info=data_info)
    val_metrics = evaluate(model, val_loader, device, data_info=data_info)
    test_metrics = evaluate(model, test_loader, device, data_info=data_info)
    
    print(f"    Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    if data_info['metric'] == 'auc':
        print(f"    Train AUC: {train_metrics.get('auc', 0):.4f}")
        print(f"    Val AUC: {val_metrics.get('auc', 0):.4f}")
        print(f"    Test AUC: {test_metrics.get('auc', 0):.4f}")
    
    print(f"    Attention Sparsity: {test_metrics['sparsity']:.4f}")
    
    # Only show causal metrics for datasets with ground-truth masks
    if 'causal_f1' in test_metrics:
        print(f"    Causal Node Detection:")
        print(f"      Precision: {test_metrics['causal_precision']:.4f}")
        print(f"      Recall: {test_metrics['causal_recall']:.4f}")
        print(f"      F1: {test_metrics['causal_f1']:.4f}")
    
    # =========================================================================
    # Comprehensive Robustness Evaluation (only for synthetic datasets)
    # =========================================================================
    
    robustness_results = None
    
    if data_info['has_masks']:
        # Only run intervention-based robustness evaluation for synthetic datasets
        # that have ground-truth causal masks
        print(f"\n[6] Comprehensive Robustness Evaluation...")
        print(f"    (Available for datasets with causal masks)")
        
        try:
            # Get datasets from factory data
            train_dataset = data.get('train_dataset')
            test_dataset = data.get('test_dataset')
            
            if train_dataset is not None and test_dataset is not None:
                robustness_results = test_model(
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    device=device,
                    batch_size=args.batch_size,
                    verbose=True
                )
                history['final_robustness'] = robustness_results
            else:
                print("    Skipping: Dataset objects not available")
        except Exception as e:
            print(f"    Skipping: {e}")
    else:
        print(f"\n[6] Robustness Evaluation skipped")
        print(f"    (Not available for real datasets without causal masks)")
        print(f"    For real datasets, evaluate OOD generalization via test set performance.")
    
    # =========================================================================
    # Save Results and Plots
    # =========================================================================
    
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Save final model (handle DataParallel)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), f'{args.save_dir}/final_model.pt')
        
        # Save dataset info for loading
        torch.save({
            'data_info': data_info,
            'args': vars(args),
            'input_dim': input_dim,
            'output_dim': output_dim
        }, f'{args.save_dir}/config.pt')
        
        # Save history
        torch.save(history, f'{args.save_dir}/history.pt')
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['contrastive_loss'], label='Contrastive Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Primary Metric (Accuracy or AUC)
        if data_info['metric'] == 'auc':
            axes[0, 1].plot(history['val_metric'], label='Val AUC')
            axes[0, 1].plot(history['test_metric'], label='Test AUC')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].set_title('ROC-AUC')
        else:
            axes[0, 1].plot(history['train_acc'], label='Train')
            axes[0, 1].plot(history['val_acc'], label='Val')
            axes[0, 1].plot(history['test_acc'], label='Test')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sparsity
        axes[1, 0].plot(history['sparsity'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Sparsity')
        axes[1, 0].set_title('Attention Sparsity (% nodes < 0.5)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Causal F1 (only for datasets with masks)
        if history['causal_f1']:
            axes[1, 1].plot(history['causal_f1'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Causal Node Detection F1')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show accuracy comparison for real datasets
            axes[1, 1].plot(history['val_acc'], label='Val')
            axes[1, 1].plot(history['test_acc'], label='Test')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Validation vs Test Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{args.save_dir}/training_curves.png', dpi=150)
        plt.close()
        
        print(f"\n    Saved results to {args.save_dir}/")
    
    print("\n" + "=" * 70)
    
    return model, history


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Causal GNN with Counterfactual Augmentation'
    )
    
    # Dataset selection (NEW: multi-dataset support)
    parser.add_argument('--dataset', type=str, default='spurious_motif',
                        choices=['spurious_motif', 'ogbg-molhiv', 'ogbg-molbbbp', 
                                 'good-hiv', 'visual_genome', 'visual_genome_synthetic'],
                        help='Dataset to use for training')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    
    # Dataset-specific options
    parser.add_argument('--num_graphs', type=int, default=1000,
                        help='Number of graphs (for synthetic datasets)')
    parser.add_argument('--bias', type=float, default=0.9,
                        help='Spurious correlation strength (for spurious_motif)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GIN layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--use_cross_view', action='store_true', default=True,
                        help='Use cross-view interaction')
    
    # Augmentation
    parser.add_argument('--causal_threshold', type=float, default=0.5,
                        help='Threshold for causal node selection')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs with random augmentation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Loss weights
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--cf_cls_weight', type=float, default=0.5,
                        help='Counterfactual classification loss weight (creates info bottleneck)')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Contrastive loss weight (reduced to allow diversity)')
    parser.add_argument('--diversity_weight', type=float, default=0.1,
                        help='Diversity loss weight (penalizes G_cf too similar to G)')
    parser.add_argument('--use_intervention_training', action='store_true',
                        help='Use intervention-based training (swaps backgrounds during training)')
    parser.add_argument('--intervention_weight', type=float, default=1.0,
                        help='Weight for intervention classification loss')
    parser.add_argument('--sparsity_weight', type=float, default=0.3,
                        help='Sparsity loss weight (higher to prevent selecting everything)')
    parser.add_argument('--entropy_weight', type=float, default=0.1,
                        help='Entropy loss weight (pushes scores to 0 or 1, prevents gray area)')
    parser.add_argument('--coverage_weight', type=float, default=0.1,
                        help='Coverage loss weight (prevents attention collapse)')
    parser.add_argument('--target_sparsity', type=float, default=0.3,
                        help='Target fraction of nodes to select (0.3 = 30%%)')
    parser.add_argument('--supervision_weight', type=float, default=0.0,
                        help='Weight for causal mask supervision (0.0 for real research, >0 for debugging)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss')
    
    # Multi-GPU (auto-enabled when multiple GPUs available)
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Force DataParallel (auto-enabled if 2+ GPUs available)')
    parser.add_argument('--single_gpu', action='store_true', default=False,
                        help='Force single GPU even if multiple available')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                        help='Comma-separated GPU IDs to use (default: 0,1)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Log every N epochs')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save model checkpoints')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model, history = train(args)

