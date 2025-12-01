"""
GIN (Graph Isomorphism Network) with Causal Attention Pooling

This implementation includes:
- GIN encoder for node embeddings
- CausalAttention layer for node importance scoring
- Weighted graph pooling based on attention scores
- Combined loss with CrossEntropy + L1 sparsity penalty

Designed for use with the Spurious-Motif benchmark to discover
invariant rationales for graph classification.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

# OGB AtomEncoder for molecular datasets
try:
    from ogb.graphproppred.mol_encoder import AtomEncoder
    OGB_AVAILABLE = True
except ImportError:
    AtomEncoder = None
    OGB_AVAILABLE = False

# =============================================================================
# GIN Encoder
# =============================================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder for learning node embeddings.
    
    GIN uses MLPs to update node features and is proven to be as powerful
    as the Weisfeiler-Lehman graph isomorphism test.
    
    Supports both:
    - Continuous features (Spurious-Motif): Uses nn.Linear projection
    - Categorical features (OGB molecules): Uses AtomEncoder embedding
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of GIN layers
        dropout: Dropout probability
        eps: Initial epsilon value for GIN (learnable)
        train_eps: Whether to learn epsilon
        use_atom_encoder: Whether to use OGB AtomEncoder for molecular features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        eps: float = 0.0,
        train_eps: bool = True,
        use_atom_encoder: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_atom_encoder = use_atom_encoder
        
        # FEATURE ENCODING:
        # - AtomEncoder for OGB molecules (categorical atom types)
        # - Linear for continuous features (Spurious-Motif, etc.)
        if use_atom_encoder and OGB_AVAILABLE and AtomEncoder is not None:
            self.feature_encoder = AtomEncoder(hidden_dim)
            print("    Using OGB AtomEncoder for molecular features")
        else:
            self.feature_encoder = nn.Linear(input_dim, hidden_dim)
            if use_atom_encoder and not OGB_AVAILABLE:
                print("    Warning: AtomEncoder requested but OGB not installed. Using Linear.")
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # MLP for each GIN layer
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
            conv = GINConv(mlp, eps=eps, train_eps=train_eps)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GIN encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Handle features based on encoder type
        if self.use_atom_encoder:
            # AtomEncoder expects LongTensor (categorical integers)
            if x.dtype == torch.float32 or x.dtype == torch.float64:
                x = x.long()
            x = self.feature_encoder(x)
        else:
            # Linear expects FloatTensor (continuous)
            if x.dtype != torch.float32:
                x = x.float()
            x = self.feature_encoder(x)
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GIN message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# =============================================================================
# Causal Attention Layer
# =============================================================================

class CausalAttention(nn.Module):
    """
    Causal Attention layer for node importance scoring.
    
    This layer learns to assign an importance score s_i ∈ [0, 1] to each node,
    indicating whether the node is part of the "causal" subgraph that
    determines the graph label.
    
    The layer uses a small MLP followed by sigmoid to produce scores,
    encouraging the model to focus on a sparse set of important nodes.
    
    Args:
        hidden_dim: Dimension of node embeddings
        attention_heads: Number of attention heads (for multi-head variant)
        temperature: Temperature for sigmoid sharpening (higher = sharper)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int = 1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.temperature = temperature
        
        # Attention MLP: maps node embedding to importance score
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),  # LeakyReLU to prevent dead neurons
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize the final layer bias to output ~0.3-0.5 after sigmoid
        # sigmoid(0) = 0.5, so initialize bias to 0 for balanced start
        # But we want some nodes to have high scores, so use small positive bias
        nn.init.zeros_(self.attention_mlp[-1].weight)
        nn.init.constant_(self.attention_mlp[-1].bias, 2.0)  # sigmoid(0.5) ≈ 0.62
        
        # Optional: learnable temperature
        self.temp_param = nn.Parameter(torch.tensor(temperature))
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention scores for each node.
        
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            Attention scores s_i ∈ [0, 1] for each node [num_nodes, 1]
        """
        # Compute raw attention logits
        attn_logits = self.attention_mlp(x)

        attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
        # Apply temperature-scaled sigmoid to get scores in [0, 1]
        scores = torch.sigmoid(attn_logits * self.temp_param)
        
        return scores


# =============================================================================
# Causal Attention Pooling
# =============================================================================

class CausalAttentionPooling(nn.Module):
    """
    Pooling layer that combines CausalAttention with weighted aggregation.
    
    Computes graph-level embeddings as:
        h_G = Σ_i (s_i * h_i)
    
    where s_i is the attention score and h_i is the node embedding.
    
    Args:
        hidden_dim: Dimension of node embeddings
        temperature: Temperature for attention sigmoid
        normalize: Whether to normalize by sum of attention scores
    """
    
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.0,
        normalize: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.normalize = normalize
        
        self.attention = CausalAttention(
            hidden_dim=hidden_dim,
            temperature=temperature
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted graph embeddings.
        
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            graph_embed: Graph-level embeddings [batch_size, hidden_dim]
            scores: Node attention scores [num_nodes, 1]
        """
        # Get attention scores for each node
        scores = self.attention(x, batch)  # [num_nodes, 1]
        
        # Weighted node embeddings
        weighted_x = x * scores  # [num_nodes, hidden_dim]
        
        # Aggregate by graph (sum)
        graph_embed = global_add_pool(weighted_x, batch)  # [batch_size, hidden_dim]
        
        if self.normalize:
            # Normalize by sum of attention scores per graph
            score_sum = global_add_pool(scores, batch)  # [batch_size, 1]
            graph_embed = graph_embed / (score_sum + 1e-8)
        
        return graph_embed, scores


# =============================================================================
# Cross-View Interaction Module
# =============================================================================

class CrossViewInteraction(nn.Module):
    """
    Cross-View Interaction module using Multi-Head Attention.
    
    This module enables information exchange between two graph views
    (e.g., original and counterfactual graphs) using attention mechanism.
    View1 serves as Query, View2 serves as Key/Value.
    
    The module:
    1. Converts sparse graph representations to dense batched tensors
    2. Applies multi-head attention with proper masking for padding
    3. Converts output back to sparse format
    
    Args:
        hidden_dim: Dimension of node embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability for attention
        add_residual: Whether to add residual connection
        add_layer_norm: Whether to apply layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        add_residual: bool = True,
        add_layer_norm: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.add_residual = add_residual
        self.add_layer_norm = add_layer_norm
        
        # Multi-head attention: View1 (Q) attends to View2 (K, V)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input/output shape: (batch, seq, feature)
        )
        
        # Optional layer normalization
        if add_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Optional projection after attention
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if add_layer_norm:
            self.output_norm = nn.LayerNorm(hidden_dim)
    
    def _create_key_padding_mask(
        self,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create key padding mask for attention.
        
        In PyTorch MultiheadAttention, True means IGNORE this position.
        
        Args:
            mask: Boolean mask from to_dense_batch [batch, max_nodes]
                  True = valid node, False = padding
                  
        Returns:
            key_padding_mask: [batch, max_nodes] where True = padding (ignore)
        """
        # Invert: to_dense_batch returns True for valid, attention needs True for padding
        return ~mask
    
    def forward(
        self,
        x_view1: torch.Tensor,
        batch_view1: torch.Tensor,
        x_view2: torch.Tensor,
        batch_view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-view attention from View1 to View2.
        
        Args:
            x_view1: Node embeddings for View1 [num_nodes_v1, hidden_dim]
            batch_view1: Batch assignment for View1 [num_nodes_v1]
            x_view2: Node embeddings for View2 [num_nodes_v2, hidden_dim]
            batch_view2: Batch assignment for View2 [num_nodes_v2]
            
        Returns:
            x_out: Updated View1 embeddings [num_nodes_v1, hidden_dim]
            attn_weights: Attention weights [batch, num_heads, max_nodes_v1, max_nodes_v2]
        """
        # Convert sparse to dense batch format
        # x_dense: [batch_size, max_num_nodes, hidden_dim]
        # mask: [batch_size, max_num_nodes] where True = valid node
        x_view1_dense, mask_view1 = to_dense_batch(x_view1, batch_view1)
        x_view2_dense, mask_view2 = to_dense_batch(x_view2, batch_view2)
        
        batch_size = x_view1_dense.size(0)
        max_nodes_v1 = x_view1_dense.size(1)
        max_nodes_v2 = x_view2_dense.size(1)
        
        # Create key padding mask for View2 (the keys/values)
        # True means this position should be IGNORED
        key_padding_mask = self._create_key_padding_mask(mask_view2)
        
        # Apply multi-head attention
        # Query: View1, Key/Value: View2
        attn_output, attn_weights = self.multihead_attn(
            query=x_view1_dense,      # [batch, max_nodes_v1, hidden_dim]
            key=x_view2_dense,        # [batch, max_nodes_v2, hidden_dim]
            value=x_view2_dense,      # [batch, max_nodes_v2, hidden_dim]
            key_padding_mask=key_padding_mask,  # [batch, max_nodes_v2]
            need_weights=True,
            average_attn_weights=False  # Return per-head weights
        )
        # attn_output: [batch, max_nodes_v1, hidden_dim]
        # attn_weights: [batch, num_heads, max_nodes_v1, max_nodes_v2]
        
        # Residual connection (before converting back to sparse)
        if self.add_residual:
            attn_output = attn_output + x_view1_dense
        
        # Layer normalization
        if self.add_layer_norm:
            attn_output = self.layer_norm(attn_output)
        
        # Feed-forward projection
        ff_output = self.output_proj(attn_output)
        
        # Another residual + norm
        if self.add_residual:
            ff_output = ff_output + attn_output
        if self.add_layer_norm:
            ff_output = self.output_norm(ff_output)
        
        # Convert back to sparse format using mask
        # Flatten and select only valid (non-padding) nodes
        x_out = ff_output[mask_view1]  # [num_nodes_v1, hidden_dim]
        
        return x_out, attn_weights


class BidirectionalCrossViewInteraction(nn.Module):
    """
    Bidirectional Cross-View Interaction module.
    
    Applies cross-attention in both directions:
    - View1 attends to View2
    - View2 attends to View1
    
    This enables mutual information exchange between views.
    
    Args:
        hidden_dim: Dimension of node embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
        share_weights: Whether to share attention weights for both directions
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        share_weights: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.share_weights = share_weights
        
        # View1 -> View2 attention
        self.cross_attn_1to2 = CrossViewInteraction(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # View2 -> View1 attention
        if share_weights:
            self.cross_attn_2to1 = self.cross_attn_1to2
        else:
            self.cross_attn_2to1 = CrossViewInteraction(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Fusion layer to combine original and cross-attended features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x_view1: torch.Tensor,
        batch_view1: torch.Tensor,
        x_view2: torch.Tensor,
        batch_view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Apply bidirectional cross-view attention.
        
        Args:
            x_view1: Node embeddings for View1 [num_nodes_v1, hidden_dim]
            batch_view1: Batch assignment for View1 [num_nodes_v1]
            x_view2: Node embeddings for View2 [num_nodes_v2, hidden_dim]
            batch_view2: Batch assignment for View2 [num_nodes_v2]
            
        Returns:
            x_view1_out: Updated View1 embeddings
            x_view2_out: Updated View2 embeddings
            attn_info: Dictionary with attention weights
        """
        # Cross-attention: View1 queries View2
        x_view1_crossed, attn_1to2 = self.cross_attn_1to2(
            x_view1, batch_view1, x_view2, batch_view2
        )
        
        # Cross-attention: View2 queries View1
        x_view2_crossed, attn_2to1 = self.cross_attn_2to1(
            x_view2, batch_view2, x_view1, batch_view1
        )
        
        # Fuse original and cross-attended features
        x_view1_fused = torch.cat([x_view1, x_view1_crossed], dim=-1)
        x_view2_fused = torch.cat([x_view2, x_view2_crossed], dim=-1)
        
        x_view1_out = self.layer_norm(self.fusion(x_view1_fused) + x_view1)
        x_view2_out = self.layer_norm(self.fusion(x_view2_fused) + x_view2)
        
        attn_info = {
            'attn_1to2': attn_1to2,
            'attn_2to1': attn_2to1
        }
        
        return x_view1_out, x_view2_out, attn_info


class CrossViewContrastiveLoss(nn.Module):
    """
    Contrastive loss for cross-view learning.
    
    Encourages graph embeddings from different views of the same graph
    to be similar, while being dissimilar from other graphs.
    
    Args:
        temperature: Temperature for softmax scaling
        normalize: Whether to L2 normalize embeddings
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        z_view1: torch.Tensor,
        z_view2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two views.
        
        Args:
            z_view1: Graph embeddings from View1 [batch_size, hidden_dim]
            z_view2: Graph embeddings from View2 [batch_size, hidden_dim]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = z_view1.size(0)
        
        # Normalize embeddings
        if self.normalize:
            z_view1 = F.normalize(z_view1, p=2, dim=1)
            z_view2 = F.normalize(z_view2, p=2, dim=1)
        
        # Compute similarity matrix
        # sim[i, j] = similarity between z_view1[i] and z_view2[j]
        sim_matrix = torch.mm(z_view1, z_view2.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_view1.device)
        
        # Cross-entropy loss in both directions
        loss_1to2 = F.cross_entropy(sim_matrix, labels)
        loss_2to1 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_1to2 + loss_2to1) / 2


# =============================================================================
# Full GIN Model with Causal Attention
# =============================================================================

class GINWithCausalAttention(nn.Module):
    """
    Complete GIN model with Causal Attention pooling for graph classification.
    
    Architecture:
        1. GIN Encoder: Learn node embeddings through message passing
        2. Causal Attention Pooling: Score nodes and create graph embedding
        3. Classifier: Predict graph label from embedding
    
    The model outputs both predictions and attention scores, allowing
    supervision or analysis of which nodes it considers important.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        output_dim: Number of output classes
        num_gin_layers: Number of GIN message passing layers
        dropout: Dropout probability
        attention_temperature: Temperature for attention sigmoid
        normalize_attention: Whether to normalize pooled embedding
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_gin_layers: int = 3,
        dropout: float = 0.5,
        attention_temperature: float = 1.0,
        normalize_attention: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GIN Encoder
        self.encoder = GINEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gin_layers,
            dropout=dropout
        )
        
        # Causal Attention Pooling
        self.pooling = CausalAttentionPooling(
            hidden_dim=hidden_dim,
            temperature=attention_temperature,
            normalize=normalize_attention
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            logits: Class logits [batch_size, output_dim]
            graph_embed: Graph embeddings [batch_size, hidden_dim]
            scores: Node attention scores [num_nodes, 1]
        """
        # Encode nodes with GIN
        node_embed = self.encoder(x, edge_index)
        
        # Pool with causal attention
        graph_embed, scores = self.pooling(node_embed, batch)
        
        # Classify
        logits = self.classifier(graph_embed)
        
        return logits, graph_embed, scores
    
    def get_attention_scores(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Get attention scores for a single graph (inference helper).
        
        Args:
            data: PyG Data object
            
        Returns:
            Attention scores for each node
        """
        self.eval()
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
            _, _, scores = self.forward(data.x, data.edge_index, batch)
        return scores.squeeze()


# =============================================================================
# Loss Functions
# =============================================================================

class CausalAttentionLoss(nn.Module):
    """
    Combined loss for training GIN with Causal Attention.
    
    Loss = CrossEntropy(predictions, labels) + λ * L1(attention_scores)
    
    The L1 penalty encourages sparse attention, pushing the model to
    select a small subgraph as the "rationale" for its predictions.
    
    Args:
        sparsity_weight: Weight λ for the L1 sparsity penalty
        reduction: Reduction method ('mean' or 'sum')
    """
    
    def __init__(
        self,
        sparsity_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.sparsity_weight = sparsity_weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_scores: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the combined loss.
        
        Args:
            logits: Predicted class logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            attention_scores: Node attention scores [num_nodes, 1]
            batch: Batch assignment for nodes (for per-graph normalization)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Sparsity loss (L1 norm of attention scores)
        if batch is not None:
            # Normalize L1 by number of graphs
            batch_size = batch.max().item() + 1
            sparsity_loss = attention_scores.abs().sum() / batch_size
        else:
            sparsity_loss = attention_scores.abs().mean()
        
        # Combined loss
        total_loss = ce_loss + self.sparsity_weight * sparsity_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'sparsity': sparsity_loss.item(),
            'mean_attention': attention_scores.mean().item()
        }
        
        return total_loss, loss_dict


class CausalRationaleLoss(nn.Module):
    """
    Advanced loss with optional supervision from ground truth causal masks.
    
    Loss = CE + λ_sparse * L1(s) + λ_entropy * H(s) + λ_connect * Disconnect(s) + λ_sup * BCE(s, causal_mask)
    
    This allows semi-supervised training where some graphs have
    ground truth annotations for which nodes are truly causal.
    
    Args:
        sparsity_weight: Weight for L1 sparsity penalty
        entropy_weight: Weight for entropy loss (pushes scores to 0 or 1)
        connectivity_weight: Weight for disconnected rationale penalty
        supervision_weight: Weight for BCE supervision loss
        continuity_weight: Weight for encouraging smooth scores between neighbors
    """
    
    def __init__(
        self,
        sparsity_weight: float = 0.1,
        entropy_weight: float = 0.1,
        connectivity_weight: float = 0.1,
        supervision_weight: float = 1.0,
        continuity_weight: float = 0.0
    ):
        super().__init__()
        
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.connectivity_weight = connectivity_weight
        self.supervision_weight = supervision_weight
        self.continuity_weight = continuity_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def _compute_entropy_loss(
        self,
        scores: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute entropy loss to push scores toward 0 or 1.
        
        Entropy H(s) = -s*log(s) - (1-s)*log(1-s)
        Minimizing entropy encourages binary (crisp) masks.
        
        Note: Low entropy = scores near 0 or 1 (good for binary selection)
              High entropy = scores near 0.5 (ambiguous)
        
        Args:
            scores: Attention scores in [0, 1], shape [num_nodes, 1]
            eps: Small constant for numerical stability (use 1e-6 for float32)
            
        Returns:
            Mean entropy loss
        """
        scores = scores.squeeze(-1)
        
        # Clamp to avoid log(0) - use larger eps for numerical stability
        # In float32, 1e-8 is too small and 1-1e-8 ≈ 1.0
        p = scores.clamp(min=eps, max=1.0 - eps)
        
        # Binary entropy: H(s) = -s*log(s) - (1-s)*log(1-s)
        # Equivalent to: -p*log(p) - (1-p)*log(1-p)
        entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
        
        return entropy.mean()
    
    def _compute_connectivity_loss(
        self,
        scores: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Compute disconnected rationale penalty.
        
        Penalizes selecting isolated nodes that have no selected neighbors.
        This encourages finding contiguous subgraph motifs.
        
        For each selected node (score > threshold), check if it has at least
        one selected neighbor. Penalize if isolated.
        
        Args:
            scores: Attention scores [num_nodes, 1]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            threshold: Threshold for considering a node as "selected"
            
        Returns:
            Disconnection penalty loss
        """
        scores = scores.squeeze(-1)
        num_nodes = scores.size(0)
        
        # Soft selection (differentiable)
        # Use sigmoid-like soft thresholding instead of hard threshold
        # selected_soft[i] approaches 1 if score[i] > threshold
        temperature = 0.1
        selected_soft = torch.sigmoid((scores - threshold) / temperature)
        
        # Build adjacency information
        src, dst = edge_index
        
        # For each node, compute the maximum score of its neighbors
        # This tells us if the node has any highly-scored neighbors
        neighbor_max_scores = torch.zeros(num_nodes, device=scores.device)
        
        # Scatter max: for each destination node, get max score of sources
        # Using scatter_reduce for efficiency
        neighbor_scores_expanded = scores[src]
        
        # For each node, sum the soft-selected neighbors
        neighbor_selected_sum = torch.zeros(num_nodes, device=scores.device)
        neighbor_selected_sum.scatter_add_(0, dst, selected_soft[src])
        
        # Count number of neighbors per node
        neighbor_count = torch.zeros(num_nodes, device=scores.device)
        neighbor_count.scatter_add_(0, dst, torch.ones_like(src, dtype=torch.float))
        neighbor_count = neighbor_count.clamp(min=1)  # Avoid division by zero
        
        # Fraction of neighbors that are selected
        neighbor_selected_frac = neighbor_selected_sum / neighbor_count
        
        # Penalty: node is selected but has no selected neighbors
        # isolation_penalty[i] = selected_soft[i] * (1 - neighbor_selected_frac[i])
        # High when node is selected but neighbors are not
        isolation_penalty = selected_soft * (1 - neighbor_selected_frac)
        
        # Average over all nodes
        return isolation_penalty.mean()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_scores: torch.Tensor,
        batch: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the advanced combined loss.
        
        Args:
            logits: Predicted class logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            attention_scores: Node attention scores [num_nodes, 1]
            batch: Batch assignment for nodes
            causal_mask: Optional ground truth causal node mask [num_nodes]
            edge_index: Optional edge indices for continuity loss
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        batch_size = batch.max().item() + 1
        
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Sparsity loss (L1)
        sparsity_loss = attention_scores.abs().sum() / batch_size
        
        # Initialize total loss
        total_loss = ce_loss + self.sparsity_weight * sparsity_loss
        
        loss_dict = {
            'ce': ce_loss.item(),
            'sparsity': sparsity_loss.item()
        }
        
        # Entropy loss (push scores to 0 or 1)
        if self.entropy_weight > 0:
            entropy_loss = self._compute_entropy_loss(attention_scores)
            total_loss = total_loss + self.entropy_weight * entropy_loss
            loss_dict['entropy'] = entropy_loss.item()
        
        # Connectivity loss (penalize isolated selected nodes)
        if edge_index is not None and self.connectivity_weight > 0:
            connectivity_loss = self._compute_connectivity_loss(
                attention_scores, edge_index, batch
            )
            total_loss = total_loss + self.connectivity_weight * connectivity_loss
            loss_dict['connectivity'] = connectivity_loss.item()
        
        # Supervision loss (if ground truth available)
        if causal_mask is not None and self.supervision_weight > 0:
            causal_mask = causal_mask.float().unsqueeze(-1)
            supervision_loss = self.bce_loss(attention_scores, causal_mask).mean()
            total_loss = total_loss + self.supervision_weight * supervision_loss
            loss_dict['supervision'] = supervision_loss.item()
        
        # Continuity loss (encourage smooth scores between neighbors)
        if edge_index is not None and self.continuity_weight > 0:
            src, dst = edge_index
            score_diff = (attention_scores[src] - attention_scores[dst]).abs()
            continuity_loss = score_diff.mean()
            total_loss = total_loss + self.continuity_weight * continuity_loss
            loss_dict['continuity'] = continuity_loss.item()
        
        loss_dict['total'] = total_loss.item()
        loss_dict['mean_attention'] = attention_scores.mean().item()
        
        return total_loss, loss_dict


# =============================================================================
# Threshold Scheduler for Annealing
# =============================================================================

class ThresholdScheduler:
    """
    Scheduler for annealing the causal attention threshold.
    
    Starts with a low threshold (selecting more nodes) and gradually
    increases to the target threshold. This helps stabilize early training
    by being more permissive initially.
    
    Scheduling strategies:
    - 'linear': Linear interpolation from start to end
    - 'cosine': Cosine annealing (slower at start and end)
    - 'exponential': Exponential decay towards target
    - 'step': Step function at specific epochs
    
    Args:
        start_threshold: Initial threshold (typically low, e.g., 0.2)
        end_threshold: Final threshold (typically 0.5)
        warmup_epochs: Number of epochs to reach end_threshold
        schedule_type: Type of scheduling ('linear', 'cosine', 'exponential', 'step')
    """
    
    def __init__(
        self,
        start_threshold: float = 0.2,
        end_threshold: float = 0.5,
        warmup_epochs: int = 20,
        schedule_type: str = 'cosine'
    ):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        
        self.current_threshold = start_threshold
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None) -> float:
        """
        Update and return the current threshold.
        
        Args:
            epoch: Current epoch (if None, increments internal counter)
            
        Returns:
            Current threshold value
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch >= self.warmup_epochs:
            self.current_threshold = self.end_threshold
        else:
            progress = self.current_epoch / self.warmup_epochs
            
            if self.schedule_type == 'linear':
                self.current_threshold = (
                    self.start_threshold + 
                    progress * (self.end_threshold - self.start_threshold)
                )
            
            elif self.schedule_type == 'cosine':
                # Cosine annealing: slower at start and end
                import math
                cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
                self.current_threshold = (
                    self.start_threshold + 
                    cosine_progress * (self.end_threshold - self.start_threshold)
                )
            
            elif self.schedule_type == 'exponential':
                # Exponential approach to target
                import math
                decay = 1 - math.exp(-3 * progress)  # ~95% at progress=1
                self.current_threshold = (
                    self.start_threshold + 
                    decay * (self.end_threshold - self.start_threshold)
                )
            
            elif self.schedule_type == 'step':
                # Step at 50% of warmup
                if progress < 0.5:
                    self.current_threshold = self.start_threshold
                else:
                    self.current_threshold = self.end_threshold
            
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return self.current_threshold
    
    def get_threshold(self) -> float:
        """Get current threshold without stepping."""
        return self.current_threshold
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_threshold = self.start_threshold
        self.current_epoch = 0
    
    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_threshold': self.current_threshold,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_threshold = state_dict['current_threshold']
        self.current_epoch = state_dict['current_epoch']


# =============================================================================
# Causal Augmenter for Counterfactual Data Generation
# =============================================================================

class CausalAugmenter:
    """
    Generates counterfactual graphs by preserving causal subgraphs
    and replacing spurious parts with random noise graphs.
    
    This augmentation technique helps models learn invariant representations
    by showing them that the label should remain the same when only the
    causal subgraph is preserved and the spurious part is randomized.
    
    Supports dynamic thresholding via ThresholdScheduler for training stability.
    
    Args:
        threshold: Score threshold to identify causal nodes (default: 0.5)
        noise_type: Type of noise graph ('erdos_renyi', 'barabasi_albert', 'random')
        edge_prob: Edge probability for Erdos-Renyi noise graphs
        num_connections: Number of edges connecting noise to causal subgraph
        preserve_features: Whether to preserve original node features for causal nodes
        feature_noise_std: Standard deviation for noise node features
        threshold_scheduler: Optional ThresholdScheduler for dynamic thresholding
        min_causal_nodes: Minimum number of causal nodes to preserve (fallback)
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        noise_type: str = 'erdos_renyi',
        edge_prob: float = 0.2,
        num_connections: int = 1,
        preserve_features: bool = True,
        feature_noise_std: float = 1.0,
        threshold_scheduler: Optional['ThresholdScheduler'] = None,
        min_causal_nodes: int = 3
    ):
        self.threshold = threshold
        self.noise_type = noise_type
        self.edge_prob = edge_prob
        self.num_connections = num_connections
        self.preserve_features = preserve_features
        self.feature_noise_std = feature_noise_std
        self.threshold_scheduler = threshold_scheduler
        self.min_causal_nodes = min_causal_nodes
    
    def get_threshold(self) -> float:
        """Get current threshold (from scheduler if available)."""
        if self.threshold_scheduler is not None:
            return self.threshold_scheduler.get_threshold()
        return self.threshold
    
    def set_threshold(self, threshold: float):
        """Manually set the threshold."""
        self.threshold = threshold
    
    def step_scheduler(self, epoch: Optional[int] = None) -> float:
        """Step the threshold scheduler and return new threshold."""
        if self.threshold_scheduler is not None:
            return self.threshold_scheduler.step(epoch)
        return self.threshold
    
    def _generate_noise_graph(
        self,
        num_nodes: int,
        feature_dim: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random noise graph.
        
        Args:
            num_nodes: Number of nodes in the noise graph
            feature_dim: Dimension of node features
            device: Device to create tensors on
            
        Returns:
            edge_index: Edge indices for noise graph [2, num_edges]
            x: Node features for noise graph [num_nodes, feature_dim]
        """
        if num_nodes <= 0:
            # Return empty graph
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, feature_dim), dtype=torch.float, device=device)
            )
        
        edges = []
        
        if self.noise_type == 'erdos_renyi':
            # Erdos-Renyi random graph
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if random.random() < self.edge_prob:
                        edges.append((i, j))
                        edges.append((j, i))  # Undirected
            
            # Ensure connectivity with spanning tree backbone
            for i in range(1, num_nodes):
                parent = random.randint(0, i - 1)
                if (parent, i) not in edges:
                    edges.append((parent, i))
                    edges.append((i, parent))
        
        elif self.noise_type == 'barabasi_albert':
            # Barabasi-Albert preferential attachment
            m = min(2, num_nodes - 1)  # edges to attach from new node
            
            # Start with complete graph on m+1 nodes
            for i in range(min(m + 1, num_nodes)):
                for j in range(i + 1, min(m + 1, num_nodes)):
                    edges.append((i, j))
                    edges.append((j, i))
            
            # Add remaining nodes with preferential attachment
            degrees = [m] * min(m + 1, num_nodes) + [0] * max(0, num_nodes - m - 1)
            
            for new_node in range(m + 1, num_nodes):
                # Select m nodes with probability proportional to degree
                total_degree = sum(degrees[:new_node])
                if total_degree == 0:
                    targets = list(range(min(m, new_node)))
                else:
                    probs = [d / total_degree for d in degrees[:new_node]]
                    targets = np.random.choice(
                        new_node, size=min(m, new_node), replace=False, p=probs
                    ).tolist()
                
                for target in targets:
                    edges.append((new_node, target))
                    edges.append((target, new_node))
                    degrees[new_node] += 1
                    degrees[target] += 1
        
        else:  # 'random' - simple random edges
            num_edges = max(num_nodes - 1, int(num_nodes * 1.5))
            for _ in range(num_edges):
                i, j = random.sample(range(num_nodes), 2)
                edges.append((i, j))
                edges.append((j, i))
        
        # Create edge index tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Generate node features (constant with small noise, matching dataset)
        # IMPORTANT: Do NOT encode node type in features - this would be data leakage!
        x = torch.ones(num_nodes, feature_dim, device=device)
        x = x + self.feature_noise_std * 0.1 * torch.randn(num_nodes, feature_dim, device=device)
        
        return edge_index, x
    
    def _extract_causal_subgraph(
        self,
        data: Data,
        causal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Extract the causal subgraph from a graph.
        
        Args:
            data: Original graph data
            causal_mask: Boolean mask indicating causal nodes
            
        Returns:
            causal_edge_index: Edges within causal subgraph
            causal_x: Features of causal nodes
            causal_indices: Original indices of causal nodes
            num_spurious: Number of removed spurious nodes
        """
        causal_indices = torch.where(causal_mask)[0]
        num_causal = len(causal_indices)
        num_spurious = data.num_nodes - num_causal
        
        # Create mapping from old indices to new indices
        old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long, 
                                 device=data.x.device)
        old_to_new[causal_indices] = torch.arange(num_causal, device=data.x.device)
        
        # Filter edges to only include those between causal nodes
        src, dst = data.edge_index
        causal_edge_mask = causal_mask[src] & causal_mask[dst]
        
        causal_edges = data.edge_index[:, causal_edge_mask]
        
        # Remap edge indices
        causal_edge_index = old_to_new[causal_edges]
        
        # Extract causal node features
        causal_x = data.x[causal_indices]
        
        return causal_edge_index, causal_x, causal_indices, num_spurious
    
    def generate_counterfactual(
        self,
        data: Data,
        node_scores: torch.Tensor
    ) -> Data:
        """
        Generate a counterfactual for a single graph.
        
        Args:
            data: Original graph data
            node_scores: Causal scores for each node [num_nodes] or [num_nodes, 1]
            
        Returns:
            Counterfactual graph with causal nodes preserved and spurious replaced
        """
        device = data.x.device
        feature_dim = data.x.size(1)
        
        # Flatten scores if needed
        if node_scores.dim() > 1:
            node_scores = node_scores.squeeze(-1)
        
        # Get current threshold (from scheduler if available)
        current_threshold = self.get_threshold()
        
        # Threshold to get causal mask
        causal_mask = node_scores > current_threshold
        
        # Handle edge cases
        num_causal = causal_mask.sum().item()
        if num_causal < self.min_causal_nodes:
            # Too few causal nodes - keep top-k nodes instead
            k = max(self.min_causal_nodes, data.num_nodes // 4)
            k = min(k, data.num_nodes)  # Don't exceed total nodes
            _, top_indices = node_scores.topk(k)
            causal_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            causal_mask[top_indices] = True
            num_causal = k
        
        if num_causal == data.num_nodes:
            # All nodes are causal - just add small noise graph
            num_spurious = max(3, data.num_nodes // 4)
        else:
            num_spurious = data.num_nodes - num_causal
        
        # Extract causal subgraph
        causal_edge_index, causal_x, causal_indices, _ = self._extract_causal_subgraph(
            data, causal_mask
        )
        
        # Generate noise graph to replace spurious part
        noise_edge_index, noise_x = self._generate_noise_graph(
            num_spurious, feature_dim, device
        )
        
        # Offset noise edges
        noise_edge_index = noise_edge_index + num_causal
        
        # Combine causal and noise graphs
        combined_x = torch.cat([causal_x, noise_x], dim=0)
        combined_edge_index = torch.cat([causal_edge_index, noise_edge_index], dim=1)
        
        # Add connecting edges between causal and noise
        if num_spurious > 0 and num_causal > 0:
            connecting_edges = []
            for _ in range(self.num_connections):
                causal_node = random.randint(0, num_causal - 1)
                noise_node = num_causal + random.randint(0, num_spurious - 1)
                connecting_edges.append([causal_node, noise_node])
                connecting_edges.append([noise_node, causal_node])
            
            connecting_edge_index = torch.tensor(
                connecting_edges, dtype=torch.long, device=device
            ).t()
            combined_edge_index = torch.cat(
                [combined_edge_index, connecting_edge_index], dim=1
            )
        
        # Create new causal mask for counterfactual
        new_causal_mask = torch.zeros(num_causal + num_spurious, dtype=torch.bool, 
                                       device=device)
        new_causal_mask[:num_causal] = True
        
        # Create counterfactual data object
        counterfactual = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            y=data.y,  # Label stays the same!
            num_nodes=num_causal + num_spurious,
            causal_mask=new_causal_mask,
            original_causal_indices=causal_indices
        )
        
        # Copy any additional attributes
        for key in ['causal_motif', 'spurious_motif']:
            if hasattr(data, key):
                setattr(counterfactual, key, getattr(data, key))
        
        return counterfactual
    
    def generate_counterfactuals(
        self,
        data: Batch,
        node_scores: torch.Tensor
    ) -> Batch:
        """
        Generate counterfactuals for a batch of graphs.
        
        Args:
            data: Batched graph data
            node_scores: Causal scores for all nodes [total_nodes] or [total_nodes, 1]
            
        Returns:
            Batch of counterfactual graphs
        """
        device = data.x.device
        
        # Flatten scores if needed
        if node_scores.dim() > 1:
            node_scores = node_scores.squeeze(-1)
        
        # Get batch information
        batch_indices = data.batch
        batch_size = batch_indices.max().item() + 1
        
        # Process each graph in the batch
        counterfactual_list = []
        
        for graph_idx in range(batch_size):
            # Get nodes belonging to this graph
            node_mask = batch_indices == graph_idx
            graph_node_indices = torch.where(node_mask)[0]
            
            # Extract single graph data
            num_nodes = node_mask.sum().item()
            
            # Get node features for this graph
            graph_x = data.x[node_mask]
            
            # Get edges for this graph
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            graph_edges = data.edge_index[:, edge_mask]
            
            # Remap edge indices to local graph
            min_idx = graph_node_indices.min()
            graph_edge_index = graph_edges - min_idx
            
            # Get scores for this graph
            graph_scores = node_scores[node_mask]
            
            # Get label
            graph_y = data.y[graph_idx] if data.y.dim() > 0 else data.y
            
            # Create single graph Data object
            single_graph = Data(
                x=graph_x,
                edge_index=graph_edge_index,
                y=graph_y.unsqueeze(0) if graph_y.dim() == 0 else graph_y,
                num_nodes=num_nodes
            )
            
            # Copy additional attributes if available
            if hasattr(data, 'causal_mask'):
                single_graph.causal_mask = data.causal_mask[node_mask]
            if hasattr(data, 'causal_motif'):
                single_graph.causal_motif = data.causal_motif[graph_idx]
            if hasattr(data, 'spurious_motif'):
                single_graph.spurious_motif = data.spurious_motif[graph_idx]
            
            # Generate counterfactual
            counterfactual = self.generate_counterfactual(single_graph, graph_scores)
            counterfactual_list.append(counterfactual)
        
        # Batch counterfactuals
        return Batch.from_data_list(counterfactual_list)
    
    def augment_batch(
        self,
        data: Batch,
        node_scores: torch.Tensor,
        augment_ratio: float = 0.5
    ) -> Batch:
        """
        Augment a batch by mixing original and counterfactual graphs.
        
        Args:
            data: Original batched graph data
            node_scores: Causal scores for all nodes
            augment_ratio: Fraction of graphs to replace with counterfactuals
            
        Returns:
            Mixed batch of original and counterfactual graphs
        """
        batch_size = data.batch.max().item() + 1
        num_augment = int(batch_size * augment_ratio)
        
        if num_augment == 0:
            return data
        
        # Select graphs to augment
        augment_indices = random.sample(range(batch_size), num_augment)
        
        # Separate scores by graph
        graph_list = []
        
        for graph_idx in range(batch_size):
            node_mask = data.batch == graph_idx
            graph_node_indices = torch.where(node_mask)[0]
            
            # Extract graph
            graph_x = data.x[node_mask]
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            graph_edges = data.edge_index[:, edge_mask]
            min_idx = graph_node_indices.min()
            graph_edge_index = graph_edges - min_idx
            
            graph_y = data.y[graph_idx] if data.y.dim() > 0 else data.y
            
            single_graph = Data(
                x=graph_x,
                edge_index=graph_edge_index,
                y=graph_y.unsqueeze(0) if graph_y.dim() == 0 else graph_y,
                num_nodes=node_mask.sum().item()
            )
            
            if hasattr(data, 'causal_mask'):
                single_graph.causal_mask = data.causal_mask[node_mask]
            
            if graph_idx in augment_indices:
                # Generate counterfactual
                graph_scores = node_scores[node_mask]
                single_graph = self.generate_counterfactual(single_graph, graph_scores)
            
            graph_list.append(single_graph)
        
        return Batch.from_data_list(graph_list)


# =============================================================================
# Training Utilities
# =============================================================================

def train_epoch(
    model: GINWithCausalAttention,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: CausalAttentionLoss,
    device: torch.device
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: GIN model with causal attention
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Dictionary with average metrics for the epoch
    """
    model.train()
    
    total_loss = 0
    total_ce = 0
    total_sparsity = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits, graph_embed, scores = model(
            batch.x, batch.edge_index, batch.batch
        )
        
        # Compute loss
        loss, loss_dict = criterion(
            logits, batch.y, scores, batch.batch
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss_dict['total'] * batch.num_graphs
        total_ce += loss_dict['ce'] * batch.num_graphs
        total_sparsity += loss_dict['sparsity'] * batch.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch.num_graphs
    
    return {
        'loss': total_loss / total_samples,
        'ce_loss': total_ce / total_samples,
        'sparsity_loss': total_sparsity / total_samples,
        'accuracy': total_correct / total_samples
    }


@torch.no_grad()
def evaluate(
    model: GINWithCausalAttention,
    loader,
    criterion: CausalAttentionLoss,
    device: torch.device
) -> dict:
    """
    Evaluate the model.
    
    Args:
        model: GIN model with causal attention
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_scores = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        logits, graph_embed, scores = model(
            batch.x, batch.edge_index, batch.batch
        )
        
        # Compute loss
        loss, loss_dict = criterion(
            logits, batch.y, scores, batch.batch
        )
        
        # Track metrics
        total_loss += loss_dict['total'] * batch.num_graphs
        
        pred = logits.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch.num_graphs
        
        all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores, dim=0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'mean_attention': all_scores.mean().item(),
        'attention_std': all_scores.std().item(),
        'sparsity': (all_scores < 0.5).float().mean().item()  # % of low-attention nodes
    }


# =============================================================================
# Demo / Main
# =============================================================================

def main():
    """Demo of GIN with Causal Attention on Spurious-Motif dataset."""
    import matplotlib.pyplot as plt
    from torch_geometric.loader import DataLoader

    from spurious_motif_dataset import SpuriousMotif, visualize_graph
    
    print("=" * 70)
    print("  GIN WITH CAUSAL ATTENTION - DEMO")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\n[1] Loading Spurious-Motif dataset...")
    root = './data/spurious_motif'
    train_dataset = SpuriousMotif(root, mode='train', bias=0.9, num_graphs=1000)
    test_dataset = SpuriousMotif(root, mode='test', bias=0.9, num_graphs=1000)
    
    print(f"    Training graphs: {len(train_dataset)}")
    print(f"    Test graphs: {len(test_dataset)}")
    print(f"    Node feature dim: {train_dataset[0].x.size(1)}")
    print(f"    Number of classes: 3")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\n[2] Creating model...")
    input_dim = train_dataset[0].x.size(1)
    model = GINWithCausalAttention(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=3,
        num_gin_layers=3,
        dropout=0.5,
        attention_temperature=1.0
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model parameters: {num_params:,}")
    
    # Create loss and optimizer
    # Use lower sparsity weight initially to allow model to learn first
    criterion = CausalAttentionLoss(sparsity_weight=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    # Training loop
    print("\n[3] Training...")
    num_epochs = 150
    best_test_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            print(f"    Epoch {epoch:3d} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} | "
                  f"Mean Attn: {test_metrics['mean_attention']:.4f} | "
                  f"Sparsity: {test_metrics['sparsity']:.4f}")
            
            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
    
    print(f"\n    Best Test Accuracy: {best_test_acc:.4f}")
    
    # Final evaluation
    print("\n[4] Final Evaluation...")
    train_metrics = evaluate(model, train_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"    Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"    Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Mean Attention: {test_metrics['mean_attention']:.4f}")
    print(f"    Attention Std:  {test_metrics['attention_std']:.4f}")
    
    # Visualize attention on sample graphs
    print("\n[5] Visualizing attention scores...")
    
    model.eval()
    sample = test_dataset[0].to(device)
    batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, _, scores = model(sample.x, sample.edge_index, batch)
    
    scores = scores.cpu().squeeze().numpy()
    
    # Create visualization with attention scores
    import matplotlib.patches as mpatches
    import networkx as nx
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a2e')
    
    # Left: Ground truth causal nodes
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    
    G = nx.Graph()
    G.add_nodes_from(range(sample.num_nodes))
    edge_index = sample.edge_index.cpu().numpy()
    edges = [(edge_index[0, i], edge_index[1, i]) 
             for i in range(edge_index.shape[1]) 
             if edge_index[0, i] < edge_index[1, i]]
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    causal_mask = sample.causal_mask.cpu().numpy()
    colors = ['#E74C3C' if causal_mask[i] else '#7FB3D5' for i in range(sample.num_nodes)]
    
    nx.draw_networkx_edges(G, pos, edge_color='#4a4a6a', width=1.5, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=400, 
                           edgecolors='white', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', ax=ax)
    
    ax.set_title('Ground Truth Causal Nodes', fontsize=14, color='white', fontweight='bold')
    ax.axis('off')
    
    # Right: Learned attention scores
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    
    # Color by attention score (blue=low, red=high)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(s) for s in scores]
    sizes = [300 + 400 * s for s in scores]
    
    nx.draw_networkx_edges(G, pos, edge_color='#4a4a6a', width=1.5, alpha=0.6, ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=scores, cmap=cmap,
                                    node_size=sizes, edgecolors='white', 
                                    linewidths=2, ax=ax, vmin=0, vmax=1)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', ax=ax)
    
    # Colorbar
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
    cbar.set_label('Attention Score', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title('Learned Attention Scores', fontsize=14, color='white', fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'Label: {SpuriousMotif.CAUSAL_MOTIFS[sample.y.item()]} | '
                 f'Predicted: {SpuriousMotif.CAUSAL_MOTIFS[model(sample.x, sample.edge_index, batch)[0].argmax().item()]}',
                 fontsize=16, color='white', fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('./visualizations/attention_comparison.png', dpi=150, 
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("    Saved: ./visualizations/attention_comparison.png")
    
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE!")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()

