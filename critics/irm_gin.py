from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import scatter


@dataclass
class CriticConfig:
    in_dim: int
    hidden_dim: int = 64
    edge_hidden_dim: int = 64
    dropout: float = 0.1
    num_domains: int = 3


class _TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeScoringHead(nn.Module):
    def __init__(self, hidden_dim: int, edge_hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        src_h, dst_h = node_embeddings[src], node_embeddings[dst]
        pair_feat = torch.cat([src_h, dst_h, torch.abs(src_h - dst_h)], dim=-1)
        scores = torch.sigmoid(self.proj(pair_feat)).squeeze(-1)
        return scores


class DomainHead(nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_domains)

    def forward(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(graph_embeddings))


class IRMEdgeCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        gin_mlp1 = _TwoLayerMLP(config.in_dim, config.hidden_dim, config.hidden_dim)
        gin_mlp2 = _TwoLayerMLP(config.hidden_dim, config.hidden_dim, config.hidden_dim)

        self.conv1 = GINConv(gin_mlp1)
        self.conv2 = GINConv(gin_mlp2)
        self.dropout = nn.Dropout(config.dropout)

        self.edge_head = EdgeScoringHead(config.hidden_dim, config.edge_hidden_dim)
        self.domain_head = DomainHead(config.hidden_dim, config.num_domains, config.dropout)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return self.dropout(h)

    def _node_weights(self, edge_scores: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index
        weights = torch.zeros(num_nodes, device=edge_scores.device)
        weights = weights.scatter_add(0, src, edge_scores)
        weights = weights.scatter_add(0, dst, edge_scores)
        return weights + 1e-6

    def _graph_embeddings(
        self, node_embeddings: torch.Tensor, edge_scores: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        node_weights = self._node_weights(edge_scores, edge_index, node_embeddings.size(0))
        weighted_nodes = node_embeddings * node_weights.unsqueeze(-1)
        pooled = scatter(weighted_nodes, batch, dim=0, reduce="sum")
        norm = scatter(node_weights, batch, dim=0, reduce="sum").unsqueeze(-1).clamp(min=1e-6)
        return pooled / norm

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_embeddings = self.encode_nodes(x, edge_index)
        edge_scores = self.edge_head(node_embeddings, edge_index)
        graph_embeddings = self._graph_embeddings(node_embeddings, edge_scores, edge_index, batch)
        domain_logits = self.domain_head(graph_embeddings)
        return {
            "edge_scores": edge_scores,
            "domain_logits": domain_logits,
            "node_embeddings": node_embeddings,
            "graph_embeddings": graph_embeddings,
        }

