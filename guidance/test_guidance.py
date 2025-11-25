import torch
import torch.nn as nn
from torch_geometric.data import Data

from guidance.interface import (
    DiffusionGuidanceInterface,
    DiffusionModuleProto,
    GuidanceConfig,
)


class DummyCritic(nn.Module):
    def forward(self, data):
        num_edges = data.edge_index.size(1)
        edge_scores = torch.rand(num_edges, device=data.edge_index.device)
        batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long, device=data.edge_index.device))
        domain_logits = torch.zeros(batch.max().item() + 1 if batch.numel() else 1, 3, device=edge_scores.device)
        return {"edge_scores": edge_scores, "domain_logits": domain_logits}


class DummyDiffusion(DiffusionModuleProto):
    def predict(self, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return latents + mask.mean()


def build_dummy_graph(num_nodes: int = 12, feat_dim: int = 8) -> Data:
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    x = torch.randn(num_nodes, feat_dim)
    return Data(x=x, edge_index=edge_index)


def run_dummy_guidance():
    graph = build_dummy_graph()
    critic = DummyCritic()
    diffusion = DummyDiffusion()
    interface = DiffusionGuidanceInterface(critic, GuidanceConfig(), diffusion)

    g_inv, g_var = interface.generate_views(graph)
    assert g_inv.edge_index.size(1) != g_var.edge_index.size(1), "Views should differ when thresholds separate edges."

    latents = torch.randn(10, 16)
    guided = interface.classifier_free_guidance(latents, graph)
    assert guided.shape == latents.shape
    return {
        "inv_edges": g_inv.edge_index.size(1),
        "var_edges": g_var.edge_index.size(1),
        "guided_mean": guided.mean().item(),
    }


if __name__ == "__main__":
    stats = run_dummy_guidance()
    print("Dummy guidance stats:", stats)

