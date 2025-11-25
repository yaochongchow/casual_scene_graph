from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data


@dataclass
class GuidanceConfig:
    invariant_threshold: float = 0.6
    spurious_threshold: float = 0.4
    guidance_scale: float = 1.5


class DiffusionModuleProto:
    """Placeholder interface for Person B's diffusion module."""

    def predict(self, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Person B's diffusion module should implement `predict`.")


class DiffusionGuidanceInterface:
    def __init__(self, critic_model, config: GuidanceConfig, diffusion_module: Optional[DiffusionModuleProto] = None):
        self.critic = critic_model
        self.config = config
        self.diffusion_module = diffusion_module

    def attach_diffusion_module(self, module: DiffusionModuleProto):
        self.diffusion_module = module

    def _score_edges(self, data: Data) -> torch.Tensor:
        self.critic.eval()
        with torch.no_grad():
            batch = data.clone()
            batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
            outputs = self.critic(batch)
            return outputs["edge_scores"]

    def generate_views(self, data: Data) -> Tuple[Data, Data]:
        scores = self._score_edges(data)
        invariant_mask = scores >= self.config.invariant_threshold
        spurious_mask = scores <= self.config.spurious_threshold

        g_inv = self._apply_mask(data, invariant_mask)
        g_var = self._apply_mask(data, spurious_mask)
        return g_inv, g_var

    @staticmethod
    def _apply_mask(data: Data, mask: torch.Tensor) -> Data:
        masked_edge_index = data.edge_index[:, mask]
        masked_edge_attr = data.edge_attr[mask] if getattr(data, "edge_attr", None) is not None else None
        target_y = data.y.clone() if getattr(data, "y", None) is not None else None
        new_data = Data(
            x=data.x.clone(),
            edge_index=masked_edge_index.clone(),
            edge_attr=masked_edge_attr.clone() if masked_edge_attr is not None else None,
            y=target_y,
        )
        new_data.num_nodes = data.num_nodes
        return new_data

    def classifier_free_guidance(self, latents: torch.Tensor, graph_data: Data) -> torch.Tensor:
        if self.diffusion_module is None:
            raise RuntimeError("Diffusion module not attached.")

        scores = self._score_edges(graph_data)
        mask = (scores >= self.config.invariant_threshold).float()

        cond = self.diffusion_module.predict(latents, mask)
        uncond = self.diffusion_module.predict(latents, torch.zeros_like(mask))
        return uncond + self.config.guidance_scale * (cond - uncond)
##*** End Patch

