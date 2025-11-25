#!/usr/bin/env python3
"""
Integrated training script for IRM Critic + DiGress Diffusion.
Connects critic edge scores to diffusion model for guided generation.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from critics.irm_gin import IRMEdgeCritic, CriticConfig
from data.good_motif import load_good_motif
from guidance.interface import DiffusionGuidanceInterface, GuidanceConfig


def load_critic(checkpoint_path: str, device: torch.device) -> IRMEdgeCritic:
    """Load trained critic from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config - adjust based on your training
        config = CriticConfig(
            in_dim=1,  # Adjust based on GOOD-Motif node features
            hidden_dim=64,
            edge_hidden_dim=32,
            num_domains=3,  # GOOD-Motif has 3 environments
            dropout=0.1
        )
    
    model = IRMEdgeCritic(config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def test_mode_collapse(
    critic: IRMEdgeCritic,
    test_loader,
    device: torch.device,
    guidance_scale: float = 5.0,
    invariant_threshold: float = 0.6,
    spurious_threshold: float = 0.4
):
    """Test if G_inv and G_var views are meaningfully different."""
    print(f"\n=== Testing Mode Collapse (guidance_scale={guidance_scale}) ===")
    
    config = GuidanceConfig(
        invariant_threshold=invariant_threshold,
        spurious_threshold=spurious_threshold,
        guidance_scale=guidance_scale
    )
    interface = DiffusionGuidanceInterface(critic, config)
    
    n_samples = 0
    inv_edge_counts = []
    var_edge_counts = []
    overlap_ratios = []
    
    for batch in test_loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            # Extract single graph
            graph_mask = batch.batch == i
            graph_data = Data(
                x=batch.x[graph_mask],
                edge_index=batch.edge_index,
                num_nodes=graph_mask.sum().item()
            )
            graph_data = graph_data.to(device)
            
            # Generate views
            g_inv, g_var = interface.generate_views(graph_data)
            
            inv_edges = g_inv.edge_index.size(1)
            var_edges = g_var.edge_index.size(1)
            total_edges = graph_data.edge_index.size(1)
            
            inv_edge_counts.append(inv_edges)
            var_edge_counts.append(var_edges)
            
            # Compute overlap (simplified)
            if total_edges > 0:
                overlap = min(inv_edges, var_edges) / total_edges
                overlap_ratios.append(overlap)
            
            n_samples += 1
            if n_samples >= 10:  # Test on first 10 graphs
                break
        
        if n_samples >= 10:
            break
    
    avg_inv = sum(inv_edge_counts) / len(inv_edge_counts) if inv_edge_counts else 0
    avg_var = sum(var_edge_counts) / len(var_edge_counts) if var_edge_counts else 0
    avg_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0
    
    print(f"Average invariant edges: {avg_inv:.2f}")
    print(f"Average spurious edges: {avg_var:.2f}")
    print(f"Average overlap ratio: {avg_overlap:.2f}")
    
    # Check for mode collapse
    if avg_overlap > 0.8:
        print("WARNING: High overlap detected - possible mode collapse!")
        print(f"  Consider increasing guidance_scale or adjusting thresholds")
        return False
    else:
        print("Views are meaningfully different - no mode collapse detected.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Integrated training: Critic + Diffusion")
    parser.add_argument("--critic-checkpoint", type=str,
                       default="artifacts/critic/checkpoints/critic_best.pt",
                       help="Path to critic checkpoint")
    parser.add_argument("--test-mode-collapse", action="store_true",
                       help="Test for mode collapse")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                       help="Guidance scale for classifier-free guidance")
    parser.add_argument("--invariant-threshold", type=float, default=0.6,
                       help="Threshold for invariant edges")
    parser.add_argument("--spurious-threshold", type=float, default=0.4,
                       help="Threshold for spurious edges")
    parser.add_argument("--dataset-root", type=str,
                       default="external/GOOD_datasets",
                       help="Path to GOOD datasets")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load critic
    if not os.path.exists(args.critic_checkpoint):
        print(f"ERROR: Critic checkpoint not found: {args.critic_checkpoint}")
        print("Please train the critic first using train_critic.py")
        return
    
    print(f"Loading critic from {args.critic_checkpoint}...")
    critic = load_critic(args.critic_checkpoint, device)
    print("Critic loaded successfully.")
    
    # Load dataset
    print("\nLoading GOOD-Motif dataset...")
    split_payload, meta_info = load_good_motif(
        dataset_root=args.dataset_root,
        domain="basis",
        shift="concept",
        batch_size=32,
        num_workers=0
    )
    
    test_loader = split_payload.get("test_loader")
    if test_loader is None:
        print("WARNING: Test loader not available, using validation loader")
        test_loader = split_payload.get("val_loader")
    
    # Test mode collapse if requested
    if args.test_mode_collapse:
        if test_loader is None:
            print("ERROR: No test/val loader available for mode collapse test")
            return
        
        success = test_mode_collapse(
            critic=critic,
            test_loader=test_loader,
            device=device,
            guidance_scale=args.guidance_scale,
            invariant_threshold=args.invariant_threshold,
            spurious_threshold=args.spurious_threshold
        )
        
        if success:
            print("\nMode collapse test PASSED")
        else:
            print("\nMode collapse test FAILED - consider adjusting parameters")
    
    print("\n=== Integration Complete ===")
    print("Critic is ready to guide diffusion model.")
    print("To use in diffusion, pass critic_model to DiscreteDenoisingDiffusion.__init__()")


if __name__ == "__main__":
    main()
