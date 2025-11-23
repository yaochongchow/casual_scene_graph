import sys
import os
import torch
import unittest
from unittest.mock import MagicMock
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also add src to path so we can import modules directly if needed, but mainly for 'from src' to work we need root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from diffusion_model_discrete import DiscreteDenoisingDiffusion

class TestDDIM(unittest.TestCase):
    def test_ddim_sampler(self):
        # Mock config
        cfg = OmegaConf.create({
            'general': {'name': 'test', 'log_every_steps': 10, 'number_chain_steps': 10, 'sample_every_val': 1, 'samples_to_generate': 2, 'samples_to_save': 2, 'chains_to_save': 1},
            'model': {
                'type': 'discrete',
                'diffusion_steps': 100,
                'n_layers': 2,
                'hidden_mlp_dims': {'X': 32, 'E': 16, 'y': 32},
                'hidden_dims': {'dx': 32, 'de': 16, 'dy': 32, 'n_head': 2, 'dim_ffX': 32, 'dim_ffE': 16, 'dim_ffy': 32},
                'lambda_train': [5, 0],
                'diffusion_noise_schedule': 'cosine',
                'transition': 'marginal',
                'extra_features': None
            },
            'train': {'lr': 1e-4, 'weight_decay': 1e-4, 'batch_size': 2},
            'dataset': {'name': 'qm9', 'remove_h': True}
        })

        # Mock dataset_infos
        dataset_infos = MagicMock()
        dataset_infos.input_dims = {'X': 5, 'E': 4, 'y': 1}
        dataset_infos.output_dims = {'X': 5, 'E': 4, 'y': 1}
        dataset_infos.node_types = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0])
        dataset_infos.edge_types = torch.tensor([200.0, 50.0, 10.0, 5.0])
        
        # Mock nodes distribution
        nodes_dist = MagicMock()
        nodes_dist.sample_n.return_value = torch.tensor([10, 12]) # batch size 2
        dataset_infos.nodes_dist = nodes_dist

        # Mock other components
        train_metrics = MagicMock()
        sampling_metrics = MagicMock()
        visualization_tools = MagicMock()
        extra_features = MagicMock()
        extra_features.return_value = MagicMock(X=torch.zeros(2, 12, 0), E=torch.zeros(2, 12, 12, 0), y=torch.zeros(2, 0))
        domain_features = MagicMock()
        domain_features.return_value = MagicMock(X=torch.zeros(2, 12, 0), E=torch.zeros(2, 12, 12, 0), y=torch.zeros(2, 0))
        
        # Initialize model
        model = DiscreteDenoisingDiffusion(cfg, dataset_infos, train_metrics, sampling_metrics, 
                                           visualization_tools, extra_features, domain_features)
        
        # Move to CPU for testing
        # model.device is a property in LightningModule, usually
        # But we can just use 'cpu'
        
        # Test sample_batch_ddim with Guidance
        batch_size = 2
        ddim_steps = 10
        guidance_scale = 2.0
        
        print(f"Running DDIM sampling with guidance_scale={guidance_scale}...")
        # We need to ensure model is on CPU
        model.eval()
        samples = model.sample_batch_ddim(batch_id=0, batch_size=batch_size, ddim_steps=ddim_steps, guidance_scale=guidance_scale)
        
        self.assertEqual(len(samples), batch_size)
        print("DDIM sampling with Guidance successful!")
        
        # Verify output structure
        for sample in samples:
            atom_types, edge_types = sample
            self.assertTrue(isinstance(atom_types, torch.Tensor))
            self.assertTrue(isinstance(edge_types, torch.Tensor))
            print(f"Sampled graph with {atom_types.shape[0]} atoms")

if __name__ == '__main__':
    unittest.main()
