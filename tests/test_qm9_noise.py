import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noise import QM9Noise

def test_qm9_noise():
    print("Testing QM9Noise implementation...")
    
    # Initialize
    noise = QM9Noise()
    print("QM9Noise initialized.")
    print(f"Node marginals: {noise.node_marginals}")
    print(f"Edge marginals: {noise.edge_marginals}")
    
    # Test get_Qt
    bs = 5
    beta_t = torch.rand(bs)
    device = torch.device('cpu')
    
    qt = noise.get_Qt(beta_t, device)
    
    print("\nTesting get_Qt:")
    print(f"Qt.X shape: {qt.X.shape}")
    print(f"Qt.E shape: {qt.E.shape}")
    
    # Check shapes
    assert qt.X.shape == (bs, 4, 4), f"Expected (5, 4, 4), got {qt.X.shape}"
    assert qt.E.shape == (bs, 5, 5), f"Expected (5, 5, 5), got {qt.E.shape}"
    
    # Check sums (should sum to 1 along the last dimension)
    x_sums = qt.X.sum(dim=-1)
    e_sums = qt.E.sum(dim=-1)
    
    assert torch.allclose(x_sums, torch.ones_like(x_sums)), "Qt.X rows do not sum to 1"
    assert torch.allclose(e_sums, torch.ones_like(e_sums)), "Qt.E rows do not sum to 1"
    print("Qt shapes and probability sums verified.")
    
    # Test get_Qt_bar
    alpha_bar_t = torch.rand(bs)
    qt_bar = noise.get_Qt_bar(alpha_bar_t, device)
    
    print("\nTesting get_Qt_bar:")
    print(f"Qt_bar.X shape: {qt_bar.X.shape}")
    print(f"Qt_bar.E shape: {qt_bar.E.shape}")
    
    # Check shapes
    assert qt_bar.X.shape == (bs, 4, 4), f"Expected (5, 4, 4), got {qt_bar.X.shape}"
    assert qt_bar.E.shape == (bs, 5, 5), f"Expected (5, 5, 5), got {qt_bar.E.shape}"
    
    # Check sums
    x_bar_sums = qt_bar.X.sum(dim=-1)
    e_bar_sums = qt_bar.E.sum(dim=-1)
    
    assert torch.allclose(x_bar_sums, torch.ones_like(x_bar_sums)), "Qt_bar.X rows do not sum to 1"
    assert torch.allclose(e_bar_sums, torch.ones_like(e_bar_sums)), "Qt_bar.E rows do not sum to 1"
    print("Qt_bar shapes and probability sums verified.")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_qm9_noise()
