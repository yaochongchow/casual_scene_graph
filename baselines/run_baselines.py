import torch
import os.path as osp
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

# --- Model Definitions ---

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index)
            z = F.relu(z)
        return z

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = GConv(input_dim, hidden_dim, num_layers=2)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        g = global_add_pool(z, batch)
        return self.project(g)

# --- Augmentations ---

def drop_edge(edge_index, p=0.2):
    if p == 0:
        return edge_index
    edge_index, _ = dropout_edge(edge_index, p=p)
    return edge_index

def mask_feature(x, p=0.2):
    if p == 0:
        return x
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, mask] = 0
    return x

# --- Loss Function ---

def info_nce_loss(z1, z2, temperature=0.5):
    # z1, z2: (batch_size, proj_dim)
    batch_size = z1.size(0)
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # Positive pairs are on the diagonal
    positives = torch.diag(sim_matrix)
    
    # Denominator: sum of exp of all similarities
    # We want to maximize positive similarity and minimize negative similarity
    # Loss = -log( exp(pos) / (sum(exp(all)) )
    
    # For numerical stability
    max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0]
    sim_matrix = sim_matrix - max_sim.detach()
    positives = positives - max_sim.squeeze().detach()
    
    exp_sim = torch.exp(sim_matrix)
    sum_exp_sim = torch.sum(exp_sim, dim=1)
    
    loss = -positives + torch.log(sum_exp_sim)
    return loss.mean()

# --- Training Loop ---

def train_baseline(model, dataloader, optimizer, aug_type='grace', device=None):
    model.train()
    if device is None:
        device = next(model.parameters()).device
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        if data.x is None:
             num_nodes = data.batch.size(0)
             data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

        # Augmentation 1
        if aug_type == 'grace':
            # GRACE: Edge drop + Feature mask
            x1 = mask_feature(data.x, p=0.2)
            edge_index1 = drop_edge(data.edge_index, p=0.2)
            
            x2 = mask_feature(data.x, p=0.2)
            edge_index2 = drop_edge(data.edge_index, p=0.2)
            
        elif aug_type == 'graphcl':
            # GraphCL: Stronger augmentations
            x1 = mask_feature(data.x, p=0.3)
            edge_index1 = drop_edge(data.edge_index, p=0.3)
            
            x2 = mask_feature(data.x, p=0.4)
            edge_index2 = drop_edge(data.edge_index, p=0.4)

        z1 = model(x1, edge_index1, data.batch)
        z2 = model(x2, edge_index2, data.batch)
        
        loss = info_nce_loss(z1, z2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)
def main():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.good_motif import load_good_motif
    
    print("Running Baselines on GOOD-Motif...")
    
    # Load GOOD-Motif dataset
    dataset_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'external', 'GOOD_datasets')
    split_payload, meta_info = load_good_motif(
        dataset_root=dataset_root,
        domain="basis",
        shift="concept",
        batch_size=32,
        num_workers=0
    )
    
    train_loaders = split_payload["train_env_loaders"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get input dimension from first sample
    if len(train_loaders) > 0:
        first_loader = list(train_loaders.values())[0]
        sample = next(iter(first_loader))
        input_dim = max(sample.num_node_features, 1)
    else:
        input_dim = 1
    
    hidden_dim = 64
    proj_dim = 64
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 1. GRACE
    print("\n--- Training GRACE on GOOD-Motif ---")
    model_grace = Encoder(input_dim, hidden_dim, proj_dim).to(device)
    opt_grace = Adam(model_grace.parameters(), lr=0.001)
    
    for epoch in range(1, 21):
        epoch_loss = 0
        n_batches = 0
        for env_id, loader in train_loaders.items():
            loss = train_baseline(model_grace, loader, opt_grace, aug_type='grace', device=device)
            epoch_loss += loss
            n_batches += 1
        epoch_loss /= n_batches if n_batches > 0 else 1
        if epoch % 5 == 0:
            print(f'Epoch {epoch:02d}, Loss: {epoch_loss:.4f}')
    
    print("GRACE training complete")
    
    # 2. GraphCL
    print("\n--- Training GraphCL on GOOD-Motif ---")
    model_graphcl = Encoder(input_dim, hidden_dim, proj_dim).to(device)
    opt_graphcl = Adam(model_graphcl.parameters(), lr=0.001)
    
    for epoch in range(1, 21):
        epoch_loss = 0
        n_batches = 0
        for env_id, loader in train_loaders.items():
            loss = train_baseline(model_graphcl, loader, opt_graphcl, aug_type='graphcl', device=device)
            epoch_loss += loss
            n_batches += 1
        epoch_loss /= n_batches if n_batches > 0 else 1
        if epoch % 5 == 0:
            print(f'Epoch {epoch:02d}, Loss: {epoch_loss:.4f}')
    
    print("GraphCL training complete")
    
    # Save results
    with open('results/week1.txt', 'w') as f:
        f.write("Baseline Results on GOOD-Motif (Concept Shift)\n")
        f.write("=" * 50 + "\n\n")
        f.write("GRACE: Training completed successfully.\n")
        f.write("GraphCL: Training completed successfully.\n\n")
        f.write("Note: Full OOD accuracy evaluation requires training a classifier head.\n")
        f.write("This is a simplified baseline implementation.\n")
    
    print("\nBaselines run complete. Results saved to results/week1.txt")


if __name__ == '__main__':
    import os
    main()
