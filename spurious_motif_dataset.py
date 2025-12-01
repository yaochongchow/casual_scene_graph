"""
Spurious-Motif Benchmark Dataset Implementation
Based on: "Discovering Invariant Rationales for Graph Neural Networks" (DIR Paper)

This dataset generates synthetic graphs where:
- Causal Motifs (House, Cycle, Grid) determine the graph label
- Spurious Motifs (Star, Wheel, Ladder) act as confounders
- Training data has strong spurious correlations
- Test data has randomized correlations
"""

import os
import os.path as osp
import random
from typing import Callable, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data, InMemoryDataset

# =============================================================================
# Motif Generation Functions
# =============================================================================

def generate_house_motif() -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a House motif (5 nodes).
    Structure: A square with a triangular roof.
    
        4
       / \
      3---2
      |   |
      0---1
    
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Square base
        (2, 4), (3, 4)                     # Triangular roof
    ]
    return edges, 5


def generate_cycle_motif(size: int = 6) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Cycle motif.
    Structure: A ring of connected nodes.
    
      1---2
     /     \
    0       3
     \     /
      5---4
    
    Args:
        size: Number of nodes in the cycle
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    edges = [(i, (i + 1) % size) for i in range(size)]
    return edges, size


def generate_grid_motif(rows: int = 3, cols: int = 3) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Grid motif.
    Structure: A 2D grid of connected nodes.
    
    0---1---2
    |   |   |
    3---4---5
    |   |   |
    6---7---8
    
    Args:
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    edges = []
    num_nodes = rows * cols
    
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            # Connect to right neighbor
            if j < cols - 1:
                edges.append((node, node + 1))
            # Connect to bottom neighbor
            if i < rows - 1:
                edges.append((node, node + cols))
    
    return edges, num_nodes


def generate_star_motif(num_leaves: int = 5) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Star motif.
    Structure: A central hub connected to multiple leaf nodes.
    
        1
        |
    4---0---2
        |
        3
    
    Args:
        num_leaves: Number of leaf nodes
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    # Node 0 is the center, nodes 1 to num_leaves are leaves
    edges = [(0, i) for i in range(1, num_leaves + 1)]
    return edges, num_leaves + 1


def generate_wheel_motif(num_spokes: int = 5) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Wheel motif.
    Structure: A cycle with a central hub connected to all cycle nodes.
    
        1
       /|\
      5-0-2
       \|/
      4---3
    
    Args:
        num_spokes: Number of nodes on the wheel rim
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    edges = []
    # Node 0 is center, nodes 1 to num_spokes form the rim
    
    # Connect center to all rim nodes
    for i in range(1, num_spokes + 1):
        edges.append((0, i))
    
    # Connect rim nodes in a cycle
    for i in range(1, num_spokes + 1):
        next_node = (i % num_spokes) + 1
        edges.append((i, next_node))
    
    return edges, num_spokes + 1


def generate_ladder_motif(rungs: int = 3) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Ladder motif.
    Structure: Two parallel chains connected by rungs.
    
    0---1
    |   |
    2---3
    |   |
    4---5
    
    Args:
        rungs: Number of rungs in the ladder
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes in the motif
    """
    edges = []
    num_nodes = rungs * 2
    
    for i in range(rungs):
        left = i * 2
        right = i * 2 + 1
        
        # Rung connection
        edges.append((left, right))
        
        # Vertical connections (rails)
        if i < rungs - 1:
            edges.append((left, left + 2))
            edges.append((right, right + 2))
    
    return edges, num_nodes


def generate_base_graph(num_nodes: int = 10, edge_prob: float = 0.2) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a random Erdos-Renyi base graph.
    
    Args:
        num_nodes: Number of nodes in the base graph
        edge_prob: Probability of edge between any two nodes
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes
    """
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edges.append((i, j))
    
    # Ensure connectivity with a spanning tree backbone
    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        if (parent, i) not in edges and (i, parent) not in edges:
            edges.append((parent, i))
    
    return edges, num_nodes


def generate_ba_base_graph(num_nodes: int = 15, m: int = 2) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate a Barabasi-Albert preferential attachment base graph.
    
    Args:
        num_nodes: Number of nodes
        m: Number of edges to attach from a new node
        
    Returns:
        edges: List of edge tuples
        num_nodes: Number of nodes
    """
    G = nx.barabasi_albert_graph(num_nodes, m)
    edges = list(G.edges())
    return edges, num_nodes


# =============================================================================
# Graph Combination Functions
# =============================================================================

def combine_graphs(
    base_edges: List[Tuple[int, int]],
    base_num_nodes: int,
    motif_edges: List[Tuple[int, int]],
    motif_num_nodes: int,
    num_connections: int = 1
) -> Tuple[List[Tuple[int, int]], int, List[int]]:
    """
    Combine a base graph with a motif by attaching the motif to random nodes.
    
    Args:
        base_edges: Edges of the base graph
        base_num_nodes: Number of nodes in base graph
        motif_edges: Edges of the motif
        motif_num_nodes: Number of nodes in motif
        num_connections: Number of edges connecting base to motif
        
    Returns:
        combined_edges: All edges of combined graph
        total_nodes: Total number of nodes
        motif_node_indices: Indices of nodes belonging to the motif
    """
    # Offset motif node indices
    offset = base_num_nodes
    offset_motif_edges = [(u + offset, v + offset) for u, v in motif_edges]
    
    # Combine edges
    combined_edges = base_edges + offset_motif_edges
    
    # Add connecting edges between base and motif
    base_nodes = list(range(base_num_nodes))
    motif_nodes = list(range(offset, offset + motif_num_nodes))
    
    for _ in range(num_connections):
        base_node = random.choice(base_nodes)
        motif_node = random.choice(motif_nodes)
        combined_edges.append((base_node, motif_node))
    
    total_nodes = base_num_nodes + motif_num_nodes
    motif_node_indices = list(range(offset, offset + motif_num_nodes))
    
    return combined_edges, total_nodes, motif_node_indices


def create_synthetic_graph(
    causal_motif_type: int,
    spurious_motif_type: int,
    base_graph_size: int = 15
) -> Tuple[Data, List[int], List[int]]:
    """
    Create a synthetic graph with causal and spurious motifs.
    
    Args:
        causal_motif_type: 0=House, 1=Cycle, 2=Grid
        spurious_motif_type: 0=Star, 1=Wheel, 2=Ladder
        base_graph_size: Number of nodes in base graph
        
    Returns:
        data: PyTorch Geometric Data object
        causal_nodes: List of node indices in causal motif
        spurious_nodes: List of node indices in spurious motif
    """
    # Generate causal motif
    causal_generators = [generate_house_motif, generate_cycle_motif, generate_grid_motif]
    causal_edges, causal_num_nodes = causal_generators[causal_motif_type]()
    
    # Generate spurious motif
    spurious_generators = [generate_star_motif, generate_wheel_motif, generate_ladder_motif]
    spurious_edges, spurious_num_nodes = spurious_generators[spurious_motif_type]()
    
    # Generate base graph
    base_edges, base_num_nodes = generate_ba_base_graph(base_graph_size, m=2)
    
    # Combine base with causal motif
    combined_edges, total_nodes, causal_nodes = combine_graphs(
        base_edges, base_num_nodes,
        causal_edges, causal_num_nodes,
        num_connections=1
    )
    
    # Combine with spurious motif
    combined_edges, total_nodes, spurious_nodes = combine_graphs(
        combined_edges, total_nodes,
        spurious_edges, spurious_num_nodes,
        num_connections=1
    )
    
    # Create edge index tensor
    edge_index = torch.tensor(combined_edges, dtype=torch.long).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create node type labels (for ground truth only, NOT used in features!)
    # Node types: 0=base, 1=causal, 2=spurious
    node_types = torch.zeros(total_nodes, dtype=torch.long)
    for idx in causal_nodes:
        node_types[idx] = 1
    for idx in spurious_nodes:
        node_types[idx] = 2
    
    # IMPORTANT: Node features must NOT contain type information!
    # The GNN must learn to identify motifs from graph TOPOLOGY only.
    # Using type embeddings would be DATA LEAKAGE - the model could trivially
    # identify causal nodes by looking at features instead of structure.
    #
    # Options for node features:
    # 1. Constant features (all ones) - forces pure structural learning
    # 2. Random features - adds noise, still forces structural learning
    # 3. Degree-based features - topological but not revealing node types
    
    # Use constant features (all ones) to force the model to learn from topology
    x = torch.ones(total_nodes, 10)  # Constant features
    
    # Alternative: Add small random noise to break symmetry
    x = x + 0.1 * torch.randn(total_nodes, 10)
    
    # Create ground truth mask for causal nodes
    causal_mask = torch.zeros(total_nodes, dtype=torch.bool)
    for idx in causal_nodes:
        causal_mask[idx] = True
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([causal_motif_type], dtype=torch.long),
        causal_mask=causal_mask,
        node_types=node_types,
        num_nodes=total_nodes
    )
    
    return data, causal_nodes, spurious_nodes


# =============================================================================
# Spurious-Motif Dataset Class
# =============================================================================

class SpuriousMotif(InMemoryDataset):
    """
    Spurious-Motif Benchmark Dataset from DIR paper.
    
    Generates synthetic graphs where:
    - Causal motifs (House, Cycle, Grid) determine the label (3 classes)
    - Spurious motifs (Star, Wheel, Ladder) act as confounders
    - Training set has strong correlation between causal and spurious motifs
    - Test set has randomized correlation
    
    Args:
        root: Root directory for the dataset
        mode: 'train', 'val', or 'test'
        bias: Correlation strength for training (0.0-1.0)
        num_graphs: Total number of graphs to generate
        transform: Optional transform function
        pre_transform: Optional pre-transform function
    """
    
    CAUSAL_MOTIFS = ['House', 'Cycle', 'Grid']
    SPURIOUS_MOTIFS = ['Star', 'Wheel', 'Ladder']
    
    def __init__(
        self,
        root: str,
        mode: str = 'train',
        bias: float = 0.9,
        num_graphs: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        seed: int = 42
    ):
        self.mode = mode
        self.bias = bias
        self.num_graphs = num_graphs
        self.seed = seed
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load appropriate processed file (compatible with older PyG versions)
        if mode == 'train':
            path = self.processed_paths[0]
        elif mode == 'val':
            path = self.processed_paths[1]
        else:  # test
            path = self.processed_paths[2]
        
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_bias{self.bias}')
    
    @property
    def raw_file_names(self) -> List[str]:
        return []  # No raw files needed
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        pass  # No download needed
    
    def process(self):
        """Generate and process the dataset."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Split sizes (60% train, 20% val, 20% test)
        train_size = int(0.6 * self.num_graphs)
        val_size = int(0.2 * self.num_graphs)
        test_size = self.num_graphs - train_size - val_size
        
        # Generate training data (biased)
        train_data = self._generate_biased_data(train_size, self.bias)
        
        # Generate validation data (slightly biased)
        val_data = self._generate_biased_data(val_size, self.bias * 0.5)
        
        # Generate test data (unbiased/random)
        test_data = self._generate_biased_data(test_size, 0.0)
        
        # Apply pre-filter and pre-transform
        for data_list in [train_data, val_data, test_data]:
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
        
        # Save processed data (compatible with older PyG versions)
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
    
    def _generate_biased_data(
        self,
        num_samples: int,
        bias: float
    ) -> List[Data]:
        """
        Generate graphs with specified spurious correlation strength.
        
        Args:
            num_samples: Number of graphs to generate
            bias: Probability of matching causal and spurious motif types
            
        Returns:
            List of Data objects
        """
        data_list = []
        
        # Ensure balanced classes
        samples_per_class = num_samples // 3
        remainder = num_samples % 3
        
        for causal_type in range(3):
            n_samples = samples_per_class + (1 if causal_type < remainder else 0)
            
            for _ in range(n_samples):
                # Determine spurious motif based on bias
                if random.random() < bias:
                    # Strong correlation: spurious type matches causal type
                    spurious_type = causal_type
                else:
                    # Random spurious motif
                    spurious_type = random.randint(0, 2)
                
                # Create the graph
                data, causal_nodes, spurious_nodes = create_synthetic_graph(
                    causal_type, spurious_type
                )
                
                # Store additional info
                data.causal_motif = torch.tensor([causal_type], dtype=torch.long)
                data.spurious_motif = torch.tensor([spurious_type], dtype=torch.long)
                data.causal_node_list = causal_nodes
                data.spurious_node_list = spurious_nodes
                
                data_list.append(data)
        
        # Shuffle the data
        random.shuffle(data_list)
        
        return data_list
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        stats = {
            'num_graphs': len(self),
            'num_classes': 3,
            'avg_num_nodes': np.mean([d.num_nodes for d in self]),
            'avg_num_edges': np.mean([d.edge_index.size(1) // 2 for d in self]),
            'class_distribution': {},
            'spurious_correlation': {}
        }
        
        # Class distribution
        labels = [d.y.item() for d in self]
        for i in range(3):
            stats['class_distribution'][self.CAUSAL_MOTIFS[i]] = labels.count(i)
        
        # Spurious correlation
        matches = sum(1 for d in self if d.causal_motif.item() == d.spurious_motif.item())
        stats['spurious_correlation']['match_rate'] = matches / len(self)
        
        return stats


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_graph(
    data: Data,
    highlight_causal: bool = True,
    highlight_spurious: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show_legend: bool = True
) -> plt.Figure:
    """
    Visualize a graph with optional highlighting of causal and spurious nodes.
    
    Args:
        data: PyTorch Geometric Data object
        highlight_causal: Whether to highlight causal motif nodes
        highlight_spurious: Whether to highlight spurious motif nodes
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        show_legend: Whether to show legend
        
    Returns:
        matplotlib Figure object
    """
    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    edge_index = data.edge_index.numpy()
    edges = [(edge_index[0, i], edge_index[1, i]) 
             for i in range(edge_index.shape[1]) 
             if edge_index[0, i] < edge_index[1, i]]  # Avoid duplicate edges
    G.add_edges_from(edges)
    
    # Set up colors
    node_colors = []
    node_sizes = []
    
    causal_mask = data.causal_mask.numpy() if hasattr(data, 'causal_mask') else None
    spurious_nodes = set(data.spurious_node_list) if hasattr(data, 'spurious_node_list') else set()
    
    # Color scheme
    BASE_COLOR = '#7FB3D5'       # Light blue for base nodes
    CAUSAL_COLOR = '#E74C3C'     # Red for causal motif
    SPURIOUS_COLOR = '#F39C12'   # Orange for spurious motif
    
    for i in range(data.num_nodes):
        is_causal = causal_mask[i] if causal_mask is not None else False
        is_spurious = i in spurious_nodes
        
        if highlight_causal and is_causal:
            node_colors.append(CAUSAL_COLOR)
            node_sizes.append(500)
        elif highlight_spurious and is_spurious:
            node_colors.append(SPURIOUS_COLOR)
            node_sizes.append(500)
        else:
            node_colors.append(BASE_COLOR)
            node_sizes.append(300)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#4a4a6a',
        width=1.5,
        alpha=0.6,
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='white',
        linewidths=2,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color='white',
        font_weight='bold',
        ax=ax
    )
    
    # Title
    if title is None:
        causal_name = SpuriousMotif.CAUSAL_MOTIFS[data.y.item()]
        spurious_name = SpuriousMotif.SPURIOUS_MOTIFS[data.spurious_motif.item()] \
            if hasattr(data, 'spurious_motif') else 'Unknown'
        title = f"Label: {causal_name} (Causal) | Spurious: {spurious_name}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Legend
    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor=BASE_COLOR, edgecolor='white', label='Base Graph'),
        ]
        if highlight_causal:
            legend_elements.append(
                mpatches.Patch(facecolor=CAUSAL_COLOR, edgecolor='white', label='Causal Motif')
            )
        if highlight_spurious:
            legend_elements.append(
                mpatches.Patch(facecolor=SPURIOUS_COLOR, edgecolor='white', label='Spurious Motif')
            )
        
        legend = ax.legend(
            handles=legend_elements,
            loc='upper left',
            facecolor='#2d2d44',
            edgecolor='white',
            fontsize=11
        )
        plt.setp(legend.get_texts(), color='white')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    return fig


def visualize_motif_comparison(
    dataset: SpuriousMotif,
    num_samples: int = 3,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize multiple graphs showing different motif combinations.
    
    Args:
        dataset: SpuriousMotif dataset
        num_samples: Number of samples per class to show
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(5*num_samples, 15), 
                             facecolor='#1a1a2e')
    
    # Get samples for each class
    samples_by_class = {0: [], 1: [], 2: []}
    for data in dataset:
        label = data.y.item()
        if len(samples_by_class[label]) < num_samples:
            samples_by_class[label].append(data)
    
    for class_idx in range(3):
        for sample_idx in range(num_samples):
            ax = axes[class_idx, sample_idx]
            ax.set_facecolor('#1a1a2e')
            
            if sample_idx < len(samples_by_class[class_idx]):
                data = samples_by_class[class_idx][sample_idx]
                
                # Convert to NetworkX
                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                edge_index = data.edge_index.numpy()
                edges = [(edge_index[0, i], edge_index[1, i]) 
                         for i in range(edge_index.shape[1]) 
                         if edge_index[0, i] < edge_index[1, i]]
                G.add_edges_from(edges)
                
                # Colors
                causal_mask = data.causal_mask.numpy()
                spurious_nodes = set(data.spurious_node_list)
                
                node_colors = []
                for i in range(data.num_nodes):
                    if causal_mask[i]:
                        node_colors.append('#E74C3C')
                    elif i in spurious_nodes:
                        node_colors.append('#F39C12')
                    else:
                        node_colors.append('#7FB3D5')
                
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                
                nx.draw_networkx_edges(G, pos, edge_color='#4a4a6a', 
                                       width=1, alpha=0.6, ax=ax)
                nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                       node_size=200, edgecolors='white',
                                       linewidths=1, ax=ax)
                
                causal_name = SpuriousMotif.CAUSAL_MOTIFS[data.y.item()]
                spurious_name = SpuriousMotif.SPURIOUS_MOTIFS[data.spurious_motif.item()]
                ax.set_title(f'{causal_name} + {spurious_name}', 
                           fontsize=12, color='white', fontweight='bold')
            
            ax.axis('off')
    
    # Row labels
    for class_idx in range(3):
        axes[class_idx, 0].set_ylabel(
            f'Class {class_idx}: {SpuriousMotif.CAUSAL_MOTIFS[class_idx]}',
            fontsize=14, color='white', fontweight='bold'
        )
    
    plt.suptitle('Spurious-Motif Dataset Samples\n(Red=Causal, Orange=Spurious, Blue=Base)',
                 fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    return fig


def visualize_individual_motifs(save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize all individual motif types.
    
    Args:
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='#1a1a2e')
    
    # Causal motifs
    causal_generators = [
        (generate_house_motif, 'House'),
        (generate_cycle_motif, 'Cycle'),
        (generate_grid_motif, 'Grid')
    ]
    
    # Spurious motifs
    spurious_generators = [
        (generate_star_motif, 'Star'),
        (generate_wheel_motif, 'Wheel'),
        (generate_ladder_motif, 'Ladder')
    ]
    
    for idx, (gen_func, name) in enumerate(causal_generators):
        ax = axes[0, idx]
        ax.set_facecolor('#1a1a2e')
        
        edges, num_nodes = gen_func()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw_networkx_edges(G, pos, edge_color='#4a4a6a', width=2, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color='#E74C3C', node_size=400,
                               edgecolors='white', linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', ax=ax)
        
        ax.set_title(f'Causal: {name}', fontsize=14, color='white', fontweight='bold')
        ax.axis('off')
    
    for idx, (gen_func, name) in enumerate(spurious_generators):
        ax = axes[1, idx]
        ax.set_facecolor('#1a1a2e')
        
        edges, num_nodes = gen_func()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw_networkx_edges(G, pos, edge_color='#4a4a6a', width=2, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color='#F39C12', node_size=400,
                               edgecolors='white', linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', ax=ax)
        
        ax.set_title(f'Spurious: {name}', fontsize=14, color='white', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Motif Types in Spurious-Motif Dataset',
                 fontsize=18, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    return fig


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """Demo of the Spurious-Motif dataset."""
    print("=" * 70)
    print("  SPURIOUS-MOTIF BENCHMARK DATASET")
    print("  Based on: Discovering Invariant Rationales (DIR) Paper")
    print("=" * 70)
    
    # Create dataset
    root = './data/spurious_motif'
    
    print("\n[1] Creating datasets...")
    print(f"    Root directory: {root}")
    print(f"    Total graphs: 1000")
    print(f"    Bias (spurious correlation): 0.9")
    
    # Create train, val, test datasets
    train_dataset = SpuriousMotif(root, mode='train', bias=0.9, num_graphs=1000)
    val_dataset = SpuriousMotif(root, mode='val', bias=0.9, num_graphs=1000)
    test_dataset = SpuriousMotif(root, mode='test', bias=0.9, num_graphs=1000)
    
    print(f"\n[2] Dataset Statistics:")
    print(f"    Training set: {len(train_dataset)} graphs")
    print(f"    Validation set: {len(val_dataset)} graphs")
    print(f"    Test set: {len(test_dataset)} graphs")
    
    # Training statistics
    train_stats = train_dataset.get_statistics()
    print(f"\n[3] Training Set Details (Biased):")
    print(f"    Average nodes per graph: {train_stats['avg_num_nodes']:.1f}")
    print(f"    Average edges per graph: {train_stats['avg_num_edges']:.1f}")
    print(f"    Class distribution: {train_stats['class_distribution']}")
    print(f"    Spurious correlation (match rate): {train_stats['spurious_correlation']['match_rate']:.2%}")
    
    # Test statistics
    test_stats = test_dataset.get_statistics()
    print(f"\n[4] Test Set Details (Unbiased):")
    print(f"    Average nodes per graph: {test_stats['avg_num_nodes']:.1f}")
    print(f"    Average edges per graph: {test_stats['avg_num_edges']:.1f}")
    print(f"    Class distribution: {test_stats['class_distribution']}")
    print(f"    Spurious correlation (match rate): {test_stats['spurious_correlation']['match_rate']:.2%}")
    
    # Sample data
    print(f"\n[5] Sample Graph:")
    sample = train_dataset[0]
    print(f"    Number of nodes: {sample.num_nodes}")
    print(f"    Number of edges: {sample.edge_index.size(1) // 2}")
    print(f"    Feature dimension: {sample.x.size(1)}")
    print(f"    Label (Causal Motif): {SpuriousMotif.CAUSAL_MOTIFS[sample.y.item()]}")
    print(f"    Spurious Motif: {SpuriousMotif.SPURIOUS_MOTIFS[sample.spurious_motif.item()]}")
    print(f"    Causal nodes: {sample.causal_node_list}")
    print(f"    Spurious nodes: {sample.spurious_node_list}")
    
    # Visualizations
    print(f"\n[6] Creating visualizations...")
    
    os.makedirs('./visualizations', exist_ok=True)
    
    # Individual motifs
    visualize_individual_motifs('./visualizations/motif_types.png')
    print("    Saved: ./visualizations/motif_types.png")
    
    # Single graph visualization
    visualize_graph(
        sample,
        highlight_causal=True,
        highlight_spurious=True,
        save_path='./visualizations/sample_graph.png'
    )
    print("    Saved: ./visualizations/sample_graph.png")
    
    # Comparison visualization
    visualize_motif_comparison(
        train_dataset,
        num_samples=3,
        save_path='./visualizations/dataset_samples.png'
    )
    print("    Saved: ./visualizations/dataset_samples.png")
    
    print("\n" + "=" * 70)
    print("  DATASET READY FOR USE!")
    print("=" * 70)
    
    # Show example usage
    print("\n[Example Usage]")
    print("""
    from spurious_motif_dataset import SpuriousMotif, visualize_graph
    
    # Load datasets
    train_data = SpuriousMotif(root='./data/spurious_motif', mode='train')
    test_data = SpuriousMotif(root='./data/spurious_motif', mode='test')
    
    # Use with DataLoader
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Visualize a graph
    visualize_graph(train_data[0], highlight_causal=True)
    plt.show()
    """)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_data, val_data, test_data = main()
    plt.show()

