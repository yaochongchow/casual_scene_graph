"""
Visual Genome Scene Graph Dataset Loader

Adapts the Visual Genome Scene Graph dataset for Graph Classification.
Original task: Scene Graph Generation (predict objects and relationships)
Adapted task: Scene Classification (predict scene type from graph structure)

Scene Classification Heuristic:
- Class 0 (Bedroom): Contains 'bed', 'pillow', 'blanket', 'lamp', 'dresser'
- Class 1 (Bathroom): Contains 'toilet', 'sink', 'bathtub', 'shower', 'mirror'
- Class 2 (Outdoor): Contains 'tree', 'sky', 'grass', 'cloud', 'mountain'
- Class 3 (Kitchen): Contains 'stove', 'refrigerator', 'oven', 'microwave', 'cabinet'
- Class 4 (Other): Default for unclassified scenes

Reference: Visual Genome (Krishna et al., 2017)
https://visualgenome.org/
"""

import json
import os
import os.path as osp
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


# =============================================================================
# Scene Classification Rules
# =============================================================================

# Keywords for each scene class (lowercase)
SCENE_KEYWORDS = {
    0: {  # Bedroom
        'bed', 'pillow', 'blanket', 'lamp', 'dresser', 'nightstand',
        'bedroom', 'mattress', 'comforter', 'headboard', 'alarm clock'
    },
    1: {  # Bathroom
        'toilet', 'sink', 'bathtub', 'shower', 'mirror', 'bathroom',
        'towel', 'faucet', 'tile', 'soap', 'toothbrush'
    },
    2: {  # Outdoor
        'tree', 'sky', 'grass', 'cloud', 'mountain', 'outdoor', 'street',
        'road', 'building', 'car', 'sun', 'flower', 'bush', 'sidewalk',
        'beach', 'ocean', 'water', 'park', 'field'
    },
    3: {  # Kitchen
        'stove', 'refrigerator', 'oven', 'microwave', 'cabinet', 'kitchen',
        'counter', 'dish', 'pot', 'pan', 'plate', 'cup', 'bowl', 'fork',
        'knife', 'spoon', 'food'
    },
}

SCENE_NAMES = {
    0: 'Bedroom',
    1: 'Bathroom',
    2: 'Outdoor',
    3: 'Kitchen',
    4: 'Other'
}


def classify_scene(objects: List[str]) -> int:
    """
    Classify a scene based on objects present.
    
    Uses keyword matching with priority ordering.
    
    Args:
        objects: List of object names in the scene
        
    Returns:
        Scene class index (0-4)
    """
    objects_lower = {obj.lower() for obj in objects}
    
    # Count matches for each scene type
    scores = {}
    for class_id, keywords in SCENE_KEYWORDS.items():
        matches = objects_lower & keywords
        scores[class_id] = len(matches)
    
    # Get class with most matches
    if max(scores.values()) > 0:
        best_class = max(scores, key=scores.get)
        return best_class
    
    # Default to "Other"
    return 4


# =============================================================================
# Visual Genome Dataset
# =============================================================================

class VisualGenomeSceneGraphs(InMemoryDataset):
    """
    Visual Genome Scene Graphs adapted for Graph Classification.
    
    Converts scene graphs (objects + relationships) into graph classification
    where the task is to predict the scene type (Bedroom, Bathroom, etc.).
    
    This tests whether GNNs can learn structural patterns that indicate
    scene types, rather than relying on node features (which are randomized).
    
    Args:
        root: Root directory for the dataset
        split: 'train', 'val', or 'test'
        transform: Optional transform function
        pre_transform: Optional pre-transform function
        pre_filter: Optional pre-filter function
        max_graphs: Maximum number of graphs to process (for efficiency)
        feature_dim: Dimension of node features (random initialization)
        feature_type: 'random', 'onehot', or 'degree'
        min_nodes: Minimum number of nodes to include a graph
        min_edges: Minimum number of edges to include a graph
    """
    
    SCENE_NAMES = SCENE_NAMES
    NUM_CLASSES = 5
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        max_graphs: int = 5000,
        feature_dim: int = 64,
        feature_type: str = 'random',
        min_nodes: int = 3,
        min_edges: int = 2,
        seed: int = 42
    ):
        self.split = split
        self.max_graphs = max_graphs
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.seed = seed
        
        # Object name to ID mapping (built during processing)
        self.object_to_id: Dict[str, int] = {}
        self.num_object_classes = 0
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load appropriate split (compatible with older PyG versions)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:  # test
            path = self.processed_paths[2]
        
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_dim{self.feature_dim}')
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['scene_graphs.json']
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt', 'metadata.pt']
    
    def download(self):
        """
        Provide instructions for downloading Visual Genome data.
        """
        if not osp.exists(osp.join(self.raw_dir, 'scene_graphs.json')):
            raise FileNotFoundError(
                f"Visual Genome scene_graphs.json not found in {self.raw_dir}.\n"
                f"Please download from: https://visualgenome.org/api/v0/api_home.html\n"
                f"Direct link: https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip\n"
                f"Extract and place scene_graphs.json in: {self.raw_dir}"
            )
    
    def process(self):
        """
        Process raw scene graphs into PyG Data objects.
        """
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Load raw scene graphs
        json_path = osp.join(self.raw_dir, 'scene_graphs.json')
        
        print(f"Loading scene graphs from {json_path}...")
        with open(json_path, 'r') as f:
            scene_graphs = json.load(f)
        
        print(f"Found {len(scene_graphs)} scene graphs")
        
        # Build object vocabulary
        print("Building object vocabulary...")
        self._build_vocabulary(scene_graphs[:self.max_graphs])
        
        # Process graphs
        print(f"Processing up to {self.max_graphs} graphs...")
        data_list = []
        skipped = {'empty': 0, 'disconnected': 0, 'small': 0}
        
        for sg in tqdm(scene_graphs[:self.max_graphs], desc="Processing"):
            try:
                data = self._process_scene_graph(sg)
                if data is not None:
                    data_list.append(data)
                else:
                    skipped['small'] += 1
            except Exception as e:
                skipped['empty'] += 1
        
        print(f"Processed {len(data_list)} valid graphs")
        print(f"Skipped: {skipped}")
        
        if len(data_list) == 0:
            raise ValueError("No valid graphs found! Check the scene_graphs.json format.")
        
        # Shuffle and split (70/15/15)
        random.shuffle(data_list)
        
        n = len(data_list)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size + val_size]
        test_data = data_list[train_size + val_size:]
        
        print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Print class distribution
        self._print_class_distribution(train_data, "Train")
        self._print_class_distribution(test_data, "Test")
        
        # Apply pre-filter and pre-transform
        for data_split in [train_data, val_data, test_data]:
            if self.pre_filter is not None:
                data_split = [d for d in data_split if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_split = [self.pre_transform(d) for d in data_split]
        
        # Save processed data (compatible with older PyG versions)
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        
        # Save metadata
        metadata = {
            'object_to_id': self.object_to_id,
            'num_object_classes': self.num_object_classes,
            'feature_dim': self.feature_dim,
            'num_scene_classes': self.NUM_CLASSES,
            'scene_names': self.SCENE_NAMES
        }
        torch.save(metadata, self.processed_paths[3])
    
    def _build_vocabulary(self, scene_graphs: List[dict]):
        """Build object name to ID mapping."""
        all_objects = set()
        
        for sg in scene_graphs:
            objects = sg.get('objects', [])
            for obj in objects:
                names = obj.get('names', [obj.get('name', 'unknown')])
                if isinstance(names, list) and len(names) > 0:
                    all_objects.add(names[0].lower())
                elif isinstance(names, str):
                    all_objects.add(names.lower())
        
        self.object_to_id = {name: idx for idx, name in enumerate(sorted(all_objects))}
        self.num_object_classes = len(self.object_to_id)
        print(f"Vocabulary size: {self.num_object_classes} objects")
    
    def _process_scene_graph(self, sg: dict) -> Optional[Data]:
        """
        Convert a single scene graph to PyG Data.
        
        Args:
            sg: Scene graph dictionary from Visual Genome JSON
            
        Returns:
            Data object or None if invalid
        """
        objects = sg.get('objects', [])
        relationships = sg.get('relationships', [])
        
        # Skip empty graphs
        if len(objects) < self.min_nodes:
            return None
        
        # Build object ID to index mapping (contiguous indices)
        obj_id_to_idx = {}
        object_names = []
        
        for idx, obj in enumerate(objects):
            obj_id = obj.get('object_id', idx)
            obj_id_to_idx[obj_id] = idx
            
            # Get object name
            names = obj.get('names', [obj.get('name', 'unknown')])
            if isinstance(names, list) and len(names) > 0:
                name = names[0]
            elif isinstance(names, str):
                name = names
            else:
                name = 'unknown'
            object_names.append(name)
        
        num_nodes = len(objects)
        
        # Build edge index from relationships
        edge_list = []
        for rel in relationships:
            subj_id = rel.get('subject_id', rel.get('subject', {}).get('object_id'))
            obj_id = rel.get('object_id', rel.get('object', {}).get('object_id'))
            
            if subj_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                src = obj_id_to_idx[subj_id]
                dst = obj_id_to_idx[obj_id]
                edge_list.append([src, dst])
                edge_list.append([dst, src])  # Make undirected
        
        # Skip graphs with too few edges
        if len(edge_list) < self.min_edges * 2:  # *2 because undirected
            return None
        
        # Create edge_index tensor
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create node features
        x = self._create_node_features(object_names, num_nodes)
        
        # Classify scene
        y = classify_scene(object_names)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([y], dtype=torch.long),
            num_nodes=num_nodes
        )
        
        # Store additional info
        data.image_id = sg.get('image_id', -1)
        data.object_names = object_names
        data.scene_name = self.SCENE_NAMES[y]
        
        return data
    
    def _create_node_features(
        self,
        object_names: List[str],
        num_nodes: int
    ) -> torch.Tensor:
        """
        Create node feature matrix.
        
        Args:
            object_names: List of object names
            num_nodes: Number of nodes
            
        Returns:
            Node feature tensor [num_nodes, feature_dim]
        """
        if self.feature_type == 'random':
            # Random features (tests structural learning)
            x = torch.randn(num_nodes, self.feature_dim)
            
        elif self.feature_type == 'onehot':
            # One-hot encoding of object category
            if self.num_object_classes > 0:
                x = torch.zeros(num_nodes, min(self.num_object_classes, self.feature_dim))
                for i, name in enumerate(object_names):
                    name_lower = name.lower()
                    if name_lower in self.object_to_id:
                        idx = self.object_to_id[name_lower]
                        if idx < x.size(1):
                            x[i, idx] = 1.0
            else:
                x = torch.randn(num_nodes, self.feature_dim)
                
        elif self.feature_type == 'degree':
            # Will be computed later after edge_index is available
            # For now, use placeholder
            x = torch.ones(num_nodes, self.feature_dim)
            
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return x
    
    def _print_class_distribution(self, data_list: List[Data], split_name: str):
        """Print class distribution for a split."""
        labels = [d.y.item() for d in data_list]
        counter = Counter(labels)
        
        print(f"\n{split_name} class distribution:")
        for class_id in sorted(counter.keys()):
            count = counter[class_id]
            pct = 100 * count / len(labels)
            print(f"  {self.SCENE_NAMES[class_id]}: {count} ({pct:.1f}%)")
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        labels = [d.y.item() for d in self]
        num_nodes = [d.num_nodes for d in self]
        num_edges = [d.edge_index.size(1) // 2 for d in self]
        
        return {
            'num_graphs': len(self),
            'num_classes': self.NUM_CLASSES,
            'avg_nodes': sum(num_nodes) / len(num_nodes),
            'avg_edges': sum(num_edges) / len(num_edges),
            'class_distribution': dict(Counter(labels)),
            'feature_dim': self.feature_dim
        }


# =============================================================================
# Synthetic Visual Genome (for testing without real data)
# =============================================================================

class SyntheticVisualGenome(InMemoryDataset):
    """
    Synthetic Visual Genome-like dataset for testing.
    
    Generates random scene graphs with structure similar to Visual Genome
    but without requiring the actual data download.
    
    Useful for:
    - Testing the pipeline before downloading real data
    - Ablation studies on graph structure
    - Debugging
    
    Args:
        root: Root directory
        split: 'train', 'val', or 'test'
        num_graphs: Number of graphs to generate
        feature_dim: Node feature dimension
    """
    
    SCENE_NAMES = SCENE_NAMES
    NUM_CLASSES = 5
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_graphs: int = 1000,
        feature_dim: int = 64,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 42
    ):
        self.split = split
        self.num_graphs = num_graphs
        self.feature_dim = feature_dim
        self.seed = seed
        
        super().__init__(root, transform, pre_transform)
        
        # Load appropriate split (compatible with older PyG versions)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]
        
        self.data, self.slices = torch.load(path)
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'synthetic_processed_n{self.num_graphs}')
    
    @property
    def raw_file_names(self) -> List[str]:
        return []  # No raw files needed
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        pass  # No download needed
    
    def process(self):
        """Generate synthetic scene graphs."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        print(f"Generating {self.num_graphs} synthetic scene graphs...")
        
        data_list = []
        
        for i in tqdm(range(self.num_graphs), desc="Generating"):
            data = self._generate_scene_graph()
            data_list.append(data)
        
        # Split
        random.shuffle(data_list)
        train_size = int(0.7 * len(data_list))
        val_size = int(0.15 * len(data_list))
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size + val_size]
        test_data = data_list[train_size + val_size:]
        
        # Save (compatible with older PyG versions)
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
    
    def _generate_scene_graph(self) -> Data:
        """Generate a single synthetic scene graph."""
        # Random scene type
        scene_class = random.randint(0, self.NUM_CLASSES - 1)
        
        # Number of objects (5-30)
        num_nodes = random.randint(5, 30)
        
        # Generate edges (Erdos-Renyi-like)
        edge_prob = random.uniform(0.1, 0.3)
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    edges.append([i, j])
                    edges.append([j, i])
        
        # Ensure connectivity
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            if [parent, i] not in edges:
                edges.append([parent, i])
                edges.append([i, parent])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Random features
        x = torch.randn(num_nodes, self.feature_dim)
        
        # Create data
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([scene_class], dtype=torch.long),
            num_nodes=num_nodes
        )
        data.scene_name = self.SCENE_NAMES[scene_class]
        
        return data


# =============================================================================
# Loader Function
# =============================================================================

def load_visual_genome(
    root: str,
    batch_size: int,
    max_graphs: int = 5000,
    feature_dim: int = 64,
    feature_type: str = 'random',
    num_workers: int = 4,
    use_synthetic: bool = False
) -> Dict[str, any]:
    """
    Load Visual Genome dataset for graph classification.
    
    Args:
        root: Root directory for data
        batch_size: Batch size for loaders
        max_graphs: Maximum graphs to use
        feature_dim: Node feature dimension
        feature_type: 'random', 'onehot', or 'degree'
        num_workers: Data loading workers
        use_synthetic: Use synthetic data instead of real VG
        
    Returns:
        Dictionary with loaders and metadata
    """
    from torch_geometric.loader import DataLoader
    
    dataset_root = osp.join(root, 'visual_genome')
    
    if use_synthetic:
        DatasetClass = SyntheticVisualGenome
        extra_kwargs = {'num_graphs': max_graphs}
    else:
        DatasetClass = VisualGenomeSceneGraphs
        extra_kwargs = {
            'max_graphs': max_graphs,
            'feature_type': feature_type
        }
    
    # Load splits
    train_dataset = DatasetClass(
        root=dataset_root,
        split='train',
        feature_dim=feature_dim,
        **extra_kwargs
    )
    val_dataset = DatasetClass(
        root=dataset_root,
        split='val',
        feature_dim=feature_dim,
        **extra_kwargs
    )
    test_dataset = DatasetClass(
        root=dataset_root,
        split='test',
        feature_dim=feature_dim,
        **extra_kwargs
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'input_dim': feature_dim,
        'output_dim': VisualGenomeSceneGraphs.NUM_CLASSES,
        'has_masks': False,
        'metric': 'accuracy',
        'task_type': 'multiclass',
        'dataset_info': {
            'name': 'Visual Genome Scene Graphs' + (' (Synthetic)' if use_synthetic else ''),
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_test': len(test_dataset),
            'description': 'Scene classification from Visual Genome scene graphs',
            'scene_classes': list(SCENE_NAMES.values())
        }
    }


# =============================================================================
# Demo
# =============================================================================

def main():
    """Demo of Visual Genome loader."""
    print("Visual Genome Scene Graph Loader Demo")
    print("=" * 60)
    
    # Use synthetic data for demo (no download required)
    print("\nLoading synthetic Visual Genome data...")
    
    data = load_visual_genome(
        root='./data',
        batch_size=32,
        max_graphs=500,
        feature_dim=64,
        use_synthetic=True,
        num_workers=0
    )
    
    print(f"\nDataset Info:")
    print(f"  Train: {data['dataset_info']['num_train']}")
    print(f"  Val: {data['dataset_info']['num_val']}")
    print(f"  Test: {data['dataset_info']['num_test']}")
    print(f"  Input dim: {data['input_dim']}")
    print(f"  Output dim: {data['output_dim']}")
    print(f"  Scene classes: {data['dataset_info']['scene_classes']}")
    
    # Test batch
    batch = next(iter(data['train_loader']))
    print(f"\nSample batch:")
    print(f"  x shape: {batch.x.shape}")
    print(f"  edge_index shape: {batch.edge_index.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  Labels: {batch.y.tolist()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nTo use real Visual Genome data:")
    print("1. Download scene_graphs.json from https://visualgenome.org/")
    print("2. Place in: ./data/visual_genome/raw/scene_graphs.json")
    print("3. Set use_synthetic=False")


if __name__ == '__main__':
    main()

