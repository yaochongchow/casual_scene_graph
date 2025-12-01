"""
Dataset Factory for Causal-DiffAug

Provides a unified interface for loading multiple graph classification datasets:
- spurious_motif: Synthetic benchmark with known causal/spurious structure
- ogbg-molhiv: Real molecular property prediction (OGB benchmark)
- good-hiv: GOOD benchmark for OOD generalization

Usage:
    from dataset_factory import get_dataset
    
    data = get_dataset('spurious_motif', root='./data', batch_size=32)
    train_loader = data['train_loader']
    input_dim = data['input_dim']
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader


# =============================================================================
# Dataset Registry
# =============================================================================

SUPPORTED_DATASETS = {
    'spurious_motif': {
        'description': 'Synthetic graphs with causal/spurious motifs',
        'metric': 'accuracy',
        'has_masks': True,
        'num_classes': 3,
    },
    'ogbg-molhiv': {
        'description': 'HIV inhibition prediction (OGB molecular)',
        'metric': 'auc',
        'has_masks': False,
        'num_classes': 2,
    },
    'ogbg-molbbbp': {
        'description': 'Blood-brain barrier penetration (OGB molecular)',
        'metric': 'auc',
        'has_masks': False,
        'num_classes': 2,
    },
    'good-hiv': {
        'description': 'HIV prediction with distribution shift (GOOD)',
        'metric': 'accuracy',
        'has_masks': False,
        'num_classes': 2,
    },
    'visual_genome': {
        'description': 'Scene classification from Visual Genome scene graphs',
        'metric': 'accuracy',
        'has_masks': False,
        'num_classes': 5,
    },
    'visual_genome_synthetic': {
        'description': 'Synthetic Visual Genome-like graphs (no download)',
        'metric': 'accuracy',
        'has_masks': False,
        'num_classes': 5,
    },
}


# =============================================================================
# Spurious-Motif Dataset Loader
# =============================================================================

def load_spurious_motif(
    root: str,
    batch_size: int,
    bias: float = 0.9,
    num_graphs: int = 1000,
    seed: int = 42,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Load the Spurious-Motif synthetic dataset.
    
    Args:
        root: Root directory for data
        batch_size: Batch size for loaders
        bias: Spurious correlation strength (0-1)
        num_graphs: Total number of graphs to generate
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with loaders and metadata
    """
    from spurious_motif_dataset import SpuriousMotif
    
    dataset_root = os.path.join(root, 'spurious_motif')
    
    # Load train/val/test splits
    train_dataset = SpuriousMotif(
        root=dataset_root,
        mode='train',
        bias=bias,
        num_graphs=num_graphs,
        seed=seed
    )
    val_dataset = SpuriousMotif(
        root=dataset_root,
        mode='val',
        bias=bias,
        num_graphs=num_graphs,
        seed=seed
    )
    test_dataset = SpuriousMotif(
        root=dataset_root,
        mode='test',
        bias=bias,
        num_graphs=num_graphs,
        seed=seed
    )
    
    # Create data loaders
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
    
    # Get input dimension from first sample
    sample = train_dataset[0]
    input_dim = sample.x.size(1)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'input_dim': input_dim,
        'output_dim': 3,  # House, Cycle, Grid
        'has_masks': True,
        'metric': 'accuracy',
        'task_type': 'multiclass',
        'bias': bias,
        'dataset_info': {
            'name': 'Spurious-Motif',
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_test': len(test_dataset),
            'description': 'Synthetic graphs with causal (House/Cycle/Grid) and spurious (Star/Wheel/Ladder) motifs'
        }
    }


# =============================================================================
# OGB Molecular Dataset Loader
# =============================================================================

def load_ogb_mol(
    name: str,
    root: str,
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Load an OGB molecular property prediction dataset.
    
    Uses standard OGB scaffold splitting for train/val/test.
    
    Args:
        name: Dataset name ('ogbg-molhiv', 'ogbg-molbbbp', etc.)
        root: Root directory for data
        batch_size: Batch size for loaders
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with loaders and metadata
    """
    try:
        from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    except ImportError:
        raise ImportError(
            "OGB not installed. Please install with: pip install ogb"
        )
    
    dataset_root = os.path.join(root, 'ogb')
    
    # Load dataset
    dataset = PygGraphPropPredDataset(name=name, root=dataset_root)
    
    # Get standard OGB scaffold split
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    # Create subset datasets
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    # OGB datasets need feature conversion to float
    def transform_batch(batch):
        """Convert node features to float and handle edge attributes."""
        if batch.x is not None:
            batch.x = batch.x.float()
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        return batch
    
    # Custom collate function to apply transforms
    from torch_geometric.data import Batch
    
    def collate_fn(data_list):
        batch = Batch.from_data_list(data_list)
        return transform_batch(batch)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Get input dimension
    sample = dataset[0]
    if sample.x is not None:
        input_dim = sample.x.size(1)
    else:
        # Some OGB datasets don't have node features - use degree
        input_dim = 1
    
    # OGB evaluator for proper metric calculation
    evaluator = Evaluator(name=name)
    
    # Determine output dimension and task type
    num_tasks = dataset.num_tasks if hasattr(dataset, 'num_tasks') else 1
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'input_dim': input_dim,
        'output_dim': num_tasks,  # Binary classification
        'has_masks': False,
        'metric': 'auc',
        'task_type': 'binary',
        'evaluator': evaluator,
        'dataset_info': {
            'name': name,
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_test': len(test_dataset),
            'description': f'OGB molecular dataset: {name}'
        }
    }


# =============================================================================
# GOOD Benchmark Dataset Loader
# =============================================================================

def load_good_hiv(
    root: str,
    batch_size: int,
    domain: str = 'size',
    shift: str = 'covariate',
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Load the GOOD-HIV dataset for OOD generalization.
    
    GOOD provides multiple domain splits and shift types for testing
    out-of-distribution generalization.
    
    Args:
        root: Root directory for data
        batch_size: Batch size for loaders
        domain: Domain for splitting ('scaffold', 'size')
        shift: Type of distribution shift ('covariate', 'concept')
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with loaders and metadata
    """
    try:
        from GOOD.data import load_dataset, create_dataloader
        from GOOD import config_summoner
    except ImportError:
        # Try alternative import paths
        try:
            from GOOD.data.good_datasets import GOODHIV
            use_legacy = True
        except ImportError:
            raise ImportError(
                "GOOD benchmark not installed. Please install with:\n"
                "pip install GOOD\n"
                "or: git clone https://github.com/divelab/GOOD && pip install -e GOOD"
            )
    
    dataset_root = os.path.join(root, 'GOOD')
    
    try:
        # Try the newer GOOD API first
        from GOOD.data.good_datasets.good_hiv import GOODHIV
        
        # Load dataset with specified domain and shift
        dataset, meta_info = GOODHIV.load(
            dataset_root=dataset_root,
            domain=domain,
            shift=shift,
            generate=True  # Generate if not exists
        )
        
        train_dataset = dataset['train']
        val_dataset = dataset['val'] if 'val' in dataset else dataset['id_val']
        test_dataset = dataset['test'] if 'test' in dataset else dataset['id_test']
        
        # Also get OOD test if available
        ood_test_dataset = dataset.get('ood_test', None)
        
    except Exception as e:
        # Fallback: Try manual loading with PyG
        print(f"GOOD native loading failed ({e}), trying fallback...")
        
        try:
            from torch_geometric.datasets import MoleculeNet
            
            # Use MoleculeNet HIV as fallback
            dataset = MoleculeNet(root=os.path.join(root, 'MoleculeNet'), name='HIV')
            
            # Manual split (80/10/10)
            num_graphs = len(dataset)
            torch.manual_seed(42)
            perm = torch.randperm(num_graphs)
            
            train_size = int(0.8 * num_graphs)
            val_size = int(0.1 * num_graphs)
            
            train_idx = perm[:train_size]
            val_idx = perm[train_size:train_size + val_size]
            test_idx = perm[train_size + val_size:]
            
            train_dataset = dataset[train_idx]
            val_dataset = dataset[val_idx]
            test_dataset = dataset[test_idx]
            ood_test_dataset = None
            
        except Exception as e2:
            raise ImportError(
                f"Could not load GOOD-HIV dataset. Errors:\n"
                f"  GOOD: {e}\n"
                f"  Fallback: {e2}\n"
                f"Please install GOOD benchmark or ensure data is available."
            )
    
    # Transform function for GOOD data
    def transform_batch(batch):
        if batch.x is not None:
            batch.x = batch.x.float()
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
        # Handle labels
        if hasattr(batch, 'y') and batch.y is not None:
            if batch.y.dim() > 1:
                batch.y = batch.y.squeeze(-1)
            batch.y = batch.y.long()
        return batch
    
    from torch_geometric.data import Batch
    
    def collate_fn(data_list):
        batch = Batch.from_data_list(data_list)
        return transform_batch(batch)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # OOD test loader if available
    ood_test_loader = None
    if ood_test_dataset is not None:
        ood_test_loader = DataLoader(
            ood_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # Get input dimension
    sample = train_dataset[0]
    input_dim = sample.x.size(1) if sample.x is not None else 9  # Default for molecules
    
    result = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'input_dim': input_dim,
        'output_dim': 2,  # Binary classification
        'has_masks': False,
        'metric': 'accuracy',
        'task_type': 'binary',
        'domain': domain,
        'shift': shift,
        'dataset_info': {
            'name': f'GOOD-HIV ({domain}, {shift})',
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_test': len(test_dataset),
            'description': f'GOOD HIV benchmark with {domain} domain and {shift} shift'
        }
    }
    
    if ood_test_loader is not None:
        result['ood_test_loader'] = ood_test_loader
        result['ood_test_dataset'] = ood_test_dataset
        result['dataset_info']['num_ood_test'] = len(ood_test_dataset)
    
    return result


# =============================================================================
# Main Factory Function
# =============================================================================

def get_dataset(
    name: str,
    root: str = './data',
    batch_size: int = 32,
    bias: float = 0.9,
    seed: int = 42,
    num_workers: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Factory function to load datasets by name.
    
    Provides a unified interface for multiple graph classification benchmarks.
    
    Args:
        name: Dataset name. Supported:
            - 'spurious_motif': Synthetic causal/spurious benchmark
            - 'ogbg-molhiv': OGB HIV inhibition prediction
            - 'ogbg-molbbbp': OGB blood-brain barrier prediction
            - 'good-hiv': GOOD benchmark with distribution shift
        root: Root directory for storing datasets
        batch_size: Batch size for data loaders
        bias: Spurious correlation strength (only for spurious_motif)
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dictionary containing:
            - train_loader: Training data loader
            - val_loader: Validation data loader
            - test_loader: Test data loader
            - input_dim: Input feature dimension
            - output_dim: Number of output classes/tasks
            - has_masks: Whether dataset has ground-truth causal masks
            - metric: Primary evaluation metric ('accuracy' or 'auc')
            - task_type: 'multiclass' or 'binary'
            - dataset_info: Dictionary with dataset metadata
            
    Example:
        >>> data = get_dataset('spurious_motif', root='./data', batch_size=32)
        >>> print(f"Input dim: {data['input_dim']}, Classes: {data['output_dim']}")
        >>> for batch in data['train_loader']:
        ...     logits = model(batch.x, batch.edge_index, batch.batch)
    """
    name = name.lower()
    
    # Create root directory if needed
    os.makedirs(root, exist_ok=True)
    
    # Route to appropriate loader
    if name == 'spurious_motif' or name == 'spurious-motif':
        num_graphs = kwargs.get('num_graphs', 1000)
        return load_spurious_motif(
            root=root,
            batch_size=batch_size,
            bias=bias,
            num_graphs=num_graphs,
            seed=seed,
            num_workers=num_workers
        )
    
    elif name.startswith('ogbg-'):
        return load_ogb_mol(
            name=name,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    elif name == 'good-hiv' or name == 'goodhiv':
        domain = kwargs.get('domain', 'size')
        shift = kwargs.get('shift', 'covariate')
        return load_good_hiv(
            root=root,
            batch_size=batch_size,
            domain=domain,
            shift=shift,
            num_workers=num_workers
        )
    
    elif name == 'visual_genome' or name == 'visualgenome' or name == 'vg':
        from visual_genome_loader import load_visual_genome
        
        max_graphs = kwargs.get('max_graphs', 5000)
        feature_dim = kwargs.get('feature_dim', 64)
        feature_type = kwargs.get('feature_type', 'random')
        
        return load_visual_genome(
            root=root,
            batch_size=batch_size,
            max_graphs=max_graphs,
            feature_dim=feature_dim,
            feature_type=feature_type,
            num_workers=num_workers,
            use_synthetic=False
        )
    
    elif name == 'visual_genome_synthetic' or name == 'vg_synthetic':
        from visual_genome_loader import load_visual_genome
        
        max_graphs = kwargs.get('max_graphs', 1000)
        feature_dim = kwargs.get('feature_dim', 64)
        
        return load_visual_genome(
            root=root,
            batch_size=batch_size,
            max_graphs=max_graphs,
            feature_dim=feature_dim,
            num_workers=num_workers,
            use_synthetic=True  # Use synthetic data
        )
    
    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. Supported datasets:\n"
            f"  {list(SUPPORTED_DATASETS.keys())}"
        )


def list_datasets() -> Dict[str, Dict]:
    """List all supported datasets with their metadata."""
    return SUPPORTED_DATASETS.copy()


def print_dataset_info(data: Dict[str, Any]):
    """Pretty print dataset information."""
    info = data.get('dataset_info', {})
    
    print("=" * 60)
    print(f"  Dataset: {info.get('name', 'Unknown')}")
    print("=" * 60)
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Input Dimension: {data['input_dim']}")
    print(f"  Output Dimension: {data['output_dim']}")
    print(f"  Task Type: {data.get('task_type', 'N/A')}")
    print(f"  Metric: {data['metric']}")
    print(f"  Has Causal Masks: {data['has_masks']}")
    print()
    print(f"  Split Sizes:")
    print(f"    Train: {info.get('num_train', 'N/A')}")
    print(f"    Val:   {info.get('num_val', 'N/A')}")
    print(f"    Test:  {info.get('num_test', 'N/A')}")
    if 'num_ood_test' in info:
        print(f"    OOD Test: {info['num_ood_test']}")
    print("=" * 60)


# =============================================================================
# Evaluation Helpers
# =============================================================================

def compute_metric(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    metric: str,
    evaluator: Optional[Any] = None
) -> float:
    """
    Compute evaluation metric.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        metric: Metric name ('accuracy' or 'auc')
        evaluator: Optional OGB evaluator for proper AUC computation
        
    Returns:
        Metric value
    """
    if metric == 'accuracy':
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=-1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        return (pred_classes == labels).float().mean().item()
    
    elif metric == 'auc':
        if evaluator is not None:
            # Use OGB evaluator
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            
            result = evaluator.eval({
                'y_true': labels.cpu().numpy(),
                'y_pred': predictions.cpu().numpy()
            })
            return result.get('rocauc', result.get('auc', 0.0))
        else:
            # Manual ROC-AUC
            try:
                from sklearn.metrics import roc_auc_score
                
                if predictions.dim() > 1 and predictions.size(1) == 2:
                    # Binary classification with 2 outputs - use probability of class 1
                    probs = torch.softmax(predictions, dim=-1)[:, 1]
                elif predictions.dim() > 1:
                    probs = torch.sigmoid(predictions).squeeze(-1)
                else:
                    probs = torch.sigmoid(predictions)
                
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                
                return roc_auc_score(
                    labels.cpu().numpy(),
                    probs.cpu().numpy()
                )
            except Exception:
                return 0.0
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# Demo
# =============================================================================

def main():
    """Demo of the dataset factory."""
    print("Dataset Factory Demo")
    print("=" * 60)
    
    # List available datasets
    print("\nSupported Datasets:")
    for name, info in SUPPORTED_DATASETS.items():
        print(f"  - {name}: {info['description']}")
    
    # Load Spurious-Motif
    print("\n" + "=" * 60)
    print("Loading Spurious-Motif dataset...")
    
    try:
        data = get_dataset(
            'spurious_motif',
            root='./data',
            batch_size=32,
            bias=0.9,
            num_graphs=500
        )
        print_dataset_info(data)
        
        # Test batch
        batch = next(iter(data['train_loader']))
        print(f"\nSample batch:")
        print(f"  Num graphs: {batch.num_graphs}")
        print(f"  Node features shape: {batch.x.shape}")
        print(f"  Labels: {batch.y.tolist()}")
        
        if data['has_masks']:
            print(f"  Has causal_mask: {hasattr(batch, 'causal_mask')}")
        
    except Exception as e:
        print(f"Error loading Spurious-Motif: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == '__main__':
    main()

