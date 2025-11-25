import os
from collections import defaultdict
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from GOOD.data.good_datasets.good_motif import GOODMotif


DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "GOOD_datasets")
)


def _group_indices_by_env(dataset) -> Dict[int, list]:
    """Collect sample indices for each environment id."""
    env_to_indices: Dict[int, list] = defaultdict(list)
    for idx in range(len(dataset)):
        data = dataset[idx]
        env_id = int(data.env_id.item()) if hasattr(data, "env_id") else 0
        env_to_indices[env_id].append(idx)
    return env_to_indices


def _build_env_loaders(dataset, batch_size: int, num_workers: int, shuffle: bool):
    grouped = _group_indices_by_env(dataset)
    loaders = {}
    for env_id, indices in grouped.items():
        subset = Subset(dataset, indices)
        loaders[env_id] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def load_good_motif(
    dataset_root: str = DEFAULT_DATA_ROOT,
    domain: str = "basis",
    shift: str = "concept",
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[Dict[str, object], Any]:
    """
    Load GOOD-Motif datasets and build per-environment dataloaders.

    Returns:
        splits: {
            "datasets": {...},
            "train_env_loaders": {env_id: DataLoader},
            "val_loader": DataLoader,
            "test_loader": DataLoader,
            "id_val_loader": DataLoader | None,
            "id_test_loader": DataLoader | None,
        }
        meta_info: GOOD-provided metadata container (Munch).
    """
    datasets, meta_info = GOODMotif.load(
        dataset_root=dataset_root, domain=domain, shift=shift, generate=False
    )

    train_env_loaders = _build_env_loaders(
        datasets["train"], batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    def _loader(name: str, shuffle: bool = False):
        split_dataset = datasets.get(name)
        if split_dataset is None:
            return None
        return DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    split_payload = {
        "datasets": datasets,
        "train_env_loaders": train_env_loaders,
        "val_loader": _loader("val"),
        "test_loader": _loader("test"),
        "id_val_loader": _loader("id_val"),
        "id_test_loader": _loader("id_test"),
    }

    return split_payload, meta_info

