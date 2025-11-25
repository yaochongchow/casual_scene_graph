import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def _compute_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0], ones)
    deg.scatter_add_(0, edge_index[1], ones)
    return deg


def aggregate_edge_scores(edge_score_dir: Path) -> Dict[tuple, Dict[int, List[float]]]:
    key_env_scores: Dict[tuple, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for file in edge_score_dir.glob("*_edges.pt"):
        records = torch.load(file, map_location="cpu")
        for record in records:
            env_id = record.get("env_id", -1)
            env_id = -1 if env_id is None else env_id
            motif_id = record.get("motif_id", -1)
            edge_scores = torch.tensor(record["edge_scores"]).view(-1)
            edge_gt = record.get("edge_gt")
            edge_index = torch.tensor(record.get("edge_index"))
            if edge_gt is None:
                continue
            edge_gt_tensor = torch.tensor(edge_gt).view(-1).long()
            num_nodes = edge_index.max().item() + 1
            deg = _compute_degrees(edge_index, num_nodes)

            for e_idx, score in enumerate(edge_scores):
                src = edge_index[0, e_idx].item()
                dst = edge_index[1, e_idx].item()
                deg_pair = tuple(sorted((int(deg[src].item()), int(deg[dst].item()))))
                key = (int(edge_gt_tensor[e_idx].item()), motif_id, deg_pair[0], deg_pair[1])
                key_env_scores[key][env_id].append(float(score.item()))
    return key_env_scores


def compute_variances(key_env_scores: Dict[tuple, Dict[int, List[float]]]) -> Dict[str, float]:
    key_variances = {}
    for key, env_dict in key_env_scores.items():
        env_means = []
        for env_id in sorted(env_dict.keys()):
            env_scores = env_dict[env_id]
            if len(env_scores) == 0:
                continue
            env_means.append(float(np.mean(env_scores)))
        if len(env_means) < 2:
            continue
        key_variances[str(key)] = float(np.var(env_means))
    return key_variances


def build_histogram(variances: List[float], output_path: Path):
    if not variances:
        print("No variance values computed; skipping histogram.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(variances, bins=40, color="#1f77b4", alpha=0.85)
    plt.xlabel("Edge score variance across environments")
    plt.ylabel("Count")
    plt.title("Edge Score Variance Histogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze edge score variance across environments")
    parser.add_argument("--edge-score-dir", type=str, default="artifacts/critic/edge_scores")
    parser.add_argument("--output", type=str, default="artifacts/critic/variance_hist.png")
    parser.add_argument("--report", type=str, default="artifacts/critic/variance_report.json")
    parser.add_argument("--low-var-threshold", type=float, default=0.01)
    args = parser.parse_args()

    edge_dir = Path(args.edge_score_dir)
    if not edge_dir.exists():
        raise FileNotFoundError(f"Edge score directory not found: {edge_dir}")

    key_env_scores = aggregate_edge_scores(edge_dir)
    key_variances = compute_variances(key_env_scores)

    # Expand per-edge variance assignments to visualize.
    expanded = []
    for key, env_scores in key_env_scores.items():
        key_str = str(key)
        variance = key_variances.get(key_str)
        if variance is None:
            continue
        count = sum(len(v) for v in env_scores.values())
        expanded.extend([variance] * count)

    build_histogram(expanded, Path(args.output))

    low_var_keys = [
        {"key": key, "variance": var}
        for key, var in key_variances.items()
        if var <= args.low_var_threshold
    ]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "num_keys": len(key_variances),
                "low_variance_keys": low_var_keys,
                "threshold": args.low_var_threshold,
            },
            fp,
            indent=2,
        )

    print(f"Wrote histogram to {args.output} and report to {args.report}")


if __name__ == "__main__":
    main()

