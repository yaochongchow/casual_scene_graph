import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from critics.irm_gin import CriticConfig, IRMEdgeCritic
from data.good_motif import DEFAULT_DATA_ROOT, load_good_motif


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, targets)
    grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return grad.pow(2)


def entropy_regularizer(edge_scores: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    entropy = (
        -edge_scores * torch.log(edge_scores + eps)
        - (1 - edge_scores) * torch.log(1 - edge_scores + eps)
    )
    return entropy.mean()


def train_one_epoch(
    model: IRMEdgeCritic,
    train_env_loaders: Dict[int, DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    irm_lambda: float,
    entropy_lambda: float,
) -> Dict[str, float]:
    model.train()
    aggregated = {"domain_loss": 0.0, "irm_penalty": 0.0, "entropy": 0.0, "steps": 0}

    for env_id, loader in train_env_loaders.items():
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            targets = torch.full(
                (outputs["domain_logits"].shape[0],),
                env_id,
                dtype=torch.long,
                device=device,
            )
            ce_loss = F.cross_entropy(outputs["domain_logits"], targets)
            penalty = irm_penalty(outputs["domain_logits"], targets) if irm_lambda > 0 else 0.0
            entropy = entropy_regularizer(outputs["edge_scores"]) if entropy_lambda > 0 else 0.0

            total_loss = ce_loss
            if irm_lambda > 0:
                total_loss = total_loss + irm_lambda * penalty
            if entropy_lambda > 0:
                total_loss = total_loss + entropy_lambda * entropy

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            aggregated["domain_loss"] += ce_loss.item()
            aggregated["irm_penalty"] += penalty.item() if irm_lambda > 0 else 0.0
            aggregated["entropy"] += entropy.item() if entropy_lambda > 0 else 0.0
            aggregated["steps"] += 1

    for key in ("domain_loss", "irm_penalty", "entropy"):
        if aggregated["steps"] > 0:
            aggregated[key] /= aggregated["steps"]

    aggregated["total_loss"] = (
        aggregated["domain_loss"]
        + irm_lambda * aggregated["irm_penalty"]
        + entropy_lambda * aggregated["entropy"]
    )
    return aggregated


def evaluate(model: IRMEdgeCritic, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            env_ids = _batch_env_ids(batch, device)
            logits = outputs["domain_logits"]
            total_loss += F.cross_entropy(logits, env_ids).item()
            preds = logits.argmax(dim=-1)
            correct += (preds == env_ids).sum().item()
            samples += env_ids.numel()
    return {
        "eval_loss": total_loss / max(samples, 1),
        "eval_acc": correct / max(samples, 1),
    }


def export_edge_scores(model, datasets, device, out_dir: Path):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test", "id_val", "id_test"]

    for split in splits:
        dataset = datasets.get(split)
        if dataset is None:
            continue
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        records: List[Dict[str, object]] = []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch = batch.to(device)
                outputs = model(batch)
                env_id = (
                    int(batch.env_id.view(-1)[0].item())
                    if hasattr(batch, "env_id")
                    else None
                )
                record = {
                    "graph_idx": idx,
                    "env_id": env_id,
                    "motif_id": int(batch.motif_id.item()) if hasattr(batch, "motif_id") else None,
                    "edge_scores": outputs["edge_scores"].detach().cpu().numpy(),
                    "edge_gt": batch.edge_gt.detach().cpu().numpy() if hasattr(batch, "edge_gt") else None,
                    "edge_index": batch.edge_index.detach().cpu().numpy(),
                }
                records.append(record)
        torch.save(records, out_dir / f"{split}_edges.pt")


def save_checkpoint(model, optimizer, epoch: int, path: Path, metrics: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _batch_env_ids(batch, device):
    if hasattr(batch, "env_id"):
        env = batch.env_id
        env = env.view(-1) if env.dim() > 1 else env
        env = env[: batch.num_graphs]
        return env.to(device)
    return torch.zeros(batch.num_graphs, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train IRM-based graph critic on GOOD-Motif")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--domain", type=str, default="basis")
    parser.add_argument("--shift", type=str, default="concept")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--irm-weight", type=float, default=25.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--artifact-dir", type=str, default="artifacts/critic")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--patience", type=int, default=5, help="Epoch window for moving-loss stabilization")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, meta_info = load_good_motif(
        dataset_root=args.dataset_root,
        domain=args.domain,
        shift=args.shift,
        batch_size=args.batch_size,
    )

    num_domains = meta_info.num_envs if hasattr(meta_info, "num_envs") else 3
    config = CriticConfig(
        in_dim=meta_info.dim_node,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        dropout=args.dropout,
        num_domains=num_domains,
    )

    model = IRMEdgeCritic(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            splits["train_env_loaders"],
            optimizer,
            device,
            irm_lambda=args.irm_weight,
            entropy_lambda=args.entropy_weight,
        )
        eval_metrics = evaluate(model, splits["val_loader"], device)
        combined = {**train_metrics, **{f"val_{k}": v for k, v in eval_metrics.items()}}
        history.append(combined)

        moving_window = history[-args.patience :]
        moving_loss = sum(m["total_loss"] for m in moving_window) / len(moving_window)

        if moving_loss < best_loss:
            best_loss = moving_loss
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": combined,
            }

        print(
            f"[Epoch {epoch}] loss={combined['total_loss']:.4f} "
            f"val_loss={combined.get('val_eval_loss', 0.0):.4f} "
            f"val_acc={combined.get('val_eval_acc', 0.0):.3f}"
        )

    if best_state:
        model.load_state_dict(best_state["state_dict"])
        optimizer.load_state_dict(best_state["optimizer"])

    artifact_dir = Path(args.artifact_dir)
    checkpoints_dir = artifact_dir / "checkpoints"
    save_checkpoint(
        model,
        optimizer,
        best_state["epoch"] if best_state else args.epochs,
        checkpoints_dir / "critic_best.pt",
        best_state["metrics"] if best_state else history[-1],
    )

    export_edge_scores(model, splits["datasets"], device, artifact_dir / "edge_scores")

    history_path = artifact_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


if __name__ == "__main__":
    main()

