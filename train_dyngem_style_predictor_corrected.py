import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class TemporalGraphDataset(Dataset):
    def __init__(self, pkl_path):
        data = load_pickle(pkl_path)
        self.items = data["items"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        A_seq = item["A_seq"].astype(np.float32)        # [T, N, N]
        X_seq = item["X_seq"].astype(np.float32)        # [T, N, F]
        A_target = item["A_target"].astype(np.float32)  # [N, N]
        X_target = item["X_target"].astype(np.float32)  # [N, F]

        for t in range(A_seq.shape[0]):
            np.fill_diagonal(A_seq[t], 0.0)
        np.fill_diagonal(A_target, 0.0)

        A_last = A_seq[-1]
        X_last = X_seq[-1]

        return (
            torch.tensor(A_last),
            torch.tensor(X_last),
            torch.tensor(A_target),
            torch.tensor(X_target),
        )


class DynGEMStylePredictor(nn.Module):
    """
    Lightweight transition autoencoder:
    input  = flattened last adjacency + flattened last node features
    output = next adjacency logits (+ optional node-feature reconstruction)
    """
    def __init__(self, num_nodes, feat_dim, hidden_dims=(512, 256), latent_dim=128, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        in_dim = num_nodes * num_nodes + num_nodes * feat_dim
        out_adj_dim = num_nodes * num_nodes
        out_x_dim = num_nodes * feat_dim

        h1, h2 = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )

        self.adj_decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, out_adj_dim),
        )

        self.x_decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_x_dim),
        )

    def forward(self, A_last, X_last):
        B, N, _ = A_last.shape
        x_in = torch.cat(
            [A_last.reshape(B, -1), X_last.reshape(B, -1)],
            dim=-1
        )
        z = self.encoder(x_in)

        A_logits = self.adj_decoder(z).reshape(B, N, N)
        X_pred = self.x_decoder(z).reshape(B, N, self.feat_dim)

        return {
            "A_logits": A_logits,
            "X_pred": X_pred,
            "z": z,
        }


def compute_pos_weight(dataset):
    pos = 0.0
    total = 0.0
    for A_last, X_last, A_target, X_target in dataset:
        pos += A_target.sum().item()
        total += A_target.numel()
    neg = total - pos
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def safe_ap(y_true, y_score):
    if np.sum(y_true) == 0:
        return None
    return average_precision_score(y_true, y_score)


def graph_stats_from_adj(A):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    num_nodes = A.shape[0]
    num_edges = int(A.sum())
    density = nx.density(G) if num_nodes > 1 else 0.0

    try:
        clustering = nx.average_clustering(G.to_undirected()) if num_nodes > 1 else 0.0
    except Exception:
        clustering = 0.0

    try:
        weak_components = nx.number_weakly_connected_components(G)
    except Exception:
        weak_components = 0

    return {
        "num_edges": num_edges,
        "density": density,
        "clustering": float(clustering),
        "weak_components": int(weak_components),
    }


def shortest_path_proxy(A):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0

    UG = G.to_undirected()
    if UG.number_of_edges() == 0:
        return 0.0

    largest_cc = max(nx.connected_components(UG), key=len)
    SG = UG.subgraph(largest_cc).copy()

    if SG.number_of_nodes() < 2:
        return 0.0

    try:
        return float(nx.average_shortest_path_length(SG))
    except Exception:
        return 0.0


def evaluate(model, loader, device):
    model.eval()
    aucs, aps, accs = [], [], []

    density_errors = []
    clustering_errors = []
    sp_errors = []
    component_errors = []
    edge_count_errors = []

    with torch.no_grad():
        for A_last, X_last, A_target, X_target in loader:
            A_last = A_last.to(device)
            X_last = X_last.to(device)
            A_target = A_target.to(device)

            out = model(A_last, X_last)
            A_prob = torch.sigmoid(out["A_logits"])

            for b in range(A_last.size(0)):
                A_true = A_target[b].cpu().numpy().copy()
                A_score = A_prob[b].cpu().numpy().copy()

                np.fill_diagonal(A_true, 0.0)
                np.fill_diagonal(A_score, 0.0)

                # Exclude self-loops from edge-level evaluation
                mask = ~np.eye(A_true.shape[0], dtype=bool)

                y_true = A_true[mask]
                y_score = A_score[mask]
                y_pred = (y_score > 0.5).astype(np.float32)

                auc = safe_auc(y_true, y_score)
                ap = safe_ap(y_true, y_score)
                acc = float((y_true == y_pred).mean())

                if auc is not None:
                    aucs.append(auc)
                if ap is not None:
                    aps.append(ap)
                accs.append(acc)

                # Structural evaluation requires the full N x N adjacency matrix.
                # y_pred contains only off-diagonal entries, so it must not be reshaped.
                pred_adj = (A_score > 0.5).astype(np.float32)
                np.fill_diagonal(pred_adj, 0.0)
                pred_stats = graph_stats_from_adj(pred_adj)
                true_stats = graph_stats_from_adj(A_true)

                density_errors.append(abs(pred_stats["density"] - true_stats["density"]))
                clustering_errors.append(abs(pred_stats["clustering"] - true_stats["clustering"]))
                component_errors.append(abs(pred_stats["weak_components"] - true_stats["weak_components"]))
                edge_count_errors.append(abs(pred_stats["num_edges"] - true_stats["num_edges"]))

                true_sp = shortest_path_proxy(A_true)
                pred_sp = shortest_path_proxy(pred_adj)
                sp_errors.append(abs(pred_sp - true_sp))

    return {
        "edge_auc_mean": float(np.mean(aucs)) if aucs else None,
        "edge_ap_mean": float(np.mean(aps)) if aps else None,
        "edge_accuracy_mean": float(np.mean(accs)) if accs else None,
        "density_error_mean": float(np.mean(density_errors)) if density_errors else None,
        "clustering_error_mean": float(np.mean(clustering_errors)) if clustering_errors else None,
        "shortest_path_error_mean": float(np.mean(sp_errors)) if sp_errors else None,
        "weak_component_error_mean": float(np.mean(component_errors)) if component_errors else None,
        "edge_count_error_mean": float(np.mean(edge_count_errors)) if edge_count_errors else None,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_ds = TemporalGraphDataset(Path(args.dataset_dir) / "temporal_train.pkl")
    val_ds = TemporalGraphDataset(Path(args.dataset_dir) / "temporal_val.pkl")
    test_ds = TemporalGraphDataset(Path(args.dataset_dir) / "temporal_test.pkl")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    A_last, X_last, A_target, X_target = train_ds[0]
    num_nodes = A_last.shape[0]
    feat_dim = X_last.shape[1]

    model = DynGEMStylePredictor(
        num_nodes=num_nodes,
        feat_dim=feat_dim,
        hidden_dims=(args.hidden_dim1, args.hidden_dim2),
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_pos_weight(train_ds).to(device)

    best_val_ap = -1.0
    best_path = Path(args.output_dir) / "best_dyngem_style.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for A_last, X_last, A_target, X_target in train_loader:
            A_last = A_last.to(device)
            X_last = X_last.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            optimizer.zero_grad()

            out = model(A_last, X_last)

            loss_adj = F.binary_cross_entropy_with_logits(
                out["A_logits"], A_target, pos_weight=pos_weight
            )

            active_mask = (X_target.abs().sum(dim=-1, keepdim=True) > 0).float()
            loss_x = (((out["X_pred"] - X_target) ** 2) * active_mask).sum() / active_mask.sum().clamp(min=1.0)

            loss = loss_adj + args.alpha_x * loss_x
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(loss.item())

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_edge_auc": train_metrics["edge_auc_mean"],
            "train_edge_ap": train_metrics["edge_ap_mean"],
            "val_edge_auc": val_metrics["edge_auc_mean"],
            "val_edge_ap": val_metrics["edge_ap_mean"],
        }
        history.append(record)
        print(record)

        if val_metrics["edge_ap_mean"] is not None and val_metrics["edge_ap_mean"] > best_val_ap:
            best_val_ap = val_metrics["edge_ap_mean"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))

    final_train = evaluate(model, train_loader, device)
    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)

    results = {
        "model": "DynGEMStylePredictor",
        "dataset_dir": args.dataset_dir,
        "best_val_ap": best_val_ap,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "config": vars(args),
    }

    with open(Path(args.output_dir) / "dyngem_style_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(Path(args.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/dyngem_style")
    parser.add_argument("--hidden_dim1", type=int, default=512)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_x", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)