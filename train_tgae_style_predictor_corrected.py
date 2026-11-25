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

        return (
            torch.tensor(A_seq),
            torch.tensor(X_seq),
            torch.tensor(A_target),
            torch.tensor(X_target),
        )


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        # X: [B, N, F]
        # A: [B, N, N]
        B, N, _ = X.shape
        I = torch.eye(N, device=X.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        deg = A_hat.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        XW = X @ self.weight
        out = A_norm @ XW
        if self.bias is not None:
            out = out + self.bias
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, H]

    def forward(self, x):
        # x: [B*, T, H]
        T = x.size(1)
        return x + self.pe[:, :T]


class TGAEStylePredictor(nn.Module):
    """
    TGAE-style temporal graph predictor:
    1. graph encoder per snapshot
    2. temporal transformer over snapshot sequence
    3. next-step graph decoding
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        latent_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.gcn1 = GraphConv(in_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)

        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max(16, seq_len + 2))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.feat_decoder = nn.Linear(latent_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_snapshot(self, X, A):
        H = F.relu(self.gcn1(X, A))
        H = self.dropout(H)
        H = F.relu(self.gcn2(H, A))
        H = self.dropout(H)
        return H

    def forward(self, A_seq, X_seq):
        # A_seq: [B, T, N, N]
        # X_seq: [B, T, N, F]
        B, T, N, Fdim = X_seq.shape

        snapshot_embeds = []
        for t in range(T):
            H_t = self.encode_snapshot(X_seq[:, t], A_seq[:, t])  # [B, N, H]
            snapshot_embeds.append(H_t)

        H_seq = torch.stack(snapshot_embeds, dim=1)   # [B, T, N, H]
        H_seq = H_seq.permute(0, 2, 1, 3).contiguous()  # [B, N, T, H]
        H_seq = H_seq.view(B * N, T, self.hidden_dim)   # [B*N, T, H]

        H_seq = self.pos_enc(H_seq)
        H_temporal = self.temporal_encoder(H_seq)       # [B*N, T, H]
        H_last = H_temporal[:, -1, :]                   # [B*N, H]

        Z = self.to_latent(H_last).view(B, N, self.latent_dim)  # [B, N, Z]

        A_logits = Z @ Z.transpose(-1, -2)              # inner-product decoder
        X_pred = self.feat_decoder(Z)                   # [B, N, F]

        return {
            "A_logits": A_logits,
            "X_pred": X_pred,
            "Z": Z,
        }


def compute_pos_weight(dataset):
    pos = 0.0
    total = 0.0
    for A_seq, X_seq, A_target, X_target in dataset:
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

    out_deg = np.array(A.sum(axis=1)).astype(float)
    in_deg = np.array(A.sum(axis=0)).astype(float)

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
        "mean_out_degree": float(out_deg.mean()),
        "mean_in_degree": float(in_deg.mean()),
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
        for A_seq, X_seq, A_target, X_target in loader:
            A_seq = A_seq.to(device)
            X_seq = X_seq.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            out = model(A_seq, X_seq)
            A_prob = torch.sigmoid(out["A_logits"])

            for b in range(A_seq.size(0)):
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
                # y_pred contains only off-diagonal entries and therefore must not
                # be reshaped back to A_true.shape.
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

    sample = train_ds[0]
    A_seq, X_seq, A_target, X_target = sample
    in_dim = X_target.shape[-1]
    seq_len = A_seq.shape[0]

    model = TGAEStylePredictor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_pos_weight(train_ds).to(device)

    best_val_ap = -1.0
    best_path = Path(args.output_dir) / "best_tgae_style.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for A_seq, X_seq, A_target, X_target in train_loader:
            A_seq = A_seq.to(device)
            X_seq = X_seq.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            optimizer.zero_grad()

            out = model(A_seq, X_seq)

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
        "model": "TGAEStylePredictor",
        "dataset_dir": args.dataset_dir,
        "best_val_ap": best_val_ap,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "config": vars(args),
    }

    with open(Path(args.output_dir) / "tgae_style_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(Path(args.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/tgae_style")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_x", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)