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
        A_seq = item["A_seq"].astype(np.float32)
        X_seq = item["X_seq"].astype(np.float32)
        A_target = item["A_target"].astype(np.float32)
        X_target = item["X_target"].astype(np.float32)

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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        B, N, _ = X.shape
        I = torch.eye(N, device=X.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        deg = A_hat.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return A_norm @ (X @ self.weight) + self.bias


class VRDAGStylePredictor(nn.Module):
    """
    Task-aligned VRDAG-style predictor:
    - encode each snapshot with graph conv
    - recurrent graph-state modeling
    - graph-level variational latent for next step
    - decode next adjacency and node attributes
    """
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gcn1 = GraphConv(in_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.post_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.post_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.node_fuse = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.edge_bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_bilinear)

        self.adj_bias = nn.Parameter(torch.tensor(-2.0))
        self.feat_decoder = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_snapshot(self, X, A):
        H = F.relu(self.gcn1(X, A))
        H = self.dropout(H)
        H = F.relu(self.gcn2(H, A))
        H = self.dropout(H)
        return H

    def encode_sequence(self, A_seq, X_seq):
        B, T, N, _ = A_seq.shape
        graph_embeds = []
        node_states = []

        for t in range(T):
            H_t = self.encode_snapshot(X_seq[:, t], A_seq[:, t])  # [B, N, H]
            graph_t = H_t.mean(dim=1)                             # [B, H]
            node_states.append(H_t)
            graph_embeds.append(graph_t)

        graph_embeds = torch.stack(graph_embeds, dim=1)          # [B, T, H]
        rnn_out, _ = self.rnn(graph_embeds)
        h_last = rnn_out[:, -1, :]                               # [B, H]
        node_last = node_states[-1]                              # [B, N, H]
        x_last = X_seq[:, -1]                                    # [B, N, F]
        return h_last, node_last, x_last

    def forward(self, A_seq, X_seq, A_target=None, X_target=None, use_posterior=True):
        h_last, node_last, x_last = self.encode_sequence(A_seq, X_seq)

        prior_mu = self.prior_mu(h_last)
        prior_logvar = self.prior_logvar(h_last)

        if use_posterior and A_target is not None and X_target is not None:
            H_tgt = self.encode_snapshot(X_target, A_target)
            g_tgt = H_tgt.mean(dim=1)
            post_in = torch.cat([h_last, g_tgt], dim=-1)
            post_mu = self.post_mu(post_in)
            post_logvar = self.post_logvar(post_in)
            z = self.reparameterize(post_mu, post_logvar)
        else:
            post_mu, post_logvar = None, None
            z = prior_mu

        B, N, _ = node_last.shape
        z_expand = z.unsqueeze(1).expand(-1, N, -1)
        node_dec = self.node_fuse(torch.cat([node_last, z_expand, x_last], dim=-1))  # [B, N, H]

        temp = node_dec @ self.edge_bilinear
        A_logits = temp @ node_dec.transpose(-1, -2) + self.adj_bias
        X_pred = self.feat_decoder(node_dec)

        return {
            "A_logits": A_logits,
            "X_pred": X_pred,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }


def kl_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (
        p_logvar - q_logvar
        + (q_var + (q_mu - p_mu) ** 2) / p_var.clamp(min=1e-8)
        - 1.0
    )
    return kl.mean()


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
        for A_seq, X_seq, A_target, X_target in loader:
            A_seq = A_seq.to(device)
            X_seq = X_seq.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            out = model(A_seq, X_seq, use_posterior=False)
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

    _, X_seq, _, X_target = train_ds[0]
    in_dim = X_target.shape[-1]

    model = VRDAGStylePredictor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_pos_weight(train_ds).to(device)

    best_val_ap = -1.0
    best_path = Path(args.output_dir) / "best_vrdag_style.pt"
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

            out = model(A_seq, X_seq, A_target=A_target, X_target=X_target, use_posterior=True)

            loss_adj = F.binary_cross_entropy_with_logits(
                out["A_logits"], A_target, pos_weight=pos_weight
            )

            active_mask = (X_target.abs().sum(dim=-1, keepdim=True) > 0).float()
            loss_x = (((out["X_pred"] - X_target) ** 2) * active_mask).sum() / active_mask.sum().clamp(min=1.0)

            loss_kl = kl_gaussian(
                out["post_mu"], out["post_logvar"],
                out["prior_mu"], out["prior_logvar"]
            )

            loss = loss_adj + args.alpha_x * loss_x + args.beta_kl * loss_kl
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
        "model": "VRDAGStylePredictor",
        "dataset_dir": args.dataset_dir,
        "best_val_ap": best_val_ap,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "config": vars(args),
    }

    with open(Path(args.output_dir) / "vrdag_style_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(Path(args.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/vrdag_style")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_x", type=float, default=0.05)
    parser.add_argument("--beta_kl", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)