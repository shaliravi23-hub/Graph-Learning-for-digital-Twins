import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class StaticGraphDataset(Dataset):
    def __init__(self, pkl_path):
        data = load_pickle(pkl_path)
        self.items = data["items"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        A = item["A"].astype(np.float32)
        X = item["X"].astype(np.float32)
        np.fill_diagonal(A, 0.0)
        return torch.tensor(X), torch.tensor(A)


class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, X):
        H = F.relu(self.fc1(X))
        H = self.dropout(H)
        mu = self.fc_mu(H)
        logvar = self.fc_logvar(H)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, Z):
        A_logits = torch.matmul(Z, Z.transpose(-1, -2))
        return A_logits

    def forward(self, X):
        mu, logvar = self.encode(X)
        Z = self.reparameterize(mu, logvar)
        A_logits = self.decode(Z)
        return A_logits, mu, logvar

    def forward_deterministic(self, X):
        """Deterministic inference using the latent posterior mean."""
        mu, logvar = self.encode(X)
        Z = mu
        A_logits = self.decode(Z)
        return A_logits, mu, logvar


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(A_logits, A_true, pos_weight=None):
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(A_logits, A_true)
    return F.binary_cross_entropy_with_logits(A_logits, A_true, pos_weight=pos_weight)


def compute_pos_weight(dataset):
    pos = 0.0
    total = 0.0
    for X, A in dataset:
        pos += A.sum().item()
        total += A.numel()
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


def evaluate(model, loader, device):
    model.eval()
    aucs, aps, accs = [], [], []

    with torch.no_grad():
        for X, A in loader:
            X = X.to(device)
            A = A.to(device)

            # Deterministic evaluation: use the latent posterior mean.
            A_logits, mu, logvar = model.forward_deterministic(X)
            A_prob = torch.sigmoid(A_logits)

            for b in range(X.size(0)):
                A_true = A[b].cpu().numpy().copy()
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

    return {
        "edge_auc_mean": float(np.mean(aucs)) if aucs else None,
        "edge_ap_mean": float(np.mean(aps)) if aps else None,
        "edge_accuracy_mean": float(np.mean(accs)) if accs else None,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_ds = StaticGraphDataset(Path(args.dataset_dir) / "static_train.pkl")
    val_ds = StaticGraphDataset(Path(args.dataset_dir) / "static_val.pkl")
    test_ds = StaticGraphDataset(Path(args.dataset_dir) / "static_test.pkl")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample_X, _ = train_ds[0]
    in_dim = sample_X.shape[1]

    model = VGAE(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = compute_pos_weight(train_ds).to(device)

    best_val_ap = -1.0
    best_path = Path(args.output_dir) / "best_static_vgae.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for X, A in train_loader:
            X = X.to(device)
            A = A.to(device)

            optimizer.zero_grad()
            A_logits, mu, logvar = model(X)

            loss_rec = reconstruction_loss(A_logits, A, pos_weight=pos_weight)
            loss_kl = kl_loss(mu, logvar)
            loss = loss_rec + args.beta * loss_kl

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        train_metrics = evaluate(model, train_loader, device)

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
        "model": "StaticVGAE",
        "dataset_dir": args.dataset_dir,
        "best_val_ap": best_val_ap,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "config": vars(args),
    }

    with open(Path(args.output_dir) / "static_vgae_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(Path(args.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/static_vgae")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)