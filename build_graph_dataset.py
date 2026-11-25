import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np


NODE_FEATURES = [
    "in_degree",
    "out_degree",
    "in_bytes",
    "out_bytes",
    "in_packets",
    "out_packets",
    "active_edge_count",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
]


def load_graph_snapshots(graph_root: Path, date_id: str):
    pkl_path = graph_root / date_id / "graph_snapshots.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing: {pkl_path}")
    with open(pkl_path, "rb") as f:
        snapshots = pickle.load(f)
    snapshots = sorted(snapshots, key=lambda x: x[0])  # (window_id, G)
    return snapshots


def build_global_node_map(graph_root: Path, all_dates):
    node_set = set()
    for date_id in all_dates:
        snapshots = load_graph_snapshots(graph_root, date_id)
        for _, G in snapshots:
            node_set.update(G.nodes())
    node_list = sorted(node_set)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    return node_list, node_to_idx


def graph_to_matrices(G: nx.DiGraph, node_to_idx, node_features):
    n = len(node_to_idx)
    f = len(node_features)

    A = np.zeros((n, n), dtype=np.float32)       # binary adjacency
    W = np.zeros((n, n), dtype=np.float32)       # weighted adjacency
    X = np.zeros((n, f), dtype=np.float32)       # node features

    # Node features
    for node, attrs in G.nodes(data=True):
        i = node_to_idx[node]
        for j, feat in enumerate(node_features):
            X[i, j] = float(attrs.get(feat, 0.0))

    # Edges
    for u, v, attrs in G.edges(data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        A[i, j] = 1.0

        # lightweight weighted edge score
        # log-scaled byte_count is safer than raw values
        byte_count = float(attrs.get("byte_count", 0.0))
        packet_count = float(attrs.get("packet_count", 0.0))
        flow_count = float(attrs.get("flow_count", 0.0))

        w = np.log1p(byte_count) + 0.25 * np.log1p(packet_count) + 0.1 * np.log1p(flow_count)
        W[i, j] = w

    return A, W, X


def collect_snapshots(graph_root: Path, split_dates, node_to_idx, node_features):
    items = []
    for date_id in split_dates:
        snapshots = load_graph_snapshots(graph_root, date_id)
        for window_id, G in snapshots:
            A, W, X = graph_to_matrices(G, node_to_idx, node_features)
            items.append({
                "date": date_id,
                "window_id": int(window_id),
                "A": A,
                "W": W,
                "X": X,
            })
    return items


def compute_feature_stats(train_items, node_features):
    active_idx = node_features.index("active_edge_count")
    all_rows = []

    for item in train_items:
        X = item["X"]
        active_mask = X[:, active_idx] > 0
        if np.any(active_mask):
            all_rows.append(X[active_mask])

    if len(all_rows) == 0:
        mean = np.zeros(len(node_features), dtype=np.float32)
        std = np.ones(len(node_features), dtype=np.float32)
        return mean, std

    stacked = np.vstack(all_rows)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_snapshots(items, mean, std, node_features):
    active_idx = node_features.index("active_edge_count")
    out = []

    for item in items:
        X = item["X"].copy()
        active_mask = X[:, active_idx] > 0

        X_norm = (X - mean) / std
        X_norm[~active_mask] = 0.0  # keep inactive nodes as zero rows

        out.append({
            "date": item["date"],
            "window_id": item["window_id"],
            "A": item["A"],
            "W": item["W"],
            "X": X_norm.astype(np.float32),
        })

    return out


def build_temporal_sequences(items, seq_len):
    # build sequences date-by-date only
    by_date = {}
    for item in items:
        by_date.setdefault(item["date"], []).append(item)

    sequences = []
    for date_id, date_items in by_date.items():
        date_items = sorted(date_items, key=lambda x: x["window_id"])

        for i in range(len(date_items) - seq_len):
            input_seq = date_items[i:i + seq_len]
            target = date_items[i + seq_len]

            sequences.append({
                "date": date_id,
                "input_window_ids": [x["window_id"] for x in input_seq],
                "target_window_id": target["window_id"],
                "A_seq": np.stack([x["A"] for x in input_seq], axis=0),   # [T, N, N]
                "W_seq": np.stack([x["W"] for x in input_seq], axis=0),   # [T, N, N]
                "X_seq": np.stack([x["X"] for x in input_seq], axis=0),   # [T, N, F]
                "A_target": target["A"],
                "W_target": target["W"],
                "X_target": target["X"],
            })

    return sequences


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_root", type=str, required=True,
                        help="Path to outputs/graphs_300s or outputs/graphs_60s")
    parser.add_argument("--output_dir", type=str, default="dataset_ready")
    parser.add_argument("--seq_len", type=int, default=4)

    parser.add_argument("--train_dates", nargs="+", required=True)
    parser.add_argument("--val_dates", nargs="+", required=True)
    parser.add_argument("--test_dates", nargs="+", required=True)

    args = parser.parse_args()

    graph_root = Path(args.graph_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dates = args.train_dates + args.val_dates + args.test_dates

    # fixed global node universe
    node_list, node_to_idx = build_global_node_map(graph_root, all_dates)

    # snapshot-level datasets
    train_items = collect_snapshots(graph_root, args.train_dates, node_to_idx, NODE_FEATURES)
    val_items = collect_snapshots(graph_root, args.val_dates, node_to_idx, NODE_FEATURES)
    test_items = collect_snapshots(graph_root, args.test_dates, node_to_idx, NODE_FEATURES)

    # normalize node features using train only
    feat_mean, feat_std = compute_feature_stats(train_items, NODE_FEATURES)
    train_items = normalize_snapshots(train_items, feat_mean, feat_std, NODE_FEATURES)
    val_items = normalize_snapshots(val_items, feat_mean, feat_std, NODE_FEATURES)
    test_items = normalize_snapshots(test_items, feat_mean, feat_std, NODE_FEATURES)

    # temporal sequence datasets
    train_seq = build_temporal_sequences(train_items, args.seq_len)
    val_seq = build_temporal_sequences(val_items, args.seq_len)
    test_seq = build_temporal_sequences(test_items, args.seq_len)

    metadata = {
        "graph_root": str(graph_root),
        "seq_len": args.seq_len,
        "num_nodes_global": len(node_list),
        "node_features": NODE_FEATURES,
        "train_dates": args.train_dates,
        "val_dates": args.val_dates,
        "test_dates": args.test_dates,
        "num_train_snapshots": len(train_items),
        "num_val_snapshots": len(val_items),
        "num_test_snapshots": len(test_items),
        "num_train_sequences": len(train_seq),
        "num_val_sequences": len(val_seq),
        "num_test_sequences": len(test_seq),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
    }

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "items": train_items,
    }, output_dir / "static_train.pkl")

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "items": val_items,
    }, output_dir / "static_val.pkl")

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "items": test_items,
    }, output_dir / "static_test.pkl")

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "seq_len": args.seq_len,
        "items": train_seq,
    }, output_dir / "temporal_train.pkl")

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "seq_len": args.seq_len,
        "items": val_seq,
    }, output_dir / "temporal_val.pkl")

    save_pickle({
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "node_features": NODE_FEATURES,
        "seq_len": args.seq_len,
        "items": test_seq,
    }, output_dir / "temporal_test.pkl")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Dataset build complete.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()