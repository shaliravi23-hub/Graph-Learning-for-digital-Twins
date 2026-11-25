"""Stage 7B: build 600-second or 900-second windows by merging 2 or 3 consecutive 300-second windows of an existing dataset.

Same days, same split and same node set as the 300 s dataset. Per merged window:
  adjacency  A = edge present in any of the merged windows;  edge weights W = sum
  in_degree, out_degree       recomputed from the merged adjacency (if, in the 300 s data, they equal the adjacency's
                              column/row sums; otherwise summed, and the metadata says so)
  active_edge_count           recomputed as in_degree + out_degree if that identity holds in the 300 s data; otherwise summed
  in/out bytes, in/out packets  summed
  tcp/udp/icmp ratios         averaged with packet weights (an approximation: per-flow counts are not stored)
The 300 s feature scaling is undone with the stored mean/SD, scaling is refitted on the merged training windows only, and
history sequences (L = 4 merged windows) are built with build_graph_dataset.py. Each day must have 288 consecutive windows.
Run from ndt_stage5:  python make_merged_windows.py --project-root .. --dataset unsw --seconds 900
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

BASES = {'unsw': 'dataset_ready', 'deakin': 'dataset_ready_deakin_chronological',
         'deakin_fold2': 'dataset_ready_deakin_fold2', 'unsw_fold2': 'dataset_ready_unsw_fold2'}
L = 4


def main(args):
    root = Path(args.project_root).resolve(); k = args.seconds // 300
    src = root / BASES[args.dataset] / '300s'; out = root / BASES[args.dataset] / f'{args.seconds}s'
    if out.exists(): raise ValueError(f'{out} exists; delete it first if you really want to rebuild')
    sys.path.insert(0, str(root))
    from build_graph_dataset import NODE_FEATURES, compute_feature_stats, normalize_snapshots, build_temporal_sequences
    meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8-sig'))
    mean, std = np.asarray(meta['feature_mean'], np.float64), np.asarray(meta['feature_std'], np.float64)
    f = {n: NODE_FEATURES.index(n) for n in NODE_FEATURES}
    split_items, node_list = {}, None
    for split in ['train', 'val', 'test']:
        p = src / f'static_{split}.pkl'
        if not p.exists(): raise FileNotFoundError(f'{p} (needed to rebuild windows for the {split} days)')
        d = pickle.load(open(p, 'rb'))
        if d['node_features'] != NODE_FEATURES: raise ValueError('Feature order differs from build_graph_dataset.NODE_FEATURES')
        if node_list is None: node_list = d['node_list']
        elif d['node_list'] != node_list: raise ValueError('Node lists differ between splits')
        split_items[split] = d['items']
    # undo the 300 s scaling and check the identities used to recompute degree features
    raw, ident = {}, dict(degree=0, active=0, rows=0)
    for split, items in split_items.items():
        raw[split] = []
        for i in items:
            x = np.asarray(i['X'], np.float64); active = np.abs(x).sum(1) > 0
            r = np.zeros_like(x); r[active] = x[active] * std + mean; r = np.where(active[:, None], r, 0.)
            a = np.asarray(i['A'], np.float64); np.fill_diagonal(a, 0)
            ident['rows'] += int(active.sum())
            ident['degree'] += int((np.abs(r[active, f['in_degree']] - a.sum(0)[active]) < 1e-2).sum()
                                   + (np.abs(r[active, f['out_degree']] - a.sum(1)[active]) < 1e-2).sum())
            ident['active'] += int((np.abs(r[active, f['active_edge_count']] - r[active, f['in_degree']] - r[active, f['out_degree']]) < 1e-2).sum())
            raw[split].append(dict(date=i['date'], window_id=int(i['window_id']), A=a, W=np.asarray(i['W'], np.float64), X=r))
    degree_rule = 'recomputed' if ident['degree'] == 2 * ident['rows'] else 'summed'
    active_rule = 'recomputed' if (degree_rule == 'recomputed' and ident['active'] == ident['rows']) else 'summed'
    merged = {}
    for split, items in raw.items():
        merged[split] = []
        for date in sorted({i['date'] for i in items}):
            day = sorted((i for i in items if i['date'] == date), key=lambda i: i['window_id'])
            w = [i['window_id'] for i in day]
            if len(w) != 288 or w != list(range(w[0], w[0] + 288)): raise ValueError(f'{split} day {date}: {len(w)} windows or gaps')
            for g in range(288 // k):
                grp = day[g * k:(g + 1) * k]
                A = (np.sum([i['A'] for i in grp], 0) > 0).astype(np.float32); W = np.sum([i['W'] for i in grp], 0).astype(np.float32)
                X = np.zeros_like(grp[0]['X'])
                for name in ['in_bytes', 'out_bytes', 'in_packets', 'out_packets']: X[:, f[name]] = np.sum([i['X'][:, f[name]] for i in grp], 0)
                if degree_rule == 'recomputed': X[:, f['in_degree']] = A.sum(0); X[:, f['out_degree']] = A.sum(1)
                else:
                    for name in ['in_degree', 'out_degree']: X[:, f[name]] = np.sum([i['X'][:, f[name]] for i in grp], 0)
                X[:, f['active_edge_count']] = (X[:, f['in_degree']] + X[:, f['out_degree']]) if active_rule == 'recomputed' \
                    else np.sum([i['X'][:, f['active_edge_count']] for i in grp], 0)
                pk = np.stack([i['X'][:, f['in_packets']] + i['X'][:, f['out_packets']] for i in grp])
                act = np.stack([(np.abs(i['X']).sum(1) > 0) for i in grp]).astype(float)
                wts = np.where(pk.sum(0) > 0, pk, act)
                for name in ['tcp_ratio', 'udp_ratio', 'icmp_ratio']:
                    v = np.stack([i['X'][:, f[name]] for i in grp]); s = wts.sum(0)
                    X[:, f[name]] = np.where(s > 0, (v * wts).sum(0) / np.where(s > 0, s, 1), 0.)
                X[~(act.sum(0) > 0)] = 0.
                merged[split].append(dict(date=date, window_id=g, A=A, W=W, X=X.astype(np.float32)))
    m2, s2 = compute_feature_stats(merged['train'], NODE_FEATURES)
    norm = {s: normalize_snapshots(v, m2, s2, NODE_FEATURES) for s, v in merged.items()}
    seqs = {s: build_temporal_sequences(v, L) for s, v in norm.items()}
    out.mkdir(parents=True)
    base = dict(node_list=node_list, node_to_idx={n: j for j, n in enumerate(node_list)}, node_features=NODE_FEATURES)
    for s in ['train', 'val', 'test']:
        pickle.dump(dict(base, items=norm[s]), open(out / f'static_{s}.pkl', 'wb'))
        pickle.dump(dict(base, seq_len=L, items=seqs[s]), open(out / f'temporal_{s}.pkl', 'wb'))
    md = dict(meta, seq_len=L, window_seconds=args.seconds, merged_from='300s', merge_factor=k,
              num_train_snapshots=len(norm['train']), num_val_snapshots=len(norm['val']), num_test_snapshots=len(norm['test']),
              num_train_sequences=len(seqs['train']), num_val_sequences=len(seqs['val']), num_test_sequences=len(seqs['test']),
              feature_mean=m2.tolist(), feature_std=s2.tolist(), degree_rule=degree_rule, active_edge_count_rule=active_rule,
              identity_checks=ident, split_creation=f'{args.seconds} s windows merged from {k} consecutive 300 s windows by make_merged_windows.py')
    (out / 'metadata.json').write_text(json.dumps(md, indent=2))
    print(json.dumps({k_: md[k_] for k_ in ['window_seconds', 'num_nodes_global', 'num_train_sequences', 'num_val_sequences',
                                             'num_test_sequences', 'degree_rule', 'active_edge_count_rule'] if k_ in md}, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(BASES), required=True)
    p.add_argument('--seconds', type=int, choices=[600, 900], required=True)
    main(p.parse_args())
