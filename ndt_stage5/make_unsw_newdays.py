"""Stage 8: build the new-day UNSW test set (preregistration_stage8_newdays.md).

The twelve UNSW capture days never used before become the test days of the main UNSW split. Training and validation
(23-27 September and 8 October 2016) are copied unchanged from dataset_ready\\300s, so every trained model, threshold and
fitted simple method of the main split applies as it is.

Rules (fixed in the plan):
  nodes     the main split's 39-node universe; edges with an endpoint outside it are removed from every window;
            node features are kept as produced by the preprocessing and scaled with the main training mean and SD
  windows   0-287 of each day; windows >= 288 are dropped; a window in range with no retained traffic is inserted empty
  exclusion a day is excluded if it has fewer than 144 windows, if more than 10% of its edges are removed by the node
            restriction, or if its graph_snapshots.pkl is missing or unreadable
  history   L = 4 windows, never crossing days (build_graph_dataset.build_temporal_sequences)
Run from ndt_stage5:
  python make_unsw_newdays.py --project-root .. --graph-root D:\\unsw\\outputs\\graphs_300s --plan preregistration_stage8_newdays.md
"""
import argparse, csv, hashlib, json, pickle, shutil, sys
from pathlib import Path
import numpy as np

NEW_DAYS = ['16-09-28', '16-09-29', '16-09-30', '16-10-01', '16-10-02', '16-10-03', '16-10-04', '16-10-05', '16-10-06',
            '16-10-07', '16-10-11', '16-10-12']
L = 4; MAX_WINDOWS = 288; MIN_WINDOWS = 144; MAX_REMOVED_SHARE = 0.10


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    root = Path(args.project_root).resolve(); src = root / 'dataset_ready' / '300s'
    out = root / args.out_name / '300s'
    if out.exists(): raise ValueError(f'{out} exists; delete it first if you really want to rebuild')
    if not Path(args.plan).exists(): raise FileNotFoundError(args.plan)
    sys.path.insert(0, str(root))
    import networkx as nx
    from build_graph_dataset import NODE_FEATURES, load_graph_snapshots, graph_to_matrices, normalize_snapshots, build_temporal_sequences

    meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8-sig'))
    with open(src / 'static_train.pkl', 'rb') as f: train = pickle.load(f)
    if train['node_features'] != NODE_FEATURES or meta['node_features'] != NODE_FEATURES: raise ValueError('Feature order differs')
    node_list = list(train['node_list']); node_to_idx = {n: i for i, n in enumerate(node_list)}
    if len(node_list) != meta['num_nodes_global']: raise ValueError('Main node universe does not match metadata')
    mean = np.asarray(meta['feature_mean'], np.float32); std = np.asarray(meta['feature_std'], np.float32)
    used = set(meta['train_dates'] + meta['val_dates'] + meta['test_dates'])
    if used & set(args.days): raise ValueError('A requested day was already used: ' + str(sorted(used & set(args.days))))

    graph_root = Path(args.graph_root); report = []; included = {}
    for day in args.days:
        row = dict(day=day, status='included', reason='', windows_in_file=0, windows_beyond_287=0, windows_used=0,
                   empty_windows_inserted=0, edges=0, edges_removed=0, removed_share=0.0, unseen_nodes=0)
        try:
            snaps = load_graph_snapshots(graph_root, day)
        except Exception as e:  # missing or unreadable capture
            row.update(status='excluded', reason=f'graph_snapshots.pkl missing or unreadable ({type(e).__name__})'); report.append(row); continue
        by_id = {int(w): G for w, G in snaps}
        if len(by_id) != len(snaps): raise ValueError(f'{day}: duplicate window ids')
        row['windows_in_file'] = len(by_id); row['windows_beyond_287'] = sum(w >= MAX_WINDOWS for w in by_id)
        in_range = [w for w in by_id if 0 <= w < MAX_WINDOWS]
        last = max(in_range) if in_range else -1
        unseen = set(); items = []
        for w in range(last + 1):
            G = by_id.get(w)
            if G is None: G = nx.DiGraph(); row['empty_windows_inserted'] += 1
            unseen |= {n for n in G.nodes if n not in node_to_idx}
            e = [(u, v) for u, v in G.edges if u != v]
            row['edges'] += len(e); row['edges_removed'] += sum((u not in node_to_idx) or (v not in node_to_idx) for u, v in e)
            H = G.subgraph([n for n in G.nodes if n in node_to_idx])
            A, W, X = graph_to_matrices(H, node_to_idx, NODE_FEATURES)
            items.append(dict(date=day, window_id=w, A=A, W=W, X=X))
        row['windows_used'] = len(items); row['unseen_nodes'] = len(unseen)
        row['removed_share'] = row['edges_removed'] / row['edges'] if row['edges'] else 0.0
        if len(items) < MIN_WINDOWS:
            row.update(status='excluded', reason=f'{len(items)} windows (< {MIN_WINDOWS})')
        elif row['removed_share'] > MAX_REMOVED_SHARE:
            row.update(status='excluded', reason=f'{100 * row["removed_share"]:.1f}% of edges outside the node universe (> 10%)')
        else:
            included[day] = items
        report.append(row)
        print(f"{day}: {row['status']:8s} windows {row['windows_used']:3d} (inserted {row['empty_windows_inserted']}, beyond 287 "
              f"{row['windows_beyond_287']}), edges {row['edges']}, removed {row['edges_removed']} ({100 * row['removed_share']:.2f}%), "
              f"unseen nodes {row['unseen_nodes']} {row['reason']}", flush=True)
    if not included: raise ValueError('No day passed the inclusion rules')

    static_test = [i for d in args.days if d in included for i in included[d]]
    static_test = normalize_snapshots(static_test, mean, std, NODE_FEATURES)
    seqs = build_temporal_sequences(static_test, L)
    out.mkdir(parents=True)
    for k in ['static_train.pkl', 'static_val.pkl']: shutil.copy2(src / k, out / k)  # byte copies of the main split
    base = dict(node_list=node_list, node_to_idx=node_to_idx, node_features=NODE_FEATURES)
    with open(out / 'static_test.pkl', 'wb') as f: pickle.dump(dict(base, items=static_test), f)
    with open(out / 'temporal_test.pkl', 'wb') as f: pickle.dump(dict(base, seq_len=L, items=seqs), f)
    with open(out / 'coverage.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
    metadata = dict(meta, test_dates=[d for d in args.days if d in included], num_test_snapshots=len(static_test),
                    num_test_sequences=len(seqs), graph_root_test=str(graph_root),
                    split_creation=('Stage 8 new-day UNSW test set built by make_unsw_newdays.py: training and validation copied from '
                                    'dataset_ready/300s; test = UNSW days never used before, on the main 39-node universe with the main '
                                    'feature scaling.'),
                    excluded_days={r['day']: r['reason'] for r in report if r['status'] == 'excluded'}, coverage=report,
                    analysis_plan=dict(file=Path(args.plan).name, sha256=sha(args.plan)), builder_sha256=sha(__file__),
                    source_hashes={k: sha(src / k) for k in ['static_train.pkl', 'static_val.pkl', 'metadata.json']},
                    snapshot_hashes={d: sha(graph_root / d / 'graph_snapshots.pkl') for d in included})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Included {len(included)} of {len(args.days)} days, {len(seqs)} test targets. Saved {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--graph-root', required=True)
    p.add_argument('--plan', required=True); p.add_argument('--days', nargs='+', default=NEW_DAYS)
    p.add_argument('--out-name', default='dataset_ready_unsw_newdays', help='Testing only: build into another folder')
    main(p.parse_args())
