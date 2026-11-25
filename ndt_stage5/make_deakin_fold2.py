"""Build the second chronological Deakin split: train 30 March, validate 31 March, test 4 April 2024.

Built from the existing dataset_ready_deakin_chronological pickles (the Deakin graph folders hold CSV files,
not graph_snapshots.pkl, so build_graph_dataset.py cannot read them). Steps:
  1. undo the old feature scaling (fitted on 30+31 March) with the mean/std stored in metadata.json;
  2. keep only nodes active on at least one of the three days (nodes seen only on 21 April are dropped);
  3. refit the scaling on the training day only, with build_graph_dataset.compute_feature_stats/normalize_snapshots;
  4. build history sequences with build_graph_dataset.build_temporal_sequences (L = 4).
Run from ndt_stage5:  python make_deakin_fold2.py --project-root ..
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

TRAIN, VAL, TEST = 'IoT_2024-03-30', 'IoT_2024-03-31', 'IoT_2024-04-04'
L = 4


def main(args):
    root = Path(args.project_root).resolve()
    src = root / 'dataset_ready_deakin_chronological' / '300s'
    out = root / 'dataset_ready_deakin_fold2' / '300s'
    if out.exists():
        raise ValueError(f'{out} exists; delete it first if you really want to rebuild')
    sys.path.insert(0, str(root))
    from build_graph_dataset import NODE_FEATURES, compute_feature_stats, normalize_snapshots, build_temporal_sequences

    meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8-sig'))
    static = {k: pickle.load(open(src / f'static_{k}.pkl', 'rb')) for k in ['train', 'val']}
    if static['train']['node_features'] != NODE_FEATURES:
        raise ValueError('Feature order differs from build_graph_dataset.NODE_FEATURES')
    items = [i for d in static.values() for i in d['items']]
    by_date = {}
    for i in items:
        by_date.setdefault(i['date'], []).append(i)
    for d in (TRAIN, VAL, TEST):
        if d not in by_date:
            raise ValueError(f'{d} not found; dates available: {sorted(by_date)}')

    # 1) undo the old scaling (inactive rows were stored as zeros)
    mean, std = np.asarray(meta['feature_mean'], np.float64), np.asarray(meta['feature_std'], np.float64)
    act = NODE_FEATURES.index('active_edge_count'); deg = [NODE_FEATURES.index('in_degree'), NODE_FEATURES.index('out_degree')]
    raw = {}
    for d in (TRAIN, VAL, TEST):
        raw[d] = []
        for i in sorted(by_date[d], key=lambda x: x['window_id']):
            x = np.asarray(i['X'], np.float64); active = np.abs(x).sum(1) > 0
            r = np.zeros_like(x); r[active] = x[active] * std + mean
            if active.any():
                if r[active, act].min() < 0.5:
                    raise ValueError('Reconstructed active_edge_count < 1 for an active node: scaling mismatch')
                if np.abs(r[active][:, deg + [act]] - np.rint(r[active][:, deg + [act]])).max() > 1e-2:
                    raise ValueError('Reconstructed degree counts are not integers: scaling mismatch')
            raw[d].append(dict(date=d, window_id=int(i['window_id']), A=np.asarray(i['A'], np.float32),
                               W=np.asarray(i['W'], np.float32), X=r.astype(np.float32)))

    # 2) node universe = nodes active on at least one of the three days
    keep = np.zeros(len(static['train']['node_list']), bool)
    for d in raw:
        for i in raw[d]:
            keep |= (np.abs(i['X']).sum(1) > 0) | (i['A'].sum(0) > 0) | (i['A'].sum(1) > 0)
    idx = np.flatnonzero(keep); node_list = [static['train']['node_list'][j] for j in idx]
    node_to_idx = {n: k for k, n in enumerate(node_list)}
    for d in raw:
        for i in raw[d]:
            i['A'] = i['A'][np.ix_(idx, idx)]; i['W'] = i['W'][np.ix_(idx, idx)]; i['X'] = i['X'][idx]

    # 3) rescale on the training day only; 4) history sequences
    m2, s2 = compute_feature_stats(raw[TRAIN], NODE_FEATURES)
    norm = {d: normalize_snapshots(raw[d], m2, s2, NODE_FEATURES) for d in raw}
    seqs = {d: build_temporal_sequences(norm[d], L) for d in norm}
    out.mkdir(parents=True)
    base = dict(node_list=node_list, node_to_idx=node_to_idx, node_features=NODE_FEATURES)
    for split, d in [('train', TRAIN), ('val', VAL), ('test', TEST)]:
        pickle.dump(dict(base, items=norm[d]), open(out / f'static_{split}.pkl', 'wb'))
        pickle.dump(dict(base, seq_len=L, items=seqs[d]), open(out / f'temporal_{split}.pkl', 'wb'))
    metadata = dict(graph_root=meta.get('graph_root'), seq_len=L, num_nodes_global=len(node_list), node_features=NODE_FEATURES,
                    train_dates=[TRAIN], val_dates=[VAL], test_dates=[TEST],
                    num_train_snapshots=len(norm[TRAIN]), num_val_snapshots=len(norm[VAL]), num_test_snapshots=len(norm[TEST]),
                    num_train_sequences=len(seqs[TRAIN]), num_val_sequences=len(seqs[VAL]), num_test_sequences=len(seqs[TEST]),
                    feature_mean=m2.tolist(), feature_std=s2.tolist(),
                    split_creation=('Second chronological Deakin split built by make_deakin_fold2.py from dataset_ready_deakin_chronological: '
                                    'old scaling undone, node set restricted to nodes active on 30 Mar, 31 Mar or 4 Apr, '
                                    'scaling refitted on 30 Mar only.'),
                    nodes_dropped=int((~keep).sum()))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ['num_nodes_global', 'nodes_dropped', 'num_train_sequences', 'num_val_sequences',
                                                'num_test_sequences']}, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True)
    main(p.parse_args())
