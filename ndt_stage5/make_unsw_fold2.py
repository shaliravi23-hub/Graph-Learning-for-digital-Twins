"""Build the second chronological UNSW split: train 23-25 September, validate 26 September, test 27 September 2016.

All five days are training days of the main UNSW split (train 23-27 Sep, validate 8 Oct, test 9-10 Oct). No analysis
decision was based on results for these days, and no model trained on this split has seen 27 September.

Built from the main dataset's dataset_ready\\300s\\static_train.pkl, which holds the five days (outputs\\graphs_300s has no
graph_snapshots.pkl for 23 September; see deviations_unsw_fold2.md). Same procedure as make_deakin_fold2.py:
  1. undo the main feature scaling with the mean/std stored in dataset_ready\\300s\\metadata.json;
  2. keep only nodes active on at least one of the five days;
  3. refit the scaling on the three training days only, with build_graph_dataset.compute_feature_stats/normalize_snapshots;
  4. build history sequences with build_graph_dataset.build_temporal_sequences (L = 4).
Every day must have 288 consecutive five-minute windows. Adjacency matrices are exactly those of the main dataset.
Run from ndt_stage5:  python make_unsw_fold2.py --project-root ..
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

TRAIN, VAL, TEST = ['16-09-23', '16-09-24', '16-09-25'], ['16-09-26'], ['16-09-27']
L = 4


def main(args):
    root = Path(args.project_root).resolve()
    src = root / 'dataset_ready' / '300s'
    out = root / 'dataset_ready_unsw_fold2' / '300s'
    if out.exists(): raise ValueError(f'{out} exists; delete it first if you really want to rebuild')
    sys.path.insert(0, str(root))
    from build_graph_dataset import NODE_FEATURES, compute_feature_stats, normalize_snapshots, build_temporal_sequences

    meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8-sig'))
    static = pickle.load(open(src / 'static_train.pkl', 'rb'))
    if static['node_features'] != NODE_FEATURES: raise ValueError('Feature order differs from build_graph_dataset.NODE_FEATURES')
    by_date = {}
    for i in static['items']: by_date.setdefault(i['date'], []).append(i)
    for d in TRAIN + VAL + TEST:
        if d not in by_date: raise ValueError(f'{d} not in static_train.pkl; dates available: {sorted(by_date)}')
        w = sorted(int(i['window_id']) for i in by_date[d])
        if len(w) != 288 or w != list(range(w[0], w[0] + 288)):
            raise ValueError(f'Day {d} has {len(w)} windows or gaps; expected 288 consecutive windows')

    # 1) undo the main scaling (inactive rows were stored as zeros)
    mean, std = np.asarray(meta['feature_mean'], np.float64), np.asarray(meta['feature_std'], np.float64)
    act = NODE_FEATURES.index('active_edge_count'); deg = [NODE_FEATURES.index('in_degree'), NODE_FEATURES.index('out_degree')]
    raw = {}
    for d in TRAIN + VAL + TEST:
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

    # 2) node universe = nodes active on at least one of the five days
    keep = np.zeros(len(static['node_list']), bool)
    for d in raw:
        for i in raw[d]:
            keep |= (np.abs(i['X']).sum(1) > 0) | (i['A'].sum(0) > 0) | (i['A'].sum(1) > 0)
    idx = np.flatnonzero(keep); node_list = [static['node_list'][j] for j in idx]
    node_to_idx = {n: k for k, n in enumerate(node_list)}
    for d in raw:
        for i in raw[d]:
            i['A'] = i['A'][np.ix_(idx, idx)]; i['W'] = i['W'][np.ix_(idx, idx)]; i['X'] = i['X'][idx]

    # 3) rescale on the training days only; 4) history sequences, built day by day
    split = {'train': [i for d in TRAIN for i in raw[d]], 'val': [i for d in VAL for i in raw[d]], 'test': [i for d in TEST for i in raw[d]]}
    m2, s2 = compute_feature_stats(split['train'], NODE_FEATURES)
    norm = {k: normalize_snapshots(v, m2, s2, NODE_FEATURES) for k, v in split.items()}
    seqs = {k: build_temporal_sequences(v, L) for k, v in norm.items()}
    out.mkdir(parents=True)
    base = dict(node_list=node_list, node_to_idx=node_to_idx, node_features=NODE_FEATURES)
    for k in ['train', 'val', 'test']:
        pickle.dump(dict(base, items=norm[k]), open(out / f'static_{k}.pkl', 'wb'))
        pickle.dump(dict(base, seq_len=L, items=seqs[k]), open(out / f'temporal_{k}.pkl', 'wb'))
    metadata = dict(graph_root=meta.get('graph_root'), seq_len=L, num_nodes_global=len(node_list), node_features=NODE_FEATURES,
                    train_dates=TRAIN, val_dates=VAL, test_dates=TEST,
                    num_train_snapshots=len(norm['train']), num_val_snapshots=len(norm['val']), num_test_snapshots=len(norm['test']),
                    num_train_sequences=len(seqs['train']), num_val_sequences=len(seqs['val']), num_test_sequences=len(seqs['test']),
                    feature_mean=m2.tolist(), feature_std=s2.tolist(),
                    split_creation=('Second chronological UNSW split built by make_unsw_fold2.py from dataset_ready/300s/static_train.pkl: '
                                    'main scaling undone, node set restricted to nodes active on 23-27 Sep 2016, '
                                    'scaling refitted on 23-25 Sep only.'),
                    nodes_dropped=int((~keep).sum()))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ['num_nodes_global', 'nodes_dropped', 'train_dates', 'val_dates', 'test_dates',
                                                'num_train_sequences', 'num_val_sequences', 'num_test_sequences']}, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True)
    main(p.parse_args())
