"""Stage 7C: a copy of a 300 s dataset whose targets lie h windows after the history (h = 2: 10 minutes ahead; h = 4: 20 minutes).

The static windows (and their feature scaling) are copied unchanged; only the history/target pairing changes. History windows
t-4 .. t-1, target t+h-1, built day by day with horizon_support.validation_items_h. Written to <dataset folder>\\300s_h<h>.
Before writing, the script checks that the same builder with h = 1 reproduces the existing temporal_test.pkl exactly.
Edge-weight arrays (W) are not carried over; no stage-7 evaluation reads them.
Run from ndt_stage5:  python make_horizon_dataset.py --project-root .. --dataset unsw --horizon 2
"""
import argparse, json, pickle, shutil
from pathlib import Path
import numpy as np
from horizon_support import validation_items_h

BASES = {'unsw': 'dataset_ready', 'deakin': 'dataset_ready_deakin_chronological',
         'deakin_fold2': 'dataset_ready_deakin_fold2', 'unsw_fold2': 'dataset_ready_unsw_fold2'}
L = 4


def main(args):
    root = Path(args.project_root).resolve(); src = root / BASES[args.dataset] / '300s'
    out = root / BASES[args.dataset] / f'300s_h{args.horizon}'
    if (out / 'metadata.json').exists(): raise ValueError(f'{out} exists; delete it first if you really want to rebuild')
    if out.exists(): shutil.rmtree(out)  # an interrupted build (metadata.json is written last)
    meta = json.loads((src / 'metadata.json').read_text(encoding='utf-8-sig'))
    # guard: the same builder with h = 1 must reproduce the existing one-step test file exactly
    one = sorted(validation_items_h(pickle.load(open(src / 'static_test.pkl', 'rb')), L, 1), key=lambda i: (i['date'], i['target_window_id']))
    ref = sorted(pickle.load(open(src / 'temporal_test.pkl', 'rb'))['items'], key=lambda i: (i['date'], i['target_window_id']))
    same = len(one) == len(ref) and all(
        a['date'] == b['date'] and a['target_window_id'] == b['target_window_id'] and list(a['input_window_ids']) == list(b['input_window_ids'])
        and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ['A_seq', 'A_target', 'X_seq', 'X_target']) for a, b in zip(one, ref))
    if not same: raise ValueError('Rebuilding the one-step test items from static_test.pkl does not reproduce temporal_test.pkl')
    print('Check passed: the builder reproduces the existing one-step test file exactly')
    out.mkdir(parents=True); counts = {}
    for split in ['train', 'val', 'test']:
        shutil.copy2(src / f'static_{split}.pkl', out / f'static_{split}.pkl')
        static = pickle.load(open(src / f'static_{split}.pkl', 'rb'))
        seqs = validation_items_h(static, L, args.horizon); counts[split] = len(seqs)
        head = {k: static[k] for k in ['node_list', 'node_to_idx', 'node_features'] if k in static}
        pickle.dump(dict(head, seq_len=L, horizon=args.horizon, items=seqs), open(out / f'temporal_{split}.pkl', 'wb'))
    md = dict(meta, seq_len=L, horizon=args.horizon, num_train_sequences=counts['train'], num_val_sequences=counts['val'],
              num_test_sequences=counts['test'],
              one_step_rebuild_check='passed',
              split_creation=f'Targets {args.horizon} windows after the last history window; built by make_horizon_dataset.py from 300s')
    (out / 'metadata.json').write_text(json.dumps(md, indent=2))
    print(json.dumps(dict(dataset=args.dataset, horizon=args.horizon, sequences=counts), indent=2)); print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(BASES), required=True)
    p.add_argument('--horizon', type=int, choices=[2, 4], required=True)
    main(p.parse_args())
