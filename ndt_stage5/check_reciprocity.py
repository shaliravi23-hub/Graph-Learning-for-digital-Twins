"""How much of each dataset's traffic is two-way, and the best any symmetric decoder (score(u,v) = score(v,u))
could possibly achieve on the test windows. Run from ndt_stage5:  python check_reciprocity.py"""
import pickle
import numpy as np
from sklearn.metrics import average_precision_score

import os
for name, path in [('UNSW', r'..\dataset_ready\300s\temporal_test.pkl'),
                   ('Deakin', r'..\dataset_ready_deakin_chronological\300s\temporal_test.pkl'),
                   ('Deakin split 2 (test 4 Apr)', r'..\dataset_ready_deakin_fold2\300s\temporal_test.pkl'),
                   ('UNSW split 2 (test 27 Sep)', r'..\dataset_ready_unsw_fold2\300s\temporal_test.pkl')]:
    if not os.path.exists(path):
        print(f'{name}: not built yet ({path})'); continue
    items = pickle.load(open(path, 'rb'))['items']
    share, ap, f1 = [], [], []
    for it in items:
        a = np.asarray(it['A_target']) > 0
        np.fill_diagonal(a, False)                      # same-node pairs are never evaluated
        if not a.any():
            continue
        both, either = a & a.T, a | a.T
        share.append(both.sum() / a.sum())              # share of edges whose reverse edge also exists
        mask = ~np.eye(len(a), dtype=bool)
        best_symmetric = a.astype(int) + a.T.astype(int)  # two-way pairs first, then one-way pairs in both directions
        ap.append(average_precision_score(a[mask], best_symmetric[mask]))
        f1.append((2 * a.sum() / (a.sum() + either.sum()),     # predict every pair with traffic in either direction
                   2 * both.sum() / (both.sum() + a.sum())))   # predict only two-way pairs
    f1 = np.array(f1)
    print(f'{name}: {np.mean(share):.3f} of edges have a reverse edge; best possible symmetric decoder: '
          f'AP {np.mean(ap):.3f}, F1 {f1.mean(0).max():.3f}  ({len(share)} test windows)')
