"""History-LR: a logistic regression on simple per-pair history features (no graph neural network).

Purpose: test whether the gain of the temporal graph models comes from the architecture or simply from
*learning* how to weigh the recent history, which the fixed memorization rules (EWMA, Transition) cannot do.

The model sees exactly the information the temporal models see: the L = 4 history windows of the target
and statistics fitted on the training split. Features of the ordered pair (u, v), fixed before any result:
  lag1..lag4          edge u->v present in history window t-1 .. t-4
  rev_lag1            reverse edge v->u present in window t-1
  rev_mean            fraction of the 4 history windows with v->u
  train_freq          fraction of training windows with u->v      (leave-one-out for training rows)
  train_rev_freq      fraction of training windows with v->u      (leave-one-out for training rows)
  transition          Transition-baseline probability given the state of u->v at t-1 (leave-one-out for training rows)
  out_deg_u_lag1      out-degree of u in window t-1 / (N - 1)
  in_deg_v_lag1       in-degree of v in window t-1 / (N - 1)
  out_deg_u_mean      mean out-degree of u over the history / (N - 1)
  in_deg_v_mean       mean in-degree of v over the history / (N - 1)
  u_active, v_active  u (v) has any edge in the history
Features are standardised with training means and SDs. No class weighting. The L2 strength C is chosen
from {0.01, 0.1, 1, 10} by validation macro F1 at the exact validation threshold (ties: lower edge-count
error, then smaller C), the same rule used to tune EWMA. Same-node pairs are never used.

Output (--out): scores.npz (validation and test logits, same layout as train_seeds.py) and fit.json.
Run from ndt_stage5:
  python history_lr.py --project-root .. --dataset unsw --out ..\\history_lr_unsw
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from horizon_support import validate_h, validation_items_h
from evaluate_pilot import load, sha, nonself, select_threshold
from train_seeds import drop_self_loops
from run_analysis_seeds import DATASETS, L, record

FEATURES = ['lag1', 'lag2', 'lag3', 'lag4', 'rev_lag1', 'rev_mean', 'train_freq', 'train_rev_freq', 'transition',
            'out_deg_u_lag1', 'in_deg_v_lag1', 'out_deg_u_mean', 'in_deg_v_mean', 'u_active', 'v_active']
C_GRID = [0.01, 0.1, 1.0, 10.0]


class TrainStats:
    """Training-split statistics, with leave-one-out versions for rows built from the training split."""
    def __init__(self, train, h=1):
        n = len(train['node_list']); self.mask = nonself(n)
        A = np.stack([i['A'] for i in train['items']]).astype(np.float64)
        self.n_windows = len(A); self.edge_sum = A.sum(0)
        # transition counts over consecutive training windows of the same capture (as evaluate_pilot.fit_transition)
        by_date = {}
        for i in train['items']: by_date.setdefault(i['date'], []).append(i)
        self.den = np.zeros((2, n, n)); self.num = np.zeros((2, n, n))
        for items in by_date.values():  # h-step transitions (h = 1 is the stage-6 definition)
            items = sorted(items, key=lambda i: i['window_id'])
            for a, b in zip(items, items[1:]):
                if b['window_id'] != a['window_id'] + 1: raise ValueError('Training transition crosses a gap')
            for a, b in zip(items, items[h:]):
                for s in (0, 1):
                    m = (a['A'] == s) & self.mask; self.den[s] += m; self.num[s] += m * b['A']
        self.pool = (self.num.sum((1, 2)) + 1) / (self.den.sum((1, 2)) + 2)

    def freq(self, target=None):
        if target is None: return self.edge_sum / self.n_windows
        return (self.edge_sum - target) / (self.n_windows - 1)

    def transition(self, prev, target=None):
        den = self.den.copy(); num = self.num.copy()
        if target is not None:  # remove this row's own transition from the counts
            for s in (0, 1):
                m = (prev == s) & self.mask; den[s] -= m; num[s] -= m * target
        t = (num + self.pool[:, None, None]) / (den + 1)
        return np.where(prev > 0, t[1], t[0])


def features(item, stats, loo):
    H = item['A_seq'].astype(np.float64); n = H.shape[1]; prev = H[-1]
    target = item['A_target'].astype(np.float64) if loo else None
    freq = stats.freq(target); deg = n - 1
    out_l1 = prev.sum(1) / deg; in_l1 = prev.sum(0) / deg
    out_m = H.sum(2).mean(0) / deg; in_m = H.sum(1).mean(0) / deg
    active = (H.sum(2) + H.sum(1)).sum(0) > 0
    cols = [H[-1], H[-2], H[-3], H[-4], prev.T, H.mean(0).T, freq, freq.T, stats.transition(prev, target),
            np.repeat(out_l1[:, None], n, 1), np.repeat(in_l1[None, :], n, 0),
            np.repeat(out_m[:, None], n, 1), np.repeat(in_m[None, :], n, 0),
            np.repeat(active[:, None], n, 1).astype(float), np.repeat(active[None, :], n, 0).astype(float)]
    return np.stack([c[stats.mask] for c in cols], 1)


def score_items(items, stats, model, mu, sd):
    out = []
    for i in items:
        n = i['A_target'].shape[0]; s = np.full((n, n), -50.0)
        s[stats.mask] = model.decision_function((features(i, stats, False) - mu) / sd); out.append(s)
    return out


def main(args):
    t0 = time.time()
    project = Path(args.project_root).resolve(); ds = project.joinpath(*DATASETS[args.dataset]['path']); out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    h = DATASETS[args.dataset].get('horizon', 1)
    self_loops = drop_self_loops(train, val, test); audit = validate_h(train, val, test, meta, h)
    tr = validation_items_h(train, L, h); vi = validation_items_h(val, L, h)
    ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    if (len(vi), len(ti)) != DATASETS[args.dataset]['counts']:
        raise ValueError('Unexpected aligned validation/test cohorts ' + str((len(vi), len(ti))))
    stats = TrainStats(train, h)
    X = np.concatenate([features(i, stats, True) for i in tr]); y = np.concatenate([i['A_target'][stats.mask] for i in tr]).astype(int)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd
    print(f'{args.dataset}: {len(tr)} training windows, {len(y)} rows, {int(y.sum())} positives', flush=True)
    grid = []
    for C in C_GRID:
        m = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-6).fit(Xs, y)
        vr = record(vi, score_items(vi, stats, m, mu, sd)); th, vf, ve = select_threshold(vr)
        vap = float(np.nanmean([average_precision_score(r['truth'][stats.mask] > 0, r['score'][stats.mask])
                                if r['truth'].sum() else np.nan for r in vr]))
        grid.append(dict(C=C, validation_F1=vf, validation_edge_error=ve, validation_AP=vap, converged=bool(m.n_iter_[0] < 2000), model=m))
        print(f'  C={C}: validation F1 {vf:.4f}, AP {vap:.4f}', flush=True)
    best = max(grid, key=lambda g: (g['validation_F1'], -g['validation_edge_error'], -g['C']))
    m = best['model']
    v_scores = score_items(vi, stats, m, mu, sd); t_scores = score_items(ti, stats, m, mu, sd)  # test scored only after selection
    out.mkdir(parents=True)
    np.savez_compressed(out / 'scores.npz', validation=np.stack(v_scores).astype(np.float32), test=np.stack(t_scores).astype(np.float32),
                        validation_dates=np.array([i['date'] for i in vi]), validation_windows=np.array([int(i['target_window_id']) for i in vi]),
                        test_dates=np.array([i['date'] for i in ti]), test_windows=np.array([int(i['target_window_id']) for i in ti]))
    fit = dict(dataset=args.dataset, model='History_LR', features=FEATURES, C_grid=C_GRID,
               grid=[{k: v for k, v in g.items() if k != 'model'} for g in grid], chosen_C=best['C'],
               coefficients_standardised=dict(zip(FEATURES, m.coef_[0].tolist())), intercept=float(m.intercept_[0]),
               feature_mean=dict(zip(FEATURES, mu.tolist())), feature_sd=dict(zip(FEATURES, sd.tolist())),
               training_windows=len(tr), training_rows=int(len(y)), training_positives=int(y.sum()),
               self_loops_removed=self_loops, nodes=audit['nodes'],
               input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'temporal_test.pkl', 'metadata.json']},
               script_sha256=sha(__file__), python=platform.python_version(), numpy=np.__version__, seconds=round(time.time() - t0, 1))
    (out / 'fit.json').write_text(json.dumps(fit, indent=2))
    print(f'Saved {out.resolve()} (chosen C={best["C"]})', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--out', required=True)
    main(p.parse_args())
