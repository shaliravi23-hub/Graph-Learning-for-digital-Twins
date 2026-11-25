"""Exploratory data and representation analysis (data_analysis_plan.md). Descriptive; no hypothesis tests.

D1  Pair-history ceiling. Each ordered pair (u, v) in a target window has a state: its own presence in the last four
    windows and that of the reverse pair (v, u), 8 bits, 256 states. A table of P(edge | state) estimated on the
    evaluated days themselves is the best any method can do with this information on those days (an in-sample
    ceiling); the same table estimated on the training days is a method that could be used in practice. Both rank pairs
    by P(edge | state); within the boundary state of the edge budget, metrics are the expected values under a random
    choice among the tied pairs. They are compared with the best learned model (asymmetric TGAE, optimised and
    published; five seeds) and simple methods at the validation edge budget.
D2  Where the edges come from. Target edges and changes are split by the trace they leave in the pair's 8-bit history:
    present in the previous window; absent there but present earlier in the last four windows; only the reverse pair
    present; no trace. Edges right of each method are split the same way.
D3  Inputs used. The optimised and published asymmetric TGAE are re-scored on the test windows with (a) the node
    features set to zero, (b) the feature rows of the nodes permuted (fixed permutation, seed 0, the same in every
    window), and (c) the adjacency history set to zero with the features intact. AP and edges right at the budget are
    compared with the intact inputs.
Outputs in --out: data_analysis.json, ceiling.csv, traces.csv, features.csv.
Run from ndt_stage5:
  python data_analysis.py --project-root .. --out ..\\data_analysis
"""
import argparse, csv, importlib.util, inspect, json, platform, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import train_seeds as ts
from evaluate_pilot import load, sha, nonself, topk, metrics, reach, select_threshold, fit_transition, baseline_records
from evaluate_temporal import validation_items
from run_analysis_seeds import L, finite_ewma, tune_ewma, record
from periodicity_check import fit_transition2, score2
from history_lr import TrainStats, features, score_items, C_GRID

SEEDS = [42, 43, 44, 45, 46]
TRACES = ['previous_window', 'earlier_in_history', 'reverse_only', 'no_trace']


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def states(item):
    """8-bit state per ordered pair: bits 0-3 own presence at t-1..t-4, bits 4-7 reverse presence at t-1..t-4."""
    a = (np.asarray(item['A_seq'][-L:]) > 0)[::-1]          # a[0] is t-1
    s = np.zeros(a.shape[1:], dtype=np.int64)
    for k in range(L):
        s |= a[k].astype(np.int64) << k; s |= a[k].T.astype(np.int64) << (L + k)
    return s


def trace(item):
    s = states(item); own = s & 0xF; rev = s >> L
    t = np.full(s.shape, 3)
    t[rev > 0] = 2; t[own > 0] = 1; t[(own & 1) > 0] = 0
    return t


def fit_table(items, mask):
    pos = np.zeros(256); tot = np.zeros(256)
    for i in items:
        s = states(i)[mask]; y = (i['A_target'][mask] > 0)
        np.add.at(tot, s, 1); np.add.at(pos, s, y)
    return np.where(tot > 0, pos / np.maximum(tot, 1), 0.0)


def expected_budget(score, truth, previous, k, mask):
    """Edges right, changes recovered and F1 at budget k, expected over a random choice among tied pairs."""
    s = score[mask]; y = truth[mask] > 0; prev = previous[mask] > 0
    order = np.argsort(-s, kind='stable'); s_sorted = s[order]
    if k <= 0: sel = np.zeros_like(s, float)
    else:
        kth = s_sorted[min(k, len(s)) - 1]
        above = s > kth; tied = s == kth; r = k - above.sum()
        sel = above.astype(float) + tied * (r / max(tied.sum(), 1))
    tp = float((sel * y).sum()); new = y & ~prev; gone = ~y & prev
    return dict(edges_right=tp, changes_recovered=float((sel * new).sum() + ((1 - sel) * gone).sum()),
                F1_at_budget=2 * tp / (k + y.sum()) if (k + y.sum()) else 0.0, selection=sel)


def evaluate_scores(items, scores, k, mask, tied):
    rows, by_trace = [], np.zeros(len(TRACES))
    for i, s in zip(items, scores):
        if tied:
            m = expected_budget(s, i['A_target'], i['A_seq'][-1], k, mask); sel = np.zeros(s.shape); sel[mask] = m.pop('selection')
        else:
            p = topk(s, k); mm = metrics(i['A_target'] > 0, p, i['A_seq'][-1], reach(i['A_target'] > 0)); sel = p.astype(float)
            m = dict(edges_right=mm['true_positive'], changes_recovered=mm['new_edges_recovered'] + mm['disappearances_recovered'],
                     F1_at_budget=mm['edge_f1'])
        y = (i['A_target'] > 0) & mask
        m['AP'] = average_precision_score(y[mask], s[mask]) if y.sum() else np.nan
        t = trace(i)
        for j in range(len(TRACES)): by_trace[j] += float((sel * y * (t == j)).sum())
        rows.append(m)
    out = {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]}
    out.update({f'edges_right_{name}': by_trace[j] / len(items) for j, name in enumerate(TRACES)})
    return out


def simple_scores(train, vi, ti):
    t1 = fit_transition(train); n = len(vi[0]['A_target']); t2, _ = fit_transition2(train, nonself(n))
    alpha, _ = tune_ewma(vi, t1, True)
    out = {'EWMA_L4': [r['score'] for r in finite_ewma(ti, alpha)],
           'Seasonal_copy': [i['A_seq'][-2].astype(float) for i in ti],
           'Order2_Transition': score2(ti, t2)}
    tr = validation_items(train, L); stats = TrainStats(train)
    X = np.concatenate([features(i, stats, True) for i in tr]); y = np.concatenate([i['A_target'][stats.mask] for i in tr]).astype(int)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd; grid = []
    for C in C_GRID:
        m = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-6).fit(Xs, y)
        grid.append((select_threshold(record(vi, score_items(vi, stats, m, mu, sd)))[1], -C, m))
    m = max(grid, key=lambda g: g[:2])[2]
    out['History_LR'] = score_items(ti, stats, m, mu, sd)
    return out


def model_scores(job_dir, keys):
    out = []
    for seed in SEEDS:
        z = np.load(Path(job_dir) / f'seed{seed}' / 'scores.npz')
        if [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != keys: raise ValueError('Keys differ: ' + str(job_dir))
        out.append(list(z['test'].astype(float)))
    return out


def feature_probe(root, run_dir, te, k, mask, device):
    import torch
    proto = json.loads((Path(run_dir) / 'protocol.json').read_text()); name = proto['model']; cfg = ts.MODELS[name]
    spec = importlib.util.spec_from_file_location('probe_' + name, root / cfg['script'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); cls = getattr(module, cfg['cls'])
    n = len(te[0]['A_target'])
    kw = dict(proto['constructor'], num_nodes=n, feat_dim=10, in_dim=10, seq_len=L)
    kw = {k_: v for k_, v in kw.items() if k_ in inspect.signature(cls.__init__).parameters}
    if proto.get('decoder_variant'): cls = ts.with_decoder(cls, proto['decoder_variant'], proto['constructor']['latent_dim'])
    a = torch.tensor(np.stack([np.asarray(i['A_seq'][-L:], np.float32) for i in te]))
    x = torch.tensor(np.stack([np.asarray(i['X_seq'][-L:], np.float32) for i in te]))
    perm = np.random.default_rng(0).permutation(n)
    variants = {'intact': (a, x), 'zero_features': (a, torch.zeros_like(x)), 'permuted_features': (a, x[:, :, perm, :]),
                'zero_history': (torch.zeros_like(a), x)}
    res = {v: [] for v in variants}
    for seed in SEEDS:
        job = Path(run_dir) / f'seed{seed}'
        done = json.loads((job / 'completed.json').read_text())
        if sha(job / 'best.pt') != done['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch: ' + str(job))
        model = cls(**kw).to(device); model.load_state_dict(torch.load(job / 'best.pt', map_location=device, weights_only=True)); model.eval()
        for v, (av, xv) in variants.items():
            with torch.inference_mode():
                s = np.concatenate([model(av[j:j + 16].to(device), xv[j:j + 16].to(device))['A_logits'].float().cpu().numpy() for j in range(0, len(te), 16)])
            res[v].append(evaluate_scores(te, list(s), k, mask, tied=False))
    return {v: {key: float(np.mean([r[key] for r in rs])) for key in ['AP', 'edges_right', 'changes_recovered']} for v, rs in res.items()}


def main(args):
    t0 = time.time(); root = Path(args.project_root).resolve(); out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    ceiling, traces, feats, info = [], [], [], {}
    sets = [('unsw', root / 'dataset_ready' / '300s', 'temporal_test.pkl'), ('deakin', root.joinpath(*ts.DATASETS['deakin']['path']), 'temporal_test.pkl'),
            ('unsw_further', root / 'dataset_ready_unsw_newdays' / '300s', 'temporal_test.pkl')]
    for name, ds, test_file in sets:
        train_dir = root / 'dataset_ready' / '300s' if name == 'unsw_further' else ds
        train = load(train_dir / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / test_file)
        ts.drop_self_loops(train, val, test)
        vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id'])); tr = validation_items(train, L)
        keys = [(i['date'], int(i['target_window_id'])) for i in ti]
        n = len(ti[0]['A_target']); mask = nonself(n); k = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
        print(f'{name}: {len(ti)} test windows, {n} nodes, budget {k}', flush=True)

        # D2: traces of target edges and changes
        tot = np.zeros(len(TRACES)); new = np.zeros(len(TRACES)); gone = 0.0
        for i in ti:
            t = trace(i); y = (i['A_target'] > 0) & mask; prev = (i['A_seq'][-1] > 0) & mask
            for j in range(len(TRACES)): tot[j] += (y & (t == j)).sum(); new[j] += (y & ~prev & (t == j)).sum()
            gone += (~y & prev).sum()
        W = len(ti)
        traces.append(dict(dataset=name, windows=W, edges_per_window=tot.sum() / W, appearing_per_window=new.sum() / W,
                           disappearing_per_window=gone / W,
                           **{f'edges_{t}': tot[j] / W for j, t in enumerate(TRACES)},
                           **{f'appearing_{t}': new[j] / W for j, t in enumerate(TRACES)}))

        # D1: tables and methods at the budget
        table_in = fit_table(ti, mask); table_tr = fit_table(tr, mask)
        rows = {'Pair-history ceiling (in-sample)': (evaluate_scores(ti, [table_in[states(i)] for i in ti], k, mask, tied=True), 1),
                'Pair-history table (training days)': (evaluate_scores(ti, [table_tr[states(i)] for i in ti], k, mask, tied=True), 1)}
        for m, s in simple_scores(train, vi, ti).items(): rows[m] = (evaluate_scores(ti, s, k, mask, tied=False), 1)
        if name == 'unsw_further':
            runs = {'Asymmetric TGAE (optimised)': root / 'stage9_bo' / 'unsw' / 'newday_scores' / 'tgae_bilinear',
                    'Asymmetric TGAE (published)': root / 'scores_unsw_newdays' / 'tgae_bilinear'}
        else:
            runs = {'Asymmetric TGAE (optimised)': root / 'stage9_bo' / name / 'final' / 'tgae_bilinear',
                    'Asymmetric TGAE (published)': root / f'seeds_{name}_decoder' / 'tgae_bilinear'}
        for m, d in runs.items():
            per_seed = [evaluate_scores(ti, s, k, mask, tied=False) for s in model_scores(d, keys)]
            rows[m] = ({key: float(np.mean([r[key] for r in per_seed])) for key in per_seed[0]}, len(per_seed))
        for m, (r, seeds) in rows.items():
            ceiling.append(dict(dataset=name, method=m, seeds=seeds, budget=k, **r))
            print(f"  {m:36s} right {r['edges_right']:.2f} changes {r['changes_recovered']:.2f} F1@k {r['F1_at_budget']:.3f} AP {r['AP']:.3f}", flush=True)
        info[name] = dict(windows=W, nodes=n, budget=k, table_states_used=int((table_in > 0).sum()))

        # D3: node features (main test sets only)
        if name != 'unsw_further':
            for m, d in [('Asymmetric TGAE (optimised)', root / 'stage9_bo' / name / 'final' / 'tgae_bilinear'),
                         ('Asymmetric TGAE (published)', root / f'seeds_{name}_decoder' / 'tgae_bilinear')]:
                r = feature_probe(root, d, ti, k, mask, args.device)
                for v, vals in r.items(): feats.append(dict(dataset=name, model=m, features=v, **vals))
                print(f'  {m}: ' + '; '.join(f"{v} AP {vals['AP']:.3f} right {vals['edges_right']:.2f}" for v, vals in r.items()), flush=True)

    out.mkdir(parents=True)
    write_csv(out / 'ceiling.csv', ceiling); write_csv(out / 'traces.csv', traces); write_csv(out / 'features.csv', feats)
    (out / 'data_analysis.json').write_text(json.dumps(dict(info=info, ceiling=ceiling, traces=traces, features=feats,
        source_hashes={p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
        python=platform.python_version(), seconds=round(time.time() - t0, 1)), indent=2, default=float))
    print(f'Saved {out.resolve()} ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--out', required=True); p.add_argument('--device', default='cuda')
    main(p.parse_args())
