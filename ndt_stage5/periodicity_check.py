"""Exploratory check (specified in periodicity_check_plan.md before running): is the learned models' one clear win, on UNSW at
one step ahead, explained by connections that switch on and off every other 5-minute window?

Part A, lag profile: mean number of edges that differ between windows t and t-k (k = 1..8), within each day, for the train,
validation and test windows of the 300 s data. A two-window on/off pattern shows as a zig-zag: fewer changes at even lags.
Part B, one step ahead, the same cohorts and budget as the stage 7A tuned comparison:
  Seasonal naive       copy window t-2
  Order-2 Transition   per-pair P(edge at t | states at t-2 and t-1), counted on training days, with the same pooled
                       pseudo-count as Transition (an order-2 version of the existing Transition method)
  M*                   the tuned learned model chosen in stage 7A (read from ..\\stage7\\tuned_<test set>.json), seeds 42-46
M* minus Order-2 Transition is tested with the stage-7 rule (two-level bootstrap, Holm over 8 contrasts, margin 0.5).
Run from ndt_stage5:  python periodicity_check.py --project-root .. --out ..\\stage7\\periodicity_check.json
"""
import argparse, collections, json, platform
from pathlib import Path
import numpy as np
from evaluate_pilot import load, sha, baseline_records, reach, nonself
from horizon_support import validate_h, validation_items_h, fit_transition_h
from train_seeds import drop_self_loops
from run_analysis_seeds import DATASETS, L, record, load_runs, two_level_bootstrap, holm
from stage6_analysis import categories, predictions, window_table
from summarize_stage7 import classify

SETS = ['unsw', 'deakin', 'deakin_fold2', 'unsw_fold2']
METRICS = ['true_positive', 'changes_recovered']
LAGS = range(1, 9)


def by_day(static):
    days = collections.defaultdict(dict)
    for i in static['items']: days[i['date']][int(i['window_id'])] = i['A'] > 0
    for d, w in days.items():
        ids = sorted(w)
        if ids != list(range(ids[0], ids[0] + len(ids))): raise ValueError(f'Day {d} has gaps')
    return days


def lag_profile(static, mask):
    out = {}
    for k in LAGS:
        diffs = [((w[t] ^ w[t - k]) & mask).sum() for w in by_day(static).values() for t in w if t - k in w]
        out[k] = float(np.mean(diffs))
    return out


def fit_transition2(train, mask):
    """Per-pair P(edge at t | state at t-2, state at t-1); state index 2*a(t-2) + a(t-1)."""
    n = len(mask); den = np.zeros((4, n, n)); num = np.zeros_like(den)
    for w in by_day(train).values():
        for t in w:
            if t - 2 in w:
                s = 2 * w[t - 2].astype(int) + w[t - 1].astype(int)
                for k in range(4):
                    m = (s == k) & mask; den[k] += m; num[k] += m * w[t]
    pool = (num.sum((1, 2)) + 1) / (den.sum((1, 2)) + 2)
    return (num + pool[:, None, None]) / (den + 1), den.sum((1, 2))


def score2(items, table):
    out = []
    for i in items:
        s = 2 * (i['A_seq'][-2] > 0).astype(int) + (i['A_seq'][-1] > 0).astype(int)
        out.append(np.take_along_axis(table, s[None], 0)[0])
    return out


def main(args):
    project = Path(args.project_root).resolve(); out = Path(args.out)
    if out.exists(): raise ValueError(f'{out} exists; choose a fresh file')
    results = {}
    for d in args.datasets:
        ds = project.joinpath(*DATASETS[d]['path'])
        train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
        static_test = load(ds / 'static_test.pkl'); meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
        drop_self_loops(train, val, test); validate_h(train, val, test, meta, 1)  # lag counts below exclude the diagonal
        vi = validation_items_h(val, L, 1); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
        if (len(vi), len(ti)) != DATASETS[d]['counts']: raise ValueError('Unexpected cohorts')
        keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
        ref = json.loads((Path(args.stage7) / f'tuned_{d}.json').read_text())
        mask = nonself(len(train['node_list'])); budget = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
        if budget != ref['budget']: raise ValueError(f'Budget {budget} differs from stage 7A ({ref["budget"]})')
        # Part A
        profile = {s: lag_profile(x, mask) for s, x in [('train', train), ('validation', val), ('test', static_test)]}
        new = [(i['A_target'] > 0) & ~(i['A_seq'][-1] > 0) & mask for i in ti]
        gone = [~(i['A_target'] > 0) & (i['A_seq'][-1] > 0) & mask for i in ti]
        back = sum((n_ & (i['A_seq'][-2] > 0)).sum() for n_, i in zip(new, ti)) / max(1, sum(n_.sum() for n_ in new))
        off = sum((g & ~(i['A_seq'][-2] > 0)).sum() for g, i in zip(gone, ti)) / max(1, sum(g.sum() for g in gone))
        # Part B
        refs = [reach(i['A_target']) for i in ti]; cats = categories(ti, train, val); policy = 'budget_' + str(budget)
        t1 = fit_transition_h(train, 1); t2, state_counts = fit_transition2(train, mask)
        simple = {'Persistence': (baseline_records(vi, 'persistence', t1), baseline_records(ti, 'persistence', t1)),
                  'Transition': (baseline_records(vi, 'transition', t1), baseline_records(ti, 'transition', t1)),
                  'Seasonal_naive': (record(vi, [i['A_seq'][-2].astype(float) for i in vi]), record(ti, [i['A_seq'][-2].astype(float) for i in ti])),
                  'Order2_Transition': (record(vi, score2(vi, t2)), record(ti, score2(ti, t2)))}
        rows, tables = {}, {}
        for name, (vr, tr) in simple.items():
            p, choice = predictions(vr, tr, budget); t = window_table(tr, p[policy], refs, cats); tables[name] = [t]
            rows[name] = dict(validation_F1=choice['val_f1'], test_edges_right=float(t['true_positive'].mean()),
                              test_changes_recovered=float(t['changes_recovered'].mean()), test_F1_budget=float(np.nanmean(t['edge_f1'])))
        for name in ['Persistence', 'Transition']:  # must reproduce the stage 7A numbers exactly
            for k in ['validation_F1', 'test_edges_right', 'test_changes_recovered']:
                if not np.isclose(rows[name][k], ref['candidates'][name][k], rtol=0, atol=1e-9):
                    raise ValueError(f'{d} {name} {k} does not reproduce stage 7A')
        m_star = ref['best_learned']; runs = load_runs(Path(args.tuning_root) / f'tuning_{d}' / 'tuned', d, keys_v, keys_t, [m_star])[m_star]
        tabs, vfs = [], []
        for seed, (v, t_, info) in sorted(runs.items()):
            vr, tr = record(vi, list(v)), record(ti, list(t_)); p, choice = predictions(vr, tr, budget)
            tabs.append(window_table(tr, p[policy], refs, cats)); vfs.append(choice['val_f1'])
        rows[m_star] = dict(seeds=sorted(runs), validation_F1=float(np.mean(vfs)),
                            test_edges_right=float(np.mean([t['true_positive'].mean() for t in tabs])),
                            test_changes_recovered=float(np.mean([t['changes_recovered'].mean() for t in tabs])),
                            test_F1_budget=float(np.mean([np.nanmean(t['edge_f1']) for t in tabs])))
        for k in ['validation_F1', 'test_edges_right', 'test_changes_recovered']:
            if not np.isclose(rows[m_star][k], ref['candidates'][m_star][k], rtol=0, atol=1e-9):
                raise ValueError(f'{d} {m_star} {k} does not reproduce stage 7A')
        base = tables['Order2_Transition'][0]
        deltas = np.stack([np.stack([t[m] - base[m] for m in METRICS], 1) for t in tabs])
        dates = sorted({x for x, _ in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == x]) for x in dates]
        samples = two_level_bootstrap(deltas, captures, 8, args.bootstrap_replicates, np.random.default_rng(20261114)); B = len(samples)
        contrasts = {}
        for j, m in enumerate(METRICS):
            lo, hi = np.nanquantile(samples[:, j], [.025, .975])
            p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
            contrasts[m] = dict(difference=float(np.nanmean(deltas[:, :, j])), ci=[float(lo), float(hi)], p_value=p)
        simple_pool = {**{k: v['validation_F1'] for k, v in ref['candidates'].items() if v['kind'] == 'simple'},
                       'Seasonal_naive': rows['Seasonal_naive']['validation_F1'], 'Order2_Transition': rows['Order2_Transition']['validation_F1']}
        results[d] = dict(budget=budget, best_learned=m_star, stage7a_best_simple=ref['best_simple'],
                          best_simple_with_new_methods=max(simple_pool, key=simple_pool.get),
                          lag_profile=profile, returned_from_t_minus_2=float(back), disappeared_after_absent_at_t_minus_2=float(off),
                          order2_state_counts=state_counts.tolist(), methods=rows, contrast_M_minus_Order2=contrasts,
                          observed=dict(edges_per_window=float(base['true_edges'].mean()), changes_per_window=float(base['changes'].mean())),
                          input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'static_test.pkl', 'temporal_test.pkl', 'metadata.json']})
    flat = [(d, m) for d in results for m in METRICS]
    for (d, m), a in zip(flat, holm([results[d]['contrast_M_minus_Order2'][m]['p_value'] for d, m in flat])):
        results[d]['contrast_M_minus_Order2'][m]['p_holm'] = float(a)
    for d in results: results[d]['classification_vs_Order2'] = classify(results[d]['contrast_M_minus_Order2'])

    # pre-specified expectations (periodicity_check_plan.md)
    def zigzag(c): return c[2] < c[1] and c[2] < c[3] and c[4] < c[3] and c[4] < c[5]
    exp = {}
    if {'unsw', 'unsw_fold2'} <= set(results):
        exp['E1 zig-zag on the UNSW test days'] = all(zigzag(results[d]['lag_profile']['test']) for d in ['unsw', 'unsw_fold2'])
    if 'unsw' in results:
        exp['E2 UNSW: M* - Order-2 Transition changes below +0.5'] = results['unsw']['contrast_M_minus_Order2']['changes_recovered']['difference'] < .5
    if len(results) == 4:
        exp['E3 learned clearly better than Order-2 Transition on 0 of 4'] = sum(
            r['classification_vs_Order2'] == 'learned clearly better' for r in results.values()) == 0
    summary = dict(results=results, expectations=exp, replicates=args.bootstrap_replicates, block=8,
                   plan=(dict(file=Path(args.plan).name, sha256=sha(args.plan)) if args.plan else None),
                   source_hashes={p_.name: sha(p_) for p_ in Path(__file__).parent.glob('*.py')}, python=platform.python_version())
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(summary, indent=2))

    print('Part A: mean edges that differ between windows t and t-k (test windows)')
    print('  test set       ' + ' '.join(f'k={k:<4d}' for k in LAGS) + '  new edges back from t-2   disappearances absent at t-2')
    for d, r in results.items():
        print(f'  {d:14s} ' + ' '.join(f'{r["lag_profile"]["test"][k]:6.2f}' for k in LAGS)
              + f'  {r["returned_from_t_minus_2"]:22.1%}   {r["disappeared_after_absent_at_t_minus_2"]:.1%}')
    print('\nPart B: one step ahead, edges right / changes recovered per window (validation F1)')
    for d, r in results.items():
        print(f'  {d:14s} ' + '  '.join(f'{k}: {v["test_edges_right"]:.2f} / {v["test_changes_recovered"]:.2f} ({v["validation_F1"]:.3f})'
                                        for k, v in r['methods'].items()))
        c = r['contrast_M_minus_Order2']
        print(f'  {"":14s} M* - Order-2 Transition: edges right {c["true_positive"]["difference"]:+.2f} [{c["true_positive"]["ci"][0]:+.2f}, '
              f'{c["true_positive"]["ci"][1]:+.2f}] p_holm {c["true_positive"]["p_holm"]:.4f}; changes {c["changes_recovered"]["difference"]:+.2f} '
              f'[{c["changes_recovered"]["ci"][0]:+.2f}, {c["changes_recovered"]["ci"][1]:+.2f}] p_holm {c["changes_recovered"]["p_holm"]:.4f} '
              f'-> {r["classification_vs_Order2"]}; best simple by validation with the new methods: {r["best_simple_with_new_methods"]}')
    print('\nExpectations:'); [print(f'  {k}: {"HELD" if v else "NOT HELD"}') for k, v in exp.items()]
    print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--datasets', nargs='+', choices=SETS, default=SETS)
    p.add_argument('--stage7', default='../stage7', help='Folder with the stage 7A tuned_<test set>.json files')
    p.add_argument('--tuning-root', default='..', help='Folder that holds tuning_<test set>')
    p.add_argument('--bootstrap-replicates', type=int, default=10000); p.add_argument('--plan')
    main(p.parse_args())
