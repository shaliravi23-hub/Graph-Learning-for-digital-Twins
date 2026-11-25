"""Stage 7: on one test set, compare the best learned model with the best simple method, both chosen on validation only.

Simple methods: Persistence, Recent_frequency_L4, Transition, EWMA_L4, EWMA_expanding (EWMA weights tuned on validation as in
run_analysis_seeds.py) and History-LR (history_lr.py). Learned models: those found in --runs (default candidates:
dyngem aernn vgrnn tgae vrdag static_prev tgae_bilinear; static_target is a reference and is never a candidate).
S* = simple method with the highest validation macro F1 at its exact validation threshold.
M* = learned model with the highest mean validation macro F1 over its seeds.
At the validation edge budget, per test window: edges right (true positives) and changes recovered (new edges predicted plus
disappearing edges not predicted). M* - S* per window, two-level paired bootstrap (seeds; circular moving blocks within each
capture), 10,000 replicates, 8-window blocks; two-sided p-values. summarize_stage7.py applies Holm across test sets and
classifies each one.
Run from ndt_stage5, e.g.
  python compare_best.py --project-root .. --dataset unsw --runs ..\\tuning_unsw\\tuned --history-lr ..\\history_lr_unsw --label tuned --out ..\\stage7\\tuned_unsw.json
"""
import argparse, json, platform
from pathlib import Path
import numpy as np
from horizon_support import validate_h, validation_items_h, fit_transition_h
from evaluate_pilot import load, sha, baseline_records, reach
from train_seeds import drop_self_loops
from run_analysis_seeds import DATASETS, L, finite_ewma, tune_ewma, record, load_runs, two_level_bootstrap
from stage6_analysis import categories, predictions, window_table

CANDIDATES = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bilinear']
METRICS = ['true_positive', 'changes_recovered']
SIMPLE_ORDER = ['Persistence', 'Recent_frequency_L4', 'Transition', 'EWMA_L4', 'EWMA_expanding', 'History_LR']


def main(args):
    project = Path(args.project_root).resolve(); ds = project.joinpath(*DATASETS[args.dataset]['path']); out = Path(args.out)
    if out.exists(): raise ValueError(f'{out} exists; choose a fresh file')
    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    h = DATASETS[args.dataset].get('horizon', 1)
    drop_self_loops(train, val, test); validate_h(train, val, test, meta, h)
    vi = validation_items_h(val, L, h); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    if (len(vi), len(ti)) != DATASETS[args.dataset]['counts']: raise ValueError('Unexpected cohorts ' + str((len(vi), len(ti))))
    keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    runs = {}
    for d in args.runs:
        for name, seeds in load_runs(d, args.dataset, keys_v, keys_t, args.candidates).items():
            if name in runs: raise ValueError(f'{name} appears in more than one --runs folder')
            runs[name] = seeds
    missing = [m for m in args.candidates if m not in runs]
    if missing: raise ValueError(f'Missing learned models {missing} in {args.runs}')
    transition = fit_transition_h(train, h); budget = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    refs = [reach(i['A_target']) for i in ti]; cats = categories(ti, train, val)
    simple = {}
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        simple[name] = (baseline_records(vi, kind, transition), baseline_records(ti, kind, transition))
    alphas = {}
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(vi, transition, finite); alphas[name] = alpha
        simple[name] = (vr, finite_ewma(ti, alpha) if finite else baseline_records(ti, 'ewma', transition, alpha))
    z = np.load(Path(args.history_lr) / 'scores.npz', allow_pickle=False)
    if [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != keys_t: raise ValueError('History-LR keys differ')
    lr_fit = json.loads((Path(args.history_lr) / 'fit.json').read_text())
    if lr_fit['dataset'] != args.dataset: raise ValueError('History-LR was fitted on ' + lr_fit['dataset'])
    simple['History_LR'] = (record(vi, list(z['validation'].astype(float))), record(ti, list(z['test'].astype(float))))

    policy = 'budget_' + str(budget); table = {}; rows = {}
    for name in SIMPLE_ORDER:
        vr, tr = simple[name]; p, choice = predictions(vr, tr, budget); t = window_table(tr, p[policy], refs, cats)
        table[name] = [t]
        rows[name] = dict(kind='simple', validation_F1=choice['val_f1'], test_edges_right=float(t['true_positive'].mean()),
                          test_changes_recovered=float(t['changes_recovered'].mean()), test_F1_budget=float(np.nanmean(t['edge_f1'])))
    for name in args.candidates:
        tabs, vfs = [], []
        for seed, (v, t_, info) in sorted(runs[name].items()):
            vr, tr = record(vi, list(v)), record(ti, list(t_)); p, choice = predictions(vr, tr, budget)
            tabs.append(window_table(tr, p[policy], refs, cats)); vfs.append(choice['val_f1'])
        table[name] = tabs
        rows[name] = dict(kind='learned', seeds=sorted(runs[name]), validation_F1=float(np.mean(vfs)),
                          test_edges_right=float(np.mean([t['true_positive'].mean() for t in tabs])),
                          test_changes_recovered=float(np.mean([t['changes_recovered'].mean() for t in tabs])),
                          test_F1_budget=float(np.mean([np.nanmean(t['edge_f1']) for t in tabs])))
    s_star = max(SIMPLE_ORDER, key=lambda n: (rows[n]['validation_F1'], -SIMPLE_ORDER.index(n)))
    m_star = max(args.candidates, key=lambda n: (rows[n]['validation_F1'], -args.candidates.index(n)))
    base = table[s_star][0]
    deltas = np.stack([np.stack([t[m] - base[m] for m in METRICS], 1) for t in table[m_star]])
    dates = sorted({d for d, _ in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
    samples = two_level_bootstrap(deltas, captures, args.block, args.bootstrap_replicates, np.random.default_rng(20261114)); B = len(samples)
    contrasts = {}
    for j, m in enumerate(METRICS):
        d = float(np.nanmean(deltas[:, :, j])); lo, hi = np.nanquantile(samples[:, j], [.025, .975])
        p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
        contrasts[m] = dict(difference=d, ci=[float(lo), float(hi)], p_value=p,
                            seed_differences=np.nanmean(deltas[:, :, j], axis=1).round(6).tolist())
    result = dict(dataset=args.dataset, label=args.label, budget=budget, best_simple=s_star, best_learned=m_star,
                  contrast_M_minus_S=contrasts, candidates=rows, ewma_alpha=alphas, history_lr_C=lr_fit['chosen_C'],
                  observed=dict(edges_per_window=float(base['true_edges'].mean()), changes_per_window=float(base['changes'].mean())),
                  runs=[str(r) for r in args.runs], replicates=B, block=args.block,
                  input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'temporal_test.pkl', 'metadata.json']},
                  source_hashes={p_.name: sha(p_) for p_ in Path(__file__).parent.glob('*.py')},
                  analysis_plan=(dict(file=Path(args.plan).name, sha256=sha(args.plan)) if args.plan else None),
                  python=platform.python_version())
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    c = contrasts
    print(f'{args.dataset} [{args.label}]: best simple {s_star} (val F1 {rows[s_star]["validation_F1"]:.4f}), best learned {m_star} '
          f'(val F1 {rows[m_star]["validation_F1"]:.4f}); M-S edges right {c["true_positive"]["difference"]:+.3f} '
          f'[{c["true_positive"]["ci"][0]:+.3f}, {c["true_positive"]["ci"][1]:+.3f}], changes {c["changes_recovered"]["difference"]:+.3f} '
          f'[{c["changes_recovered"]["ci"][0]:+.3f}, {c["changes_recovered"]["ci"][1]:+.3f}]  -> {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--runs', nargs='+', required=True); p.add_argument('--history-lr', required=True)
    p.add_argument('--candidates', nargs='+', default=CANDIDATES); p.add_argument('--label', required=True)
    p.add_argument('--out', required=True); p.add_argument('--bootstrap-replicates', type=int, default=10000)
    p.add_argument('--block', type=int, default=8); p.add_argument('--plan')
    main(p.parse_args())
