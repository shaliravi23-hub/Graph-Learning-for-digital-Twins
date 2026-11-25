"""Stage 5 analysis: multi-seed learned models versus history baselines under identical decoders.

Extends ndt_stage3/run_analysis.py to Deakin chronological, to several training seeds per model and
to the two Static VGAE variants. No model is re-run: learned scores come from the scores.npz files
written by train_seeds.py, whose checkpoint hashes are verified.

Outputs in --out:
  per_run_summary.csv     every model x seed x decoder (window means)
  aggregate_summary.csv   mean, SD, min and max over seeds for every metric
  headline.csv            "mean ± SD" at the validation threshold and the validation budget
  paired_intervals.csv    learned model minus EWMA_L4, two-level bootstrap (seeds and moving blocks),
                          bootstrap p-values, Holm adjustment over the primary family. With B replicates the
                          smallest p-value is 2/(B+1); the default B=10000 keeps 72 Holm-adjusted tests below 0.05.
  selection.json, audit.json
"""
import argparse, csv, json, platform, warnings
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from evaluate_pilot import load, sha, validate, nonself, topk, reach, metrics, select_threshold, write_csv, fit_transition, baseline_records
from evaluate_temporal import validation_items
from train_seeds import drop_self_loops

L = 4
ALPHAS = [.1, .3, .5, .8, 1.]
BUDGET_FACTORS = [.5, .75, 1., 1.25, 1.5, 2.]
DATASETS = {
    'unsw': dict(path=('dataset_ready', '300s'), counts=(284, 568)),
    'deakin': dict(path=('dataset_ready_deakin_chronological', '300s'), counts=(284, 284)),
    'deakin_fold2': dict(path=('dataset_ready_deakin_fold2', '300s'), counts=(284, 284)),
    'unsw_fold2': dict(path=('dataset_ready_unsw_fold2', '300s'), counts=(284, 284)),
}
# Stage 7 window-length ablation (make_merged_windows.py): validation and test cohorts per merged window length
for _base, _path, _days in [('unsw', 'dataset_ready', (1, 2)), ('deakin', 'dataset_ready_deakin_chronological', (1, 1)),
                            ('deakin_fold2', 'dataset_ready_deakin_fold2', (1, 1)), ('unsw_fold2', 'dataset_ready_unsw_fold2', (1, 1))]:
    for _sec, _per_day in [(600, 140), (900, 92)]:
        DATASETS[f'{_base}_{_sec}s'] = dict(path=(_path, f'{_sec}s'), counts=tuple(d * _per_day for d in _days))
# Stage 7C prediction-horizon ablation (make_horizon_dataset.py): validation and test cohorts, 285 - h targets per day
for _base, _path, _days in [('unsw', 'dataset_ready', (1, 2)), ('deakin', 'dataset_ready_deakin_chronological', (1, 1)),
                            ('deakin_fold2', 'dataset_ready_deakin_fold2', (1, 1)), ('unsw_fold2', 'dataset_ready_unsw_fold2', (1, 1))]:
    for _h in (2, 4):
        DATASETS[f'{_base}_h{_h}'] = dict(path=(_path, f'300s_h{_h}'), counts=tuple(d * (285 - _h) for d in _days), horizon=_h)
REFERENCE = 'EWMA_L4'
PAIRED_METRICS = ['edge_f1', 'missed_reachability', 'false_reachability', 'false_isolation', 'new_edge_recall', 'disappearance_recall']
HEADLINE_METRICS = ['AP', 'edge_f1', 'missed_reachability', 'false_isolation', 'false_reachability', 'new_edge_recall', 'disappearance_recall']
NOT_COMPARED = {'static_target', 'Degree_product_target'}  # target-information references are never ranked


def finite_ewma(items, alpha):
    records = []
    for i in items:
        history = i['A_seq']; s = history[0].astype(float).copy()
        for a in history[1:]: s = alpha * a + (1 - alpha) * s
        records.append(dict(date=i['date'], window=int(i['target_window_id']), truth=i['A_target'].copy(), previous=history[-1].copy(), score=s))
    return records


def tune_ewma(items, transition, finite):
    candidates = []
    for alpha in ALPHAS:
        r = finite_ewma(items, alpha) if finite else baseline_records(items, 'ewma', transition, alpha)
        th, f, e = select_threshold(r); candidates.append((f, -e, -alpha, alpha, r))
    best = max(candidates, key=lambda c: c[:3]); return best[3], best[4]


def record(items, scores):
    return [dict(date=i['date'], window=int(i['target_window_id']), truth=i['A_target'].copy(), previous=i['A_seq'][-1].copy(), score=s)
            for i, s in zip(items, scores)]


def degree_product(items):
    """Reference score built from the target window's own degrees (no learning)."""
    return record(items, [np.outer(i['A_target'].sum(1), i['A_target'].sum(0)).astype(float) for i in items])


def evaluate(vr, tr, is_logit, refs, budgets, persistence=False):
    threshold, vf, ve = select_threshold(vr)
    mask = nonself(len(tr[0]['score']))
    policies = [('validation_threshold', threshold)] + [('budget_' + str(k), k) for k in budgets]
    if persistence: policies = [('copy_previous', None)] + policies
    ap = np.array([average_precision_score(r['truth'][mask] > 0, expit(r['score'][mask]) if is_logit else r['score'][mask])
                   if r['truth'].sum() else np.nan for r in tr])
    tables = {}
    for policy, value in policies:
        rows = []
        for r, rr in zip(tr, refs):
            s = r['score']
            if policy.startswith('budget_'): p = topk(s, value)
            elif policy == 'copy_previous': p = r['previous'] > 0
            else: p = (s >= value) & mask
            rows.append(metrics(r['truth'] > 0, p, r['previous'], rr))
        table = {'AP': ap}; table.update({k: np.array([row[k] for row in rows], float) for k in rows[0]})
        tables[policy] = table
    choice = dict(threshold=threshold if np.isfinite(threshold) else 'inf', val_f1=vf, val_edge_error=ve)
    if is_logit: choice['sigmoid_threshold'] = float(expit(threshold))
    return tables, choice


def load_runs(runs_dir, dataset, keys_v, keys_t, wanted):
    found = {}
    for mdir in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        if wanted and mdir.name not in wanted: continue
        for job in sorted(mdir.glob('seed*')):
            if not (job / 'completed.json').exists(): continue
            info = json.loads((job / 'completed.json').read_text())
            if info.get('smoke'): raise ValueError(f'{job} is a smoke run; analyse full runs only')
            if info.get('dataset') != dataset: raise ValueError(f'{job} was trained on {info.get("dataset")}, not {dataset}')
            if sha(job / 'best.pt') != info['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch: ' + str(job))
            z = np.load(job / 'scores.npz', allow_pickle=False)
            kv = [(str(d), int(w)) for d, w in zip(z['validation_dates'], z['validation_windows'])]
            kt = [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])]
            if kv != keys_v or kt != keys_t: raise ValueError('Window keys differ from the evaluation cohort: ' + str(job))
            found.setdefault(mdir.name, {})[int(info['seed'])] = (z['validation'].astype(float), z['test'].astype(float), info)
    return found


def two_level_bootstrap(deltas, captures, block, replicates, rng):
    """deltas [S, W, M]. Each replicate resamples seeds with replacement and, within each capture,
    circular moving blocks of windows shared by all seeds."""
    S = deltas.shape[0]; out = []
    for start in range(0, replicates, 100):
        count = min(100, replicates - start); seg = []
        for ids in captures:
            b = max(1, min(block, len(ids) // 2))
            starts = rng.integers(0, len(ids), (count, int(np.ceil(len(ids) / b))))
            ix = ((starts[:, :, None] + np.arange(b)) % len(ids)).reshape(count, -1)[:, :len(ids)]
            seg.append(ids[ix])
        ix = np.concatenate(seg, axis=1)
        sd = rng.integers(0, S, (count, S))
        values = deltas[sd[:, :, None], ix[:, None, :]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out.append(np.nanmean(values, axis=(1, 2)))
    return np.concatenate(out)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); running = 0.
    for rank, j in enumerate(order):
        running = max(running, min(1., (m - rank) * p[j])); adj[j] = running
    return adj


def main(args):
    project = Path(args.project_root).resolve(); ds = project.joinpath(*DATASETS[args.dataset]['path']); out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    if args.plan and not Path(args.plan).exists(): raise FileNotFoundError(args.plan)
    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    self_loops = drop_self_loops(train, val, test); audit = validate(train, val, test, meta); audit['self_loops_removed'] = self_loops
    print('Self-loops removed (same-node pairs are never evaluated):', self_loops, flush=True)
    vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    if meta['seq_len'] != L or (len(vi), len(ti)) != DATASETS[args.dataset]['counts']:
        raise ValueError('Unexpected aligned validation/test cohorts ' + str((len(vi), len(ti))))
    keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    runs = load_runs(args.runs, args.dataset, keys_v, keys_t, set(args.models or []))
    if not runs: raise ValueError('No completed runs found in ' + str(args.runs))
    print('Runs found:', {m: sorted(s) for m, s in runs.items()}, flush=True)

    transition = fit_transition(train)
    budget = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    budgets = sorted(set(int(np.rint(budget * f)) for f in BUDGET_FACTORS))
    refs = [reach(i['A_target']) for i in ti]
    sources = {}  # name -> list of (seed, val_records, test_records, is_logit)
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        sources[name] = [(None, baseline_records(vi, kind, transition), baseline_records(ti, kind, transition), False)]
    alphas = {}
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(vi, transition, finite); alphas[name] = alpha
        sources[name] = [(None, vr, finite_ewma(ti, alpha) if finite else baseline_records(ti, 'ewma', transition, alpha), False)]
    sources['Degree_product_target'] = [(None, degree_product(vi), degree_product(ti), False)]
    for name, seeds in runs.items():
        sources[name] = [(seed, record(vi, list(v)), record(ti, list(t)), True) for seed, (v, t, _) in sorted(seeds.items())]

    results = {}; choices = {}; per_run = []
    for name, entries in sources.items():
        for seed, vr, tr, is_logit in entries:
            tables, choice = evaluate(vr, tr, is_logit, refs, budgets, persistence=(name == 'Persistence'))
            results[(name, seed)] = tables; choices.setdefault(name, {})['seed_' + str(seed) if seed is not None else 'deterministic'] = choice
            for policy, table in tables.items():
                row = dict(dataset=args.dataset, model=name, seed='' if seed is None else seed, decoder=policy, n=len(ti))
                for k, v in table.items():
                    good = v[np.isfinite(v)]; row[k + '_mean'] = float(good.mean()) if len(good) else None; row[k + '_valid_n'] = len(good)
                per_run.append(row)
        print('Evaluated', name, flush=True)
    for name, a in alphas.items(): choices[name]['alpha'] = a

    aggregate = []; headline = []
    metric_keys = [k[:-5] for k in per_run[0] if k.endswith('_mean')]
    for name in sources:
        for policy in sorted({r['decoder'] for r in per_run if r['model'] == name}):
            rows = [r for r in per_run if r['model'] == name and r['decoder'] == policy]
            row = dict(dataset=args.dataset, model=name, decoder=policy, n_seeds=len(rows))
            for k in metric_keys:
                v = np.array([r[k + '_mean'] for r in rows if r[k + '_mean'] is not None], float)
                row[k + '_mean'] = float(v.mean()) if len(v) else None
                row[k + '_sd'] = float(v.std(ddof=1)) if len(v) > 1 else None
                row[k + '_min'] = float(v.min()) if len(v) else None; row[k + '_max'] = float(v.max()) if len(v) else None
            aggregate.append(row)
            if policy in ('validation_threshold', 'budget_' + str(budget), 'copy_previous'):
                h = dict(dataset=args.dataset, model=name, decoder=policy, n_seeds=len(rows),
                         role='reference (target information)' if name in NOT_COMPARED else ('baseline' if rows[0]['seed'] == '' else 'learned'))
                for k in HEADLINE_METRICS:
                    m, s = row[k + '_mean'], row[k + '_sd']
                    h[k] = '' if m is None else (f'{m:.4f}' if s is None else f'{m:.4f} ± {s:.4f}')
                headline.append(h)

    intervals = []
    rng = np.random.default_rng(20261019)
    dates = sorted({d for d, _ in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
    compared = [n for n in runs if n not in NOT_COMPARED]
    for policy in ['validation_threshold'] + ['budget_' + str(k) for k in budgets]:
        base = results[(REFERENCE, None)][policy]
        for name in compared:
            seeds = sorted(runs[name])
            deltas = np.stack([np.stack([results[(name, s)][policy][m] - base[m] for m in PAIRED_METRICS], axis=1) for s in seeds])
            per_seed = np.nanmean(deltas, axis=1)
            for block in [4, 8, 16]:
                samples = two_level_bootstrap(deltas, captures, block, args.bootstrap_replicates, rng)
                for j, m in enumerate(PAIRED_METRICS):
                    d = float(np.nanmean(deltas[:, :, j])); lo, hi = np.nanquantile(samples[:, j], [.025, .975])
                    B = len(samples); p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
                    intervals.append(dict(dataset=args.dataset, model=name, reference=REFERENCE, decoder=policy, metric=m, block_length=block,
                        n_seeds=len(seeds), mean_difference=d, ci_low=float(lo), ci_high=float(hi),
                        seed_min_difference=float(per_seed[:, j].min()), seed_max_difference=float(per_seed[:, j].max()),
                        seeds_same_sign=int((np.sign(per_seed[:, j]) == np.sign(d)).sum()), p_value=p, p_holm=None,
                        family='primary' if (block == args.primary_block and policy in ('validation_threshold', 'budget_' + str(budget))) else 'exploratory',
                        valid_n=int(np.isfinite(deltas[0, :, j]).sum()), replicates=B,
                        scope='Two-level paired bootstrap: training seeds resampled with replacement and within-capture circular moving blocks; '
                              'per-seed thresholds and checkpoints fixed (selection uncertainty excluded).'))
    primary = [r for r in intervals if r['family'] == 'primary']
    if primary:
        for r, a in zip(primary, holm([r['p_value'] for r in primary])): r['p_holm'] = float(a)

    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return x
    audit.update(dataset=args.dataset, dataset_dir=str(ds.relative_to(project)), aligned_val_targets=len(vi), test_targets=len(ti),
                 validation_k=budget, budgets=budgets, alpha_grid=ALPHAS, bootstrap_seed=20261019,
                 bootstrap_replicates=args.bootstrap_replicates, primary_block=args.primary_block,
                 smallest_attainable_p=2 / (args.bootstrap_replicates + 1),
                 holm_family='All learned-vs-EWMA_L4 contrasts at the primary block length for validation_threshold and budget_' + str(budget),
                 runs={m: {str(s): dict(checkpoint_sha256=i['checkpoint_sha256'], best_epoch=i['best_epoch']) for s, (_, _, i) in v.items()} for m, v in runs.items()},
                 input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'temporal_test.pkl', 'metadata.json']},
                 source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                 numpy=np.__version__, python=platform.python_version(),
                 analysis_plan=(dict(file=Path(args.plan).name, sha256=sha(args.plan)) if args.plan else None),
                 limitations='Test captures were inspected in earlier analyses. EWMA_L4 restarts from the first snapshot of each four-window input. '
                             'Transition uses training-fitted parameters. static_target and Degree_product_target use target-window information and are references, never ranked.')
    out.mkdir(parents=True)
    write_csv(out / 'per_run_summary.csv', per_run); write_csv(out / 'aggregate_summary.csv', aggregate)
    with open(out / 'headline.csv', 'w', newline='', encoding='utf-8-sig') as f:  # BOM so Excel shows the ± sign
        w = csv.DictWriter(f, fieldnames=list(headline[0])); w.writeheader(); w.writerows(headline)
    if intervals: write_csv(out / 'paired_intervals.csv', intervals)
    (out / 'selection.json').write_text(json.dumps(clean(choices), indent=2)); (out / 'audit.json').write_text(json.dumps(clean(audit), indent=2))
    print('Saved', out.resolve(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--runs', required=True, help='The --out folder of train_seeds.py for this dataset')
    p.add_argument('--out', required=True); p.add_argument('--models', nargs='*', help='Default: every model found in --runs')
    p.add_argument('--bootstrap-replicates', type=int, default=10000); p.add_argument('--primary-block', type=int, default=8)
    p.add_argument('--plan', help='Pre-registered analysis plan; its SHA-256 is recorded in audit.json')
    main(p.parse_args())
