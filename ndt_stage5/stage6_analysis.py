"""Stage 6 analysis: edges per window, returning versus never-seen new edges, History-LR, and calibration.

No model is re-run. Learned scores come from the scores.npz files of train_seeds.py (checkpoint hashes are
verified) and History-LR scores from history_lr.py. Baselines and decoders are exactly those of
run_analysis_seeds.py: the exact validation threshold and the validation edge budget.

1. Counts per window (counts_summary.csv, metrics_summary.csv)
   edges right (true positives), false edges, new edges recovered, disappearing edges recovered,
   changes recovered (= new + disappearing recovered), continuing edges kept.
2. New edges by where they were last seen (new_edge_categories.csv). A new edge is present in the target
   window and absent from the previous window. Each new edge gets the first category that applies:
     in_history      present in one of the three earlier history windows (t-2 .. t-4)
     earlier_today   present earlier on the same capture day, before the history
     in_training     present in a training window
     in_validation   present in a validation window
     never_seen      none of the above
3. Paired intervals on counts (count_intervals.csv): model minus reference per window, two-level bootstrap
   (seeds and 8-window circular moving blocks within each capture), as run_analysis_seeds.py.
   Primary family (Holm), at the validation budget and block length 8, metrics changes_recovered and true_positive:
     tgae - EWMA_L4, tgae_bilinear - EWMA_L4, tgae - History_LR, tgae_bilinear - History_LR, History_LR - EWMA_L4
   (contrasts whose models are absent are dropped; the family actually used is recorded in audit.json).
   Everything else is exploratory.
4. Calibration (calibration.csv, calibration_summary.csv, reliability.csv), on test, pooled over windows and
   non-self pairs, for three versions of each model's probabilities:
     raw               sigmoid(logit) for learned models and History-LR; the score itself for baselines
     weight_corrected  sigmoid(logit - log w+), learned models only (w+ = training positive-class weight)
     platt             one-feature logistic recalibration fitted on validation, applied to test
   Metrics: Brier score, log loss, ECE (10 equal-width bins), expected edges per window (sum of
   probabilities) against observed edges, and edges and F1 when the calibrated probability is cut at 0.5.

Run from ndt_stage5, e.g.
  python stage6_analysis.py --project-root .. --dataset unsw --runs ..\\seeds_unsw ..\\seeds_unsw_decoder --history-lr ..\\history_lr_unsw --out ..\\stage6_unsw
"""
import argparse, json, platform
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from evaluate_pilot import load, sha, validate, nonself, topk, reach, metrics, select_threshold, write_csv, fit_transition, baseline_records
from evaluate_temporal import validation_items
from train_seeds import drop_self_loops
from run_analysis_seeds import DATASETS, L, finite_ewma, tune_ewma, record, load_runs, two_level_bootstrap, holm

CATEGORIES = ['in_history', 'earlier_today', 'in_training', 'in_validation', 'never_seen']
COUNT_METRICS = ['true_positive', 'false_positive', 'new_edges_recovered', 'disappearances_recovered', 'changes_recovered', 'continuing_kept']
PAIRED = ['changes_recovered', 'true_positive', 'new_edges_recovered', 'disappearances_recovered', 'false_positive']
REFERENCES = ['EWMA_L4', 'Transition', 'History_LR']
PRIMARY = [('tgae', 'EWMA_L4'), ('tgae_bilinear', 'EWMA_L4'), ('tgae', 'History_LR'), ('tgae_bilinear', 'History_LR'), ('History_LR', 'EWMA_L4')]
PRIMARY_METRICS = ['changes_recovered', 'true_positive']
NOT_COMPARED = {'static_target'}


def categories(ti, train, val):
    """Per test window: an int array with the category index of every new edge (-1 elsewhere)."""
    train_any = np.any([i['A'] > 0 for i in train['items']], axis=0)
    val_any = np.any([i['A'] > 0 for i in val['items']], axis=0)
    days = {}
    for i in ti:
        w = days.setdefault(i['date'], {})
        for wid, a in zip(i['input_window_ids'], i['A_seq']): w[int(wid)] = a > 0
        w[int(i['target_window_id'])] = i['A_target'] > 0
    out = []
    for i in ti:
        H = i['A_seq'] > 0; first = int(i['input_window_ids'][0]); w = days[i['date']]
        earlier = np.zeros_like(H[0])
        for wid, a in w.items():
            if wid < first: earlier |= a
        new = (i['A_target'] > 0) & ~H[-1]
        cat = np.full(new.shape, -1)
        rules = [H[:-1].any(0), earlier, train_any, val_any, np.ones_like(new)]
        left = new.copy()
        for k, rule in enumerate(rules):
            hit = left & rule; cat[hit] = k; left &= ~hit
        out.append(cat)
    return out


def predictions(vr, tr, budget):
    threshold, vf, ve = select_threshold(vr); mask = nonself(len(tr[0]['score']))
    return {'validation_threshold': [(r['score'] >= threshold) & mask for r in tr],
            'budget_' + str(budget): [topk(r['score'], budget) for r in tr]}, dict(threshold=threshold, val_f1=vf, val_edge_error=ve)


def window_table(tr, preds, refs, cats):
    rows = []
    for r, p, rr, c in zip(tr, preds, refs, cats):
        m = metrics(r['truth'] > 0, p, r['previous'], rr)
        m['changes_recovered'] = m['new_edges_recovered'] + m['disappearances_recovered']
        m['changes'] = m['new_edges'] + m['disappeared_edges']
        m['continuing_kept'] = m['true_positive'] - m['new_edges_recovered']
        for k, name in enumerate(CATEGORIES):
            m['new_' + name] = int((c == k).sum()); m['recovered_' + name] = int(((c == k) & p).sum())
        rows.append(m)
    return {k: np.array([row[k] for row in rows], float) for k in rows[0]}


def calibration_rows(name, seed, vr, tr, is_logit, weight):
    mask = nonself(len(tr[0]['score']))
    def probs(recs, kind, platt=None):
        out = []
        for r in recs:
            s = r['score'].astype(np.float64)
            if kind == 'raw': p = expit(s) if is_logit else np.clip(s, 0, 1)
            elif kind == 'weight_corrected': p = expit(s - np.log(weight))
            else: p = platt.predict_proba(s[mask].reshape(-1, 1))[:, 1]; q = np.zeros_like(s); q[mask] = p; p = q
            out.append(p)
        return out
    kinds = ['raw'] + (['weight_corrected'] if is_logit and weight else []) + ['platt']
    xv = np.concatenate([r['score'][mask] for r in vr]).reshape(-1, 1).astype(np.float64)
    yv = np.concatenate([r['truth'][mask] for r in vr]).astype(int)
    platt = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(xv, yv) if len(np.unique(yv)) == 2 else None
    rows = []; bins = []
    y = np.concatenate([r['truth'][mask] for r in tr]).astype(float)
    for kind in kinds:
        if kind == 'platt' and platt is None: continue
        P = probs(tr, kind, platt); p = np.clip(np.concatenate([q[mask] for q in P]), 1e-7, 1 - 1e-7)
        edges = np.array([q[mask].sum() for q in P]); cut = [(q >= .5) & mask for q in P]
        f1 = []
        for r, c in zip(tr, cut):
            a = r['truth'] > 0; tp = (a & c).sum(); den = 2 * tp + (~a & c).sum() + (a & ~c).sum()
            f1.append(2 * tp / den if den else 0.)
        f1 = np.mean(f1)
        idx = np.minimum((p * 10).astype(int), 9); ece = 0.
        for b in range(10):
            sel = idx == b
            if sel.any():
                ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
                bins.append(dict(model=name, seed='' if seed is None else seed, version=kind, bin=b, pairs=int(sel.sum()),
                                 mean_probability=float(p[sel].mean()), observed_rate=float(y[sel].mean())))
        rows.append(dict(model=name, seed='' if seed is None else seed, version=kind, brier=float(np.mean((p - y) ** 2)),
                         log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), ece_10=float(ece),
                         expected_edges_per_window=float(edges.mean()), observed_edges_per_window=float(y.sum() / len(tr)),
                         edges_at_0_5=float(np.mean([c.sum() for c in cut])), f1_at_0_5=float(f1),
                         platt_slope=float(platt.coef_[0, 0]) if (kind == 'platt' and platt is not None) else '',
                         platt_intercept=float(platt.intercept_[0]) if (kind == 'platt' and platt is not None) else ''))
    return rows, bins


def mean_sd(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return (float(v.mean()) if len(v) else None, float(v.std(ddof=1)) if len(v) > 1 else None)


def main(args):
    project = Path(args.project_root).resolve(); ds = project.joinpath(*DATASETS[args.dataset]['path']); out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    if args.plan and not Path(args.plan).exists(): raise FileNotFoundError(args.plan)
    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    self_loops = drop_self_loops(train, val, test); audit = validate(train, val, test, meta); audit['self_loops_removed'] = self_loops
    vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    if meta['seq_len'] != L or (len(vi), len(ti)) != DATASETS[args.dataset]['counts']:
        raise ValueError('Unexpected aligned validation/test cohorts ' + str((len(vi), len(ti))))
    keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    runs = {}
    for d in args.runs:
        for name, seeds in load_runs(d, args.dataset, keys_v, keys_t, None).items():
            if name in runs: raise ValueError(f'Model {name} appears in more than one --runs folder')
            runs[name] = seeds
    print('Runs found:', {m: sorted(s) for m, s in runs.items()}, flush=True)

    transition = fit_transition(train); budget = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    refs = [reach(i['A_target']) for i in ti]; cats = categories(ti, train, val)
    sources = {}
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        sources[name] = [(None, baseline_records(vi, kind, transition), baseline_records(ti, kind, transition), False, None)]
    alphas = {}
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(vi, transition, finite); alphas[name] = alpha
        sources[name] = [(None, vr, finite_ewma(ti, alpha) if finite else baseline_records(ti, 'ewma', transition, alpha), False, None)]
    lr_fit = None
    if args.history_lr:
        z = np.load(Path(args.history_lr) / 'scores.npz', allow_pickle=False)
        if [(str(d), int(w)) for d, w in zip(z['validation_dates'], z['validation_windows'])] != keys_v or \
           [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != keys_t:
            raise ValueError('History-LR window keys differ from the evaluation cohort')
        lr_fit = json.loads((Path(args.history_lr) / 'fit.json').read_text())
        if lr_fit['dataset'] != args.dataset: raise ValueError('History-LR was fitted on ' + lr_fit['dataset'])
        sources['History_LR'] = [(None, record(vi, list(z['validation'].astype(float))), record(ti, list(z['test'].astype(float))), True, None)]
    for name, seeds in runs.items():
        entries = []
        for seed, (v, t, info) in sorted(seeds.items()):
            sel = run_dir(args.runs, name, seed) / 'selection.json'
            weight = json.loads(sel.read_text()).get('positive_class_weight') if sel.exists() else None
            entries.append((seed, record(vi, list(v)), record(ti, list(t)), True, weight))
        sources[name] = entries

    tables = {}; choices = {}; cal_rows = []; rel_rows = []; ap = {}
    mask = nonself(len(ti[0]['A_target']))
    for name, entries in sources.items():
        for seed, vr, tr, is_logit, weight in entries:
            preds, choice = predictions(vr, tr, budget)
            choices.setdefault(name, {})['deterministic' if seed is None else f'seed_{seed}'] = choice
            for policy, p in preds.items(): tables[(name, seed, policy)] = window_table(tr, p, refs, cats)
            ap[(name, seed)] = float(np.nanmean([average_precision_score(r['truth'][mask] > 0, r['score'][mask]) if r['truth'].sum() else np.nan for r in tr]))
            c, b = calibration_rows(name, seed, vr, tr, is_logit, weight); cal_rows += c; rel_rows += b
        print('Evaluated', name, flush=True)

    policies = ['validation_threshold', 'budget_' + str(budget)]
    counts, mets, cat_rows = [], [], []
    for name, entries in sources.items():
        seeds = [e[0] for e in entries]
        for policy in policies:
            per_seed = [tables[(name, s, policy)] for s in seeds]
            row = dict(dataset=args.dataset, model=name, decoder=policy, n_seeds=len(seeds))
            for k in ['true_edges', 'new_edges', 'disappeared_edges', 'changes', 'predicted_edges'] + COUNT_METRICS:
                m, s = mean_sd([t[k].mean() for t in per_seed]); row[k] = m; row[k + '_sd'] = s
            counts.append(row)
            mrow = dict(dataset=args.dataset, model=name, decoder=policy, n_seeds=len(seeds))
            m, s = mean_sd([ap[(name, sd)] for sd in seeds]); mrow['AP'] = m; mrow['AP_sd'] = s
            for k in ['edge_f1', 'precision', 'recall', 'missed_reachability', 'false_reachability', 'false_isolation',
                      'new_edge_recall', 'disappearance_recall', 'persistent_edge_recall']:
                m, s = mean_sd([np.nanmean(t[k]) for t in per_seed]); mrow[k] = m; mrow[k + '_sd'] = s
            mets.append(mrow)
            total_new = per_seed[0]['new_edges'].sum()
            for c in CATEGORIES:
                n_c = per_seed[0]['new_' + c].sum()
                rec = [t['recovered_' + c].sum() / n_c if n_c else np.nan for t in per_seed]
                m, s = mean_sd(rec); mw, sw = mean_sd([t['recovered_' + c].mean() for t in per_seed])
                cat_rows.append(dict(dataset=args.dataset, model=name, decoder=policy, category=c, new_edges_total=int(n_c),
                                     share_of_new_edges=float(n_c / total_new) if total_new else None,
                                     new_per_window=float(per_seed[0]['new_' + c].mean()), recall_pooled=m, recall_pooled_sd=s,
                                     recovered_per_window=mw, recovered_per_window_sd=sw))

    rng = np.random.default_rng(20261113); intervals = []
    dates = sorted({d for d, _ in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
    # compared: every learned model (target-information references excluded) and History-LR; references are never compared with each other
    # except History-LR against EWMA_L4 and Transition
    compared = [n for n in runs if n not in NOT_COMPARED] + (['History_LR'] if 'History_LR' in sources else [])
    for policy in policies:
        for ref in [r for r in REFERENCES if r in sources]:
            base = tables[(ref, None, policy)]
            for name in compared:
                if name == ref: continue
                seeds = [e[0] for e in sources[name]]
                deltas = np.stack([np.stack([tables[(name, s, policy)][m] - base[m] for m in PAIRED], 1) for s in seeds])
                per_seed = np.nanmean(deltas, axis=1)
                for block in [4, 8, 16]:
                    samples = two_level_bootstrap(deltas, captures, block, args.bootstrap_replicates, rng); B = len(samples)
                    for j, m in enumerate(PAIRED):
                        d = float(np.nanmean(deltas[:, :, j])); lo, hi = np.nanquantile(samples[:, j], [.025, .975])
                        p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
                        primary = (block == args.primary_block and policy == 'budget_' + str(budget) and (name, ref) in PRIMARY and m in PRIMARY_METRICS)
                        intervals.append(dict(dataset=args.dataset, model=name, reference=ref, decoder=policy, metric=m, block_length=block,
                                              n_seeds=len(seeds), mean_difference_per_window=d, ci_low=float(lo), ci_high=float(hi),
                                              seed_min_difference=float(per_seed[:, j].min()), seed_max_difference=float(per_seed[:, j].max()),
                                              seeds_same_sign=int((np.sign(per_seed[:, j]) == np.sign(d)).sum()), p_value=p, p_holm=None,
                                              family='primary' if primary else 'exploratory', replicates=B))
        print('Bootstrapped', policy, flush=True)
    prim = [r for r in intervals if r['family'] == 'primary']
    for r, a in zip(prim, holm([r['p_value'] for r in prim])): r['p_holm'] = float(a)

    cal_summary = []
    for name in sources:
        for kind in ['raw', 'weight_corrected', 'platt']:
            rows = [r for r in cal_rows if r['model'] == name and r['version'] == kind]
            if not rows: continue
            row = dict(dataset=args.dataset, model=name, version=kind, n_seeds=len(rows))
            for k in ['brier', 'log_loss', 'ece_10', 'expected_edges_per_window', 'observed_edges_per_window', 'edges_at_0_5', 'f1_at_0_5']:
                m, s = mean_sd([r[k] for r in rows]); row[k] = m; row[k + '_sd'] = s
            cal_summary.append(row)

    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return str(x)
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): return float(x)
        return x
    audit.update(dataset=args.dataset, validation_k=budget, ewma_alpha=alphas, categories=CATEGORIES,
                 category_rule='first matching: t-2..t-4 history, earlier the same capture day, training split, validation split, never',
                 primary_family=[dict(model=r['model'], reference=r['reference'], metric=r['metric']) for r in prim],
                 bootstrap_seed=20261113, bootstrap_replicates=args.bootstrap_replicates, primary_block=args.primary_block,
                 smallest_attainable_p=2 / (args.bootstrap_replicates + 1),
                 runs={m: {str(s): dict(checkpoint_sha256=i['checkpoint_sha256']) for s, (_, _, i) in v.items()} for m, v in runs.items()},
                 history_lr=(dict(chosen_C=lr_fit['chosen_C'], script_sha256=lr_fit['script_sha256'],
                                  scores_sha256=sha(Path(args.history_lr) / 'scores.npz')) if lr_fit else None),
                 input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'temporal_test.pkl', 'metadata.json']},
                 source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                 analysis_plan=(dict(file=Path(args.plan).name, sha256=sha(args.plan)) if args.plan else None),
                 numpy=np.__version__, python=platform.python_version())
    out.mkdir(parents=True)
    write_csv(out / 'counts_summary.csv', counts); write_csv(out / 'metrics_summary.csv', mets)
    write_csv(out / 'new_edge_categories.csv', cat_rows); write_csv(out / 'count_intervals.csv', intervals)
    write_csv(out / 'calibration.csv', cal_rows); write_csv(out / 'calibration_summary.csv', cal_summary)
    write_csv(out / 'reliability.csv', rel_rows)
    (out / 'selection.json').write_text(json.dumps(clean(choices), indent=2)); (out / 'audit.json').write_text(json.dumps(clean(audit), indent=2))
    print('Saved', out.resolve(), flush=True)


def run_dir(runs_dirs, model, seed):
    for d in runs_dirs:
        p = Path(d) / model / f'seed{seed}'
        if p.exists(): return p
    raise FileNotFoundError(f'{model} seed{seed}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--runs', nargs='+', required=True, help='One or more --out folders of train_seeds.py for this dataset')
    p.add_argument('--history-lr', help='The --out folder of history_lr.py for this dataset')
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap-replicates', type=int, default=10000); p.add_argument('--primary-block', type=int, default=8)
    p.add_argument('--plan', help='Pre-registered analysis plan; its SHA-256 is recorded in audit.json')
    main(p.parse_args())
