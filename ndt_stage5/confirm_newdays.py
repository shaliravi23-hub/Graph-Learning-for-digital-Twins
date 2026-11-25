"""Stage 8: confirmation on unseen UNSW capture days (preregistration_stage8_newdays.md).

Inputs
  --dataset-dir   new-day dataset written by make_unsw_newdays.py (training and validation are the main split's)
  --scores        checkpoint scores written by score_newdays.py (validation logits as stored, new-day test logits)
  --main-runs     seeds_unsw: the main-split test logits, for the ranking question Q1
  --history-lr    history_lr_unsw: stored History-LR fit, for guard G2
  --stage7        stage7 folder: default_unsw.json and periodicity_check.json, for guard G2
Steps
  1. checks the plan hash recorded by the builder, the input hashes and the cohort keys;
  2. fits the nine simple methods on the training days and tunes them on the validation day as in stage 7;
     guard G2: History-LR refit reproduces the stored validation scores (<= 1e-5) and C, EWMA the same smoothing
     factors, and every stage-7 validation F1 (simple methods; learned models, mean over seeds) is reproduced (1e-9);
  3. chooses S* and M* on the validation day, counts edges right and changes recovered per new-day window at the
     validation edge budget, and runs the two-level paired bootstrap (seeds x circular 8-window blocks within each day);
  4. evaluates predictions N1-N7, questions Q1-Q2 and secondary analyses S1-S3.
Outputs in --out (must not exist): confirmation.json, methods.csv, contrasts.csv, predictions.csv, per_day.csv,
categories.csv, lag_profile.csv.
--replay-main runs the same analysis with the main test days (9-10 October) in place of the new days, from the stored
main-split scores; this gives the main-day values the predictions are based on and must reproduce stage 7 exactly.
Run from ndt_stage5:
  python confirm_newdays.py --project-root .. --plan preregistration_stage8_newdays.md --scores ..\\scores_unsw_newdays ^
      --main-runs ..\\seeds_unsw --history-lr ..\\history_lr_unsw --stage7 ..\\stage7 --out ..\\stage8
  python confirm_newdays.py --replay-main --project-root .. --plan preregistration_stage8_newdays.md ^
      --scores ..\\seeds_unsw ..\\seeds_unsw_decoder --main-runs ..\\seeds_unsw --history-lr ..\\history_lr_unsw ^
      --stage7 ..\\stage7 --out ..\\stage8_main_replay
"""
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from evaluate_pilot import load, sha, nonself, reach, select_threshold, fit_transition, baseline_records
from evaluate_temporal import validation_items
from train_seeds import drop_self_loops
from run_analysis_seeds import L, finite_ewma, tune_ewma, record, two_level_bootstrap, holm
from stage6_analysis import CATEGORIES, categories, predictions, window_table
from summarize_stage7 import classify, METRICS
from periodicity_check import lag_profile, fit_transition2, score2
from history_lr import TrainStats, features, score_items, C_GRID

SEEDS = [42, 43, 44, 45, 46]
SIX = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev']
LEARNED = SIX + ['tgae_bilinear']                       # M* candidates
SCORED = LEARNED + ['tgae_bias']                         # tgae_bias is used for N2 only
SIMPLE = ['Persistence', 'Recent_frequency_L4', 'Transition', 'EWMA_L4', 'EWMA_expanding', 'History_LR',
          'Seasonal_copy', 'Order2_Transition', 'EdgeBank']
FLOOR_MODELS = ['tgae', 'vgrnn', 'static_prev']
LATE_DAYS = ['16-10-11', '16-10-12']
BLOCK = 8; RNG_SEED = 20261118
G2_SCORE_TOLERANCE = 1e-5; G2_F1_TOLERANCE = 1e-9


def edgebank(items, train, static):
    """1 if the ordered pair was an edge in any training window or in any earlier window of the same day."""
    n = len(train['node_list']); mask = nonself(n)
    bank = np.any([i['A'] > 0 for i in train['items']], axis=0)
    days = {}
    for i in static['items']: days.setdefault(i['date'], {})[int(i['window_id'])] = i['A'] > 0
    out = []
    for i in items:
        earlier = np.zeros((n, n), bool)
        for w, a in days[i['date']].items():
            if w < int(i['target_window_id']): earlier |= a
        out.append(((bank | earlier) & mask).astype(float))
    return out


def history_lr(train, val, vi, ti):
    tr = validation_items(train, L); stats = TrainStats(train)
    X = np.concatenate([features(i, stats, True) for i in tr]); y = np.concatenate([i['A_target'][stats.mask] for i in tr]).astype(int)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd
    grid = []
    for C in C_GRID:
        m = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-6).fit(Xs, y)
        th, vf, ve = select_threshold(record(vi, score_items(vi, stats, m, mu, sd))); grid.append((vf, -ve, -C, C, m))
    vf, _, _, C, m = max(grid, key=lambda g: g[:3])
    return C, score_items(vi, stats, m, mu, sd), score_items(ti, stats, m, mu, sd)


def contrast(tab_m, tab_c, captures, replicates):
    deltas = np.stack([np.stack([t[k] - tab_c[k] for k in METRICS], 1) for t in tab_m])
    samples = two_level_bootstrap(deltas, captures, BLOCK, replicates, np.random.default_rng(RNG_SEED)); B = len(samples)
    res = {}
    for j, k in enumerate(METRICS):
        lo, hi = np.nanquantile(samples[:, j], [.025, .975])
        p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
        res[k] = dict(difference=float(np.nanmean(deltas[:, :, j])), ci=[float(lo), float(hi)], p_value=float(p),
                      seed_differences=[float(x) for x in np.nanmean(deltas[:, :, j], 1)])
    return res


def adjust(contrasts):
    """Holm over every (contrast, measure) in the dictionary; adds p_holm and the classification."""
    flat = [(c, k) for c in contrasts.values() for k in METRICS]
    for (c, k), a in zip(flat, holm([c[k]['p_value'] for c, k in flat])): c[k]['p_holm'] = float(a)
    for c in contrasts.values(): c['classification'] = classify(c)
    return contrasts


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main(args):
    t0 = time.time()
    root = Path(args.project_root).resolve(); main_ds = root / 'dataset_ready' / '300s'; out = Path(args.out)
    ds = main_ds if args.replay_main else root / args.dataset_dir
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    plan_sha = sha(args.plan)
    if not args.replay_main and meta['analysis_plan']['sha256'] != plan_sha:
        raise ValueError('The plan differs from the one recorded when the dataset was built')
    for k in ['static_train.pkl', 'static_val.pkl']:
        if sha(ds / k) != sha(main_ds / k): raise ValueError(f'{k} differs from the main split')
    hashes = {k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'static_test.pkl', 'temporal_test.pkl', 'metadata.json']}

    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    static_test = load(ds / 'static_test.pkl')
    drop_self_loops(train, val, test); drop_self_loops(dict(items=[]), static_test, dict(items=[]))
    n = len(train['node_list']); mask = nonself(n)
    vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    dates = sorted({i['date'] for i in ti})
    if dates != sorted(meta['test_dates']): raise ValueError('Test dates differ from the builder metadata')
    keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
    budget = int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    ref7 = json.loads((Path(args.stage7) / 'default_unsw.json').read_text())
    per7 = json.loads((Path(args.stage7) / 'periodicity_check.json').read_text())['results']['unsw']
    if budget != ref7['budget']: raise ValueError(f'Budget {budget} differs from stage 7 ({ref7["budget"]})')
    print(f'{len(dates)} new days, {len(ti)} test targets; validation {len(vi)} targets; budget k = {budget}', flush=True)

    # ---- learned models (score_newdays.py output)
    learned = {}
    for name in SCORED:
        for seed in args.seeds:
            found = [Path(d) / name for d in args.scores if (Path(d) / name).is_dir()]
            if len(found) != 1: raise ValueError(f'{name}: expected one folder among --scores, found {len(found)}')
            job = found[0] / f'seed{seed}'
            info = json.loads((job / 'completed.json').read_text())
            if args.replay_main:
                if info['dataset'] != 'unsw' or info.get('smoke') or sha(job / 'best.pt') != info['checkpoint_sha256']:
                    raise ValueError(f'{job} is not a full main-split run with its original checkpoint')
            elif info['dataset'] != 'unsw_newdays' or info['dataset_hashes']['temporal_test.pkl'] != hashes['temporal_test.pkl']:
                raise ValueError(f'{job} was not scored on this new-day dataset')
            z = np.load(job / 'scores.npz', allow_pickle=False)
            kv = [(str(d), int(w)) for d, w in zip(z['validation_dates'], z['validation_windows'])]
            kt = [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])]
            if kv != keys_v or kt != keys_t: raise ValueError('Window keys differ from the cohort: ' + str(job))
            learned.setdefault(name, {})[seed] = (z['validation'].astype(float), z['test'].astype(float))

    # ---- simple methods, fitted on the training days and tuned on the validation day
    t1 = fit_transition(train); t2, _ = fit_transition2(train, mask)
    simple = {}
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        simple[name] = (baseline_records(vi, kind, t1), baseline_records(ti, kind, t1))
    alphas = {}
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(vi, t1, finite); alphas[name] = alpha
        simple[name] = (vr, finite_ewma(ti, alpha) if finite else baseline_records(ti, 'ewma', t1, alpha))
    C, v_lr, t_lr = history_lr(train, val, vi, ti); simple['History_LR'] = (record(vi, v_lr), record(ti, t_lr))
    simple['Seasonal_copy'] = (record(vi, [i['A_seq'][-2].astype(float) for i in vi]), record(ti, [i['A_seq'][-2].astype(float) for i in ti]))
    simple['Order2_Transition'] = (record(vi, score2(vi, t2)), record(ti, score2(ti, t2)))
    simple['EdgeBank'] = (record(vi, edgebank(vi, train, val)), record(ti, edgebank(ti, train, static_test)))
    print('Simple methods fitted', flush=True)

    # ---- guard G2
    stored_lr = np.load(Path(args.history_lr) / 'scores.npz'); fit_lr = json.loads((Path(args.history_lr) / 'fit.json').read_text())
    lr_diff = float(np.abs(np.stack(v_lr) - stored_lr['validation']).max())
    g2 = dict(history_lr_max_abs_diff=lr_diff, history_lr_C=C, stored_C=fit_lr['chosen_C'], ewma_alpha=alphas, stored_ewma_alpha=ref7['ewma_alpha'])
    if lr_diff > G2_SCORE_TOLERANCE or C != fit_lr['chosen_C']: raise ValueError(f'G2 failed: History-LR differs ({g2})')
    if alphas != ref7['ewma_alpha']: raise ValueError(f'G2 failed: EWMA smoothing factors differ ({g2})')
    val_f1 = {name: select_threshold(vr)[1] for name, (vr, _) in simple.items()}
    for name, seeds in learned.items():
        val_f1[name] = float(np.mean([select_threshold(record(vi, list(v)))[1] for v, _ in seeds.values()]))
    reference = {k: v['validation_F1'] for k, v in ref7['candidates'].items()}
    reference.update(Seasonal_copy=per7['methods']['Seasonal_naive']['validation_F1'], Order2_Transition=per7['methods']['Order2_Transition']['validation_F1'])
    if sorted(args.seeds) != SEEDS: reference = {k: v for k, v in reference.items() if k not in learned}  # stage-7 means use all five seeds
    bad = {k: (val_f1[k], v) for k, v in reference.items() if abs(val_f1[k] - v) > G2_F1_TOLERANCE}
    if bad: raise ValueError(f'G2 failed: validation F1 differs from stage 7: {bad}')
    g2['validation_F1_reproduced'] = sorted(reference)
    print('G2 passed', g2, flush=True)

    # ---- S*, M*, window tables at the validation edge budget
    s_star = max(SIMPLE, key=lambda k: val_f1[k]); m_star = max(LEARNED, key=lambda k: val_f1[k])
    policy = f'budget_{budget}'; refs = [reach(i['A_target']) for i in ti]; cats = categories(ti, train, val)
    tables, f1_vt, choices = {}, {}, {}
    for name, (vr, tr) in simple.items():
        p, ch = predictions(vr, tr, budget); tables[name] = window_table(tr, p[policy], refs, cats); choices[name] = ch
        f1_vt[name] = float(np.mean(window_table(tr, p['validation_threshold'], refs, cats)['edge_f1']))
    for name, seeds in learned.items():
        tables[name], f1_vt[name], choices[name] = {}, {}, {}
        for seed, (v, t) in seeds.items():
            vr, tr = record(vi, list(v)), record(ti, list(t)); p, ch = predictions(vr, tr, budget)
            tables[name][seed] = window_table(tr, p[policy], refs, cats); choices[name][seed] = ch
            f1_vt[name][seed] = float(np.mean(window_table(tr, p['validation_threshold'], refs, cats)['edge_f1']))
        print('Evaluated', name, flush=True)
    seedlist = sorted(args.seeds)
    m_tabs = [tables[m_star][s] for s in seedlist]
    observed = dict(edges_per_window=float(np.mean([i['A_target'][mask].sum() for i in ti])),
                    changes_per_window=float(np.mean(m_tabs[0]['changes'])))

    def mean_count(name, k, idx=slice(None)):
        if name in learned: return float(np.mean([tables[name][s][k][idx].mean() for s in seedlist]))
        return float(tables[name][k][idx].mean())

    methods = []
    for name in SIMPLE + SCORED:
        row = dict(method=name, kind='simple' if name in simple else 'learned', validation_F1=val_f1[name],
                   edges_right=mean_count(name, 'true_positive'), changes_recovered=mean_count(name, 'changes_recovered'),
                   F1_at_budget=mean_count(name, 'edge_f1'),
                   F1_at_validation_threshold=float(np.mean(list(f1_vt[name].values()))) if name in learned else f1_vt[name])
        if name in learned: row['seeds'] = len(seedlist)
        methods.append(row)

    # ---- primary comparisons
    comparators = list(dict.fromkeys([s_star, 'EWMA_L4', 'Seasonal_copy']))
    primary = adjust({c: contrast(m_tabs, tables[c], captures, args.bootstrap_replicates) for c in comparators})
    for c, r in primary.items():
        print(f'{m_star} - {c}: edges right {r["true_positive"]["difference"]:+.2f} {np.round(r["true_positive"]["ci"], 2)}, '
              f'changes {r["changes_recovered"]["difference"]:+.2f} {np.round(r["changes_recovered"]["ci"], 2)} -> {r["classification"]}', flush=True)

    # ---- predictions
    def at_half(name, seed): return float(np.mean([((t >= 0) & mask).sum() for t in learned[name][seed][1]]))
    floor = {m: {s: at_half(m, s) for s in seedlist} for m in FLOOR_MODELS + ['tgae_bias']}
    profile = lag_profile(static_test, mask)
    two_way = sum(((a['A'] > 0) & (a['A'].T > 0) & mask).sum() for a in static_test['items']) / max(1, sum(((a['A'] > 0) & mask).sum() for a in static_test['items']))
    c = profile
    pred = {
        'N1': dict(holds=all(v >= 10 * observed['edges_per_window'] for m in FLOOR_MODELS for v in floor[m].values()),
                   predicted_edges_at_0_5=floor, threshold=10 * observed['edges_per_window']),
        'N2': dict(holds=all(floor['tgae_bias'][s] <= .2 * floor['tgae'][s] for s in seedlist),
                   reduction={s: 1 - floor['tgae_bias'][s] / floor['tgae'][s] for s in seedlist}),
        'N3': dict(holds=primary[s_star]['classification'] == 'learned clearly better', classification=primary[s_star]['classification']),
        'N4': dict(holds=primary['EWMA_L4']['classification'] == 'learned clearly better', classification=primary['EWMA_L4']['classification']),
        'N5': dict(holds=bool(c[2] < c[1] and c[2] < c[3] and c[4] < c[3] and c[4] < c[5]), lag_profile=profile),
        'N6': dict(holds=primary['Seasonal_copy']['changes_recovered']['ci'][1] < .5, upper_bound=primary['Seasonal_copy']['changes_recovered']['ci'][1]),
        'N7': dict(holds=bool(two_way >= .8), two_way_share=float(two_way)),
    }
    for k, v in pred.items(): print(k, 'HELD' if v['holds'] else 'NOT HELD', flush=True)

    # ---- questions
    main_test = load(main_ds / 'temporal_test.pkl'); drop_self_loops(dict(items=[]), dict(items=[]), main_test)
    mti = sorted(main_test['items'], key=lambda i: (i['date'], i['target_window_id'])); mrefs = [reach(i['A_target']) for i in mti]
    mkeys = [(i['date'], int(i['target_window_id'])) for i in mti]; mcats = categories(mti, train, val)
    main_f1 = {}
    for name in SIX:
        vals = []
        for seed in seedlist:
            z = np.load(Path(args.main_runs) / name / f'seed{seed}' / 'scores.npz')
            if [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != mkeys: raise ValueError('Main test keys differ')
            p, _ = predictions(record(vi, list(z['validation'].astype(float))), record(mti, list(z['test'].astype(float))), budget)
            vals.append(float(np.mean(window_table(record(mti, list(z['test'].astype(float))), p['validation_threshold'], mrefs, mcats)['edge_f1'])))
        main_f1[name] = float(np.mean(vals))
    new_f1 = {m: float(np.mean(list(f1_vt[m].values()))) for m in SIX}
    tau_all = kendalltau([main_f1[m] for m in SIX], [new_f1[m] for m in SIX])[0]
    tau_day = {}
    for d, idx in zip(dates, captures):
        day_f1 = {}
        for m in SIX:
            vals = []
            for seed in seedlist:
                v, t = learned[m][seed]; p, _ = predictions(record(vi, list(v)), record([ti[j] for j in idx], [t[j] for j in idx]), budget)
                vals.append(float(np.mean(window_table(record([ti[j] for j in idx], [t[j] for j in idx]), p['validation_threshold'],
                                                       [refs[j] for j in idx], [cats[j] for j in idx])['edge_f1'])))
            day_f1[m] = float(np.mean(vals))
        tau_day[d] = float(kendalltau([main_f1[m] for m in SIX], [day_f1[m] for m in SIX])[0])
    simple_rank = sorted(SIMPLE, key=lambda k: -val_f1[k])
    questions = {'Q1': dict(kendall_tau_pooled=float(tau_all), kendall_tau_per_day=tau_day, F1_main_test=main_f1, F1_new_days=new_f1),
                 'Q2': dict(edges_right=mean_count('EdgeBank', 'true_positive'), changes_recovered=mean_count('EdgeBank', 'changes_recovered'),
                            validation_F1=val_f1['EdgeBank'], validation_rank_among_simple=simple_rank.index('EdgeBank') + 1, of=len(SIMPLE))}

    # ---- secondary analyses
    late = [d for d in LATE_DAYS if d in dates]; secondary = {}
    if late:
        idx = np.concatenate([captures[dates.index(d)] for d in late]); pos = {j: k for k, j in enumerate(idx)}
        sub = lambda tab: {k: v[idx] for k, v in tab.items()}
        caps = [np.array([pos[j] for j in captures[dates.index(d)]]) for d in late]
        secondary['S1'] = dict(days=late, contrast=adjust({s_star: contrast([sub(t) for t in m_tabs], sub(tables[s_star]), caps, args.bootstrap_replicates)})[s_star])
    per_day, day_contrasts = [], {}
    for d, idx in zip(dates, captures):
        row = dict(date=d, windows=len(idx), observed_edges=float(np.mean([ti[j]['A_target'][mask].sum() for j in idx])),
                   observed_changes=float(m_tabs[0]['changes'][idx].mean()))
        for name in dict.fromkeys([m_star, s_star, 'EWMA_L4', 'Seasonal_copy', 'EdgeBank', 'Persistence']):
            row[f'{name}_edges_right'] = mean_count(name, 'true_positive', idx); row[f'{name}_changes_recovered'] = mean_count(name, 'changes_recovered', idx)
        per_day.append(row)
        day_contrasts[d] = contrast([{k: v[idx] for k, v in t.items()} for t in m_tabs], {k: v[idx] for k, v in tables[s_star].items()},
                                    [np.arange(len(idx))], args.bootstrap_replicates)
    adjust(day_contrasts)
    for row in per_day:
        r = day_contrasts[row['date']]; row['M_minus_S_classification'] = r['classification']
        for k in METRICS: row[f'M_minus_S_{k}'] = r[k]['difference']; row[f'M_minus_S_{k}_p_holm'] = r[k]['p_holm']
    secondary['S2'] = dict(per_day=per_day, M_minus_S_by_day=day_contrasts)
    cat_rows = []
    for name in dict.fromkeys([m_star, s_star, 'EWMA_L4', 'Seasonal_copy', 'EdgeBank', 'Persistence']):
        row = dict(method=name)
        for k in CATEGORIES:
            row[f'new_{k}'] = mean_count(name, 'new_' + k); row[f'recovered_{k}'] = mean_count(name, 'recovered_' + k)
        cat_rows.append(row)
    secondary['S3'] = cat_rows

    # ---- outputs
    out.mkdir(parents=True)
    result = dict(mode='main_replay' if args.replay_main else 'new_days', plan=dict(file=Path(args.plan).name, sha256=plan_sha), new_days=dates, excluded_days=meta.get('excluded_days', {}),
                  coverage=meta.get('coverage'), budget=budget, seeds=seedlist, best_simple=s_star, best_learned=m_star,
                  validation_F1=val_f1, observed=observed, guard_G2=g2, contrasts=primary, predictions=pred, questions=questions,
                  secondary=secondary, methods=methods, replicates=args.bootstrap_replicates, block=BLOCK, bootstrap_seed=RNG_SEED,
                  input_hashes=hashes, scores=[str(Path(d).resolve()) for d in args.scores],
                  source_hashes={p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
                  python=platform.python_version(), numpy=np.__version__, seconds=round(time.time() - t0, 1))
    (out / 'confirmation.json').write_text(json.dumps(result, indent=2, default=float))
    write_csv(out / 'methods.csv', methods)
    write_csv(out / 'contrasts.csv', [dict(analysis='primary', best_learned=m_star, comparator=c, measure=k, difference=r[k]['difference'],
                                          ci_low=r[k]['ci'][0], ci_high=r[k]['ci'][1], p_value=r[k]['p_value'], p_holm=r[k]['p_holm'],
                                          classification=r['classification']) for c, r in primary.items() for k in METRICS]
              + ([dict(analysis='S1 ' + '+'.join(late), best_learned=m_star, comparator=s_star, measure=k, difference=secondary['S1']['contrast'][k]['difference'],
                       ci_low=secondary['S1']['contrast'][k]['ci'][0], ci_high=secondary['S1']['contrast'][k]['ci'][1],
                       p_value=secondary['S1']['contrast'][k]['p_value'], p_holm=secondary['S1']['contrast'][k]['p_holm'],
                       classification=secondary['S1']['contrast']['classification']) for k in METRICS] if late else []))
    write_csv(out / 'predictions.csv', [dict(id=k, holds=v['holds'], **{a: json.dumps(b, default=float) for a, b in v.items() if a != 'holds'})
                                        for k, v in pred.items()])
    write_csv(out / 'per_day.csv', per_day); write_csv(out / 'categories.csv', cat_rows)
    write_csv(out / 'lag_profile.csv', [dict(lag=k, mean_differing_edges=v) for k, v in profile.items()])
    print(f'Saved {out.resolve()} ({result["seconds"]} s)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--plan', required=True)
    p.add_argument('--dataset-dir', default=str(Path('dataset_ready_unsw_newdays') / '300s'))
    p.add_argument('--scores', nargs='+', required=True); p.add_argument('--main-runs', required=True)
    p.add_argument('--replay-main', action='store_true', help='Main test days in place of the new days (basis of the predictions)')
    p.add_argument('--history-lr', required=True); p.add_argument('--stage7', required=True)
    p.add_argument('--out', required=True); p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--bootstrap-replicates', type=int, default=10000)
    main(p.parse_args())
