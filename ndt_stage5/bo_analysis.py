"""Stage 9 analysis (preregistration_stage9_optimisation.md): A1, A3, A4 and predictions O1-O5.

Inputs
  --bo            stage9_bo folder written by bo_search.py (<dataset>/final/<model>/seed*)
  --newday-scores optimised UNSW runs scored on the 12 further UNSW days by score_newdays.py
  --stage7        stage7 folder (default_<dataset>.json: best simple method and published validation F1)
Steps
  1. refits the six stage-7 simple methods on each dataset and checks that the best one by validation F1 is the stage-7 S*;
  2. A1: published (seeds_<dataset>, seeds_<dataset>_decoder) against optimised runs, per model: selected setting,
     parameter count, five-seed validation F1, test F1 at the validation threshold, edges right and changes recovered;
  3. A3: M*_o (best optimised model by mean validation F1) minus S* on the main test days, two-level paired bootstrap,
     Holm over 2 datasets x 2 measures, stage-7 classification rule;
  4. A4: M*_o on the 12 further UNSW days against seasonal copy and EWMA (L = 4), Holm over 2 comparators x 2 measures;
  5. predictions O1-O5.
Outputs in --out (must not exist): stage9_results.json, models.csv, contrasts.csv, predictions.csv.
Run from ndt_stage5:
  python bo_analysis.py --project-root .. --plan preregistration_stage9_optimisation.md --bo ..\\stage9_bo ^
      --newday-scores ..\\stage9_bo\\unsw\\newday_scores --stage7 ..\\stage7 --out ..\\stage9
"""
import argparse, csv, importlib.util, inspect, json, platform, time
from pathlib import Path
import numpy as np
import train_seeds as ts
from evaluate_pilot import load, sha, nonself, reach, select_threshold, fit_transition, baseline_records
from evaluate_temporal import validation_items
from run_analysis_seeds import L, finite_ewma, tune_ewma, record, load_runs, two_level_bootstrap, holm
from stage6_analysis import categories, predictions, window_table
from summarize_stage7 import classify, METRICS
from history_lr import TrainStats, features, score_items, C_GRID
from sklearn.linear_model import LogisticRegression

MODELS = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bilinear']
SIX = ['Persistence', 'Recent_frequency_L4', 'Transition', 'EWMA_L4', 'EWMA_expanding', 'History_LR']
PUBLISHED = {'unsw': ['seeds_unsw', 'seeds_unsw_decoder'], 'deakin': ['seeds_deakin', 'seeds_deakin_decoder']}
SEEDS = [42, 43, 44, 45, 46]
BLOCK = 8; RNG_SEED = 20261119; REPLICATES = 10000
O5_MARGIN = 0.02


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def simple_methods(train, va, te):
    t1 = fit_transition(train); out = {}
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        out[name] = (baseline_records(va, kind, t1), baseline_records(te, kind, t1))
    alphas = {}
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(va, t1, finite); alphas[name] = alpha
        out[name] = (vr, finite_ewma(te, alpha) if finite else baseline_records(te, 'ewma', t1, alpha))
    tr = validation_items(train, L); stats = TrainStats(train)
    X = np.concatenate([features(i, stats, True) for i in tr]); y = np.concatenate([i['A_target'][stats.mask] for i in tr]).astype(int)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd; grid = []
    for C in C_GRID:
        m = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-6).fit(Xs, y)
        th, vf, ve = select_threshold(record(va, score_items(va, stats, m, mu, sd))); grid.append((vf, -ve, -C, C, m))
    vf, _, _, C, m = max(grid, key=lambda g: g[:3])
    out['History_LR'] = (record(va, score_items(va, stats, m, mu, sd)), record(te, score_items(te, stats, m, mu, sd)))
    return out, alphas, C


def contrast(tab_m, tab_c, captures):
    deltas = np.stack([np.stack([t[k] - tab_c[k] for k in METRICS], 1) for t in tab_m])
    samples = two_level_bootstrap(deltas, captures, BLOCK, REPLICATES, np.random.default_rng(RNG_SEED)); B = len(samples); res = {}
    for j, k in enumerate(METRICS):
        lo, hi = np.nanquantile(samples[:, j], [.025, .975])
        p = min(1., 2 * min(((samples[:, j] <= 0).sum() + 1) / (B + 1), ((samples[:, j] >= 0).sum() + 1) / (B + 1)))
        res[k] = dict(difference=float(np.nanmean(deltas[:, :, j])), ci=[float(lo), float(hi)], p_value=float(p),
                      seed_differences=[float(x) for x in np.nanmean(deltas[:, :, j], 1)])
    return res


def adjust(contrasts):
    flat = [(c, k) for c in contrasts.values() for k in METRICS]
    for (c, k), a in zip(flat, holm([c[k]['p_value'] for c, k in flat])): c[k]['p_holm'] = float(a)
    for c in contrasts.values(): c['classification'] = classify(c)
    return contrasts


def parameter_count(root, run_dir, name, n):
    """Parameters of a model as built from its protocol.json (no weights needed)."""
    import torch
    proto = json.loads((Path(run_dir) / name / 'protocol.json').read_text()); cfg = ts.MODELS[name]
    spec = importlib.util.spec_from_file_location('count_' + name, root / cfg['script'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); cls = getattr(module, cfg['cls'])
    kw = dict(proto['constructor'], num_nodes=n, feat_dim=10, in_dim=10, seq_len=L)
    kw = {k: v for k, v in kw.items() if k in inspect.signature(cls.__init__).parameters}
    if proto.get('decoder_variant'): cls = ts.with_decoder(cls, proto['decoder_variant'], proto['constructor']['latent_dim'])
    return int(sum(p.numel() for p in cls(**kw).parameters())), proto


def evaluate_runs(runs, va, te, budget, refs, cats):
    """Per model and seed: validation F1, the window table at the budget, test F1 at the validation threshold."""
    out = {}
    for name, seeds in runs.items():
        out[name] = {}
        for seed, (v, t, info) in seeds.items():
            vr, tr = record(va, list(v)), record(te, list(t)); p, ch = predictions(vr, tr, budget)
            out[name][seed] = dict(validation_F1=select_threshold(vr)[1], table=window_table(tr, p[f'budget_{budget}'], refs, cats),
                                   F1_test=float(np.mean(window_table(tr, p['validation_threshold'], refs, cats)['edge_f1'])))
    return out


def summary(ev):
    s = sorted(ev)
    return dict(seeds=len(s), validation_F1=float(np.mean([ev[k]['validation_F1'] for k in s])),
                F1_test=float(np.mean([ev[k]['F1_test'] for k in s])),
                edges_right=float(np.mean([ev[k]['table']['true_positive'].mean() for k in s])),
                changes_recovered=float(np.mean([ev[k]['table']['changes_recovered'].mean() for k in s])))


def main(args):
    t0 = time.time(); root = Path(args.project_root).resolve(); out = Path(args.out); bo = Path(args.bo)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    plan_sha = sha(args.plan); results = dict(plan=dict(file=Path(args.plan).name, sha256=plan_sha)); rows = []; primary = {}
    best = {}; tables = {}
    for dataset in ['unsw', 'deakin']:
        tr_items, va, te, audit = ts.data_setup(root, dataset)
        ds = root.joinpath(*ts.DATASETS[dataset]['path']); train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl')
        ts.drop_self_loops(train, val, dict(items=[]))
        n = len(va[0]['A_target']); budget = int(np.rint(np.mean([i['A_target'].sum() for i in va])))
        keys_v = [(i['date'], int(i['target_window_id'])) for i in va]; keys_t = [(i['date'], int(i['target_window_id'])) for i in te]
        dates = sorted({k[0] for k in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
        refs = [reach(i['A_target']) for i in te]; cats = categories(te, train, val)
        ref7 = json.loads((Path(args.stage7) / f'default_{dataset}.json').read_text())
        if budget != ref7['budget']: raise ValueError(f'{dataset}: budget {budget} differs from stage 7')

        simple, alphas, C = simple_methods(train, va, te)
        sval = {k: select_threshold(v[0])[1] for k, v in simple.items()}
        s_star = max(SIX, key=lambda k: sval[k])
        if s_star != ref7['best_simple']: raise ValueError(f'{dataset}: best simple method {s_star} differs from stage 7 ({ref7["best_simple"]})')
        for k in SIX:
            if abs(sval[k] - ref7['candidates'][k]['validation_F1']) > 1e-9: raise ValueError(f'{dataset} {k}: validation F1 differs from stage 7')
        p, _ = predictions(*simple[s_star], budget); s_tab = window_table(simple[s_star][1], p[f'budget_{budget}'], refs, cats)

        optimised = load_runs(bo / dataset / 'final', dataset, keys_v, keys_t, set(MODELS))
        if sorted(optimised) != sorted(MODELS) or any(sorted(v) != SEEDS for v in optimised.values()):
            raise ValueError(f'{dataset}: optimised runs incomplete: { {k: sorted(v) for k, v in optimised.items()} }')
        published = {}
        for rd in PUBLISHED[dataset]: published.update(load_runs(root / rd, dataset, keys_v, keys_t, set(MODELS)))
        ev_o = evaluate_runs(optimised, va, te, budget, refs, cats); ev_p = evaluate_runs(published, va, te, budget, refs, cats)
        for name in MODELS:
            po, proto_o = parameter_count(root, bo / dataset / 'final', name, n)
            rd = [d for d in PUBLISHED[dataset] if (root / d / name).exists()][0]; pp, _ = parameter_count(root, root / rd, name, n)
            so, sp = summary(ev_o[name]), summary(ev_p[name])
            rows.append(dict(dataset=dataset, model=name, parameters_published=pp, parameters_optimised=po,
                             **{f'{k}_published': v for k, v in sp.items() if k != 'seeds'},
                             **{f'{k}_optimised': v for k, v in so.items() if k != 'seeds'},
                             selected_trial=proto_o['selected_trial'], **{f'hp_{k}': v for k, v in proto_o['hyperparameters'].items()}))
            if abs(sp['validation_F1'] - ref7['candidates'][name]['validation_F1']) > 1e-9:
                raise ValueError(f'{dataset} {name}: published validation F1 differs from stage 7')
        m_star = max(MODELS, key=lambda k: summary(ev_o[k])['validation_F1']); best[dataset] = m_star
        m_tabs = [ev_o[m_star][s]['table'] for s in SEEDS]; tables[dataset] = m_tabs
        primary[dataset] = contrast(m_tabs, s_tab, captures)
        results[dataset] = dict(budget=budget, best_simple=s_star, best_learned_optimised=m_star,
                                validation_F1_best_optimised=summary(ev_o[m_star])['validation_F1'],
                                validation_F1_stage7_best=ref7['candidates'][ref7['best_learned']]['validation_F1'],
                                stage7_best_learned=ref7['best_learned'], simple_validation_F1=sval, ewma_alpha=alphas, history_lr_C=C,
                                input_hashes=audit['input_hashes'])
        print(f'{dataset}: S* {s_star}, M*_o {m_star} (validation F1 {results[dataset]["validation_F1_best_optimised"]:.4f})', flush=True)
    adjust(primary)

    # ---- A4: 12 further UNSW days
    nd = root / 'dataset_ready_unsw_newdays' / '300s'
    train = load(root / 'dataset_ready' / '300s' / 'static_train.pkl'); val = load(nd / 'static_val.pkl'); test = load(nd / 'temporal_test.pkl')
    ts.drop_self_loops(train, val, test)
    vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    keys_v = [(i['date'], int(i['target_window_id'])) for i in vi]; keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    dates = sorted({k[0] for k in keys_t}); captures = [np.array([j for j, k in enumerate(keys_t) if k[0] == d]) for d in dates]
    budget = results['unsw']['budget']; refs = [reach(i['A_target']) for i in ti]; cats = categories(ti, train, val)
    t1 = fit_transition(train); alpha, vr = tune_ewma(vi, t1, True)
    comps = {'Seasonal_copy': (record(vi, [i['A_seq'][-2].astype(float) for i in vi]), record(ti, [i['A_seq'][-2].astype(float) for i in ti])),
             'EWMA_L4': (vr, finite_ewma(ti, alpha))}
    m_star = best['unsw']; m_tabs = []
    for seed in SEEDS:
        job = Path(args.newday_scores) / m_star / f'seed{seed}'
        info = json.loads((job / 'completed.json').read_text())
        if info['dataset'] != 'unsw_newdays' or info['dataset_hashes']['temporal_test.pkl'] != sha(nd / 'temporal_test.pkl'):
            raise ValueError(f'{job} was not scored on the further-day dataset')
        z = np.load(job / 'scores.npz')
        if [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != keys_t: raise ValueError('Further-day keys differ')
        v, t = record(vi, list(z['validation'].astype(float))), record(ti, list(z['test'].astype(float)))
        p, _ = predictions(v, t, budget); m_tabs.append(window_table(t, p[f'budget_{budget}'], refs, cats))
    further = {}
    for name, (v, t) in comps.items():
        p, _ = predictions(v, t, budget); further[name] = contrast(m_tabs, window_table(t, p[f'budget_{budget}'], refs, cats), captures)
    adjust(further)

    # ---- predictions
    mrow = {(r['dataset'], r['model']): r for r in rows}
    pred = {
        'O1': dict(holds=primary['unsw']['classification'] == 'learned clearly better', classification=primary['unsw']['classification']),
        'O2': dict(holds=primary['deakin']['classification'] != 'learned clearly better', classification=primary['deakin']['classification']),
        'O3': dict(holds=further['Seasonal_copy']['classification'] != 'learned clearly better', classification=further['Seasonal_copy']['classification']),
        'O4': dict(holds=all(mrow[(d, m)]['parameters_optimised'] < mrow[(d, m)]['parameters_published'] for d in ['unsw', 'deakin'] for m in ['dyngem', 'aernn']),
                   parameters={f'{d}_{m}': [mrow[(d, m)]['parameters_published'], mrow[(d, m)]['parameters_optimised']] for d in ['unsw', 'deakin'] for m in ['dyngem', 'aernn']}),
        'O5': dict(holds=all(results[d]['validation_F1_best_optimised'] - results[d]['validation_F1_stage7_best'] <= O5_MARGIN for d in ['unsw', 'deakin']),
                   gain={d: results[d]['validation_F1_best_optimised'] - results[d]['validation_F1_stage7_best'] for d in ['unsw', 'deakin']}),
    }
    for k, v in pred.items(): print(k, 'HELD' if v['holds'] else 'NOT HELD', flush=True)

    out.mkdir(parents=True)
    results.update(primary=primary, further_days=dict(best_learned=m_star, comparisons=further, days=dates, ewma_alpha=alpha),
                   predictions=pred, models=rows, replicates=REPLICATES, block=BLOCK, bootstrap_seed=RNG_SEED,
                   source_hashes={p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
                   python=platform.python_version(), seconds=round(time.time() - t0, 1))
    (out / 'stage9_results.json').write_text(json.dumps(results, indent=2, default=float))
    write_csv(out / 'models.csv', rows)
    crow = [dict(analysis='A3', dataset=d, learned=best[d], comparator=results[d]['best_simple'], measure=k, difference=c[k]['difference'],
                 ci_low=c[k]['ci'][0], ci_high=c[k]['ci'][1], p_holm=c[k]['p_holm'], classification=c['classification'])
            for d, c in primary.items() for k in METRICS]
    crow += [dict(analysis='A4 further UNSW days', dataset='unsw_newdays', learned=m_star, comparator=n, measure=k, difference=c[k]['difference'],
                  ci_low=c[k]['ci'][0], ci_high=c[k]['ci'][1], p_holm=c[k]['p_holm'], classification=c['classification'])
             for n, c in further.items() for k in METRICS]
    write_csv(out / 'contrasts.csv', crow)
    write_csv(out / 'predictions.csv', [dict(id=k, holds=v['holds'], **{a: json.dumps(b, default=float) for a, b in v.items() if a != 'holds'}) for k, v in pred.items()])
    print(f'Saved {out.resolve()} ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--plan', required=True); p.add_argument('--bo', required=True)
    p.add_argument('--newday-scores', required=True); p.add_argument('--stage7', required=True); p.add_argument('--out', required=True)
    main(p.parse_args())
