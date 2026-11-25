"""Apply the pre-registered decision rules (preregistration_deakin_fold2.md, H1-H6) to the split-2 results.
Run from ndt_stage5 after the analysis:
  python evaluate_preregistration.py --project-root .. --runs ..\\seeds_deakin_fold2 --analysis ..\\analysis_deakin_fold2
Writes prereg_results.json into the analysis folder and prints one PASS/FAIL line per hypothesis.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEEDS = [42, 43, 44, 45, 46]
INNER = ['tgae', 'vgrnn', 'static_prev', 'static_target']
LEARNED = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bias', 'tgae_bilinear']  # static_target is a reference
BASELINES = ['Persistence', 'Recent_frequency_L4', 'Transition', 'EWMA_L4', 'EWMA_expanding']


def reciprocity(path):
    share, ap, f1 = [], [], []
    for it in pickle.load(open(path, 'rb'))['items']:
        a = np.asarray(it['A_target']) > 0; np.fill_diagonal(a, False)
        if not a.any(): continue
        both, either = a & a.T, a | a.T; mask = ~np.eye(len(a), dtype=bool)
        share.append(both.sum() / a.sum())
        ap.append(average_precision_score(a[mask], (a.astype(int) + a.T.astype(int))[mask]))
        f1.append((2 * a.sum() / (a.sum() + either.sum()), 2 * both.sum() / (both.sum() + a.sum())))
    return float(np.mean(share)), float(np.mean(ap)), float(np.array(f1).mean(0).max())


def main(args):
    root = Path(args.project_root); runs = Path(args.runs); an = Path(args.analysis)
    ar = pd.read_csv(runs / 'all_runs_summary.csv')
    hl = pd.read_csv(an / 'headline.csv', encoding='utf-8-sig')
    pi = pd.read_csv(an / 'paired_intervals.csv')
    missing = {m: sorted(set(SEEDS) - set(ar[ar.model == m].seed)) for m in LEARNED + ['static_target']}
    missing = {m: s for m, s in missing.items() if s}
    if missing: raise ValueError(f'Incomplete runs, missing seeds: {missing}')

    def per_seed(model, decoder, col):
        x = ar[(ar.model == model) & (ar.decoder == decoder)].set_index('seed')[col]
        return x.loc[SEEDS].astype(float)
    def head(model, decoder, col):
        return float(str(hl[(hl.model == model) & (hl.decoder == decoder)][col].iloc[0]).split(' ± ')[0])
    budget = [d for d in pi[pi.family == 'primary'].decoder.unique() if d.startswith('budget_')][0]
    true_edges = float(ar.true_edges_mean.mean())
    share, ceil_ap, ceil_f1 = reciprocity(root / 'dataset_ready_deakin_fold2' / '300s' / 'temporal_test.pkl')
    res = {}

    edges = {m: float(per_seed(m, 'raw_0.5', 'predicted_edges_mean').mean()) for m in INNER}
    res['H1a'] = dict(rule='mean edges at 0.5 >= 10 x observed for tgae, vgrnn, static_prev, static_target',
                      values=dict(edges, observed=true_edges), passed=all(v >= 10 * true_edges for v in edges.values()))
    cut = 1 - per_seed('tgae_bias', 'raw_0.5', 'predicted_edges_mean') / per_seed('tgae', 'raw_0.5', 'predicted_edges_mean')
    dap = head('tgae_bias', 'validation_threshold', 'AP') - head('tgae', 'validation_threshold', 'AP')
    res['H1b'] = dict(rule='tgae_bias predicts >= 80% fewer edges at 0.5 than tgae in every seed, and |mean AP difference| <= 0.02',
                      values=dict(reduction_per_seed=cut.round(4).tolist(), AP_difference=dap),
                      passed=bool((cut >= 0.8).all() and abs(dap) <= 0.02))
    res['H2a'] = dict(rule='share of test edges with a reverse edge <= 0.10', values=dict(two_way_share=share, ceiling_AP=ceil_ap, ceiling_F1=ceil_f1),
                      passed=share <= 0.10)
    bil = per_seed('tgae_bilinear', 'validation_threshold', 'edge_f1_mean'); tg = per_seed('tgae', 'validation_threshold', 'edge_f1_mean')
    res['H2b'] = dict(rule='tgae_bilinear validation-threshold F1 > symmetric ceiling F1 in >= 4 of 5 seeds',
                      values=dict(f1_per_seed=bil.round(4).tolist(), ceiling_F1=ceil_f1), passed=int((bil > ceil_f1).sum()) >= 4)
    res['H2c'] = dict(rule='tgae_bilinear validation-threshold F1 > tgae F1 in every seed (seed-matched)',
                      values=dict(difference_per_seed=(bil - tg).round(4).tolist()), passed=bool((bil > tg).all()))
    best_base = max(BASELINES, key=lambda m: head(m, 'validation_threshold', 'edge_f1'))
    learned_f1 = {m: head(m, 'validation_threshold', 'edge_f1') for m in LEARNED}
    res['H3a'] = dict(rule='best history baseline validation-threshold F1 >= mean F1 of every learned model',
                      values=dict(best_baseline=best_base, best_baseline_F1=head(best_base, 'validation_threshold', 'edge_f1'), learned_F1=learned_f1),
                      passed=all(head(best_base, 'validation_threshold', 'edge_f1') >= v for v in learned_f1.values()))
    prim = pi[pi.family == 'primary']
    better = prim[(prim.metric == 'edge_f1') & (prim.mean_difference > 0) & (prim.p_holm < 0.05)]
    res['H3b'] = dict(rule='no learned model has a positive F1 difference from EWMA_L4 with Holm p < 0.05 (validation threshold or validation budget)',
                      values=dict(significant_positive=better[['model', 'decoder', 'mean_difference', 'p_holm']].to_dict('records')), passed=better.empty)
    r = prim[(prim.model == 'tgae_bilinear') & (prim.decoder == budget) & (prim.metric == 'new_edge_recall')].iloc[0]
    res['H4'] = dict(rule=f'tgae_bilinear new-edge recall minus EWMA_L4 at {budget} > 0 with Holm p < 0.05',
                     values=dict(difference=float(r.mean_difference), ci=[float(r.ci_low), float(r.ci_high)], p_holm=float(r.p_holm)),
                     passed=bool(r.mean_difference > 0 and r.p_holm < 0.05))
    nv = head('vrdag', 'validation_threshold', 'new_edge_recall')
    res['H5'] = dict(rule='VRDAG mean new-edge recall at the validation threshold < 0.10', values=dict(new_edge_recall=nv), passed=nv < 0.10)
    sd = {m: float(per_seed(m, 'validation_threshold', 'edge_f1_mean').std(ddof=1)) for m in LEARNED + ['static_target']}
    res['H6'] = dict(rule='SD over seeds of validation-threshold F1 <= 0.05 for every trained model', values=sd, passed=all(v <= 0.05 for v in sd.values()))

    (an / 'prereg_results.json').write_text(json.dumps(res, indent=2, default=float))
    for h, v in res.items():
        print(f'{h}: {"PASS" if v["passed"] else "FAIL"}  {v["rule"]}')
    print(f'{sum(v["passed"] for v in res.values())} of {len(res)} pass. Details: {an / "prereg_results.json"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--runs', required=True); p.add_argument('--analysis', required=True)
    main(p.parse_args())
