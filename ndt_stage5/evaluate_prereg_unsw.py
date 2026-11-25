"""Apply the pre-registered decision rules of preregistration_unsw_fold2.md (U1-U8 and question Q1) to the UNSW split-2 results.
Run from ndt_stage5 after training and both analyses:
  python evaluate_prereg_unsw.py --project-root .. --runs ..\\seeds_unsw_fold2 --analysis ..\\analysis_unsw_fold2 --stage6 ..\\stage6_unsw_fold2
Writes prereg_results.json into the stage-6 folder and prints one line per rule.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
INNER = ['tgae', 'vgrnn', 'static_prev', 'static_target']
TRAINED = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'static_target', 'tgae_bias', 'tgae_bilinear']
MARGIN = 0.5  # edges per window


def reciprocity(path):
    share, f1 = [], []
    for it in pickle.load(open(path, 'rb'))['items']:
        a = np.asarray(it['A_target']) > 0; np.fill_diagonal(a, False)
        if not a.any(): continue
        both, either = a & a.T, a | a.T
        share.append(both.sum() / a.sum())
        f1.append((2 * a.sum() / (a.sum() + either.sum()), 2 * both.sum() / (both.sum() + a.sum())))
    return float(np.mean(share)), float(np.array(f1).mean(0).max())


def main(args):
    root = Path(args.project_root); runs = Path(args.runs); an = Path(args.analysis); s6 = Path(args.stage6)
    ar = pd.read_csv(runs / 'all_runs_summary.csv')
    hl = pd.read_csv(an / 'headline.csv', encoding='utf-8-sig'); pi = pd.read_csv(an / 'paired_intervals.csv')
    ci = pd.read_csv(s6 / 'count_intervals.csv')
    missing = {m: sorted(set(SEEDS) - set(ar[ar.model == m].seed)) for m in TRAINED}
    missing = {m: s for m, s in missing.items() if s}
    if missing: raise ValueError(f'Incomplete runs, missing seeds: {missing}')
    plan_an = json.loads((an / 'audit.json').read_text()).get('analysis_plan'); plan_s6 = json.loads((s6 / 'audit.json').read_text()).get('analysis_plan')

    def per_seed(model, decoder, col):
        return ar[(ar.model == model) & (ar.decoder == decoder)].set_index('seed')[col].loc[SEEDS].astype(float)
    def head(model, decoder, col):
        return float(str(hl[(hl.model == model) & (hl.decoder == decoder)][col].iloc[0]).split(' ± ')[0])
    budget = [d for d in pi[pi.family == 'primary'].decoder.unique() if d.startswith('budget_')][0]
    def contrast(model, ref, metric):
        r = ci[(ci.model == model) & (ci.reference == ref) & (ci.decoder == budget) & (ci.metric == metric) & (ci.family == 'primary')]
        if len(r) != 1: raise ValueError(f'Primary contrast {model} - {ref} {metric} not found')
        r = r.iloc[0]; return dict(difference=float(r.mean_difference_per_window), ci=[float(r.ci_low), float(r.ci_high)], p_holm=float(r.p_holm))
    true_edges = float(ar.true_edges_mean.mean()); res = {}

    edges = {m: float(per_seed(m, 'raw_0.5', 'predicted_edges_mean').mean()) for m in INNER}
    res['U1'] = dict(rule='mean edges at 0.5 >= 10 x observed for tgae, vgrnn, static_prev, static_target',
                     values=dict(edges, observed=true_edges), passed=all(v >= 10 * true_edges for v in edges.values()))
    cut = 1 - per_seed('tgae_bias', 'raw_0.5', 'predicted_edges_mean') / per_seed('tgae', 'raw_0.5', 'predicted_edges_mean')
    dap = head('tgae_bias', 'validation_threshold', 'AP') - head('tgae', 'validation_threshold', 'AP')
    res['U2'] = dict(rule='tgae_bias predicts >= 80% fewer edges at 0.5 than tgae in every seed, and |mean AP difference| <= 0.02',
                     values=dict(reduction_per_seed=cut.round(4).tolist(), AP_difference=dap), passed=bool((cut >= .8).all() and abs(dap) <= .02))
    share, ceil_f1 = reciprocity(root / 'dataset_ready_unsw_fold2' / '300s' / 'temporal_test.pkl')
    res['U3'] = dict(rule='share of test edges with a reverse edge >= 0.80 and symmetric-ceiling F1 >= 0.90',
                     values=dict(two_way_share=share, ceiling_F1=ceil_f1), passed=bool(share >= .8 and ceil_f1 >= .9))
    prim = pi[pi.family == 'primary']; better = prim[(prim.metric == 'edge_f1') & (prim.mean_difference > 0) & (prim.p_holm < .05)]
    res['U4'] = dict(rule='no learned model has a positive F1 difference from EWMA_L4 with Holm p < 0.05 (validation threshold or validation budget)',
                     values=dict(significant_positive=better[['model', 'decoder', 'mean_difference', 'p_holm']].to_dict('records')), passed=bool(better.empty))
    tp = contrast('tgae', 'EWMA_L4', 'true_positive')
    res['U5'] = dict(rule=f'tgae minus EWMA_L4 edges right per window at {budget}: lower 95% bound > -{MARGIN}', values=tp, passed=tp['ci'][0] > -MARGIN)
    ch = contrast('tgae', 'EWMA_L4', 'changes_recovered')
    res['U6'] = dict(rule=f'tgae minus EWMA_L4 changes recovered per window at {budget} > 0 with Holm p < 0.05 (stage-6 family)',
                     values=ch, passed=bool(ch['difference'] > 0 and ch['p_holm'] < .05))
    nv = head('vrdag', 'validation_threshold', 'new_edge_recall')
    res['U7'] = dict(rule='VRDAG mean new-edge recall at the validation threshold < 0.15', values=dict(new_edge_recall=nv), passed=nv < .15)
    sd = {m: float(per_seed(m, 'validation_threshold', 'edge_f1_mean').std(ddof=1)) for m in TRAINED}
    res['U8'] = dict(rule='SD over seeds of validation-threshold F1 <= 0.05 for every trained model', values=sd, passed=all(v <= .05 for v in sd.values()))
    q = {m: dict(changes=contrast(m, 'History_LR', 'changes_recovered'), edges_right=contrast(m, 'History_LR', 'true_positive'))
         for m in ['tgae', 'tgae_bilinear']}
    credited = {m: bool(v['changes']['difference'] > 0 and v['changes']['p_holm'] < .05 and v['edges_right']['ci'][0] > -MARGIN) for m, v in q.items()}
    res['Q1'] = dict(rule='Question (no prediction). The architecture is credited with the change gain only if model minus History_LR changes '
                          f'recovered > 0 with Holm p < 0.05 and the lower 95% bound of edges right > -{MARGIN}; otherwise the gain is '
                          'attributed to learning from history', values=dict(q, credited=credited),
                     answer={m: ('architecture adds value beyond History-LR' if c else 'gain attributed to learning from history') for m, c in credited.items()})
    res['plan_hashes'] = dict(run_analysis_seeds=plan_an, stage6_analysis=plan_s6)
    (s6 / 'prereg_results.json').write_text(json.dumps(res, indent=2, default=float))
    for h, v in res.items():
        if h == 'plan_hashes': continue
        status = v['answer'] if h == 'Q1' else ('PASS' if v['passed'] else 'FAIL')
        print(f'{h}: {status}  {v["rule"]}')
    rules = [k for k in res if k.startswith('U')]
    print(f'{sum(res[k]["passed"] for k in rules)} of {len(rules)} predictions held. Plan recorded: {plan_an} / {plan_s6}')
    print('Details:', s6 / 'prereg_results.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--runs', required=True)
    p.add_argument('--analysis', required=True); p.add_argument('--stage6', required=True)
    main(p.parse_args())
