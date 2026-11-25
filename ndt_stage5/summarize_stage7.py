"""Stage 7: classify each test set for one setting and check the pre-registered prediction.
Holm adjustment over the 8 contrasts of a setting (4 test sets x {edges right, changes recovered}). Margin m = 0.5 edges per window.
  learned clearly better : one measure favours M* (difference > 0, Holm p < 0.05) and the other is not worse by more than m
                           (lower 95% bound > -m)
  simple clearly better  : the mirror image (difference < 0, Holm p < 0.05; other measure's upper 95% bound < +m)
  trade-off              : both of the above hold (small gains in opposite directions)
  no clear winner        : neither holds
Prediction (every setting): learned clearly better on at most 1 of the 4 test sets.
Run from ndt_stage5, e.g.
  python summarize_stage7.py --setting tuned --files ..\\stage7\\tuned_unsw.json ..\\stage7\\tuned_deakin.json ..\\stage7\\tuned_deakin_fold2.json ..\\stage7\\tuned_unsw_fold2.json
"""
import argparse, csv, json
from pathlib import Path
from run_analysis_seeds import holm

MARGIN = 0.5
METRICS = ['true_positive', 'changes_recovered']


def classify(c):
    def favours(sign):
        for a, b in [(METRICS[0], METRICS[1]), (METRICS[1], METRICS[0])]:
            x, y = c[a], c[b]
            if sign > 0 and x['difference'] > 0 and x['p_holm'] < .05 and y['ci'][0] > -MARGIN: return True
            if sign < 0 and x['difference'] < 0 and x['p_holm'] < .05 and y['ci'][1] < MARGIN: return True
        return False
    l, s = favours(1), favours(-1)
    return 'trade-off' if (l and s) else 'learned clearly better' if l else 'simple clearly better' if s else 'no clear winner'


def main(args):
    res = [json.loads(Path(f).read_text()) for f in args.files]
    if len({r['dataset'] for r in res}) != len(res): raise ValueError('Each test set must appear once')
    flat = [(r, m) for r in res for m in METRICS]
    for (r, m), a in zip(flat, holm([r['contrast_M_minus_S'][m]['p_value'] for r, m in flat])):
        r['contrast_M_minus_S'][m]['p_holm'] = float(a)
    rows = []
    for r in res:
        c = r['contrast_M_minus_S']; cls = classify(c)
        rows.append(dict(setting=args.setting, dataset=r['dataset'], best_simple=r['best_simple'], best_learned=r['best_learned'],
                         edges_right_difference=c['true_positive']['difference'], edges_right_ci_low=c['true_positive']['ci'][0],
                         edges_right_ci_high=c['true_positive']['ci'][1], edges_right_p_holm=c['true_positive']['p_holm'],
                         changes_difference=c['changes_recovered']['difference'], changes_ci_low=c['changes_recovered']['ci'][0],
                         changes_ci_high=c['changes_recovered']['ci'][1], changes_p_holm=c['changes_recovered']['p_holm'],
                         observed_edges=r['observed']['edges_per_window'], observed_changes=r['observed']['changes_per_window'],
                         classification=cls))
        print(f"{r['dataset']:>14}  {r['best_learned']:>13} vs {r['best_simple']:<19} edges right {c['true_positive']['difference']:+.2f} "
              f"[{c['true_positive']['ci'][0]:+.2f}, {c['true_positive']['ci'][1]:+.2f}]  changes {c['changes_recovered']['difference']:+.2f} "
              f"[{c['changes_recovered']['ci'][0]:+.2f}, {c['changes_recovered']['ci'][1]:+.2f}]  -> {cls}")
    n = sum(r['classification'] == 'learned clearly better' for r in rows)
    print(f"Prediction (learned clearly better on at most 1 of {len(rows)}): {'HELD' if n <= 1 else 'NOT HELD'} ({n} of {len(rows)})")
    out = Path(args.out or Path(args.files[0]).parent / f'summary_{args.setting}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setting', required=True); p.add_argument('--files', nargs='+', required=True); p.add_argument('--out')
    main(p.parse_args())
