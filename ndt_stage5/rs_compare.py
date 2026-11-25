"""Optimiser check (preregistration_stage10_random_search.md): random search against the TPE search of stage 9.

Reads the stage9_results.json written by bo_analysis.py for the TPE runs (--tpe) and for the random-search runs (--rs),
and evaluates predictions R1-R4. Outputs in --out: comparison.json, comparison.csv (per model and dataset).
Run from ndt_stage5:
  python rs_compare.py --plan preregistration_stage10_random_search.md --tpe ..\\stage9 --rs ..\\stage10 --out ..\\stage10\\comparison
"""
import argparse, csv, json
from pathlib import Path
from evaluate_pilot import sha

R2_MARGIN = 0.02; R4_LIMIT = 0.80
MAIN = [('unsw', 'primary', 'unsw'), ('deakin', 'primary', 'deakin')]


def main(args):
    out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    tpe = json.loads((Path(args.tpe) / 'stage9_results.json').read_text()); rs = json.loads((Path(args.rs) / 'stage9_results.json').read_text())
    rows = []; key = lambda r: (r['dataset'], r['model'])
    t_rows = {key(r): r for r in tpe['models']}; r_rows = {key(r): r for r in rs['models']}
    for k in sorted(t_rows):
        a, b = t_rows[k], r_rows[k]
        rows.append(dict(dataset=k[0], model=k[1], validation_F1_published=a['validation_F1_published'],
                         validation_F1_tpe=a['validation_F1_optimised'], validation_F1_random=b['validation_F1_optimised'],
                         difference_random_minus_tpe=b['validation_F1_optimised'] - a['validation_F1_optimised'],
                         F1_test_tpe=a['F1_test_optimised'], F1_test_random=b['F1_test_optimised'],
                         edges_right_tpe=a['edges_right_optimised'], edges_right_random=b['edges_right_optimised'],
                         changes_recovered_tpe=a['changes_recovered_optimised'], changes_recovered_random=b['changes_recovered_optimised'],
                         parameters_published=a['parameters_published'], parameters_tpe=a['parameters_optimised'], parameters_random=b['parameters_optimised']))
    classes = {}
    for d, sect, _ in MAIN:
        classes[f'{d} main vs {tpe[d]["best_simple"]}'] = (tpe[sect][d]['classification'], rs[sect][d]['classification'])
    for comp in ['Seasonal_copy', 'EWMA_L4']:
        classes[f'further UNSW days vs {comp}'] = (tpe['further_days']['comparisons'][comp]['classification'], rs['further_days']['comparisons'][comp]['classification'])
    big = [r for r in rows if r['model'] in ('dyngem', 'aernn')]
    pred = {
        'R1': dict(holds=all(rs[d]['best_learned_optimised'] == 'tgae_bilinear' for d in ['unsw', 'deakin']),
                   best_random={d: rs[d]['best_learned_optimised'] for d in ['unsw', 'deakin']}),
        'R2': dict(holds=all(abs(r['difference_random_minus_tpe']) <= R2_MARGIN for r in rows),
                   largest_difference=max(rows, key=lambda r: abs(r['difference_random_minus_tpe']))),
        'R3': dict(holds=all(t == r for t, r in classes.values()), classifications={k: dict(tpe=t, random=r) for k, (t, r) in classes.items()}),
        'R4': dict(holds=all(r['validation_F1_random'] < R4_LIMIT for r in big), values={f"{r['dataset']}_{r['model']}": r['validation_F1_random'] for r in big}),
    }
    for k, v in pred.items(): print(k, 'HELD' if v['holds'] else 'NOT HELD', flush=True)
    for r in rows:
        print(f"{r['dataset']:6s} {r['model']:13s} validation F1 published {r['validation_F1_published']:.3f} TPE {r['validation_F1_tpe']:.3f} "
              f"random {r['validation_F1_random']:.3f} ({r['difference_random_minus_tpe']:+.3f})", flush=True)
    out.mkdir(parents=True)
    keys = list(rows[0]);
    with open(out / 'comparison.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out / 'comparison.json').write_text(json.dumps(dict(plan=dict(file=Path(args.plan).name, sha256=sha(args.plan)), predictions=pred,
                                                         models=rows, tpe_plan=tpe['plan'], random_plan=rs['plan']), indent=2, default=float))
    print(f'Saved {out.resolve()}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', required=True); p.add_argument('--tpe', required=True); p.add_argument('--rs', required=True); p.add_argument('--out', required=True)
    main(p.parse_args())
