"""Stage 7B/7D/7E: train the scope and ablation settings with the configurations chosen in stage 7A.

Settings
  window600, window900   TGAE, asymmetric TGAE and VRDAG with their tuned settings, on the merged-window dataset
                         (build it first with make_merged_windows.py)
  horizon2, horizon4     the same three models with their tuned settings, predicting 2 or 4 windows ahead
                         (build the dataset first with make_horizon_dataset.py)
  history1, history2     the same three models with their tuned settings, given only the last 1 or 2 history windows
  noweight               TGAE, VGRNN and Static VGAE (previous window) in the default setting, trained without class weighting;
                         prints the pre-registered score-floor check (edges predicted at 0.5 against observed edges)
Run from ndt_stage5, e.g.
  python run_scope.py --project-root .. --dataset deakin_fold2 --setting window900 --tuning ..\\tuning_deakin_fold2 --out ..\\scope_deakin_fold2_window900
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

SCOPE_MODELS = ['tgae', 'tgae_bilinear', 'vrdag']
FLOOR_MODELS = ['tgae', 'vgrnn', 'static_prev']


def main(args):
    choice = None
    if args.setting != 'noweight':
        if not args.tuning: raise ValueError('--tuning is required for this setting')
        choice = json.loads((Path(args.tuning) / 'choice.json').read_text())
        if choice['dataset'] != args.dataset: raise ValueError(f'Tuning folder is for {choice["dataset"]}, not {args.dataset}')
    dataset = (f'{args.dataset}_{args.setting[6:]}s' if args.setting.startswith('window') else
               f'{args.dataset}_h{args.setting[7:]}' if args.setting.startswith('horizon') else args.dataset)
    models = FLOOR_MODELS if args.setting == 'noweight' else SCOPE_MODELS
    for m in models:
        if args.setting == 'noweight': lr, width = 1e-3, 1.0
        else: lr, width = choice['models'][m]['learning_rate'], choice['models'][m]['width']
        cmd = [sys.executable, str(Path(__file__).parent / 'train_seeds.py'), '--project-root', args.project_root, '--dataset', dataset,
               '--models', m, '--learning-rate', str(lr), '--width', str(width), '--out', args.out, '--device', args.device, '--resume']
        if args.setting.startswith('history'): cmd += ['--history', args.setting[7:]]
        if args.setting == 'noweight': cmd += ['--class-weight', 'none']
        if args.epochs: cmd += ['--epochs', str(args.epochs)]
        print('>>', ' '.join(cmd), flush=True); subprocess.run(cmd, check=True)
    if args.setting == 'noweight':
        with open(Path(args.out) / 'all_runs_summary.csv', newline='', encoding='utf-8') as fh:
            rows = [r for r in csv.DictReader(fh) if r['decoder'] == 'raw_0.5']
        for m in models:
            r = [x for x in rows if x['model'] == m]
            edges = sum(float(x['predicted_edges_mean']) for x in r) / len(r); obs = sum(float(x['true_edges_mean']) for x in r) / len(r)
            print(f'{args.dataset} {m} without class weighting: {edges:.1f} edges at 0.5 against {obs:.1f} observed '
                  f'({edges / obs:.1f}x) -> floor prediction {"HELD" if edges >= 10 * obs else "NOT HELD"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', required=True,
                                                                     choices=['unsw', 'deakin', 'deakin_fold2', 'unsw_fold2'])
    p.add_argument('--setting', required=True, choices=['window600', 'window900', 'horizon2', 'horizon4', 'history1', 'history2', 'noweight'])
    p.add_argument('--tuning', help='The --out folder of tune_models.py for this test set (not needed for noweight)')
    p.add_argument('--out', required=True); p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, help='Testing only')
    main(p.parse_args())
