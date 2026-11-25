"""Stage 7A: give every learned model the same kind of validation tuning that EWMA and History-LR received.

Grid (one factor at a time around the published setting, 4 settings per model, chosen before any tuning run):
  default     learning rate 1e-3, layer widths x1   (the stage-5 runs, reused, not retrained)
  lr3e-4      learning rate 3e-4, widths x1
  width0.5    learning rate 1e-3, widths x0.5
  width2      learning rate 1e-3, widths x2
Selection: seed 42 of every setting; the setting with the highest validation macro F1 at the exact validation threshold
wins (ties: higher validation AP, then the default). Only validation data are used. The winner is then trained with
seeds 43-46 (seed 42 is reused: training is deterministic). Models: dyngem aernn vgrnn tgae vrdag static_prev tgae_bilinear.

Steps (each resumable; rerun the same command after an interruption):
  1. train seed 42 for the three new settings          -> <out>/<setting>/<model>/seed42
  2. choose a setting per model                         -> <out>/choice.json
  3. train seeds 43-46 for winners that are not default -> <out>/<setting>/<model>/seed43..46
  4. assemble the tuned models in one folder            -> <out>/tuned/<model>/seed42..46 and all_runs_summary.csv
Run from ndt_stage5, e.g.
  python tune_models.py --project-root .. --dataset deakin_fold2 --existing-runs ..\\seeds_deakin_fold2 --out ..\\tuning_deakin_fold2
"""
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path

MODELS = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bilinear']
SETTINGS = {'default': (1e-3, 1.0), 'lr3e-4': (3e-4, 1.0), 'width0.5': (1e-3, 0.5), 'width2': (1e-3, 2.0)}
SEEDS = [42, 43, 44, 45, 46]


def summary_row(job):
    with open(job / 'summary.csv', newline='', encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r['decoder'] == 'validation_threshold']
    return float(rows[0]['validation_F1']), float(rows[0]['validation_AP'])


def find_default(existing, model):
    for d in existing:
        p = Path(d) / model
        if (p / 'seed42' / 'completed.json').exists(): return p
    raise FileNotFoundError(f'No completed default run of {model} (seed 42) in {existing}')


def check_default(mdir, dataset, project_root):
    from evaluate_pilot import sha
    from train_seeds import DATASETS, MODELS as REGISTRY
    proto = json.loads((mdir / 'protocol.json').read_text())
    bad = {k: proto.get(k) for k, v in [('learning_rate', 1e-3), ('history', 4), ('weighting', 'original'), ('smoke', False), ('dataset', dataset)]
           if proto.get(k) != v}
    if proto.get('constructor') != json.loads(json.dumps(REGISTRY[mdir.name]['kwargs'])): bad['constructor'] = proto.get('constructor')
    ds = Path(project_root).resolve().joinpath(*DATASETS[dataset]['path'])
    now = {k: sha(ds / k) for k in proto['input_hashes']}
    if now != proto['input_hashes']: bad['input_hashes'] = 'the dataset files differ from those the run was trained on'
    if bad: raise ValueError(f'{mdir} is not a default stage-5 run on the current {dataset} data: {bad}')


def train(args, setting, models, seeds):
    lr, width = SETTINGS[setting]
    cmd = [sys.executable, str(Path(__file__).parent / 'train_seeds.py'), '--project-root', args.project_root, '--dataset', args.dataset,
           '--models', *models, '--seeds', *map(str, seeds), '--learning-rate', str(lr), '--width', str(width),
           '--out', str(Path(args.out) / setting), '--device', args.device, '--resume'] + (['--epochs', str(args.epochs)] if args.epochs else [])
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    defaults = {m: find_default(args.existing_runs, m) for m in MODELS}
    for m, p in defaults.items(): check_default(p, args.dataset, args.project_root)
    for setting in ['lr3e-4', 'width0.5', 'width2']:
        train(args, setting, MODELS, [42])
    choice = {}
    for m in MODELS:
        cand = {'default': summary_row(defaults[m] / 'seed42')}
        for s in ['lr3e-4', 'width0.5', 'width2']: cand[s] = summary_row(out / s / m / 'seed42')
        order = list(SETTINGS)
        best = max(order, key=lambda s: (cand[s][0], cand[s][1], s == 'default', -order.index(s)))
        choice[m] = dict(chosen=best, learning_rate=SETTINGS[best][0], width=SETTINGS[best][1],
                         validation={s: dict(F1=v[0], AP=v[1]) for s, v in cand.items()})
        print(f'{m}: chosen {best}  ' + '  '.join(f'{s} F1 {v[0]:.4f}' for s, v in cand.items()), flush=True)
    (out / 'choice.json').write_text(json.dumps(dict(dataset=args.dataset, grid=SETTINGS, rule='max validation F1; ties: validation AP, then default',
                                                     models=choice), indent=2))
    for m, c in choice.items():
        if c['chosen'] != 'default': train(args, c['chosen'], [m], SEEDS[1:])
    tuned = out / 'tuned'
    for m, c in choice.items():
        src = defaults[m] if c['chosen'] == 'default' else out / c['chosen'] / m
        dst = tuned / m
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / 'protocol.json', dst / 'protocol.json')
        for seed in SEEDS:
            if not (src / f'seed{seed}' / 'completed.json').exists(): raise FileNotFoundError(src / f'seed{seed}')
            shutil.copytree(src / f'seed{seed}', dst / f'seed{seed}')
    rows = []
    for f in sorted(tuned.glob('*/seed*/summary.csv')):
        with f.open(newline='', encoding='utf-8') as fh: rows.extend(csv.DictReader(fh))
    with open(tuned / 'all_runs_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Tuned models assembled in', tuned.resolve(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--existing-runs', nargs='+', required=True, help='Folders with the default stage-5 runs of the seven models')
    p.add_argument('--out', required=True); p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, help='Testing only')
    main(p.parse_args())
