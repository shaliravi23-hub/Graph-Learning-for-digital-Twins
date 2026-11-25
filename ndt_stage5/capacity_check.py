"""Overfitting and capacity check: training versus validation versus test performance of the existing checkpoints.

No model is trained or tuned. For each dataset (the main UNSW and Deakin splits) and every learned model and seed:
  1. rebuilds the model from protocol.json and loads best.pt (SHA-256 checked against completed.json);
  2. re-scores the validation day and stops unless it reproduces the stored validation logits (max |diff| <= 0.01 and
     validation F1 within 0.001, as guard G1 of stage 8);
  3. scores the training windows, i.e. the targets the model was fitted to;
  4. evaluates training, validation and test windows with the same decision rules: F1 at the validation threshold,
     edges right and changes recovered at the validation edge budget, and AP over all ordered pairs;
  5. reads history.json: best epoch, training loss at epoch 1, the best epoch and the last epoch, validation AP curve.
The simple methods of stage 7 (fitted on the training days, tuned on the validation day) are evaluated on the same
three sets as a reference: a method with no capacity to overfit shows how much of a gap comes from the days themselves.

Outputs in --out (must not exist): per_run.csv (model x seed x split), summary.csv (mean over seeds),
curves.csv (training loss and validation AP per epoch), capacity_check.json.
Run from ndt_stage5:
  python capacity_check.py --project-root .. --out ..\\capacity_check
  python capacity_check.py --project-root .. --out ..\\stage9_bo\\capacity_check --runs unsw=stage9_bo/unsw/final deakin=stage9_bo/deakin/final
(--runs replaces the run folders of a dataset, given relative to the project root.)
"""
import argparse, csv, importlib.util, inspect, json, platform, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scipy.special import expit
import train_seeds as ts
from evaluate_pilot import sha, nonself, reach, metrics, select_threshold, fit_transition, baseline_records
from run_analysis_seeds import L, finite_ewma, tune_ewma, record
from stage6_analysis import predictions
from periodicity_check import fit_transition2, score2
from history_lr import TrainStats, features, score_items, C_GRID

RUNS = {'unsw': ['seeds_unsw', 'seeds_unsw_decoder'], 'deakin': ['seeds_deakin', 'seeds_deakin_decoder']}
LEARNED = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bias', 'tgae_bilinear']
SIMPLE = ['Persistence', 'Recent_frequency_L4', 'Transition', 'EWMA_L4', 'EWMA_expanding', 'History_LR', 'Seasonal_copy',
          'Order2_Transition']
SPLITS = ['train', 'validation', 'test']
G1_LOGIT_TOLERANCE = 1e-2; G1_F1_TOLERANCE = 1e-3


def evaluate(recs, vr, budget):
    """Per-window means at the validation threshold and the validation edge budget, and mean AP."""
    p, _ = predictions(vr, recs, budget)
    mask = nonself(len(recs[0]['truth']))
    at_t = [metrics(r['truth'] > 0, q, r['previous'], reach(r['truth'] > 0)) for r, q in zip(recs, p['validation_threshold'])]
    at_k = [metrics(r['truth'] > 0, q, r['previous'], reach(r['truth'] > 0)) for r, q in zip(recs, p[f'budget_{budget}'])]
    ap = [average_precision_score(r['truth'][mask] > 0, r['score'][mask]) for r in recs if r['truth'][mask].sum()]
    return dict(F1=float(np.mean([m['edge_f1'] for m in at_t])),
                edges_right=float(np.mean([m['true_positive'] for m in at_k])),
                changes_recovered=float(np.mean([m['new_edges_recovered'] + m['disappearances_recovered'] for m in at_k])),
                F1_at_budget=float(np.mean([m['edge_f1'] for m in at_k])), AP=float(np.mean(ap)),
                predicted_edges_at_threshold=float(np.mean([m['predicted_edges'] for m in at_t])))


def simple_methods(train, tr, va, te):
    t1 = fit_transition(train); n = len(tr[0]['A_target']); t2, _ = fit_transition2(train, nonself(n))
    out = {}
    for kind, name in [('persistence', 'Persistence'), ('recent_frequency', 'Recent_frequency_L4'), ('transition', 'Transition')]:
        out[name] = [baseline_records(x, kind, t1) for x in (tr, va, te)]
    for finite, name in [(True, 'EWMA_L4'), (False, 'EWMA_expanding')]:
        alpha, vr = tune_ewma(va, t1, finite)
        out[name] = [finite_ewma(tr, alpha) if finite else baseline_records(tr, 'ewma', t1, alpha), vr,
                     finite_ewma(te, alpha) if finite else baseline_records(te, 'ewma', t1, alpha)]
    stats = TrainStats(train)
    X = np.concatenate([features(i, stats, True) for i in tr]); y = np.concatenate([i['A_target'][stats.mask] for i in tr]).astype(int)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd; grid = []
    for C in C_GRID:
        m = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-6).fit(Xs, y)
        th, vf, ve = select_threshold(record(va, score_items(va, stats, m, mu, sd))); grid.append((vf, -ve, -C, m))
    m = max(grid, key=lambda g: g[:3])[3]
    out['History_LR'] = [record(x, score_items(x, stats, m, mu, sd)) for x in (tr, va, te)]
    out['Seasonal_copy'] = [record(x, [i['A_seq'][-2].astype(float) for i in x]) for x in (tr, va, te)]
    out['Order2_Transition'] = [record(x, score2(x, t2)) for x in (tr, va, te)]
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main(args):
    import torch
    t0 = time.time(); root = Path(args.project_root).resolve(); out = Path(args.out)
    if out.exists(): raise ValueError('Output exists; choose a fresh directory')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; use --device cpu')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    for flag, value in [('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True)]:
        if hasattr(torch.backends.cuda, flag): getattr(torch.backends.cuda, flag)(value)

    runs = dict(RUNS)
    for spec in args.runs or []:
        d, dirs = spec.split('=', 1); runs[d] = dirs.split(',')
    rows, curves, info = [], [], {}
    for dataset in args.datasets:
        tr, va, te, audit = ts.data_setup(root, dataset)
        ds = root.joinpath(*ts.DATASETS[dataset]['path'])
        train = ts.load(ds / 'static_train.pkl'); ts.drop_self_loops(train, dict(items=[]), dict(items=[]))
        n = len(tr[0]['A_target']); budget = int(np.rint(np.mean([i['A_target'].sum() for i in va])))
        keys_v = [(i['date'], int(i['target_window_id'])) for i in va]; keys_t = [(i['date'], int(i['target_window_id'])) for i in te]
        info[dataset] = dict(train_targets=len(tr), validation_targets=len(va), test_targets=len(te), nodes=n, budget=budget,
                             training_edges=int(sum(i['A_target'][nonself(n)].sum() for i in tr)),
                             observed_edges=dict(train=float(np.mean([i['A_target'][nonself(n)].sum() for i in tr])),
                                                 validation=float(np.mean([i['A_target'][nonself(n)].sum() for i in va])),
                                                 test=float(np.mean([i['A_target'][nonself(n)].sum() for i in te]))),
                             input_hashes=audit['input_hashes'])
        print(f'{dataset}: {len(tr)} training, {len(va)} validation, {len(te)} test targets; budget {budget}', flush=True)

        for name, recs in simple_methods(train, tr, va, te).items():
            for split, r in zip(SPLITS, recs):
                rows.append(dict(dataset=dataset, method=name, kind='simple', seed='', parameters='', split=split, **evaluate(r, recs[1], budget)))
            print(f'  {name} done', flush=True)

        def arrays(items, H):
            return (torch.tensor(np.stack([np.asarray(i['A_seq'][-H:], np.float32) for i in items])),
                    torch.tensor(np.stack([np.asarray(i['X_seq'][-H:], np.float32) for i in items])))

        def score(model, kind, arr):
            model.eval(); ss = []
            with torch.inference_mode():
                for j in range(0, len(arr[0]), 16):
                    a, x = arr[0][j:j + 16].to(args.device), arr[1][j:j + 16].to(args.device)
                    if kind == 'last': s = model(a[:, -1], x[:, -1])['A_logits']
                    elif kind == 'variational': s = model(a, x, use_posterior=False)['A_logits']
                    elif kind == 'sequence': s = model(a, x)['A_logits']
                    elif kind == 'static_prev': s = model.forward_deterministic(x[:, -1])[0]
                    else: raise ValueError(kind)
                    ss.extend(s.float().cpu().numpy())
            ss = np.stack(ss)
            if not np.isfinite(ss).all(): raise ValueError('Nonfinite logits')
            return ss

        for run_dir in runs[dataset]:
            for name in LEARNED:
                mdir = root / run_dir / name
                if not mdir.exists(): continue
                proto = json.loads((mdir / 'protocol.json').read_text())
                if proto.get('dataset') != dataset: raise ValueError(f'{mdir} was trained on {proto.get("dataset")}')
                cfg = ts.MODELS[name]; source = root / cfg['script']
                if sha(source) != proto['source_sha256']: raise ValueError(f'{source.name} differs from the script {name} was trained with')
                H = int(proto['history']); kind = proto['input']
                spec = importlib.util.spec_from_file_location('cap_' + name, source)
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                cls = getattr(module, cfg['cls'])
                kw = dict(proto['constructor'], num_nodes=n, feat_dim=10, in_dim=10, seq_len=H)
                kw = {k: v for k, v in kw.items() if k in inspect.signature(cls.__init__).parameters}
                if proto.get('decoder_variant'): cls = ts.with_decoder(cls, proto['decoder_variant'], proto['constructor']['latent_dim'])
                tr_arr, va_arr = arrays(tr, H), arrays(va, H)
                for job in sorted(mdir.glob('seed*')):
                    if not (job / 'completed.json').exists(): continue
                    done = json.loads((job / 'completed.json').read_text()); seed = int(done['seed'])
                    if seed not in args.seeds or done.get('smoke'): continue
                    if sha(job / 'best.pt') != done['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch: ' + str(job))
                    z = np.load(job / 'scores.npz', allow_pickle=False)
                    if [(str(d), int(w)) for d, w in zip(z['validation_dates'], z['validation_windows'])] != keys_v or \
                       [(str(d), int(w)) for d, w in zip(z['test_dates'], z['test_windows'])] != keys_t:
                        raise ValueError('Window keys differ: ' + str(job))
                    model = cls(**kw).to(args.device)
                    model.load_state_dict(torch.load(job / 'best.pt', map_location=args.device, weights_only=True), strict=True)
                    params = int(sum(p.numel() for p in model.parameters()))
                    v = score(model, kind, va_arr); diff = float(np.abs(v - z['validation']).max())
                    vr = record(va, list(z['validation'].astype(float)))
                    f_old = select_threshold(vr)[1]; f_new = select_threshold(record(va, list(v)))[1]
                    if diff > G1_LOGIT_TOLERANCE or abs(f_new - f_old) > G1_F1_TOLERANCE:
                        raise ValueError(f'Validation logits not reproduced for {dataset} {name} seed {seed}: {diff:.2e}')
                    recs = [record(tr, list(score(model, kind, tr_arr))), vr, record(te, list(z['test'].astype(float)))]
                    hist = json.loads((job / 'history.json').read_text()); best = max(hist, key=lambda h: h['validation_AP'])
                    extra = dict(epochs=len(hist), best_epoch=best['epoch'], loss_first=hist[0]['mean_batch_loss'],
                                 loss_best=best['mean_batch_loss'], loss_last=hist[-1]['mean_batch_loss'],
                                 validation_AP_last=hist[-1]['validation_AP'])
                    for split, r in zip(SPLITS, recs):
                        rows.append(dict(dataset=dataset, method=name, kind='learned', seed=seed, parameters=params, split=split,
                                         **evaluate(r, vr, budget), **extra))
                    curves += [dict(dataset=dataset, method=name, seed=seed, **h) for h in hist]
                    print(f'  {name} seed {seed}: {params} parameters, best epoch {best["epoch"]}/{len(hist)}, validation check {diff:.1e}', flush=True)

    out.mkdir(parents=True)
    write_csv(out / 'per_run.csv', rows); write_csv(out / 'curves.csv', curves)
    summary = []
    groups = {}
    for r in rows: groups.setdefault((r['dataset'], r['method']), []).append(r)
    for (dataset, method), rs in groups.items():
        row = dict(dataset=dataset, method=method, kind=rs[0]['kind'], seeds=len({r['seed'] for r in rs}),
                   parameters=rs[0]['parameters'])
        for split in SPLITS:
            sub = [r for r in rs if r['split'] == split]
            for k in ['F1', 'edges_right', 'changes_recovered', 'AP', 'predicted_edges_at_threshold']:
                row[f'{k}_{split}'] = float(np.mean([r[k] for r in sub]))
        row['F1_gap_train_minus_validation'] = row['F1_train'] - row['F1_validation']
        row['AP_gap_train_minus_validation'] = row['AP_train'] - row['AP_validation']
        if rs[0]['kind'] == 'learned':
            for k in ['best_epoch', 'epochs', 'loss_first', 'loss_best', 'loss_last', 'validation_AP_last']:
                row[k] = float(np.mean([r[k] for r in rs if r['split'] == 'train']))
        summary.append(row)
    write_csv(out / 'summary.csv', summary)
    (out / 'capacity_check.json').write_text(json.dumps(dict(datasets=info, summary=summary, seeds=args.seeds, run_folders=runs,
        source_hashes={p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
        python=platform.python_version(), torch=torch.__version__, device=args.device, seconds=round(time.time() - t0, 1)), indent=2))
    for r in summary:
        print(f"{r['dataset']:7s} {r['method']:20s} F1 train {r['F1_train']:.3f} val {r['F1_validation']:.3f} test {r['F1_test']:.3f} | "
              f"AP {r['AP_train']:.3f} {r['AP_validation']:.3f} {r['AP_test']:.3f} | right {r['edges_right_train']:.2f} "
              f"{r['edges_right_validation']:.2f} {r['edges_right_test']:.2f}" + (f" | best epoch {r['best_epoch']:.0f}/{r['epochs']:.0f}" if r['kind'] == 'learned' else ''))
    print(f'Saved {out.resolve()} ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--datasets', nargs='+', default=['unsw', 'deakin']); p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--device', default='cuda')
    p.add_argument('--runs', nargs='+', help='dataset=folder[,folder] relative to the project root, replacing the default run folders')
    main(p.parse_args())
