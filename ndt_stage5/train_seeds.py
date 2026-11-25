"""Stage 5: multi-seed training of every learned model on UNSW or Deakin chronological.

Generalises ndt_stage4/train_controls.py from TGAE to all models. History length L=4 and the
original positive-class weight are fixed. Architectures and optimiser settings follow the
manuscript (Table 2) and the corrected training scripts; no architecture search.

Models
  dyngem, aernn, vgrnn, tgae, vrdag   historical-input predictors (never see the target window)
  static_prev                          Static VGAE given the PREVIOUS window's node features (fair predictor)
  static_target                        Static VGAE given the TARGET window's node features (reference only)

Decoder ablation (not in the default set; run them into their own --out folder)
  tgae_bias       TGAE-style encoder, decoder z_u.z_v + b         (removes the score floor at 0.5)
  tgae_bilinear   TGAE-style encoder, decoder z_u^T W z_v + b     (also lets u->v and v->u differ)
  Both start exactly as TGAE-style (b = 0, W = identity), so seed-matched runs differ only in the decoder.

Per model and seed this writes, under <out>/<model>/seed<k>/:
  best.pt, history.json, selection.json, scores.npz (validation and test logits with their keys),
  summary.csv, per_snapshot.csv, completed.json
and <out>/all_runs_summary.csv collects every completed run.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, hashlib, importlib.util, inspect, json, platform, random
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from horizon_support import validate_h, validation_items_h
from evaluate_pilot import load, sha, select_threshold, nonself, topk, reach, metrics, write_csv

L = 4
DATASETS = {
    'unsw': dict(path=('dataset_ready', '300s'), counts=(1420, 284, 568)),
    'deakin': dict(path=('dataset_ready_deakin_chronological', '300s'), counts=(568, 284, 284)),
    # Second chronological split (train 30 Mar, validate 31 Mar, test 4 Apr), built by make_deakin_fold2.py
    'deakin_fold2': dict(path=('dataset_ready_deakin_fold2', '300s'), counts=(284, 284, 284)),
    # Second chronological UNSW split (train 23-25 Sep, validate 26 Sep, test 27 Sep 2016), built by make_unsw_fold2.py
    'unsw_fold2': dict(path=('dataset_ready_unsw_fold2', '300s'), counts=(852, 284, 284)),
}
# Stage 7 window-length ablation: 600 s and 900 s windows made by merging consecutive 300 s windows (make_merged_windows.py)
for _base, _path, _days in [('unsw', 'dataset_ready', (5, 1, 2)), ('deakin', 'dataset_ready_deakin_chronological', (2, 1, 1)),
                            ('deakin_fold2', 'dataset_ready_deakin_fold2', (1, 1, 1)), ('unsw_fold2', 'dataset_ready_unsw_fold2', (3, 1, 1))]:
    for _sec, _per_day in [(600, 140), (900, 92)]:
        DATASETS[f'{_base}_{_sec}s'] = dict(path=(_path, f'{_sec}s'), counts=tuple(d * _per_day for d in _days))
# Stage 7C prediction-horizon ablation (make_horizon_dataset.py): targets h windows after the history; 285 - h targets per day
for _base, _path, _days in [('unsw', 'dataset_ready', (5, 1, 2)), ('deakin', 'dataset_ready_deakin_chronological', (2, 1, 1)),
                            ('deakin_fold2', 'dataset_ready_deakin_fold2', (1, 1, 1)), ('unsw_fold2', 'dataset_ready_unsw_fold2', (3, 1, 1))]:
    for _h in (2, 4):
        DATASETS[f'{_base}_h{_h}'] = dict(path=(_path, f'300s_h{_h}'), counts=tuple(d * (285 - _h) for d in _days), horizon=_h)
# kind: how the model is called. lr is 1e-3 for every model.
MODELS = {
    'dyngem': dict(script='train_dyngem_style_predictor_corrected.py', cls='DynGEMStylePredictor',
                   kwargs=dict(hidden_dims=(768, 384), latent_dim=128, dropout=.1),
                   kind='last', epochs=50, batch=32, weight_decay=1e-5, grad_clip=5., alpha_x=.05, beta_kl=0.),
    'aernn': dict(script='train_dyngraph2vec_aernn_style_corrected.py', cls='Dyngraph2VecAERNNStyle',
                  kwargs=dict(enc_dim=512, embed_dim=256, rnn_hidden=256, dropout=0.),
                  kind='sequence', epochs=50, batch=32, weight_decay=1e-5, grad_clip=5., alpha_x=.05, beta_kl=0.),
    'vgrnn': dict(script='train_vgrnn_predictor_corrected.py', cls='VGRNNPredictor',
                  kwargs=dict(hidden_dim=128, latent_dim=32, dropout=.1),
                  kind='variational', epochs=60, batch=16, weight_decay=0., grad_clip=5., alpha_x=.05, beta_kl=1e-4),
    'tgae': dict(script='train_tgae_style_predictor_corrected.py', cls='TGAEStylePredictor',
                 kwargs=dict(hidden_dim=128, latent_dim=32, num_heads=4, num_layers=2, dropout=.1),
                 kind='sequence', epochs=60, batch=16, weight_decay=1e-5, grad_clip=5., alpha_x=.05, beta_kl=0.),
    'vrdag': dict(script='train_vrdag_style_predictor_corrected.py', cls='VRDAGStylePredictor',
                  kwargs=dict(hidden_dim=64, latent_dim=32, dropout=.1),
                  kind='variational', epochs=60, batch=16, weight_decay=1e-5, grad_clip=5., alpha_x=.05, beta_kl=1e-4),
    'static_prev': dict(script='train_static_vgae_updated.py', cls='VGAE',
                        kwargs=dict(hidden_dim=64, latent_dim=32, dropout=0.),
                        kind='static_prev', epochs=50, batch=32, weight_decay=0., grad_clip=None, alpha_x=0., beta_kl=1e-4),
    'static_target': dict(script='train_static_vgae_updated.py', cls='VGAE',
                          kwargs=dict(hidden_dim=64, latent_dim=32, dropout=0.),
                          kind='static_target', epochs=50, batch=32, weight_decay=0., grad_clip=None, alpha_x=0., beta_kl=1e-4),
}
# Recorded, but a change in these does not alter training, so it does not block --resume.
IGNORED_ON_RESUME = {'training_runner_sha256', 'self_loops_removed'}
DEFAULT_MODELS = list(MODELS)
MODELS['tgae_bias'] = dict(MODELS['tgae'], decoder_variant='bias')
MODELS['tgae_bilinear'] = dict(MODELS['tgae'], decoder_variant='bilinear')
WIDTH_KEYS = ('hidden_dims', 'hidden_dim', 'latent_dim', 'enc_dim', 'embed_dim', 'rnn_hidden')


def scaled(kwargs, width):
    """Constructor arguments with every layer width multiplied by `width` (stage 7 tuning grid)."""
    out = dict(kwargs)
    for k in WIDTH_KEYS:
        if k in out:
            out[k] = tuple(int(round(v * width)) for v in out[k]) if isinstance(out[k], (tuple, list)) else int(round(out[k] * width))
    return out


DECODERS = ['validation_threshold', 'validation_topk', 'previous_topk', 'raw_0.5', 'weight_corrected_0.5']


def cohort_id(items):
    return hashlib.sha256(json.dumps([(i['date'], int(i['target_window_id'])) for i in items], separators=(',', ':')).encode()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def with_decoder(base, variant, latent_dim):
    """TGAE-style encoder with a different edge decoder. At initialisation (b = 0, W = identity) the logits equal the
    original inner product, and the base parameters are drawn exactly as for TGAE-style with the same seed."""
    import torch
    from torch import nn

    class DecoderVariant(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.edge_bias = nn.Parameter(torch.zeros(()))
            if variant == 'bilinear':
                self.edge_weight = nn.Parameter(torch.eye(latent_dim))

        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            z = out['Z']
            if variant == 'bilinear':
                logits = z @ self.edge_weight @ z.transpose(-1, -2)
            else:
                logits = z @ z.transpose(-1, -2)
            out['A_logits'] = logits + self.edge_bias
            return out

    DecoderVariant.__name__ = f'{base.__name__}_{variant}'
    return DecoderVariant


def drop_self_loops(train, val, test):
    """Zero adjacency diagonals in place, as the original training scripts did on load
    (np.fill_diagonal(..., 0)) and as the evaluation does (same-node pairs excluded).
    Returns the number of self-loop entries removed per split (test: target windows only)."""
    removed = {}
    for split, data, keys in [('train', train, ['A', 'W']), ('val', val, ['A', 'W']),
                              ('test', test, ['A_target', 'W_target', 'A_seq', 'W_seq'])]:
        count = 0
        for item in data['items']:
            for k in keys:
                if k not in item: continue
                a = np.array(item[k], copy=True)
                if k in ('A', 'A_target'):
                    bad = ~np.isin(a, [0, 1])
                    if bad.any():
                        raise ValueError(f'{split} {k} has {int(bad.sum())} non-binary entries, e.g. {np.unique(a[bad])[:5]}')
                    count += int(np.count_nonzero(np.diagonal(a)))
                if a.ndim == 2: np.fill_diagonal(a, 0)
                else:
                    for t in range(a.shape[0]): np.fill_diagonal(a[t], 0)
                item[k] = a
        removed[split] = count
    return removed


def data_setup(root, dataset):
    ds = root.joinpath(*DATASETS[dataset]['path'])
    train = load(ds / 'static_train.pkl'); val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    self_loops = drop_self_loops(train, val, test)
    h = DATASETS[dataset].get('horizon', 1)
    audit = validate_h(train, val, test, meta, h); audit['self_loops_removed'] = self_loops
    tr = validation_items_h(train, L, h); va = validation_items_h(val, L, h)
    te = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    counts = (len(tr), len(va), len(te))
    if meta['seq_len'] != L or counts != DATASETS[dataset]['counts']:
        raise ValueError(f'Unexpected {dataset} aligned cohorts {counts}; expected {DATASETS[dataset]["counts"]} with L={L}')
    for i in va + te:
        if i['X_seq'].shape != (L, len(train['node_list']), 10) or not np.isfinite(i['X_seq']).all():
            raise ValueError('Invalid historical features')
    pos = sum(i['A_target'].sum(dtype=float) for i in tr); total = sum(i['A_target'].size for i in tr)
    audit.update(dataset=dataset, dataset_dir=str(ds.relative_to(root)), train_targets=len(tr), val_targets=len(va),
                 test_targets=len(te), positive_class_weight=(total - pos) / pos,
                 cohort_hashes={k: cohort_id(v) for k, v in [('train', tr), ('val', va), ('test', te)]},
                 input_hashes={k: sha(ds / k) for k in ['static_train.pkl', 'static_val.pkl', 'temporal_test.pkl', 'metadata.json']})
    return tr, va, te, audit


def main(args):
    root = Path(args.project_root).resolve(); out = Path(args.out)
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise ValueError('Unknown models ' + str(unknown) + '; choose from ' + str(list(MODELS)))
    for m in args.models:
        if not (root / MODELS[m]['script']).exists():
            raise FileNotFoundError(root / MODELS[m]['script'])
    tr, va, te, audit = data_setup(root, args.dataset)
    print(json.dumps({k: audit[k] for k in ['dataset', 'dataset_dir', 'nodes', 'train_targets', 'val_targets', 'test_targets',
                                            'test_captures', 'positive_class_weight', 'self_loops_removed']}, indent=2), flush=True)
    if args.preflight_only:
        print('Preflight passed; no training performed.'); return

    import torch
    import torch.nn.functional as F
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu explicitly')
    # warn_only: an op without a deterministic CUDA kernel warns instead of stopping a long batch.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    for flag, value in [('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True)]:
        if hasattr(torch.backends.cuda, flag): getattr(torch.backends.cuda, flag)(value)
    if args.smoke:
        tr, va, te = tr[:32], va[:16], te[:16]
    seeds = args.seeds[:1] if args.smoke else args.seeds
    n = len(tr[0]['A_target']); mask = nonself(n)
    budget = int(np.rint(np.mean([i['A_target'].sum() for i in va])))
    weight = float(audit['positive_class_weight']) if args.class_weight == 'original' else 1.0
    H = args.history

    def tensors(items):
        return tuple(torch.tensor(np.stack(v), dtype=torch.float32) for v in [
            [np.asarray(i['A_seq'][-H:], np.float32) for i in items], [np.asarray(i['X_seq'][-H:], np.float32) for i in items],
            [np.asarray(i['A_target'], np.float32) for i in items], [np.asarray(i['X_target'], np.float32) for i in items]])
    arrays, v_arrays, t_arrays = tensors(tr), tensors(va), tensors(te)
    refs = [reach(i['A_target']) for i in te]
    w = torch.tensor(weight, dtype=torch.float32, device=args.device)

    def logits(model, kind, a, x, xt):
        # Target-window arrays reach the model only for the static_target reference.
        if kind == 'last': return model(a[:, -1], x[:, -1])['A_logits']
        if kind == 'variational': return model(a, x, use_posterior=False)['A_logits']
        if kind == 'sequence': return model(a, x)['A_logits']
        if kind == 'static_prev': return model.forward_deterministic(x[:, -1])[0]
        if kind == 'static_target': return model.forward_deterministic(xt)[0]
        raise ValueError(kind)

    def scores(model, kind, arr, items):
        model.eval(); ss = []
        with torch.inference_mode():
            for j in range(0, len(items), 16):
                a, x = arr[0][j:j + 16].to(args.device), arr[1][j:j + 16].to(args.device)
                xt = arr[3][j:j + 16].to(args.device) if kind == 'static_target' else None
                ss.extend(logits(model, kind, a, x, xt).float().cpu().numpy())
        if not all(np.isfinite(s).all() for s in ss): raise ValueError('Nonfinite logits')
        return [dict(date=i['date'], window=int(i['target_window_id']), truth=i['A_target'].copy(),
                     previous=i['A_seq'][-1].copy(), score=s) for i, s in zip(items, ss)]

    def loss_fn(model, cfg, module, a, x, target, xt):
        kind = cfg['kind']
        if kind in ('static_prev', 'static_target'):
            out, mu, logvar = model(xt if kind == 'static_target' else x[:, -1])
            return F.binary_cross_entropy_with_logits(out, target, pos_weight=w) + cfg['beta_kl'] * module.kl_loss(mu, logvar)
        if kind == 'last': pred = model(a[:, -1], x[:, -1])
        elif kind == 'variational': pred = model(a, x, A_target=target, X_target=xt, use_posterior=True)
        else: pred = model(a, x)
        loss = F.binary_cross_entropy_with_logits(pred['A_logits'], target, pos_weight=w)
        activity = (xt.abs().sum(-1, keepdim=True) > 0).float()
        loss = loss + cfg['alpha_x'] * (((pred['X_pred'] - xt) ** 2) * activity).sum() / activity.sum().clamp(min=1.)
        if kind == 'variational':
            loss = loss + cfg['beta_kl'] * module.kl_gaussian(pred['post_mu'], pred['post_logvar'], pred['prior_mu'], pred['prior_logvar'])
        return loss

    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        cfg = dict(MODELS[name]); cfg['kwargs'] = scaled(cfg['kwargs'], args.width); source = root / cfg['script']; mdir = out / name
        epochs = 1 if args.smoke else (args.epochs or cfg['epochs'])
        protocol = dict(model=name, history=H, weighting=args.class_weight, epochs=epochs,
                        batch_size=cfg['batch'], learning_rate=args.learning_rate, weight_decay=cfg['weight_decay'], alpha_x=cfg['alpha_x'],
                        beta_kl=cfg['beta_kl'], gradient_clip=cfg['grad_clip'], constructor=cfg['kwargs'], input=cfg['kind'],
                        role='target-information reference (not a predictor)' if name == 'static_target' else 'historical-input predictor',
                        **({'decoder_variant': cfg['decoder_variant']} if cfg.get('decoder_variant') else {}),
                        loss_mask='All N*N adjacency entries; feature-loss activity mask as in the corrected scripts',
                        checkpoint_selection='Highest macro validation sigmoid-score AP; earlier epoch on ties',
                        threshold_selection='Exact aligned validation macro-F1 boundaries; frozen for test',
                        determinism='torch.use_deterministic_algorithms(True, warn_only=True)',
                        source_sha256=sha(source), training_runner_sha256=sha(__file__),
                        metric_module_sha256=sha(Path(__file__).parent / 'evaluate_pilot.py'),
                        device=args.device, smoke=args.smoke, torch=torch.__version__, numpy=np.__version__,
                        python=platform.python_version(),
                        accelerator=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU', **audit)
        protocol = json.loads(json.dumps(protocol, default=str))
        if (mdir / 'protocol.json').exists():
            if not args.resume:
                raise ValueError(f'{mdir} exists; add --resume to continue the same protocol, or choose a new --out')
            saved = json.loads((mdir / 'protocol.json').read_text())
            if {k: v for k, v in saved.items() if k not in IGNORED_ON_RESUME} != {k: v for k, v in protocol.items() if k not in IGNORED_ON_RESUME}:
                raise ValueError(f'Resume protocol mismatch for {name}; choose a new --out')
        else:
            mdir.mkdir(parents=True, exist_ok=True); (mdir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        spec = importlib.util.spec_from_file_location('stage5_' + name, source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cls = getattr(module, cfg['cls'])
        kw = dict(cfg['kwargs'], num_nodes=n, feat_dim=10, in_dim=10, seq_len=H)
        kw = {k: v for k, v in kw.items() if k in inspect.signature(cls.__init__).parameters}
        if cfg.get('decoder_variant'):  # wrap after filtering: the wrapper's signature is (*args, **kwargs)
            cls = with_decoder(cls, cfg['decoder_variant'], cfg['kwargs']['latent_dim'])
        for seed in seeds:
            job = mdir / f'seed{seed}'
            if (job / 'completed.json').exists():
                print('Resume: already completed', name, 'seed', seed, flush=True); continue
            job.mkdir(exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            model = cls(**kw).to(args.device); initial = state_hash(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=cfg['weight_decay'])
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*arrays), batch_size=cfg['batch'], shuffle=True,
                                                 generator=torch.Generator().manual_seed(seed), num_workers=0)
            history = []; best = -np.inf; best_epoch = None
            for epoch in range(1, epochs + 1):
                model.train(); losses = []
                for a, x, target, xt in loader:
                    a, x, target, xt = (v.to(args.device) for v in [a, x, target, xt]); optimizer.zero_grad()
                    loss = loss_fn(model, cfg, module, a, x, target, xt)
                    if not torch.isfinite(loss): raise ValueError(f'Nonfinite training loss ({name}, seed {seed}, epoch {epoch})')
                    loss.backward()
                    if cfg['grad_clip']: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    optimizer.step(); losses.append(float(loss.detach().cpu()))
                vr = scores(model, cfg['kind'], v_arrays, va)
                ap = float(np.mean([average_precision_score(r['truth'][mask], expit(r['score'][mask])) for r in vr if r['truth'].sum()]))
                if not np.isfinite(ap): raise ValueError('Nonfinite validation AP')
                history.append(dict(epoch=epoch, mean_batch_loss=float(np.mean(losses)), validation_AP=ap))
                if ap > best:
                    best, best_epoch = ap, epoch; torch.save(model.state_dict(), job / 'best.pt')
                (job / 'history.json').write_text(json.dumps(history, indent=2))
                print(f'{args.dataset} {name} seed{seed} epoch {epoch} val AP {ap:.6f}', flush=True)
            model.load_state_dict(torch.load(job / 'best.pt', map_location=args.device, weights_only=True), strict=True)
            vr = scores(model, cfg['kind'], v_arrays, va); threshold, vf, ve = select_threshold(vr)
            # Test windows are scored only after checkpoint and decoder selection.
            tr_rec = scores(model, cfg['kind'], t_arrays, te); details = []; summary = []
            for decoder in DECODERS:
                group = []
                for r, rr in zip(tr_rec, refs):
                    s = r['score']
                    if decoder == 'validation_threshold': p = (s >= threshold) & mask
                    elif decoder == 'raw_0.5': p = (s >= 0) & mask
                    elif decoder == 'weight_corrected_0.5': p = (s >= np.log(weight)) & mask
                    else: p = topk(s, budget if decoder == 'validation_topk' else r['previous'].sum())
                    row = dict(model=name, seed=seed, decoder=decoder, date=r['date'], window=r['window'],
                               AP=average_precision_score(r['truth'][mask], expit(s[mask])) if r['truth'].sum() else np.nan,
                               **metrics(r['truth'], p, r['previous'], rr))
                    details.append(row); group.append(row)
                row = dict(dataset=args.dataset, model=name, seed=seed, decoder=decoder, n=len(group), best_epoch=best_epoch,
                           validation_AP=best, validation_F1=vf)
                for k in group[0]:
                    if k in ['model', 'seed', 'decoder', 'date', 'window']: continue
                    v = np.array([g[k] for g in group], float); v = v[np.isfinite(v)]
                    row[k + '_mean'] = float(v.mean()) if len(v) else None; row[k + '_valid_n'] = len(v)
                summary.append(row)
            write_csv(job / 'per_snapshot.csv', details); write_csv(job / 'summary.csv', summary)
            (job / 'selection.json').write_text(json.dumps(dict(
                logit_threshold=threshold if np.isfinite(threshold) else 'inf', sigmoid_threshold=float(expit(threshold)),
                validation_F1=vf, validation_edge_error=ve, k=budget, positive_class_weight=weight,
                log_positive_class_weight=float(np.log(weight))), indent=2))
            np.savez_compressed(job / 'scores.npz',
                                validation=np.stack([r['score'] for r in vr]).astype(np.float32),
                                test=np.stack([r['score'] for r in tr_rec]).astype(np.float32),
                                validation_dates=np.array([r['date'] for r in vr]), validation_windows=np.array([r['window'] for r in vr]),
                                test_dates=np.array([r['date'] for r in tr_rec]), test_windows=np.array([r['window'] for r in tr_rec]))
            (job / 'completed.json').write_text(json.dumps(dict(
                model=name, dataset=args.dataset, seed=seed, smoke=args.smoke, best_epoch=best_epoch,
                checkpoint_sha256=sha(job / 'best.pt'), initial_state_hash=initial, training_runner_sha256=sha(__file__)), indent=2))
            vt = [s for s in summary if s['decoder'] == 'validation_threshold'][0]
            print(f'DONE {args.dataset} {name} seed{seed}: test AP {vt["AP_mean"]:.4f}, F1 {vt["edge_f1_mean"]:.4f} '
                  f'(validation threshold {float(expit(threshold)):.4f})', flush=True)
            del model, optimizer
    rows = []
    for f in sorted(out.glob('*/seed*/summary.csv')):
        with f.open(newline='', encoding='utf-8') as fh: rows.extend(csv.DictReader(fh))
    if rows: write_csv(out / 'all_runs_summary.csv', rows)
    print('Finished. Results in', out.resolve(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True, help='Folder that contains dataset_ready* and the training scripts')
    p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                   help='Default: the seven main models. Decoder ablation: tgae_bias tgae_bilinear')
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--epochs', type=int, default=None, help='Testing only: override every model\'s epoch count')
    p.add_argument('--smoke', action='store_true', help='1 epoch, first seed, 32/16/16 items: checks the pipeline only')
    p.add_argument('--preflight-only', action='store_true', help='Validate the data split and stop')
    p.add_argument('--resume', action='store_true', help='Skip completed runs in an existing --out')
    # Stage 7 options. The defaults reproduce the stage-5 protocol exactly.
    p.add_argument('--learning-rate', type=float, default=1e-3, help='Adam learning rate (default 1e-3)')
    p.add_argument('--width', type=float, default=1.0, help='Multiply every layer width by this factor (default 1)')
    p.add_argument('--history', type=int, choices=[1, 2, 3, 4], default=4, help='Use only the last H history windows (default 4)')
    p.add_argument('--class-weight', choices=['original', 'none'], default='original',
                   help='original: positive-class weight from training data; none: unweighted loss')
    main(p.parse_args())
