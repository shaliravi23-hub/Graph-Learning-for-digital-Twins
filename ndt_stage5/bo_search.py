"""Stage 9: Bayesian optimisation of model size and training hyperparameters (preregistration_stage9_optimisation.md).

Two steps, both resumable after an interruption:
  search  For each dataset and model: an Optuna study (multivariate TPE, fixed sampler seed) of --trials trials, each a
          full training run with seed 42. The first trial is the published setting. Every epoch reports the validation
          AP, and the median pruner stops trials that are clearly worse. A trial's value is the validation F1 at the
          exact validation threshold of its best-AP checkpoint, the rule used to choose models throughout the study.
          Studies are stored in SQLite, so a rerun continues where it stopped.
  final   Retrains the best trial's setting of every model with seeds 42-46 and writes run folders in the same format as
          seeds_unsw (protocol.json, best.pt, scores.npz, history.json, selection.json, completed.json), so the existing
          analysis and new-day scoring scripts read them unchanged.
Training is that of train_seeds.py (same loss, data order, checkpoint rule and determinism settings); with the
published setting and the published epoch count, train_run reproduces train_seeds.py exactly (checked by --self-test).
Only these change: width (all layer widths, rounded to multiples of 4), transformer depth (TGAE family), dropout,
learning rate, weight decay, batch size, positive-class weight, and training length (up to 120 epochs, stopping when
validation AP has not improved for 20 epochs).
Run from ndt_stage5:
  python bo_search.py --project-root .. --plan preregistration_stage9_optimisation.md --dataset unsw --step search --out ..\\stage9_bo
  python bo_search.py --project-root .. --plan preregistration_stage9_optimisation.md --dataset unsw --step final --out ..\\stage9_bo
The plan's SHA-256 is stored with each study when it is created; a later run with a different plan file stops.
--sampler random replaces the TPE sampler by plain random search (Optuna RandomSampler, seed 20261120), with everything
else unchanged (search space, published first trial, pruner, trial value, final runs); used for the optimiser check.
"""
import argparse, csv, importlib.util, inspect, json, platform, random, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.special import expit
import train_seeds as ts
from evaluate_pilot import sha, nonself, select_threshold

MODELS = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bilinear']
SEEDS = [42, 43, 44, 45, 46]
MAX_EPOCHS = 120; PATIENCE = 20
SAMPLER_SEED = 20261119
RANDOM_SEARCH_SEED = 20261120   # a different seed, so the random search shares no proposals with the TPE start-up trials
WIDTH_RANGE = {'dyngem': (0.125, 2.0), 'aernn': (0.125, 2.0)}   # others (0.25, 4.0)
DEPTH_MODELS = {'tgae', 'tgae_bilinear'}


def width_scaled(kwargs, width):
    out = dict(kwargs)
    for k in ts.WIDTH_KEYS:
        if k in out:
            f = lambda v: max(4, 4 * int(round(v * width / 4)))
            out[k] = tuple(f(v) for v in out[k]) if isinstance(out[k], (tuple, list)) else f(out[k])
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def published(name, w_orig, for_search=True):
    cfg = ts.MODELS[name]; kw = cfg['kwargs']
    # In the search, a published weight decay of 0 is entered as 1e-8, the lower end of the log-scale range.
    wd = max(cfg['weight_decay'], 1e-8) if for_search else cfg['weight_decay']
    hp = dict(width=1.0, learning_rate=1e-3, weight_decay=wd, batch_size=cfg['batch'], class_weight_power=1.0)
    if 'dropout' in kw: hp['dropout'] = kw['dropout']
    if name in DEPTH_MODELS: hp['num_layers'] = kw['num_layers']
    return hp


def suggest(trial, name):
    cfg = ts.MODELS[name]; lo, hi = WIDTH_RANGE.get(name, (0.25, 4.0))
    hp = dict(width=trial.suggest_float('width', lo, hi, log=True),
              learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
              weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
              batch_size=trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
              class_weight_power=trial.suggest_float('class_weight_power', 0.0, 1.0))
    if 'dropout' in cfg['kwargs']: hp['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    if name in DEPTH_MODELS: hp['num_layers'] = trial.suggest_int('num_layers', 1, 4)
    return hp


class Setup:
    """Data, tensors and model classes for one dataset, shared by all trials."""
    def __init__(self, root, dataset, device):
        import torch
        self.torch = torch; self.root = root; self.dataset = dataset; self.device = device
        self.tr, self.va, self.te, self.audit = ts.data_setup(root, dataset)
        self.n = len(self.tr[0]['A_target']); self.mask = nonself(self.n); self.H = ts.L
        self.w_orig = float(self.audit['positive_class_weight'])
        self.budget = int(np.rint(np.mean([i['A_target'].sum() for i in self.va])))
        def tensors(items):
            return tuple(torch.tensor(np.stack(v), dtype=torch.float32) for v in [
                [np.asarray(i['A_seq'][-self.H:], np.float32) for i in items], [np.asarray(i['X_seq'][-self.H:], np.float32) for i in items],
                [np.asarray(i['A_target'], np.float32) for i in items], [np.asarray(i['X_target'], np.float32) for i in items]])
        self.arrays, self.v_arrays, self.t_arrays = tensors(self.tr), tensors(self.va), tensors(self.te)
        self.modules = {}

    def model_class(self, name):
        if name not in self.modules:
            cfg = ts.MODELS[name]; source = self.root / cfg['script']
            spec = importlib.util.spec_from_file_location('stage9_' + name, source)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.modules[name] = (module, getattr(module, cfg['cls']), sha(source))
        return self.modules[name]

    def constructor(self, name, hp):
        cfg = ts.MODELS[name]; kw = width_scaled(cfg['kwargs'], hp['width'])
        if 'dropout' in hp: kw['dropout'] = hp['dropout']
        if 'num_layers' in hp: kw['num_layers'] = hp['num_layers']
        return kw


def logits(model, kind, a, x):
    if kind == 'last': return model(a[:, -1], x[:, -1])['A_logits']
    if kind == 'variational': return model(a, x, use_posterior=False)['A_logits']
    if kind == 'sequence': return model(a, x)['A_logits']
    if kind == 'static_prev': return model.forward_deterministic(x[:, -1])[0]
    raise ValueError(kind)


def score(S, model, kind, arr):
    torch = S.torch; model.eval(); ss = []
    with torch.inference_mode():
        for j in range(0, len(arr[0]), 16):
            ss.extend(logits(model, kind, arr[0][j:j + 16].to(S.device), arr[1][j:j + 16].to(S.device)).float().cpu().numpy())
    ss = np.stack(ss)
    if not np.isfinite(ss).all(): raise ValueError('Nonfinite logits')
    return ss


def train_run(S, name, hp, seed, epochs=MAX_EPOCHS, patience=PATIENCE, trial=None):
    """One training run. Returns the best-AP state, its validation logits, the history and the parameter count."""
    torch = S.torch; F = torch.nn.functional
    cfg = ts.MODELS[name]; module, cls, source_sha = S.model_class(name)
    kw = dict(S.constructor(name, hp), num_nodes=S.n, feat_dim=10, in_dim=10, seq_len=S.H)
    kw = {k: v for k, v in kw.items() if k in inspect.signature(cls.__init__).parameters}
    if cfg.get('decoder_variant'): cls = ts.with_decoder(cls, cfg['decoder_variant'], S.constructor(name, hp)['latent_dim'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = cls(**kw).to(S.device); initial = ts.state_hash(model.state_dict())
    params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*S.arrays), batch_size=hp['batch_size'], shuffle=True,
                                         generator=torch.Generator().manual_seed(seed), num_workers=0)
    w = torch.tensor(S.w_orig ** hp['class_weight_power'], dtype=torch.float32, device=S.device)

    def loss_fn(a, x, target, xt):
        kind = cfg['kind']
        if kind == 'static_prev':
            out, mu, logvar = model(x[:, -1])
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

    history = []; best = -np.inf; best_epoch = None; best_state = None; since = 0
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for a, x, target, xt in loader:
            a, x, target, xt = (v.to(S.device) for v in [a, x, target, xt]); optimizer.zero_grad()
            loss = loss_fn(a, x, target, xt)
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite training loss ({name}, seed {seed}, epoch {epoch})')
            loss.backward()
            if cfg['grad_clip']: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        v = score(S, model, cfg['kind'], S.v_arrays)
        ap = float(np.mean([average_precision_score(i['A_target'][S.mask], expit(s[S.mask])) for i, s in zip(S.va, v) if i['A_target'].sum()]))
        if not np.isfinite(ap): raise ValueError('Nonfinite validation AP')
        history.append(dict(epoch=epoch, mean_batch_loss=float(np.mean(losses)), validation_AP=ap))
        if ap > best:
            best, best_epoch, since = ap, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
        if trial is not None:
            import optuna
            trial.report(ap, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
        if since >= patience: break
    model.load_state_dict(best_state, strict=True)
    v = score(S, model, cfg['kind'], S.v_arrays)
    th, vf, ve = select_threshold([dict(truth=i['A_target'], score=s) for i, s in zip(S.va, v)])
    return dict(model=model, state=best_state, validation=v, threshold=th, validation_F1=vf, validation_edge_error=ve,
                best_epoch=best_epoch, validation_AP=best, history=history, parameters=params, initial_state_hash=initial,
                constructor=S.constructor(name, hp), source_sha256=source_sha, positive_class_weight=float(w.item()))


def study_for(out, dataset, name, w_orig, plan_sha, sampler_name='tpe'):
    import optuna
    storage = f"sqlite:///{(out / dataset / 'studies' / f'{name}.db').resolve().as_posix()}"
    # After an interruption, the sampler seed is offset by the number of stored trials, so that a resumed study does not
    # repeat the random proposals it has already made; an uninterrupted study uses SAMPLER_SEED as is.
    existing = 0
    if f'{dataset}_{name}' in optuna.get_all_study_names(storage):
        existing = len(optuna.load_study(study_name=f'{dataset}_{name}', storage=storage).trials)
    if sampler_name == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED + existing, multivariate=True, n_startup_trials=10)
    else:
        sampler = optuna.samplers.RandomSampler(seed=RANDOM_SEARCH_SEED + existing)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10)
    study = optuna.create_study(study_name=f'{dataset}_{name}', storage=storage, direction='maximize', sampler=sampler,
                                pruner=pruner, load_if_exists=True)
    if len(study.trials) == 0:
        study.set_user_attr('analysis_plan_sha256', plan_sha); study.set_user_attr('sampler', sampler_name)
        study.enqueue_trial(published(name, w_orig))
    elif study.user_attrs.get('analysis_plan_sha256') != plan_sha:
        raise ValueError(f'{dataset} {name}: the study was started under another analysis plan')
    elif study.user_attrs.get('sampler', 'tpe') != sampler_name:
        raise ValueError(f'{dataset} {name}: the study was started with another sampler')
    return study


def search(args, S, out):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (out / args.dataset / 'studies').mkdir(parents=True, exist_ok=True)
    for name in args.models:
        study = study_for(out, args.dataset, name, S.w_orig, args.plan_sha, args.sampler)
        done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
        for t in study.trials:   # a trial cut off by an interruption is closed as failed and does not count
            if t.state == optuna.trial.TrialState.RUNNING: study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        remaining = args.trials - len(done)
        print(f'{args.dataset} {name}: {len(done)} trials done, {max(remaining, 0)} to run', flush=True)

        def objective(trial):
            hp = suggest(trial, name); t0 = time.time()
            r = train_run(S, name, hp, 42, epochs=args.max_epochs, trial=trial)
            for k in ['best_epoch', 'validation_AP', 'parameters']: trial.set_user_attr(k, r[k])
            trial.set_user_attr('epochs_run', len(r['history'])); trial.set_user_attr('seconds', round(time.time() - t0, 1))
            print(f'  {name} trial {trial.number}: validation F1 {r["validation_F1"]:.4f}, AP {r["validation_AP"]:.4f}, '
                  f'{r["parameters"]} parameters, best epoch {r["best_epoch"]}/{len(r["history"])}, {time.time() - t0:.0f} s', flush=True)
            return r['validation_F1']
        if remaining > 0: study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        rows = [dict(dataset=args.dataset, model=name, trial=t.number, state=t.state.name, validation_F1=t.value, **t.params, **t.user_attrs)
                for t in study.trials]
        if rows: write_csv(out / args.dataset / f'trials_{name}.csv', rows)
        best = study.best_trial
        print(f'{args.dataset} {name}: best trial {best.number}, validation F1 {best.value:.4f}, {best.params}', flush=True)


def final(args, S, out):
    import optuna, torch
    for name in args.models:
        study = study_for(out, args.dataset, name, S.w_orig, args.plan_sha, args.sampler)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(complete) + len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) < args.trials:
            raise ValueError(f'{name}: the search has not finished ({len(study.trials)} trials)')
        best = study.best_trial; hp = dict(published(name, S.w_orig)); hp.update(best.params)
        mdir = out / args.dataset / 'final' / name; mdir.mkdir(parents=True, exist_ok=True)
        cfg = ts.MODELS[name]
        protocol = dict(model=name, dataset=args.dataset, history=S.H, input=cfg['kind'], hyperparameters=hp, sampler=args.sampler,
                        selected_trial=best.number, selected_validation_F1_seed42=best.value, constructor=S.constructor(name, hp),
                        max_epochs=MAX_EPOCHS, patience=PATIENCE, alpha_x=cfg['alpha_x'], beta_kl=cfg['beta_kl'],
                        gradient_clip=cfg['grad_clip'], positive_class_weight=S.w_orig ** hp['class_weight_power'],
                        checkpoint_selection='Highest macro validation sigmoid-score AP; earlier epoch on ties',
                        source_sha256=S.model_class(name)[2], training_runner_sha256=sha(__file__),
                        analysis_plan=dict(file=Path(args.plan).name, sha256=args.plan_sha),
                        **({'decoder_variant': cfg['decoder_variant']} if cfg.get('decoder_variant') else {}),
                        device=args.device, torch=torch.__version__, python=platform.python_version(),
                        input_hashes=S.audit['input_hashes'])
        if (mdir / 'protocol.json').exists():
            saved = json.loads((mdir / 'protocol.json').read_text())
            if saved['hyperparameters'] != hp: raise ValueError(f'{mdir} holds runs of another setting')
        else:
            (mdir / 'protocol.json').write_text(json.dumps(protocol, indent=2, default=str))
        for seed in SEEDS:
            job = mdir / f'seed{seed}'
            if (job / 'completed.json').exists(): continue
            job.mkdir(parents=True, exist_ok=True); t0 = time.time()
            r = train_run(S, name, hp, seed, epochs=args.max_epochs)
            torch.save(r['state'], job / 'best.pt')
            t = score(S, r['model'], cfg['kind'], S.t_arrays)
            np.savez_compressed(job / 'scores.npz', validation=r['validation'].astype(np.float32), test=t.astype(np.float32),
                                validation_dates=np.array([i['date'] for i in S.va]), validation_windows=np.array([int(i['target_window_id']) for i in S.va]),
                                test_dates=np.array([i['date'] for i in S.te]), test_windows=np.array([int(i['target_window_id']) for i in S.te]))
            (job / 'history.json').write_text(json.dumps(r['history'], indent=2))
            (job / 'selection.json').write_text(json.dumps(dict(logit_threshold=r['threshold'] if np.isfinite(r['threshold']) else 'inf',
                sigmoid_threshold=float(expit(r['threshold'])), validation_F1=r['validation_F1'], validation_edge_error=r['validation_edge_error'],
                k=S.budget, positive_class_weight=r['positive_class_weight']), indent=2))
            (job / 'completed.json').write_text(json.dumps(dict(model=name, dataset=args.dataset, seed=seed, smoke=False,
                best_epoch=r['best_epoch'], epochs_run=len(r['history']), parameters=r['parameters'],
                checkpoint_sha256=sha(job / 'best.pt'), initial_state_hash=r['initial_state_hash'], training_runner_sha256=sha(__file__),
                seconds=round(time.time() - t0, 1)), indent=2))
            print(f'FINAL {args.dataset} {name} seed {seed}: validation F1 {r["validation_F1"]:.4f}, best epoch {r["best_epoch"]}, '
                  f'{time.time() - t0:.0f} s', flush=True)


def self_test(args, S):
    """Published setting and epoch count, seed 42, 2 epochs: the state after training equals train_seeds.py's."""
    import torch, subprocess, sys, tempfile
    name = args.models[0]; cfg = ts.MODELS[name]
    r = train_run(S, name, published(name, S.w_orig, for_search=False), 42, epochs=2, patience=10 ** 6)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(Path(ts.__file__)), '--project-root', str(S.root), '--dataset', args.dataset, '--models', name,
                        '--out', tmp, '--device', args.device, '--seeds', '42', '--epochs', '2'], check=True, capture_output=True)
        ref = torch.load(Path(tmp) / name / 'seed42' / 'best.pt', map_location='cpu', weights_only=True)
    if ref.keys() != r['state'].keys(): raise ValueError('train_run builds a different model from train_seeds.py')
    diff = max(float((ref[k].float() - r['state'][k].float()).abs().max()) for k in ref)
    print(f'Self-test {name}: best epoch {r["best_epoch"]}, largest weight difference from train_seeds.py {diff:.2e}', flush=True)
    # Exact on the CPU; on a GPU, atomic operations may differ slightly between processes.
    if diff > (0 if args.device == 'cpu' else 1e-3): raise ValueError('train_run does not reproduce train_seeds.py')


def main(args):
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    for flag, value in [('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True)]:
        if hasattr(torch.backends.cuda, flag): getattr(torch.backends.cuda, flag)(value)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; use --device cpu')
    root = Path(args.project_root).resolve(); out = Path(args.out)
    args.plan_sha = sha(args.plan) if args.plan else None
    if args.step != 'self-test' and not args.plan: raise ValueError('--plan is required for search and final')
    S = Setup(root, args.dataset, args.device)
    print(f'{args.dataset}: {len(S.tr)} training, {len(S.va)} validation, {len(S.te)} test targets; '
          f'positive weight {S.w_orig:.1f}; budget {S.budget}', flush=True)
    if args.step == 'self-test': self_test(args, S); return
    if args.step in ('search', 'both'): search(args, S, out)
    if args.step in ('final', 'both'): final(args, S, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--out', required=True); p.add_argument('--plan')
    p.add_argument('--dataset', choices=['unsw', 'deakin'], required=True)
    p.add_argument('--step', choices=['search', 'final', 'both', 'self-test'], required=True)
    p.add_argument('--models', nargs='+', default=MODELS); p.add_argument('--trials', type=int, default=40)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS, help='Testing only: shorter training')
    p.add_argument('--sampler', choices=['tpe', 'random'], default='tpe')
    main(p.parse_args())
