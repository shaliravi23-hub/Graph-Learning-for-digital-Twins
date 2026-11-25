"""Stage 8: score the new UNSW days with the existing main-split checkpoints (preregistration_stage8_newdays.md).

No model is trained or tuned. For every model and seed in the given run folders this
  1. rebuilds the model from its protocol.json and loads best.pt (its SHA-256 must match completed.json, and the model
     script must be the one it was trained with);
  2. re-scores the validation day (8 October 2016) and stops unless it reproduces the stored validation logits (guard G1:
     maximum absolute difference <= 0.01, and the same validation F1 at the exact validation threshold, to within 0.001);
  3. scores the new test days and writes <out>/<model>/seed<k>/scores.npz (stored validation logits, new test logits,
     window keys), selection.json (copied) and completed.json.
Run from ndt_stage5:
  python score_newdays.py --project-root .. --runs ..\\seeds_unsw ..\\seeds_unsw_decoder --out ..\\seeds_unsw_newdays
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, importlib.util, inspect, json, platform, shutil
from pathlib import Path
import numpy as np
import train_seeds as ts
from evaluate_pilot import load, sha, select_threshold
from evaluate_temporal import validation_items

L = 4; G1_LOGIT_TOLERANCE = 1e-2; G1_F1_TOLERANCE = 1e-3
DEFAULT_MODELS = ['dyngem', 'aernn', 'vgrnn', 'tgae', 'vrdag', 'static_prev', 'tgae_bilinear', 'tgae_bias']


def main(args):
    import torch
    root = Path(args.project_root).resolve(); ds = root / args.dataset_dir; out = Path(args.out)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; use --device cpu')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    for flag, value in [('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True)]:
        if hasattr(torch.backends.cuda, flag): getattr(torch.backends.cuda, flag)(value)

    meta = json.loads((ds / 'metadata.json').read_text(encoding='utf-8-sig'))
    val = load(ds / 'static_val.pkl'); test = load(ds / 'temporal_test.pkl')
    ts.drop_self_loops(dict(items=[]), val, test)
    vi = validation_items(val, L); ti = sorted(test['items'], key=lambda i: (i['date'], i['target_window_id']))
    if {i['date'] for i in ti} != set(meta['test_dates']): raise ValueError('Test dates differ from metadata')
    n = len(val['node_list'])
    keys_t = [(i['date'], int(i['target_window_id'])) for i in ti]
    print(f'Validation targets {len(vi)}, new-day test targets {len(ti)} on {len(meta["test_dates"])} days, {n} nodes', flush=True)

    def arrays(items, H):
        return (torch.tensor(np.stack([np.asarray(i['A_seq'][-H:], np.float32) for i in items])),
                torch.tensor(np.stack([np.asarray(i['X_seq'][-H:], np.float32) for i in items])))

    def logits(model, kind, a, x):
        if kind == 'last': return model(a[:, -1], x[:, -1])['A_logits']
        if kind == 'variational': return model(a, x, use_posterior=False)['A_logits']
        if kind == 'sequence': return model(a, x)['A_logits']
        if kind == 'static_prev': return model.forward_deterministic(x[:, -1])[0]
        raise ValueError(f'Input kind {kind} is not scored here')

    def score(model, kind, arr):
        model.eval(); ss = []
        with torch.inference_mode():
            for j in range(0, len(arr[0]), 16):
                ss.extend(logits(model, kind, arr[0][j:j + 16].to(args.device), arr[1][j:j + 16].to(args.device)).float().cpu().numpy())
        ss = np.stack(ss)
        if not np.isfinite(ss).all(): raise ValueError('Nonfinite logits')
        return ss

    out.mkdir(parents=True, exist_ok=True); report = []
    for run_dir in args.runs:
        for name in args.models:
            mdir = Path(run_dir) / name
            if not mdir.exists(): continue
            proto = json.loads((mdir / 'protocol.json').read_text())
            if proto.get('dataset') != 'unsw': raise ValueError(f'{mdir} was trained on {proto.get("dataset")}, not the main UNSW split')
            cfg = ts.MODELS[name]; source = root / cfg['script']
            if sha(source) != proto['source_sha256']: raise ValueError(f'{source.name} differs from the script {name} was trained with')
            H = int(proto['history']); kind = proto['input']
            spec = importlib.util.spec_from_file_location('stage8_' + name, source)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            cls = getattr(module, cfg['cls'])
            kw = dict(proto['constructor'], num_nodes=n, feat_dim=10, in_dim=10, seq_len=H)
            kw = {k: v for k, v in kw.items() if k in inspect.signature(cls.__init__).parameters}
            if proto.get('decoder_variant'): cls = ts.with_decoder(cls, proto['decoder_variant'], proto['constructor']['latent_dim'])
            v_arr, t_arr = arrays(vi, H), arrays(ti, H)
            for job in sorted(mdir.glob('seed*')):
                info_path = job / 'completed.json'
                if not info_path.exists(): continue
                info = json.loads(info_path.read_text()); seed = int(info['seed'])
                if seed not in args.seeds: continue
                if sha(job / 'best.pt') != info['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch: ' + str(job))
                stored = np.load(job / 'scores.npz', allow_pickle=False)
                kv = [(str(d), int(w)) for d, w in zip(stored['validation_dates'], stored['validation_windows'])]
                if kv != [(i['date'], int(i['target_window_id'])) for i in vi]: raise ValueError('Validation keys differ: ' + str(job))
                model = cls(**kw).to(args.device)
                model.load_state_dict(torch.load(job / 'best.pt', map_location=args.device, weights_only=True), strict=True)
                v = score(model, kind, v_arr); diff = float(np.abs(v - stored['validation']).max())
                f_new = select_threshold([dict(truth=i['A_target'], score=s) for i, s in zip(vi, v)])[1]
                f_old = select_threshold([dict(truth=i['A_target'], score=s) for i, s in zip(vi, stored['validation'])])[1]
                if diff > G1_LOGIT_TOLERANCE or abs(f_new - f_old) > G1_F1_TOLERANCE:
                    raise ValueError(f'G1 failed for {name} seed {seed}: validation logits differ by up to {diff:.2e}, '
                                     f'validation F1 {f_new:.6f} against stored {f_old:.6f}')
                t = score(model, kind, t_arr)
                dest = out / name / f'seed{seed}'; dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / 'scores.npz', validation=stored['validation'].astype(np.float32), test=t.astype(np.float32),
                                    validation_dates=stored['validation_dates'], validation_windows=stored['validation_windows'],
                                    test_dates=np.array([d for d, _ in keys_t]), test_windows=np.array([w for _, w in keys_t]))
                if (job / 'selection.json').exists(): shutil.copy2(job / 'selection.json', dest / 'selection.json')
                rec = dict(model=name, dataset='unsw_newdays', seed=seed, smoke=False, source_run=str(job.resolve()),
                           checkpoint_sha256=info['checkpoint_sha256'], history=H, input=kind,
                           decoder_variant=proto.get('decoder_variant'), validation_max_abs_diff=diff, validation_F1=f_old,
                           test_dates=meta['test_dates'], test_targets=len(ti), scorer_sha256=sha(__file__),
                           dataset_hashes={k: sha(ds / k) for k in ['static_val.pkl', 'temporal_test.pkl', 'metadata.json']},
                           device=args.device, torch=torch.__version__, python=platform.python_version())
                (dest / 'completed.json').write_text(json.dumps(rec, indent=2)); report.append(rec)
                print(f'{name} seed{seed}: G1 passed (max |validation diff| {diff:.2e}, validation F1 {f_old:.4f}); scored {len(ti)} new-day targets', flush=True)
                del model
    if not report: raise ValueError('No checkpoints were scored; check --runs and --models')
    (out / 'scoring_report.json').write_text(json.dumps(report, indent=2))
    print(f'Scored {len(report)} model-seed checkpoints -> {out.resolve()}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', required=True); p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True); p.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--dataset-dir', default=str(Path('dataset_ready_unsw_newdays') / '300s'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    main(p.parse_args())
