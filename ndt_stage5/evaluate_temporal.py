"""Extend the confirmed UNSW pilot to existing historical-input checkpoints."""
import argparse,importlib.util,inspect,json,platform
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
from evaluate_pilot import load,sha,validate,make_sequences,nonself,topk,reach,metrics,select_threshold,write_csv

SPECS={
'dyngem':('DynGEMStylePredictor',dict(hidden_dims=(768,384),latent_dim=128,dropout=.1)),
'aernn':('Dyngraph2VecAERNNStyle',dict(enc_dim=512,embed_dim=256,rnn_hidden=256,dropout=0.)),
'vgrnn':('VGRNNPredictor',dict(hidden_dim=128,latent_dim=32,dropout=.1)),
'tgae':('TGAEStylePredictor',dict(hidden_dim=128,latent_dim=32,num_heads=4,num_layers=2,dropout=.1)),
'vrdag':('VRDAGStylePredictor',dict(hidden_dim=64,latent_dim=32,dropout=.1))}

def validation_items(val,L):
    lookup={(i['date'],i['window_id']):i for i in val['items']}
    items=make_sequences(val,L)
    for i in items:
        i['X_seq']=np.stack([lookup[(i['date'],w)]['X'] for w in i['input_window_ids']])
    return items


def infer(model,name,item,torch,device):
    # Only historical inputs cross this boundary. Never pass target arrays.
    a=torch.tensor(item['A_seq'][None],dtype=torch.float32,device=device)
    x=torch.tensor(item['X_seq'][None],dtype=torch.float32,device=device)
    with torch.inference_mode():
        if name=='dyngem': out=model(a[:,-1],x[:,-1])
        elif name in ['vgrnn','vrdag']: out=model(a,x,use_posterior=False)
        else: out=model(a,x)
    s=out['A_logits'][0].detach().cpu().numpy()
    if s.shape!=item['A_target'].shape or not np.isfinite(s).all(): raise ValueError('Invalid logits')
    return s


def main(args):
    import torch
    root=Path(args.project_root).resolve(); ds=root/'dataset_ready'/'300s';out=Path(args.out)
    if out.exists(): raise ValueError('Output exists; use a fresh directory')
    train=load(ds/'static_train.pkl');val=load(ds/'static_val.pkl');test=load(ds/'temporal_test.pkl')
    meta=json.loads((ds/'metadata.json').read_text(encoding='utf-8-sig'));audit=validate(train,val,test,meta)
    vi=validation_items(val,meta['seq_len']);ti=sorted(test['items'],key=lambda i:(i['date'],i['target_window_id']))
    for i in vi+ti:
        if i['X_seq'].shape!=(meta['seq_len'],len(train['node_list']),10) or not np.isfinite(i['X_seq']).all(): raise ValueError('Invalid historical features')
    mask=nonself(len(train['node_list']));budget=int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    metadata={}
    for folder in ['runtime_unsw_corrected','runtime_unsw_recovery']:
        for f in sorted((root/folder).glob('*_metadata.json')):
            md=json.loads(f.read_text(encoding='utf-8-sig'));md=md.get('runtime_metadata',md)
            name=md.get('arguments',{}).get('model')
            if name not in SPECS: continue
            if name in metadata: raise ValueError('Ambiguous metadata for '+name)
            metadata[name]=(f,md)
    if set(metadata)!=set(SPECS): raise ValueError('Missing model metadata: '+str(sorted(set(SPECS)-set(metadata))))
    records=[];summary=[];selections={};provenance={};refs=[reach(i['A_target']) for i in ti]
    for name in SPECS:
        f,md=metadata[name];cfg=md['arguments'];script=root/cfg['script'];checkpoint=root/cfg['checkpoint']
        for p,key in [(script,'model_script_sha256'),(checkpoint,'checkpoint_sha256'),(ds/'temporal_test.pkl','test_data_sha256')]:
            if sha(p)!=md[key]: raise ValueError('Hash mismatch: '+str(p))
        spec=importlib.util.spec_from_file_location('verified_'+name,script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cname,kw=SPECS[name];cls=getattr(module,cname);kw=dict(kw)
        kw.update(num_nodes=len(train['node_list']),feat_dim=10,in_dim=10,seq_len=meta['seq_len'])
        allowed=inspect.signature(cls.__init__).parameters;kw={k:v for k,v in kw.items() if k in allowed}
        model=cls(**kw).to(args.device)
        model.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True),strict=True);model.eval()
        def scores(items):
            return [dict(date=i['date'],window=i['target_window_id'],truth=i['A_target'].copy(),previous=i['A_seq'][-1].copy(),score=infer(model,name,i,torch,args.device)) for i in items]
        vr=scores(vi);tr=scores(ti)
        threshold,vf,ve=select_threshold(vr)
        selections[name]=dict(logit_threshold=threshold,score_threshold=float(expit(threshold)),val_f1=vf,val_edge_count_error=ve,k=budget)
        # Confirm target-array independence using an altered-target copy.
        original=vi[0];altered=dict(original,A_target=1-original['A_target'],X_target=np.zeros_like(original['X_target']))
        if not np.allclose(infer(model,name,original,torch,args.device),infer(model,name,altered,torch,args.device),rtol=0,atol=0): raise ValueError('Target independence/determinism failed')
        # For symmetric decoders, tie directions are positional decisions, not learned direction.
        asym=max(float(np.max(np.abs(r['score']-r['score'].T))) for r in tr)
        provenance[name]=dict(metadata=str(f),checkpoint_sha256=sha(checkpoint),script_sha256=sha(script),constructor=kw,max_logit_asymmetry=asym,target_independence_check=True)
        for rule in ['validation_threshold','validation_topk','previous_topk','raw_0.5']:
            group=[]
            for r,rr in zip(tr,refs):
                s=r['score']
                if rule=='validation_threshold':p=(s>=threshold)&mask
                elif rule=='raw_0.5':p=(s>=0)&mask
                else:p=topk(s,budget if rule=='validation_topk' else r['previous'].sum())
                row=dict(model=name,task='historical_prediction',decoder=rule,date=r['date'],window=r['window'],
                         AP=average_precision_score(r['truth'][mask],expit(s[mask])) if r['truth'].sum() else np.nan,
                         **metrics(r['truth'],p,r['previous'],rr))
                records.append(row);group.append(row)
            row=dict(model=name,task='historical_prediction',decoder=rule,n=len(group))
            for key in group[0]:
                if key in ['model','task','decoder','date','window']:continue
                v=np.array([r[key] for r in group],float);v=v[np.isfinite(v)]
                row[key+'_mean']=float(v.mean()) if len(v) else None;row[key+'_valid_n']=len(v)
            summary.append(row)
        print(name,'validation F1',round(vf,6),'score threshold',expit(threshold),flush=True)
        del model
    audit.update(aligned_val_targets=len(vi),validation_k=budget,task='historical_prediction',device=args.device,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),evaluator_sha256=sha(__file__),
        metric_module_sha256=sha(Path(__file__).parent/'evaluate_pilot.py'),
        input_hashes={k:sha(ds/k) for k in ['static_train.pkl','static_val.pkl','temporal_test.pkl','metadata.json']},models=provenance,
        limitations='Existing single-seed checkpoints; original checkpoint selection retained. No retraining, new periods, seed uncertainty or latency benchmark. Compare matching decoders with pilot baselines.')
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,float) and not np.isfinite(x):return str(x)
        return x
    out.mkdir(parents=True);write_csv(out/'summary.csv',summary);write_csv(out/'per_snapshot.csv',records)
    (out/'selection.json').write_text(json.dumps(clean(selections),indent=2));(out/'audit.json').write_text(json.dumps(clean(audit),indent=2))
    print('Saved',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');main(p.parse_args())
