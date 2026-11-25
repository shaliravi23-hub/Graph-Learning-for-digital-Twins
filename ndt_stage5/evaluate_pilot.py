"""UNSW pilot: separated attribute reconstruction and causal historical baselines.
No training, input mutation, target activity masking, or test-based selection.
"""
import argparse, collections, csv, hashlib, io, json, pickle, platform, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import average_precision_score


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def load(p):
    # Only load the user's trusted dataset pickles.
    with open(p,'rb') as f: return pickle.load(f)


def nonself(n): return ~np.eye(n,dtype=bool)


def topk(s,k):
    ids=np.flatnonzero(nonself(len(s)).ravel())
    # Descending score, ascending row-major pair index for ties; no diagonal.
    ids=ids[np.lexsort((ids,-s.ravel()[ids]))]
    out=np.zeros(s.size,dtype=bool)
    out[ids[:int(np.clip(k,0,len(ids)))]]=True
    return out.reshape(s.shape)


def select_threshold(records):
    """Exact search over attainable validation boundaries; macro snapshot F1.
    Tie: lower mean cardinality error, then higher threshold. Empty/empty F1=0.
    """
    mask=nonself(len(records[0]['truth']))
    grid=np.r_[np.unique(np.concatenate([r['score'][mask] for r in records])),np.inf]
    fs=np.zeros(len(grid)); es=np.zeros(len(grid))
    for r in records:
        ss=r['score'][mask]; yy=r['truth'][mask]
        order=np.argsort(ss,kind='stable'); ss=ss[order]; yy=yy[order]
        cum=np.r_[0.,np.cumsum(yy,dtype=float)]
        ix=np.searchsorted(ss,grid,side='left'); pred=len(ss)-ix
        tp=cum[-1]-cum[ix]; den=pred+yy.sum()
        fs+=np.divide(2*tp,den,out=np.zeros(len(grid)),where=den>0)
        es+=np.abs(pred-yy.sum())
    best=np.flatnonzero(np.isclose(fs,fs.max(),rtol=0,atol=1e-10))
    best=best[np.isclose(es[best],es[best].min(),rtol=0,atol=1e-10)]
    j=best[-1]
    return float(grid[j]),float(fs[j]/len(records)),float(es[j]/len(records))


def reach(a):
    return np.isfinite(shortest_path(csr_matrix(a.astype(float)),directed=True,unweighted=True)) & nonself(len(a))


def metrics(a,p,previous,ref_reach):
    a=a.astype(bool); p=p.astype(bool)&nonself(len(a)); previous=previous.astype(bool)
    tp=int((a&p).sum()); fp=int((~a&p).sum()); fn=int((a&~p).sum())
    def rate(x,y): return float(x/y) if y else np.nan
    rp=reach(p); mask=nonself(len(a)); active=a.any(0)|a.any(1)
    new=a&~previous; persistent=a&previous; gone=previous&~a
    return dict(edge_f1=rate(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        precision=rate(tp,tp+fp),recall=rate(tp,tp+fn),edge_count_error=abs(int(p.sum())-int(a.sum())),
        predicted_edges=int(p.sum()),true_edges=int(a.sum()),true_positive=tp,false_positive=fp,false_negative=fn,
        false_reachability=rate((rp&~ref_reach&mask).sum(),(~ref_reach&mask).sum()),
        missed_reachability=rate((ref_reach&~rp).sum(),ref_reach.sum()),
        false_isolation=rate((active&~(p.any(0)|p.any(1))).sum(),active.sum()),
        new_edge_recall=rate((p&new).sum(),new.sum()),new_edges=int(new.sum()),new_edges_recovered=int((p&new).sum()),
        persistent_edge_recall=rate((p&persistent).sum(),persistent.sum()),persistent_edges=int(persistent.sum()),
        disappearance_recall=rate((~p&gone).sum(),gone.sum()),disappeared_edges=int(gone.sum()),disappearances_recovered=int((~p&gone).sum()))


def validate(train,val,test,meta):
    nodes=train['node_list']; n=len(nodes)
    for d in [val,test]:
        if d['node_list']!=nodes: raise ValueError('Node registry/order mismatch')
        if d['node_features']!=train['node_features']: raise ValueError('Feature order mismatch')
    if n!=meta['num_nodes_global']: raise ValueError('Metadata N mismatch')
    if test['seq_len']!=meta['seq_len']: raise ValueError('History length mismatch')
    dates=[set(i['date'] for i in d['items']) for d in [train,val,test]]
    if any(dates[i]&dates[j] for i,j in [(0,1),(0,2),(1,2)]): raise ValueError('Split capture overlap')
    for got,key in zip(dates,['train_dates','val_dates','test_dates']):
        if got!=set(meta[key]): raise ValueError('Split dates mismatch: '+key)
    def date_value(s):
        for fmt in ['%y-%m-%d','IoT_%Y-%m-%d']:
            try: return datetime.strptime(s,fmt)
            except ValueError: pass
        raise ValueError('This chronological pilot requires explicit date capture IDs')
    spans=[sorted(date_value(s) for s in group) for group in dates]
    if not (spans[0][-1]<spans[1][0] and spans[1][-1]<spans[2][0]):
        raise ValueError('Train/validation/test dates are not strictly forward chronological')
    counts=[]
    for d in [train,val]:
        for i in d['items']:
            a=i['A'];x=i['X']
            if a.shape!=(n,n) or x.shape!=(n,10) or not np.isfinite(x).all(): raise ValueError('Invalid static array')
            if not np.isin(a,[0,1]).all() or np.trace(a)!=0: raise ValueError('Adjacency must be binary with zero diagonal')
    seen=set();previous_item={}
    for i in sorted(test['items'],key=lambda x:(x['date'],x['target_window_id'])):
        key=(i['date'],i['target_window_id'])
        if key in seen: raise ValueError('Duplicate test targets')
        seen.add(key)
        a=i['A_target']; history=i['A_seq']; x=i['X_target']
        if a.shape!=(n,n) or history.shape!=(meta['seq_len'],n,n) or x.shape!=(n,10):
            raise ValueError('Invalid test shape')
        if not np.isfinite(x).all() or not np.isin(a,[0,1]).all() or np.trace(a)!=0:
            raise ValueError('Invalid test target')
        if not np.isin(history,[0,1]).all() or np.trace(history,axis1=1,axis2=2).any():
            raise ValueError('Invalid historical adjacency')
        prior=previous_item.get(i['date'])
        if prior is not None:
            if i['target_window_id']!=prior['target_window_id']+1:
                raise ValueError('Test target gap; cannot update causal EWMA without missing observations')
            if not np.array_equal(history[-1],prior['A_target']) or not np.array_equal(history[:-1],prior['A_seq'][1:]):
                raise ValueError('Inconsistent overlapping test history')
        previous_item[i['date']]=i
        ids=i['input_window_ids']+[i['target_window_id']]
        if len(ids)!=meta['seq_len']+1 or any(b-a!=1 for a,b in zip(ids,ids[1:])):
            raise ValueError('Non-consecutive windows; resolve observation gaps first')
        counts.append(i['date'])
    return dict(nodes=n,train_snapshots=len(train['items']),val_snapshots=len(val['items']),
                test_targets=len(test['items']),test_captures=dict(collections.Counter(counts)))


def make_sequences(static,L):
    groups=collections.defaultdict(list)
    for i in static['items']: groups[i['date']].append(i)
    out=[]
    for date,items in sorted(groups.items()):
        items=sorted(items,key=lambda i:i['window_id'])
        if len({i['window_id'] for i in items})!=len(items): raise ValueError('Duplicate static windows')
        for j in range(L,len(items)):
            history=items[j-L:j]; target=items[j]
            ids=[i['window_id'] for i in history]+[target['window_id']]
            if any(b-a!=1 for a,b in zip(ids,ids[1:])): raise ValueError('Static history has gaps')
            out.append(dict(date=date,target_window_id=target['window_id'],input_window_ids=ids[:-1],
                A_seq=np.stack([i['A'] for i in history]),A_target=target['A'],X_target=target['X']))
    return out


def fit_transition(train):
    n=len(train['items'][0]['A']); groups=collections.defaultdict(list)
    for i in train['items']: groups[i['date']].append(i)
    den=np.zeros((2,n,n)); num=np.zeros_like(den); mask=nonself(n)
    for items in groups.values():
        items=sorted(items,key=lambda i:i['window_id'])
        for a,b in zip(items,items[1:]):
            if b['window_id']!=a['window_id']+1: raise ValueError('Training transition crosses a gap')
            for state in [0,1]:
                m=(a['A']==state)&mask; den[state]+=m; num[state]+=m*b['A']
    pool=(num.sum((1,2))+1)/(den.sum((1,2))+2)
    # One pseudo-observation at the pooled training-only transition probability.
    return (num+pool[:,None,None])/(den+1)


def baseline_records(items,kind,transition,alpha=.5):
    records=[]; ew=None; lastdate=None
    for i in sorted(items,key=lambda i:(i['date'],i['target_window_id'])):
        history=i['A_seq']; prev=history[-1].copy()
        if kind=='recent_frequency': s=history.mean(0,dtype=np.float64)
        elif kind=='transition': s=np.where(prev>0,transition[1],transition[0])
        elif kind=='ewma':
            if lastdate!=i['date']:
                ew=history[0].astype(float).copy()
                for a in history[1:]: ew=alpha*a+(1-alpha)*ew
            else: ew=alpha*prev+(1-alpha)*ew
            s=ew.copy(); lastdate=i['date']
        else: s=prev.copy()
        records.append(dict(date=i['date'],window=i['target_window_id'],truth=i['A_target'].copy(),previous=prev,score=s))
    return records


def numpy_weights(path):
    # Restricted reader for this six-tensor float32 state_dict, no torch imports.
    z=zipfile.ZipFile(path); prefix=z.namelist()[0].split('/')[0]
    if z.read(prefix+'/byteorder').decode()!='little': raise ValueError('Unsupported byte order')
    def rebuild(storage,offset,shape,strides,*args):
        return np.ndarray(shape,dtype='<f4',buffer=storage,offset=offset*4,strides=tuple(s*4 for s in strides)).copy()
    class Reader(pickle.Unpickler):
        def find_class(self,m,n):
            if (m,n)==('torch._utils','_rebuild_tensor_v2'): return rebuild
            if (m,n)==('torch','FloatStorage'): return 'float'
            if (m,n)==('collections','OrderedDict'): return collections.OrderedDict
            raise ValueError('Unsupported checkpoint global '+m+'.'+n)
        def persistent_load(self,p):
            if p[0]!='storage' or p[1]!='float': raise ValueError('Unsupported tensor storage')
            return z.read(prefix+'/data/'+p[2])
    w=Reader(io.BytesIO(z.read(prefix+'/data.pkl'))).load()
    expected={'fc1.weight':(64,10),'fc1.bias':(64,),'fc_mu.weight':(32,64),'fc_mu.bias':(32,),
              'fc_logvar.weight':(32,64),'fc_logvar.bias':(32,)}
    if set(w)!=set(expected) or any(w[k].shape!=v for k,v in expected.items()): raise ValueError('Checkpoint architecture mismatch')
    return w


def static_scorer(path,backend,device):
    if backend=='numpy':
        w=numpy_weights(path)
        def score(x):
            h=np.maximum(x@w['fc1.weight'].T+w['fc1.bias'],0)
            mu=h@w['fc_mu.weight'].T+w['fc_mu.bias']; return mu@mu.T
        return score
    import torch
    class AttributeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc1=torch.nn.Linear(10,64);self.fc_mu=torch.nn.Linear(64,32); self.fc_logvar=torch.nn.Linear(64,32)
        def forward(self,x):
            mu=self.fc_mu(torch.relu(self.fc1(x))); return mu@mu.T
    model=AttributeVAE().to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True);model.eval()
    def score(x):
        with torch.inference_mode(): return model(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
    return score


def static_records(items,scorer):
    return [dict(date=i['date'],window=i['target_window_id'],truth=i['A_target'].copy(),
                 previous=i['A_seq'][-1].copy(),score=scorer(i['X_target'])) for i in items]


def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    root=Path(args.dataset_dir); out=Path(args.out)
    if out.exists(): raise ValueError('Output directory already exists; choose a fresh path')
    train=load(root/'static_train.pkl'); val=load(root/'static_val.pkl');test=load(root/'temporal_test.pkl')
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8-sig'))
    audit=validate(train,val,test,meta); L=meta['seq_len']; vi=make_sequences(val,L)
    ti=sorted(test['items'],key=lambda i:(i['date'],i['target_window_id']))
    if not vi: raise ValueError('No aligned validation targets')
    runtime=json.loads(Path(args.runtime_metadata).read_text(encoding='utf-8-sig'))
    if sha(args.checkpoint)!=runtime['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    if sha(root/'temporal_test.pkl')!=runtime['test_data_sha256']: raise ValueError('Test-data hash mismatch')
    positive=sum(i['A'].sum(dtype=float) for i in train['items']); total=sum(i['A'].size for i in train['items'])
    weight=(total-positive)/positive
    scorer=static_scorer(args.checkpoint,args.backend,args.device)
    transition=fit_transition(train); selections={}; details=[]; summaries=[]
    budget=int(np.rint(np.mean([i['A_target'].sum() for i in vi])))
    audit.update(aligned_val_targets=len(vi),positive_class_weight=weight,validation_k=budget,
        task='Historical baselines forecast next window; attribute VAE reconstructs current window; never rank these as one task.',
        backend=args.backend,device=args.device,python=platform.python_version(),numpy=np.__version__,
        input_hashes={k:sha(root/k) for k in ['static_train.pkl','static_val.pkl','temporal_test.pkl','metadata.json']},
        checkpoint_hash=sha(args.checkpoint),evaluator_hash=sha(__file__),
        threshold_selection='Exact attainable validation score boundaries; macro F1; lower edge-count error then higher threshold',
        topk_ties='descending raw score then ascending row-major non-self pair index',
        uncertainty='Single checkpoint and observed captures; no seed or deployment uncertainty quantified')
    refs=[reach(i['A_target']) for i in ti]
    def evaluate(name,task,vr,tr,logits=False):
        for r in vr+tr:
            if not np.isfinite(r['score']).all(): raise ValueError('Nonfinite scores')
        threshold,vf,ve=select_threshold(vr)
        selections[name]=dict(threshold=threshold,val_f1=vf,val_edge_count_error=ve,
                              probability_threshold=float(expit(threshold)) if logits else threshold)
        rules=['validation_threshold','validation_topk','previous_topk']
        if name=='Persistence': rules=['copy_previous']
        if logits: rules+=['raw_0.5','weight_corrected_0.5']
        for rule in rules:
            group=[]
            for r,rr in zip(tr,refs):
                s=r['score']; mask=nonself(len(s))
                if rule=='copy_previous': p=r['previous']>0
                elif rule=='validation_threshold': p=(s>=threshold)&mask
                elif rule=='validation_topk': p=topk(s,budget)
                elif rule=='previous_topk': p=topk(s,r['previous'].sum())
                elif rule=='raw_0.5': p=(s>=0)&mask
                else: p=(s>=np.log(weight))&mask
                ap=average_precision_score(r['truth'][mask],expit(s[mask]) if logits else s[mask]) if r['truth'].sum() else np.nan
                row=dict(model=name,task=task,decoder=rule,date=r['date'],window=r['window'],AP=ap,
                         **metrics(r['truth'],p,r['previous'],rr))
                details.append(row);group.append(row)
            row=dict(model=name,task=task,decoder=rule,n=len(group))
            for key in group[0]:
                if key in ['model','task','decoder','date','window']: continue
                values=np.array([g[key] for g in group],float);good=values[np.isfinite(values)]
                row[key+'_mean']=float(good.mean()) if len(good) else None;row[key+'_valid_n']=len(good)
            summaries.append(row)
        print(name,'validation F1',round(vf,6),'threshold',selections[name]['probability_threshold'],flush=True)
    for kind,name in [('persistence','Persistence'),('recent_frequency','Recent_frequency'),('transition','Transition')]:
        evaluate(name,'historical_prediction',baseline_records(vi,kind,transition),baseline_records(ti,kind,transition))
    options=[]
    for alpha in [.1,.3,.5,.8,1.]:
        records=baseline_records(vi,'ewma',transition,alpha);th,f,e=select_threshold(records)
        options.append((f,-e,-alpha,alpha,records))
    best=max(options,key=lambda x:x[:3]);alpha=best[3]
    evaluate('EWMA','historical_prediction',best[4],baseline_records(ti,'ewma',transition,alpha))
    selections['EWMA']['alpha']=alpha
    evaluate('Attribute_VAE','current_window_reconstruction',static_records(vi,scorer),static_records(ti,scorer),True)
    out.mkdir(parents=True)
    write_csv(out/'summary.csv',summaries);write_csv(out/'per_snapshot.csv',details)
    # JSON permits Infinity in Python, but use explicit strings for portable manifests.
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return str(x)
        return x
    (out/'selection.json').write_text(json.dumps(clean(selections),indent=2))
    (out/'audit.json').write_text(json.dumps(clean(audit),indent=2))
    print('Saved',out.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',required=True); parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--runtime-metadata',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--backend',choices=['torch','numpy'],default='torch')
    parser.add_argument('--device',default='cpu')
    run(parser.parse_args())
