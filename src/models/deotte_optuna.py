"""Optuna search around Deotte's best parameters — 3-fold CV, 15 trials"""
import pandas as pd, numpy as np, math, gc, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import optuna, warnings; warnings.filterwarnings('ignore')
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

DEVICE = torch.device('cuda'); SEED = 42; FOLDS = 3

# ====== Data (same as deotte_realmlp.py) ======
pr("Loading data...")
train = pd.read_csv("train_fe.csv", index_col='id'); test = pd.read_csv("test_fe.csv", index_col='id')
CLASSES=['GALAXY','QSO','STAR']; LABEL_MAP={c:i for i,c in enumerate(CLASSES)}
train['class'] = train['class'].map(LABEL_MAP).astype('int8')
y = train['class'].values.astype(int)
X = train.drop(['class'], axis=1); X_test = test.copy()

base_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
base_num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
category_map = {}
cp=[('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]
ic=[('alpha_cat_','delta_cat_'),('u_cat_','z_cat_')]

def fe(df,fit=False):
    df=df.copy()
    df['_gdr']=(df['g']/(df['redshift']+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    df['_idr']=(df['i']/(df['redshift']+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    for a,b in cp: df[f'_{a}-{b}']=(df[a]-df[b]).astype('float32')
    mags=df[['u','g','r','i','z']].astype('float32')
    df['_mm']=mags.mean(axis=1).astype('float32'); df['_mr']=(mags.max(axis=1)-mags.min(axis=1)).astype('float32')
    sr=df['redshift'].astype('float32')-min(0.0,float(df['redshift'].min()))+1e-4
    df['_lz']=np.log1p(sr).astype('float32')
    for col in base_cat_cols:
        if fit: codes,uniques=pd.factorize(df[col],sort=False); category_map[col]=uniques
        else: uniques=category_map[col]; cm={cat:i for i,cat in enumerate(uniques)}; codes=df[col].map(cm).fillna(-1).astype('int32')
        df[col]=pd.Series(codes,index=df.index).astype('int32').astype('category')
    for col in base_num_cols:
        cn=f'{col}_cat_'; fl=np.floor(df[col]).astype('float32')
        if fit: codes,uniques=pd.factorize(fl,sort=False); category_map[cn]=uniques
        else: uniques=category_map[cn]; cm={cat:i for i,cat in enumerate(uniques)}; codes=fl.map(cm).fillna(-1).astype('int32')
        df[cn]=pd.Series(codes,index=df.index).astype('int32').astype('category')
    for n in[100,500]:
        bn=f'd{n}_b_'
        if fit: kb=KBinsDiscretizer(n_bins=n,encode='ordinal',strategy='quantile',subsample=None); b=kb.fit_transform(df[['delta']]).ravel().astype('int32'); category_map[bn]=kb
        else: kb=category_map[bn]; b=kb.transform(df[['delta']]).ravel().astype('int32')
        df[bn]=pd.Series(b,index=df.index).astype('int32').astype('category')
    cns=[]
    for cols in ic:
        cn='__'.join(cols)+'__'; cns.append(cn)
        combo=df[cols[0]].astype(str); [combo:=combo+'|'+df[col].astype(str) for col in cols[1:]]
        if fit: codes,uniques=pd.factorize(combo,sort=False); category_map[cn]=uniques
        else: uniques=category_map[cn]; cm={cat:i for i,cat in enumerate(uniques)}; codes=combo.map(cm).fillna(-1).astype('int32')
        df[cn]=pd.Series(codes,index=df.index).astype('int32').astype('category')
    ncc=[c for c in df.columns if str(df[c].dtype)=='category' and c not in base_cat_cols]
    nnc=[c for c in df.columns if c.startswith('_') and str(df[c].dtype)!='category']
    return df,ncc,nnc,cns

X,ncc,nnc,cns=fe(X,fit=True); X_test,_,_,_=fe(X_test,fit=False)
cats=sorted(base_cat_cols+ncc); nums_f=sorted(base_num_cols+nnc)
X=X.reindex(sorted(X.columns),axis=1); X_test=X_test.reindex(sorted(X_test.columns),axis=1)

class RobustPrep:
    def __init__(self,tfms): self._t=[t for t in tfms if t in('median_center','robust_scale')]
    def fit(self,X):
        if self._t: self._m=np.median(X,axis=0); qd=np.quantile(X,0.75,axis=0)-np.quantile(X,0.25,axis=0); qd[qd==0]=1.0; self._s=1.0/(qd+1e-30)
    def transform(self,X):
        X=X.copy().astype(np.float32)
        for t in self._t:
            if t=='median_center': X-=self._m
            elif t=='robust_scale': X*=self._s
        return X

prep=RobustPrep(['median_center','robust_scale']); prep.fit(X[nums_f].values.astype(np.float32))
Xn=prep.transform(X[nums_f].values.astype(np.float32)); Xtn=prep.transform(X_test[nums_f].values.astype(np.float32))
NF=Xn.shape[1]; pr(f"Features: {NF}")

# ====== Model classes ======
class SL(nn.Module):
    def __init__(self,ne,nf): super().__init__(); self.s=nn.Parameter(torch.ones(ne,nf))
    def forward(self,x): return x*self.s[None,:,:]
class NL(nn.Module):
    def __init__(self,ne,inf,ouf,b=True):
        super().__init__(); self.inf=inf
        self.w=nn.Parameter(torch.randn(ne,inf,ouf)); self.b=nn.Parameter(torch.randn(ne,ouf)) if b else None
    def forward(self,x): x=torch.einsum('bki,kio->bko',x,self.w)/math.sqrt(self.inf); return x+self.b if self.b is not None else x
class PE(nn.Module):
    def __init__(self,ne,nf,hd=16,od=5,fs=2.33):
        super().__init__(); self.act=nn.PReLU()
        self.w1=nn.Parameter(torch.randn(ne,nf,hd)*fs); self.b1=nn.Parameter(torch.randn(ne,nf,hd))
        self.w2=nn.Parameter(torch.randn(ne,nf,hd,od-1)/math.sqrt(hd)); self.b2=nn.Parameter(torch.zeros(ne,nf,od-1))
        nn.init.uniform_(self.b1,-math.pi,math.pi)
    def forward(self,x):
        p=torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w1.unsqueeze(0)+self.b1.unsqueeze(0)))
        t=self.act(torch.einsum('bkfh,kfhd->bkfd',p,self.w2)+self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1),t],dim=-1).flatten(start_dim=2)

class RM(nn.Module):
    def __init__(self,nf,nc,do):
        super().__init__(); self.ne=8
        self.emb=PE(self.ne,nf)
        d=nf*5; dims=[512,512,512]
        layers=[SL(self.ne,d)]; self._dr=[]
        for o in dims:
            layers+=[NL(self.ne,d,o),nn.GELU()]; dr=nn.Dropout(do); layers.append(dr); self._dr.append(dr); d=o
        self.h=nn.Sequential(*layers); self.o=NL(self.ne,d,nc)
    def forward(self,x):
        x=x.unsqueeze(1).expand(-1,self.ne,-1); x=self.emb(x); x=self.h(x); return F.softmax(self.o(x),dim=2)

def sched(init,prog):
    if prog<0.2: return init
    t=(prog-0.2)/0.8; return init*(math.cos(math.pi*t)+1)/2

# ====== Optuna objective ======
def objective(trial):
    do = trial.suggest_float('dropout', 0.02, 0.08)
    ls = trial.suggest_float('ls_eps', 0.01, 0.08)
    lpp = trial.suggest_float('loss_prior_power', 0.5, 2.0)
    epochs = trial.suggest_int('epochs', 4, 10)

    all_scores = []
    for fold,(tri,vai) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=42).split(Xn,y)):
        np.random.seed(SEED+fold*100); torch.manual_seed(SEED+fold*100)
        Xtr=X.iloc[tri].copy(); Xva=X.iloc[vai].copy()
        ytr=y[tri]; yva=y[vai]
        for cn in cns:
            te=TargetEncoder(target_type='multiclass',cv=5,smooth='auto',shuffle=True,random_state=SEED+fold*100)
            tre=te.fit_transform(Xtr[[cn]],ytr); vae=te.transform(Xva[[cn]])
            for cl in range(3):
                tn=f'_{cn}TE_c{cl}'; Xtr[tn]=tre[:,cl] if tre.ndim>1 else tre; Xva[tn]=vae[:,cl] if vae.ndim>1 else vae
        te_cols=[c for c in Xtr.columns if '_TE_' in c]
        Xtr=np.column_stack([Xtr[nums_f].values,Xtr[te_cols].values]).astype(np.float32)
        Xva=np.column_stack([Xva[nums_f].values,Xva[te_cols].values]).astype(np.float32)
        p2=RobustPrep(['median_center','robust_scale']); p2.fit(Xtr); Xtr=p2.transform(Xtr); Xva=p2.transform(Xva)

        xt=torch.tensor(Xtr).to(DEVICE); yt=torch.tensor(ytr).long().to(DEVICE)
        xv=torch.tensor(Xva).to(DEVICE)

        model=RM(Xtr.shape[1],3,do).to(DEVICE)
        sp,pp,fw,ow,bp=[],[],[],[],[]
        for n,p in model.named_parameters():
            if 'emb' in n: pp.append(p)
            elif '.s' in n: sp.append(p)
            elif p is model.o.weight: fw.append(p)
            elif 'bias' in n: bp.append(p)
            else: ow.append(p)
        lr=0.01; wd=0.0125
        groups=[{'params':sp,'lr':lr*10,'wd':wd*0.1},{'params':pp,'lr':lr*0.115,'wd':wd},
                {'params':fw,'lr':lr,'wd':wd*0.1},{'params':ow,'lr':lr,'wd':wd},
                {'params':bp,'lr':lr*0.1,'wd':wd*0.5}]
        groups=[g for g in groups if g['params']]
        opt=torch.optim.AdamW(groups,lr=lr,weight_decay=wd,betas=(0.9,0.98))
        ts=epochs*len(yt); step=0; best=-np.inf; bst=None
        ema={k:v.detach().clone() for k,v in model.state_dict().items()}

        class_counts = np.bincount(ytr,minlength=3).astype(np.float64)
        class_counts = class_counts/np.exp(np.log(class_counts).mean())
        loss_mult = torch.tensor(np.power(class_counts,lpp),dtype=torch.float32).to(DEVICE)

        for ep in range(epochs):
            model.train(); perm=torch.randperm(len(yt))
            for s in range(0,len(yt),256):
                idx=perm[s:s+256]; prog=step/ts; step+=1
                for g in opt.param_groups: g['lr']=sched(g.get('lr_base',g['lr']),prog)
                opt.zero_grad(); yp=model(xt[idx])
                ls2=sched(ls,prog)
                ys=torch.full_like(yp[:,0,:],ls2/3); ys.scatter_(1,yt[idx].unsqueeze(1),1-ls2+ls2/3)
                ys=ys.unsqueeze(1).expand(-1,8,-1)
                yp2 = yp * loss_mult[None,None,:]; yp2 = yp2/yp2.sum(dim=2,keepdim=True).clamp_min(1e-15)
                loss=-(ys*torch.log(yp2.clamp(1e-15,1))).sum(dim=2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                for k,v in model.state_dict().items():
                    if torch.is_floating_point(v): ema[k].mul_(0.997875).add_(v.detach(),alpha=0.002125)
                for dm in model._dr: dm.p=do*math.exp(-4*prog)

            model.eval(); live={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(ema,strict=True)
            with torch.no_grad(): vp=model(xv).mean(dim=1).cpu().numpy()
            model.load_state_dict(live,strict=True)
            sc=balanced_accuracy_score(yva,np.argmax(vp,axis=1))
            if sc>best: best=sc; bst={k:v.detach().clone() for k,v in ema.items()}
        all_scores.append(best)
        del model; gc.collect(); torch.cuda.empty_cache()

    return np.mean(all_scores)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15, show_progress_bar=True)

pr(f"\nBest trial: {study.best_trial.number}")
pr(f"Best BA: {study.best_value:.5f}")
pr(f"Best params: {study.best_params}")
pr(f"Baseline (Deotte original): ~0.9674")
pr(f"Improvement: {study.best_value - 0.96745:+.5f}")
