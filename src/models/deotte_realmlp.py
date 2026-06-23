"""
Deotte's RealMLP R2-103 — exact notebook port
===============================================
Data: raw train.csv + test.csv
Features: Deotte's exact pipeline (color diffs, ratios, floor bins, TE)
Model: PBLD + NTPLinear + n_ens=8 + EMA + 5-group params
Training: 6 epochs, flat_cos lr, cos ls, expm4t dropout, AdamW(betas=0.9,0.98)
"""
import pandas as pd, numpy as np, math, gc, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; pr(f"Device: {DEVICE}")

# ==============================
# 1. Data + Deotte Feature Engineering
# ==============================
pr("Loading data...")
train = pd.read_csv("train_fe.csv", index_col='id')
test  = pd.read_csv("test_fe.csv", index_col='id')
sub  = pd.read_csv("sample_submission.csv")  # placeholder

CLASSES = ['GALAXY','QSO','STAR']
LABEL_MAP = {c:i for i,c in enumerate(CLASSES)}
INV_MAP = {v:k for k,v in LABEL_MAP.items()}
train['class'] = train['class'].map(LABEL_MAP).astype('int8')
y = train['class'].values.astype(int)
X = train.drop(['class'], axis=1)
X_test = test.copy()

base_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
base_num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
pr(f"Raw: {X.shape}  Cat:{base_cat_cols}  Num:{len(base_num_cols)}")

# Feature engineering (exact Deotte pipeline)
category_map = {}
color_pairs = [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]
important_combos = [('alpha_cat_','delta_cat_'),('u_cat_','z_cat_')]

def feature_engineering(df, fit=False):
    df = df.copy()
    df['_g_div_redshift'] = (df['g']/(df['redshift']+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    df['_i_div_redshift'] = (df['i']/(df['redshift']+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    for a,b in color_pairs: df[f'_{a}-{b}'] = (df[a]-df[b]).astype('float32')
    mags = df[['u','g','r','i','z']].astype('float32')
    df['_mag_mean'] = mags.mean(axis=1).astype('float32')
    df['_mag_range'] = (mags.max(axis=1)-mags.min(axis=1)).astype('float32')
    shifted_redshift = df['redshift'].astype('float32') - min(0.0, float(df['redshift'].min())) + 1e-4
    df['_log1p_redshift'] = np.log1p(shifted_redshift).astype('float32')
    for col in base_cat_cols:
        if fit: codes, uniques = pd.factorize(df[col], sort=False); category_map[col] = uniques
        else: uniques = category_map[col]; code_map = {cat:i for i,cat in enumerate(uniques)}; codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = pd.Series(codes, index=df.index).astype('int32').astype('category')
    for col in base_num_cols:
        cat_name = f'{col}_cat_'; floored = np.floor(df[col]).astype('float32')
        if fit: codes, uniques = pd.factorize(floored, sort=False); category_map[cat_name] = uniques
        else: uniques = category_map[cat_name]; code_map = {cat:i for i,cat in enumerate(uniques)}; codes = floored.map(code_map).fillna(-1).astype('int32')
        df[cat_name] = pd.Series(codes,index=df.index).astype('int32').astype('category')
    for n_bins in [100,500]:
        bin_name = f'delta_{n_bins}_quantile_bin_'
        if fit: kb = KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='quantile',subsample=None); binned = kb.fit_transform(df[['delta']]).ravel().astype('int32'); category_map[bin_name] = kb
        else: kb = category_map[bin_name]; binned = kb.transform(df[['delta']]).ravel().astype('int32')
        df[bin_name] = pd.Series(binned,index=df.index).astype('int32').astype('category')
    combo_names = []
    for cols in important_combos:
        combo_name = '__'.join(cols)+'__'; combo_names.append(combo_name)
        combo = df[cols[0]].astype(str); [combo := combo + '|' + df[col].astype(str) for col in cols[1:]]
        if fit: codes, uniques = pd.factorize(combo, sort=False); category_map[combo_name] = uniques
        else: uniques = category_map[combo_name]; code_map = {cat:i for i,cat in enumerate(uniques)}; codes = combo.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = pd.Series(codes,index=df.index).astype('int32').astype('category')
    new_cat_cols = [c for c in df.columns if str(df[c].dtype)=='category' and c not in base_cat_cols]
    new_num_cols = [c for c in df.columns if c.startswith('_') and str(df[c].dtype)!='category']
    return df, new_cat_cols, new_num_cols, combo_names

X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, _, _, _ = feature_engineering(X_test, fit=False)
cat_cols = sorted(base_cat_cols + new_cat_cols)
num_cols_final = sorted(base_num_cols + new_num_cols)
X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
pr(f"Features: {X.shape[1]}  ({len(cat_cols)} cat, {len(num_cols_final)} num)")

# ==============================
# 2. Numerical Preprocessing
# ==============================
class NumericalPreprocessor:
    def __init__(self, tfms): self._tfms = [t for t in tfms if t in ("median_center","robust_scale","smooth_clip","l2_normalize")]
    def fit(self,X):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X,axis=0)
            qd = np.quantile(X,0.75,axis=0)-np.quantile(X,0.25,axis=0); qd[qd==0]=0.5*(X.max(axis=0)[qd==0]-X.min(axis=0)[qd==0]); qd[qd==0]=1.0
            self._iqr = 1.0/(qd+1e-30)
    def transform(self,X):
        X = X.copy().astype(np.float32)
        for t in self._tfms:
            if t=="median_center": X -= self._median
            elif t=="robust_scale": X *= self._iqr
            elif t=="smooth_clip": X = X/np.sqrt(1+(X/3)**2)
        return X

prep = NumericalPreprocessor(['median_center','robust_scale'])
prep.fit(X[num_cols_final].values.astype(np.float32))
X_num_all = prep.transform(X[num_cols_final].values.astype(np.float32))
X_test_num = prep.transform(X_test[num_cols_final].values.astype(np.float32))
NF = X_num_all.shape[1]
pr(f"Num features: {NF}")

# ==============================
# 3. Model Architecture (Deotte exact)
# ==============================
class ScalingLayer(nn.Module):
    def __init__(self,n_ens,n_f): super().__init__(); self.scale=nn.Parameter(torch.ones(n_ens,n_f))
    def forward(self,x): return x*self.scale[None,:,:]
class NTPLinear(nn.Module):
    def __init__(self,n_ens,in_f,out_f,bias=True):
        super().__init__(); self.in_f=in_f
        self.weight=nn.Parameter(torch.randn(n_ens,in_f,out_f))
        self.bias=nn.Parameter(torch.randn(n_ens,out_f)) if bias else None
    def forward(self,x): x=torch.einsum("bki,kio->bko",x,self.weight)/math.sqrt(self.in_f); return x+self.bias if self.bias is not None else x
class PBLDEmbedding(nn.Module):
    def __init__(self,n_ens,n_f,hidden_dim=16,out_dim=5,freq_scale=2.33,activation=nn.PReLU):
        super().__init__(); self.n_f=n_f; self.out_dim=out_dim; self.act=activation()
        self.w1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim)*freq_scale)
        self.b1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim)); nn.init.uniform_(self.b1,-math.pi,math.pi)
        self.w2=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim,out_dim-1)/math.sqrt(hidden_dim))
        self.b2=nn.Parameter(torch.zeros(n_ens,n_f,out_dim-1))
    def forward(self,x):
        p=torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w1.unsqueeze(0)+self.b1.unsqueeze(0)))
        t=self.act(torch.einsum("bkfh,kfhd->bkfd",p,self.w2)+self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1),t],dim=-1).flatten(start_dim=2)

class RealMLP(nn.Module):
    def __init__(self,n_f,n_cls,cfg):
        super().__init__(); self.n_ens=cfg['n_ens']
        self.num_embed=PBLDEmbedding(self.n_ens,n_f,cfg['pbld_hidden_dim'],cfg['pbld_out_dim'],cfg['pbld_freq_scale'])
        d=n_f*cfg['pbld_out_dim']; dims=cfg['hidden_dims']; act=cfg['activation']()
        layers=[]; self._drops=[]
        if cfg['add_front_scale']: layers.append(ScalingLayer(self.n_ens,d))
        for out in dims:
            layers.append(NTPLinear(self.n_ens,d,out)); layers.append(act)
            drop=nn.Dropout(cfg['dropout']); layers.append(drop); self._drops.append(drop); d=out
        self.hidden=nn.Sequential(*layers); self.out=NTPLinear(self.n_ens,d,n_cls)
    def forward(self,x):
        x=x.unsqueeze(1).expand(-1,self.n_ens,-1); x=self.num_embed(x)
        x=self.hidden(x); return F.softmax(self.out(x),dim=2)

# ==============================
# 4. Config (Deotte R2-103)
# ==============================
CFG = dict(n_ens=8,hidden_dims=[512,512,512],dropout=0.044,add_front_scale=True,
    pbld_hidden_dim=16,pbld_out_dim=5,pbld_freq_scale=2.33,activation=nn.GELU,
    lr=0.01,weight_decay=0.0125,epochs=6,train_bs=256,eval_bs=10240,
    ls_eps=0.04,grad_clip=1.0,ema_decay=0.997875)

def apply_schedule(init, progress, sched, flat_ratio=0.2):
    if sched=='flat_cos':
        if progress<flat_ratio: return init
        t=(progress-flat_ratio)/(1-flat_ratio); return init*(math.cos(math.pi*t)+1)/2
    if sched=='cos': return init*(math.cos(math.pi*progress)+1)/2
    if sched=='expm4t': return init*math.exp(-4*progress)
    return init

# ==============================
# 5. 5-Fold CV
# ==============================
oof = np.zeros((len(y),3),dtype=np.float32); tp=np.zeros((X_test_num.shape[0],3),dtype=np.float32)
fold_scores = []
te_combo_names = ['__'.join(c)+'__' for c in important_combos] if 'X' in locals() else []  # placeholder

for fold,(tr_idx,val_idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X_num_all,y)):
    pr(f"\nFold {fold+1}/5")
    fold_seed = SEED + fold*100; torch.manual_seed(fold_seed); np.random.seed(fold_seed)

    # Fold-safe target encoding
    X_tr = X.iloc[tr_idx].copy(); X_val = X.iloc[val_idx].copy(); X_tst = X_test.copy()
    y_tr = y[tr_idx]; y_val = y[val_idx]
    te_names = []
    for combo_name in combo_names:
        te = TargetEncoder(target_type='multiclass',cv=5,smooth='auto',shuffle=True,random_state=fold_seed)
        tr_enc = te.fit_transform(X_tr[[combo_name]], y_tr)
        val_enc = te.transform(X_val[[combo_name]]); tst_enc = te.transform(X_tst[[combo_name]])
        for cls in range(3):
            cn = f'_{combo_name}TE_class{cls}'; te_names.append(cn)
            X_tr[cn] = tr_enc[:,cls] if tr_enc.ndim>1 else tr_enc
            X_val[cn] = val_enc[:,cls] if val_enc.ndim>1 else val_enc
            X_tst[cn] = tst_enc[:,cls] if tst_enc.ndim>1 else tst_enc

    X_tr = X_tr[num_cols_final+te_names].values.astype(np.float32)
    X_val = X_val[num_cols_final+te_names].values.astype(np.float32)
    X_tst = X_tst[num_cols_final+te_names].values.astype(np.float32)
    prep2 = NumericalPreprocessor(['median_center','robust_scale']); prep2.fit(X_tr)
    X_tr = prep2.transform(X_tr); X_val = prep2.transform(X_val); X_tst = prep2.transform(X_tst)

    xtr=torch.tensor(X_tr).to(DEVICE); ytr=torch.tensor(y_tr).long().to(DEVICE)
    xv=torch.tensor(X_val).to(DEVICE); yv=torch.tensor(y_val).long().to(DEVICE)

    # Class weights
    classes = np.unique(y_tr); cw = compute_class_weight('balanced',classes=classes,y=y_tr)
    class_weights = torch.tensor(cw,dtype=torch.float32).to(DEVICE)
    # loss_prior_power
    class_counts = np.bincount(y_tr,minlength=3).astype(np.float64)
    class_counts = class_counts/np.exp(np.log(class_counts).mean())
    loss_mult = torch.tensor(np.power(class_counts,1.075),dtype=torch.float32).to(DEVICE)

    model = RealMLP(X_tr.shape[1],3,CFG).to(DEVICE)
    # 5-group optimizer
    n_ens = CFG['n_ens']
    scale_p,pbld_p,first_w,other_w,bias_p=[],[],[],[],[]
    first_w_id = id(model.out.weight)
    for n,p in model.named_parameters():
        if 'num_embed' in n: pbld_p.append(p)
        elif 'scale' in n: scale_p.append(p)
        elif id(p)==first_w_id: first_w.append(p)
        elif 'bias' in n: bias_p.append(p)
        else: other_w.append(p)
    lr=CFG['lr']; wd=CFG['weight_decay']
    groups=[{'params':scale_p,'lr':lr*10,'weight_decay':wd*0.1,'lr_base':lr*10},
            {'params':pbld_p,'lr':lr*0.115,'weight_decay':wd,'lr_base':lr*0.115},
            {'params':first_w,'lr':lr,'weight_decay':wd*0.1,'lr_base':lr},
            {'params':other_w,'lr':lr,'weight_decay':wd,'lr_base':lr},
            {'params':bias_p,'lr':lr*0.1,'weight_decay':wd*0.5,'lr_base':lr*0.1}]
    opt=torch.optim.AdamW(groups,lr=lr,weight_decay=wd,betas=(0.9,0.98))
    total_steps = CFG['epochs']*len(y_tr); step=0; best_score=-np.inf; best_state=None
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}

    for ep in range(CFG['epochs']):
        model.train(); perm=torch.randperm(len(y_tr))
        for s in range(0,len(y_tr),CFG['train_bs']):
            idx=perm[s:s+CFG['train_bs']]; progress=step/total_steps; step+=1
            for g in opt.param_groups: g['lr']=apply_schedule(g['lr_base'],progress,'flat_cos')
            opt.zero_grad(); yp=model(xtr[idx])
            ls=apply_schedule(CFG['ls_eps'],progress,'cos')
            ys=torch.full_like(yp[:,0,:],ls/3); ys.scatter_(1,ytr[idx].unsqueeze(1),1-ls+ls/3)
            ys=ys.unsqueeze(1).expand(-1,n_ens,-1)
            yp2 = yp * loss_mult[None,None,:]; yp2 = yp2/yp2.sum(dim=2,keepdim=True).clamp_min(1e-15)
            loss=-(ys*torch.log(yp2.clamp(1e-15,1))).sum(dim=2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['grad_clip']); opt.step()
            for k,v in model.state_dict().items():
                if torch.is_floating_point(v): ema[k].mul_(0.997875).add_(v.detach(),alpha=0.002125)
            for dm in model._drops: dm.p=apply_schedule(CFG['dropout'],progress,'expm4t')

        model.eval(); live={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema,strict=True)
        with torch.no_grad(): vp=model(xv).mean(dim=1).cpu().numpy()
        model.load_state_dict(live,strict=True)
        sc=balanced_accuracy_score(yv.cpu(),np.argmax(vp,axis=1))
        if sc>best_score: best_score=sc; best_state={k:v.detach().clone() for k,v in ema.items()}
        pr(f"  ep{ep+1}:{sc:.5f}{' *' if sc>best_score else ''}")

    model.load_state_dict(best_state,strict=True); model.eval()
    with torch.no_grad():
        oof[val_idx]=model(xv).mean(dim=1).cpu().numpy()
        # Test preds (batched)
        for s in range(0,len(X_tst),CFG['eval_bs']):
            tp[s:s+CFG['eval_bs']] += model(torch.tensor(X_tst[s:s+CFG['eval_bs']]).to(DEVICE)).mean(dim=1).cpu().numpy()/5
    fold_scores.append(best_score)
    del model; gc.collect(); torch.cuda.empty_cache()

ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))  # y is numpy, oof is numpy
pr(f"\nDeotte RealMLP OOF: {ba:.5f} folds={[f'{s:.5f}' for s in fold_scores]}")
np.save('oof_Deotte_RealMLP.npy',oof); np.save('test_Deotte_RealMLP.npy',tp)
pr("Saved!")
