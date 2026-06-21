"""RealMLP ablation: A(current) vs B(beta2) vs C(grouped) vs D(Deotte)"""
import pandas as pd, numpy as np, math, time, gc, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; FOLDS = 5; N_CLASSES = 3
pr(f"Device: {DEVICE}")

# ====== Data ======
pr("Loading data...")
train = pd.read_csv("train_fe.csv"); test = pd.read_csv("test_fe.csv")
y_all = LabelEncoder().fit_transform(train['class'])

color_pairs = [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]
for df in [train, test]:
    df['_g_div_redshift'] = (df['g']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    df['_i_div_redshift'] = (df['i']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    for a,b in color_pairs: df[f'_{a}-{b}'] = (df[a]-df[b]).astype('float32')
    mags = df[['u','g','r','i','z']].astype('float32')
    df['_mag_mean'] = mags.mean(axis=1).astype('float32')
    df['_mag_range'] = (mags.max(axis=1)-mags.min(axis=1)).astype('float32')
    df['_log1p_redshift'] = np.log1p(df['redshift'].clip(lower=0)+1e-4).astype('float32')
    for col in ['alpha','delta','u','g','r','i','z','redshift']:
        df[f'{col}_cat_'] = np.floor(df[col]).astype('int32').astype('category')
    for n_bins in [100,500]:
        kb = KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='quantile',subsample=None)
        df[f'delta_{n_bins}_bin_'] = pd.Series(kb.fit_transform(df[['delta']]).ravel().astype('int32'),index=df.index).astype('category')

for df in [train,test]:
    df['_ca'] = df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_cz'] = df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)
for combo in ['_ca','_cz']:
    for ci in range(3):
        cn=f'{combo}_TE_{ci}'; yb=(y_all==ci).astype(int)
        te=TargetEncoder(cv=5,smooth='auto',random_state=42)
        train[cn]=te.fit_transform(train[[combo]],yb).ravel()
        te2=TargetEncoder(smooth='auto',random_state=42); te2.fit(train[[combo]],yb)
        test[cn]=te2.transform(test[[combo]]).ravel()
train.drop(['_ca','_cz'],axis=1,inplace=True); test.drop(['_ca','_cz'],axis=1,inplace=True)
for col in ['spectral_type','galaxy_population']:
    train[col+'_enc']=LabelEncoder().fit_transform(train[col]); test[col+'_enc']=LabelEncoder().fit_transform(test[col])

num_cols = ['u','g','r','i','z','redshift','alpha','delta','u_g','g_r','r_i','i_z','u_r','g_i','r_z',
            'color_curv','u_z','g_z','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = list(dict.fromkeys([c for c in num_cols+new_cols if c in train.columns]))
X_all_num = train[feat_all].values.astype(np.float32); X_test_num = test[feat_all].values.astype(np.float32)

# Preprocessing: median+robust
class NumPrep:
    def fit(self,X):
        self.median=np.median(X,axis=0); qd=np.quantile(X,0.75,axis=0)-np.quantile(X,0.25,axis=0)
        qd[qd==0]=0.5*(X.max(axis=0)[qd==0]-X.min(axis=0)[qd==0]); qd[qd==0]=1.0
        self.scale=1.0/(qd+1e-30)
    def transform(self,X): return (X.copy()-self.median)*self.scale
prep=NumPrep(); prep.fit(X_all_num); X_all_num=prep.transform(X_all_num); X_test_num=prep.transform(X_test_num)
pr(f"Features: {X_all_num.shape[1]}")

# ====== Model components ======
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
    def __init__(self,n_ens,n_f,hidden_dim=16,out_dim=5,freq_scale=2.33):
        super().__init__(); self.n_f=n_f; self.out_dim=out_dim
        self.w1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim)*freq_scale)
        self.b1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim))
        self.w2=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim,out_dim-1)/math.sqrt(hidden_dim))
        self.b2=nn.Parameter(torch.zeros(n_ens,n_f,out_dim-1))
        nn.init.uniform_(self.b1,-math.pi,math.pi)
    def forward(self,x):
        p=torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w1.unsqueeze(0)+self.b1.unsqueeze(0)))
        t=torch.einsum("bkfh,kfhd->bkfd",p,self.w2)+self.b2.unsqueeze(0)
        return torch.cat([x.unsqueeze(-1),t],dim=-1).flatten(start_dim=2)

# ====== Ablation configs ======
ABLATIONS = [
    ('A (current)',  dict(beta2=0.999, grouped=False, sched=False)),
    ('B (+beta2)',   dict(beta2=0.98,  grouped=False, sched=False)),
    ('C (+grouped)', dict(beta2=0.98,  grouped=True,  sched=False)),
    ('D (Deotte)',   dict(beta2=0.98,  grouped=True,  sched=True)),
]

BASE = dict(n_ens=8, hidden_dim=[512,512,512], dropout=0.044, add_front_scale=True,
            pbld_hidden_dim=16, pbld_out_dim=5, pbld_freq_scale=2.33,
            lr=0.01, weight_decay=0.0125, epochs=6, train_bs=256, ls_eps=0.04,
            grad_clip=1.0, ema_decay=0.997875)

def apply_schedule(init, progress, sched):
    if sched=='flat_cos':
        if progress<0.2: return init
        t=(progress-0.2)/0.8; return init*(math.cos(math.pi*t)+1)/2
    if sched=='cos': return init*(math.cos(math.pi*progress)+1)/2
    return init

results = {}

for ab_name, ab in ABLATIONS:
    pr(f"\n{'='*40}\n{ab_name}\n{'='*40}")
    oof=np.zeros((len(y_all),N_CLASSES),dtype=np.float32)
    fold_scores=[]

    for fold,(tr_idx,val_idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X_all_num,y_all)):
        torch.manual_seed(SEED+fold*100)
        X_tr=torch.tensor(X_all_num[tr_idx]).to(DEVICE); y_tr=torch.tensor(y_all[tr_idx]).long().to(DEVICE)
        X_val=torch.tensor(X_all_num[val_idx]).to(DEVICE); y_val=torch.tensor(y_all[val_idx]).long().to(DEVICE)

        class RealMLP(nn.Module):
            def __init__(self):
                super().__init__(); self.n_ens=BASE['n_ens']
                self.num_embed=PBLDEmbedding(self.n_ens,X_all_num.shape[1])
                d=X_all_num.shape[1]*5; dims=BASE['hidden_dim']; act=nn.GELU()
                layers=[]; self._drops=[]
                if BASE['add_front_scale']: layers.append(ScalingLayer(self.n_ens,d))
                for out in dims:
                    layers.append(NTPLinear(self.n_ens,d,out)); layers.append(act)
                    drop=nn.Dropout(BASE['dropout']); layers.append(drop); self._drops.append(drop); d=out
                self.hidden=nn.Sequential(*layers); self.out=NTPLinear(self.n_ens,d,N_CLASSES)
            def forward(self,x): x=x.unsqueeze(1).expand(-1,self.n_ens,-1); x=self.num_embed(x); x=self.hidden(x); return F.softmax(self.out(x),dim=2)

        model=RealMLP().to(DEVICE)

        # Optimizer
        if ab['grouped']:
            scale_p,pbld_p,first_w,other_w,bias_p=[],[],[],[],[]
            first_w_id=id(model.out.weight)
            for n,p in model.named_parameters():
                if 'num_embed' in n: pbld_p.append(p)
                elif 'scale' in n: scale_p.append(p)
                elif 'bias' in n: bias_p.append(p)
                elif id(p)==first_w_id: first_w.append(p)
                else: other_w.append(p)
            lr=BASE['lr']; wd=BASE['weight_decay']
            groups=[{'params':scale_p,'lr':lr*10,'weight_decay':wd*0.1},
                    {'params':pbld_p,'lr':lr*0.115,'weight_decay':wd},
                    {'params':first_w,'lr':lr,'weight_decay':wd*0.1},
                    {'params':other_w,'lr':lr,'weight_decay':wd},
                    {'params':bias_p,'lr':lr*0.1,'weight_decay':wd*0.5}]
            opt=torch.optim.AdamW(groups,lr=lr,weight_decay=wd,betas=(0.9,ab['beta2']))
        else:
            opt=torch.optim.AdamW(model.parameters(),lr=BASE['lr'],weight_decay=BASE['weight_decay'],betas=(0.9,ab['beta2']))

        total_steps=BASE['epochs']*len(y_tr); step=0; best_score=-np.inf; best_state=None
        ema={k:v.detach().clone() for k,v in model.state_dict().items()}

        for epoch in range(BASE['epochs']):
            model.train(); perm=torch.randperm(len(y_tr))
            for start in range(0,len(y_tr),BASE['train_bs']):
                idx=perm[start:start+BASE['train_bs']]; progress=step/total_steps; step+=1
                if ab['sched']:
                    for g in opt.param_groups: g['lr']=apply_schedule(g.get('lr_base',g['lr']),progress,'flat_cos')
                opt.zero_grad()
                yp=model(X_tr[idx])
                ls=apply_schedule(BASE['ls_eps'],progress,'cos') if ab['sched'] else BASE['ls_eps']
                ys=torch.full_like(yp[:,0,:],ls/N_CLASSES)
                ys.scatter_(1,y_tr[idx].unsqueeze(1),1.0-ls+ls/N_CLASSES)
                ys=ys.unsqueeze(1).expand(-1,BASE['n_ens'],-1)
                loss=-(ys*torch.log(yp.clamp(1e-15,1))).sum(dim=2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),BASE['grad_clip']); opt.step()
                for k,v in model.state_dict().items():
                    if torch.is_floating_point(v): ema[k].mul_(0.997875).add_(v.detach(),alpha=0.002125)
                dp=BASE['dropout']
                if ab['sched']: dp=apply_schedule(dp,progress,'expm4t')
                for dm in model._drops: dm.p=dp

            model.eval(); live={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(ema,strict=True)
            with torch.no_grad(): vp=model(X_val).mean(dim=1).cpu().numpy()
            model.load_state_dict(live,strict=True)
            sc=balanced_accuracy_score(y_val.cpu(),np.argmax(vp,axis=1))
            if sc>best_score: best_score=sc; best_state={k:v.detach().clone() for k,v in ema.items()}

        model.load_state_dict(best_state,strict=True); model.eval()
        with torch.no_grad(): oof[val_idx]=model(torch.tensor(X_all_num[val_idx]).to(DEVICE)).mean(dim=1).cpu().numpy()
        fold_scores.append(best_score)
        del model; gc.collect(); torch.cuda.empty_cache()
        pr(f"  Fold {fold+1}: {best_score:.5f}")

    ba=balanced_accuracy_score(y_all,np.argmax(oof,axis=1))
    pr(f"  OOF BA: {ba:.5f}  folds={[f'{s:.5f}' for s in fold_scores]}")
    results[ab_name]=ba

pr(f"\n{'='*40}\nSUMMARY\n{'='*40}")
prev = None
for k,v in results.items():
    delta = f"+{v-prev:.5f}" if prev else ""
    pr(f"  {k}: {v:.5f} {delta}")
    prev = v
