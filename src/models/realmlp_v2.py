"""
RealMLP v2: robust scaling + PBLD PReLU + feature selection
"""
import pandas as pd, numpy as np, math, gc, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; N_CLASSES = 3

# ====== Data ======
pr("Loading...")
train = pd.read_csv("train_fe.csv"); test = pd.read_csv("test_fe.csv")
y_all = LabelEncoder().fit_transform(train['class'])

for df in [train,test]:
    df['_g_div_r']=(df['g']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    df['_i_div_r']=(df['i']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]:
        df[f'_{a}-{b}']=(df[a]-df[b]).astype('float32')
    df['_mag_mean']=df[['u','g','r','i','z']].mean(axis=1).astype('float32')
    df['_mag_range']=(df[['u','g','r','i','z']].max(axis=1)-df[['u','g','r','i','z']].min(axis=1)).astype('float32')
    df['_log1p_z']=np.log1p(df['redshift'].clip(0)+1e-4).astype('float32')
    for col in ['alpha','delta','u','g','r','i','z','redshift']:
        df[f'{col}_cat_']=np.floor(df[col]).astype('int32').astype('category')
    for n in[100,500]:
        kb=KBinsDiscretizer(n_bins=n,encode='ordinal',strategy='quantile',subsample=None)
        df[f'd_{n}_bin_']=pd.Series(kb.fit_transform(df[['delta']]).ravel().astype('int32'),index=df.index).astype('category')
for df in [train,test]:
    df['_ca']=df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_cz']=df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)
for combo in['_ca','_cz']:
    for ci in range(3):
        cn=f'{combo}_TE_{ci}'; yb=(y_all==ci).astype(int)
        te=TargetEncoder(cv=5,smooth='auto',random_state=42)
        train[cn]=te.fit_transform(train[[combo]],yb).ravel()
        te2=TargetEncoder(smooth='auto',random_state=42); te2.fit(train[[combo]],yb)
        test[cn]=te2.transform(test[[combo]]).ravel()
train.drop(['_ca','_cz'],axis=1,inplace=True); test.drop(['_ca','_cz'],axis=1,inplace=True)
for col in['spectral_type','galaxy_population']:
    train[col+'_enc']=LabelEncoder().fit_transform(train[col]); test[col+'_enc']=LabelEncoder().fit_transform(test[col])

base = ['u','g','r','i','z','redshift','alpha','delta','u_g','g_r','r_i','i_z','u_r','g_i','r_z']
extra = ['color_curv','u_z','g_z','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc']
newc = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feats = list(dict.fromkeys([c for c in base+extra+newc if c in train.columns]))

# Feature selection: drop features that hurt XGBoost (noise)
# Keep only features that benefit both trees and MLP
DROP = ['XGB_d6','XGB_d8','XGB_d10']  # XGB underperformed with color_curv and some TE
# Simplified: remove color_curv and g_z which added noise
feats = [c for c in feats if c not in ['color_curv','g_z']]

X = train[feats].values.astype(np.float32); Xt = test[feats].values.astype(np.float32)

# ====== Robust scaling (median_center + robust_scale) ======
class RobustScaler:
    def fit(self,X):
        self.median=np.median(X,axis=0); qd=np.quantile(X,0.75,axis=0)-np.quantile(X,0.25,axis=0)
        qd[qd==0]=0.5*(X.max(axis=0)[qd==0]-X.min(axis=0)[qd==0]); qd[qd==0]=1.0
        self.scale=1.0/(qd+1e-30)
    def transform(self,X): return (X.copy()-self.median)*self.scale

rs = RobustScaler(); rs.fit(X); X = rs.transform(X); Xt = rs.transform(Xt)
NF = X.shape[1]; pr(f"Features: {NF} (robust scaled)")

# ====== Model ======
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
        super().__init__(); self.act=nn.PReLU(n_ens*n_f)  # ← PBLD PReLU!
        self.w1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim)*freq_scale)
        self.b1=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim))
        self.w2=nn.Parameter(torch.randn(n_ens,n_f,hidden_dim,out_dim-1)/math.sqrt(hidden_dim))
        self.b2=nn.Parameter(torch.zeros(n_ens,n_f,out_dim-1))
        nn.init.uniform_(self.b1,-math.pi,math.pi)
    def forward(self,x):
        p=torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w1.unsqueeze(0)+self.b1.unsqueeze(0)))
        t=torch.einsum("bkfh,kfhd->bkfd",p,self.w2)+self.b2.unsqueeze(0)
        return torch.cat([x.unsqueeze(-1),t],dim=-1).flatten(start_dim=2)

class RealMLP(nn.Module):
    def __init__(self,n_f,n_cls):
        super().__init__(); self.n_ens=8
        self.num_embed=PBLDEmbedding(8,n_f)
        d=n_f*5
        layers=[ScalingLayer(8,d)]
        self._drops=[]
        for out in[512,512,512]:
            layers.append(NTPLinear(8,d,out)); layers.append(nn.GELU())
            drop=nn.Dropout(0.044); layers.append(drop); self._drops.append(drop); d=out
        self.hidden=nn.Sequential(*layers); self.out=NTPLinear(8,d,3)
    def forward(self,x):
        x=x.unsqueeze(1).expand(-1,8,-1); x=self.num_embed(x)
        x=self.hidden(x); return F.softmax(self.out(x),dim=2)

# ====== Training (5-fold, D config) ======
oof=np.zeros((len(y_all),3),dtype=np.float32); tp=np.zeros((Xt.shape[0],3),dtype=np.float32)
fold_scores=[]
for fold,(tr,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y_all)):
    torch.manual_seed(SEED+fold*100)
    xtr=torch.tensor(X[tr]).to(DEVICE); ytr=torch.tensor(y_all[tr]).long().to(DEVICE)
    xv=torch.tensor(X[val]).to(DEVICE); yv=torch.tensor(y_all[val]).long().to(DEVICE)

    model=RealMLP(NF,N_CLASSES).to(DEVICE)
    # 5-group optimizer
    scale_p,pbld_p,other_w,bias_p=[],[],[],[]
    for n,p in model.named_parameters():
        if 'num_embed' in n: pbld_p.append(p)
        elif 'scale' in n: scale_p.append(p)
        elif 'bias' in n: bias_p.append(p)
        else: other_w.append(p)
    groups=[{'params':scale_p,'lr':0.1,'weight_decay':0.00125},
            {'params':pbld_p,'lr':0.00115,'weight_decay':0.0125},
            {'params':other_w,'lr':0.01,'weight_decay':0.0125},
            {'params':bias_p,'lr':0.001,'weight_decay':0.00625}]
    opt=torch.optim.AdamW(groups,betas=(0.9,0.98))
    total_steps=6*len(ytr); step=0; best=-np.inf; best_state=None
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}

    for ep in range(6):
        model.train(); perm=torch.randperm(len(ytr))
        for s in range(0,len(ytr),256):
            idx=perm[s:s+256]; progress=step/total_steps; step+=1
            for g in opt.param_groups: g['lr']=g['lr']*(math.cos(math.pi*max(0,min(1,(progress-0.2)/0.8)))+1)/2
            opt.zero_grad(); yp=model(xtr[idx])
            ls=0.04*(math.cos(math.pi*progress)+1)/2
            ys=torch.full_like(yp[:,0,:],ls/3); ys.scatter_(1,ytr[idx].unsqueeze(1),1-ls+ls/3)
            ys=ys.unsqueeze(1).expand(-1,8,-1)
            loss=-(ys*torch.log(yp.clamp(1e-15,1))).sum(dim=2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            for k,v in model.state_dict().items():
                if torch.is_floating_point(v): ema[k].mul_(0.997875).add_(v.detach(),alpha=0.002125)
            for dm in model._drops: dm.p=0.044*math.exp(-4*progress)

        model.eval(); live={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema,strict=True)
        with torch.no_grad(): vp=model(xv).mean(dim=1).cpu().numpy()
        model.load_state_dict(live,strict=True)
        sc=balanced_accuracy_score(yv,np.argmax(vp,axis=1))
        if sc>best: best=sc; best_state={k:v.detach().clone() for k,v in ema.items()}

    model.load_state_dict(best_state,strict=True); model.eval()
    with torch.no_grad(): oof[val]=model(xv).mean(dim=1).cpu().numpy()
    fold_scores.append(best)
    pr(f"Fold {fold+1}:{best:.5f}")
    del model; gc.collect(); torch.cuda.empty_cache()

ba=balanced_accuracy_score(y_all,np.argmax(oof,axis=1))
pr(f"\nRealMLP v2 OOF: {ba:.5f} folds={[f'{s:.5f}' for s in fold_scores]}")
pr(f"vs v1: +{ba-0.95981:.5f}")
np.save('oof_RealMLP_v2.npy',oof)
