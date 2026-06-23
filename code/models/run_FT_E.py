"""FT-E: d_token=256 + D's regularization, d_ff=512 (not 768)"""
import pandas as pd, numpy as np, math, gc, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] [E] {msg}", flush=True)

DEVICE=torch.device('cuda'); SEED=42; N_CLASSES=3
CFG=dict(n_layers=3,d_token=256,n_heads=8,d_ff=512,dropout=0.25,lr=1e-3,wd=1e-3,epochs=60,bs=256)
pr(f"CFG: {CFG}")

train=pd.read_csv("train_fe.csv"); test=pd.read_csv("test_fe.csv")
y_all=LabelEncoder().fit_transform(train['class'])

for df in [train,test]:
    df['_g_div_r']=(df['g']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    df['_i_div_r']=(df['i']/(df['redshift']+1e-6)).clip(-10,10).astype('float32')
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]:
        df[f'_{a}-{b}']=(df[a]-df[b]).astype('float32')
    df['_mag_mean']=df[['u','g','r','i','z']].astype('float32').mean(axis=1)
    df['_mag_range']=df[['u','g','r','i','z']].astype('float32').max(axis=1)-df[['u','g','r','i','z']].astype('float32').min(axis=1)
    df['_log1p_z']=np.log1p(df['redshift'].clip(lower=0)+1e-4).astype('float32')
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

nc=['u','g','r','i','z','redshift','alpha','delta','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv','u_z','g_z','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc']
newc=[c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feats=list(dict.fromkeys([c for c in nc+newc if c in train.columns]))
X=StandardScaler().fit_transform(train[feats].values.astype(np.float32))
Xt=StandardScaler().fit_transform(test[feats].values.astype(np.float32))
NF=X.shape[1]; pr(f'Features:{NF}')

class FT(nn.Module):
    def __init__(self,nf,d,nl,nh,d_ff,do):
        super().__init__()
        self.w=nn.Parameter(torch.empty(nf,d)); self.b=nn.Parameter(torch.empty(nf,d))
        nn.init.normal_(self.w,std=0.01); nn.init.zeros_(self.b)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.normal_(self.cls,std=0.01)
        self.layers=nn.ModuleList()
        for _ in range(nl):
            self.layers.append(nn.ModuleList([
                nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d),
                nn.LayerNorm(d),nn.LayerNorm(d),nn.Dropout(do),nn.Dropout(do),
                nn.Linear(d,d_ff),nn.ReLU(),nn.Linear(d_ff,d)
            ]))
        self.head=nn.Linear(d,N_CLASSES)
    def forward(self,x):
        t=x.unsqueeze(-1)*self.w+self.b; c=self.cls.expand(x.size(0),-1,-1)
        x=torch.cat([c,t],dim=1); B,N,D=x.shape; H=8; dk=D//H
        for wq,wk,wv,wo,ln1,ln2,do1,do2,ff1,act,ff2 in self.layers:
            x_ln=ln1(x); q=wq(x_ln).view(B,N,H,dk).transpose(1,2); k=wk(x_ln).view(B,N,H,dk).transpose(1,2)
            v=wv(x_ln).view(B,N,H,dk).transpose(1,2)
            attn=F.softmax((q@k.transpose(-2,-1))/math.sqrt(dk),dim=-1); attn=do1(attn)
            x=x+do2(wo((attn@v).transpose(1,2).contiguous().view(B,N,D)))
            x=x+ff2(act(ff1(do2(ln2(x)))))
        return self.head(x[:,0,:])

oof=np.zeros((len(y_all),N_CLASSES),dtype=np.float32)
tp=np.zeros((Xt.shape[0],N_CLASSES),dtype=np.float32)
for fold,(tr,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y_all)):
    torch.manual_seed(SEED+fold*100)
    xtr=torch.tensor(X[tr]).to(DEVICE); ytr=torch.tensor(y_all[tr]).long().to(DEVICE)
    model=FT(NF,CFG['d_token'],CFG['n_layers'],CFG['n_heads'],CFG['d_ff'],CFG['dropout']).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CFG['epochs'])
    best=-np.inf; bs=CFG['bs']
    for ep in range(CFG['epochs']):
        model.train(); perm=torch.randperm(len(ytr))
        for s in range(0,len(ytr),bs):
            idx=perm[s:s+bs]; opt.zero_grad()
            F.cross_entropy(model(xtr[idx]),ytr[idx]).backward(); opt.step()
        sch.step()
        model.eval(); xv=torch.tensor(X[val]).to(DEVICE); preds=[]
        with torch.no_grad():
            for s in range(0,len(xv),bs*4): preds.append(F.softmax(model(xv[s:s+bs*4]),dim=-1).cpu().numpy())
        sc=balanced_accuracy_score(y_all[val],np.argmax(np.concatenate(preds),axis=1))
        if sc>best: best=sc; bs_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(bs_state,strict=True); model.eval()
    with torch.no_grad():
        xv=torch.tensor(X[val]).to(DEVICE); p=[]
        for s in range(0,len(xv),bs*4): p.append(F.softmax(model(xv[s:s+bs*4]),dim=-1).cpu().numpy())
        oof[val]=np.concatenate(p)
        xtt=torch.tensor(Xt).to(DEVICE); pt=[]
        for s in range(0,len(xtt),bs*4): pt.append(F.softmax(model(xtt[s:s+bs*4]),dim=-1).cpu().numpy())
        tp+=np.concatenate(pt)/5
    pr(f'Fold {fold+1}:{best:.5f}')
    del model; gc.collect(); torch.cuda.empty_cache()

ba=balanced_accuracy_score(y_all,np.argmax(oof,axis=1))
pr(f'FT-E OOF:{ba:.5f}')
np.save('oof_FT_E.npy',oof); np.save('test_FT_E.npy',tp)
pr('Saved!')
