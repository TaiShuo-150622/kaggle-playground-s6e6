"""
FT-Transformer ablation: 4 configs, can run in parallel
Usage: python ablation_ft.py <config_id>
  config_id: A, B, C, or D
"""
import sys, pandas as pd, numpy as np, math, time, gc, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{config_id}] {msg}", flush=True)

config_id = sys.argv[1] if len(sys.argv) > 1 else 'A'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; FOLDS = 5; N_CLASSES = 3

# ====== 4 FT Configs ======
FT_CONFIGS = {
    'A': dict(name='baseline', n_layers=3, d_token=192, n_heads=8, d_ff=512,
              dropout=0.15, lr=3e-4, wd=1e-4, epochs=60, bs=512),
    'B': dict(name='deeper', n_layers=6, d_token=192, n_heads=8, d_ff=512,
              dropout=0.15, lr=3e-4, wd=1e-4, epochs=60, bs=512),
    'C': dict(name='wider', n_layers=3, d_token=256, n_heads=8, d_ff=768,
              dropout=0.15, lr=3e-4, wd=1e-4, epochs=60, bs=512),
    'D': dict(name='regularized', n_layers=3, d_token=192, n_heads=8, d_ff=512,
              dropout=0.25, lr=1e-3, wd=1e-3, epochs=80, bs=512),
}

CFG = FT_CONFIGS[config_id]
pr(f"Config {config_id}: {CFG['name']}")

# ====== Data (same as before) ======
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

X_all = train[feat_all].values.astype(np.float32)
X_test_arr = test[feat_all].values.astype(np.float32)
scaler = StandardScaler(); X_all = scaler.fit_transform(X_all); X_test_arr = scaler.transform(X_test_arr)
N_FEATURES = X_all.shape[1]
pr(f"Features: {N_FEATURES}")

# ====== FT-Transformer Model ======
class FeatureTokenizer(nn.Module):
    def __init__(self, n_f, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_f, d_token))
        self.bias = nn.Parameter(torch.empty(n_f, d_token))
        nn.init.normal_(self.weight, std=0.01); nn.init.zeros_(self.bias)
    def forward(self, x): return x.unsqueeze(-1) * self.weight + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads=n_heads; self.d_k=d//n_heads
        self.W_q=nn.Linear(d,d); self.W_k=nn.Linear(d,d); self.W_v=nn.Linear(d,d); self.W_o=nn.Linear(d,d)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x):
        B,N,D=x.shape; H=self.n_heads; dk=self.d_k
        q=self.W_q(x).view(B,N,H,dk).transpose(1,2); k=self.W_k(x).view(B,N,H,dk).transpose(1,2)
        v=self.W_v(x).view(B,N,H,dk).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(dk); attn=F.softmax(scores,dim=-1); attn=self.dropout(attn)
        out=(attn@v).transpose(1,2).contiguous().view(B,N,D); return self.W_o(out)

class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.attn=MultiHeadAttention(d,n_heads,dropout)
        self.ffn=nn.Sequential(nn.Linear(d,d_ff),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_ff,d),nn.Dropout(dropout))
        self.norm1=nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
    def forward(self,x): x=x+self.attn(self.norm1(x)); x=x+self.ffn(self.norm2(x)); return x

class FTTransformer(nn.Module):
    def __init__(self, n_f, n_cls, d_token=192, n_layers=3, n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.tokenizer=FeatureTokenizer(n_f,d_token)
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_token)); nn.init.normal_(self.cls_token,std=0.01)
        self.layers=nn.ModuleList([TransformerLayer(d_token,n_heads,d_ff,dropout) for _ in range(n_layers)])
        self.head=nn.Linear(d_token,n_cls)
    def forward(self, x):
        tokens=self.tokenizer(x); cls=self.cls_token.expand(x.size(0),-1,-1)
        x=torch.cat([cls,tokens],dim=1)
        for layer in self.layers: x=layer(x)
        return self.head(x[:,0,:])

# ====== Training ======
oof = np.zeros((len(y_all), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_test_arr), N_CLASSES), dtype=np.float32)
fold_scores = []

for fold,(tr_idx,val_idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X_all,y_all)):
    torch.manual_seed(SEED+fold*100)
    X_tr=torch.tensor(X_all[tr_idx]).to(DEVICE); y_tr=torch.tensor(y_all[tr_idx]).long().to(DEVICE)
    X_val=torch.tensor(X_all[val_idx]).to(DEVICE); y_val=torch.tensor(y_all[val_idx]).long().to(DEVICE)

    model = FTTransformer(N_FEATURES, N_CLASSES, CFG['d_token'], CFG['n_layers'],
                          CFG['n_heads'], CFG['d_ff'], CFG['dropout']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    best_score = -np.inf; best_state = None

    for epoch in range(CFG['epochs']):
        model.train(); perm=torch.randperm(len(y_tr))
        for start in range(0,len(y_tr),CFG['bs']):
            idx=perm[start:start+CFG['bs']]; opt.zero_grad()
            loss=F.cross_entropy(model(X_tr[idx]),y_tr[idx]); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vp=model(X_val).cpu().numpy()
        sc=balanced_accuracy_score(y_val.cpu(),np.argmax(vp,axis=1))
        if sc>best_score: best_score=sc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state,strict=True); model.eval()
    with torch.no_grad():
        oof[val_idx]=F.softmax(model(torch.tensor(X_all[val_idx]).to(DEVICE)),dim=-1).cpu().numpy()
        test_preds+=F.softmax(model(torch.tensor(X_test_arr).to(DEVICE)),dim=-1).cpu().numpy()/FOLDS
    fold_scores.append(best_score)
    pr(f"  Fold {fold+1}: {best_score:.5f}")
    del model; gc.collect(); torch.cuda.empty_cache()

ba=balanced_accuracy_score(y_all,np.argmax(oof,axis=1))
pr(f"OOF BA: {ba:.5f} folds={[f'{s:.5f}' for s in fold_scores]}")
np.save(f'oof_FT_{config_id}.npy', oof)
np.save(f'test_FT_{config_id}.npy', test_preds)
pr(f"Saved: oof_FT_{config_id}.npy, test_FT_{config_id}.npy")
