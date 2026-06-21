"""
FT-Transformer: 5-fold CV for Playground S6E6
Saves OOF predictions for ensemble
"""
import pandas as pd, numpy as np, time, gc, math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ========== 1. Data + Features (same as pipeline) ==========
pr("Loading data...")
train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
y_all = LabelEncoder().fit_transform(train['class']).astype(np.int64)

# Feature engineering (simplified from pipeline)
for df in [train, test]:
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]:
        df[f'{a}_{b}'] = df[a]-df[b]
    df['u_z']=df['u']-df['z']; df['g_z']=df['g']-df['z']
    alpha_rad=np.deg2rad(df['alpha'])
    df['alpha_sin']=np.sin(alpha_rad); df['alpha_cos']=np.cos(alpha_rad)
    df['_g_div_redshift']=(df['g']/(df['redshift']+1e-6)).clip(-10,10)
    df['_i_div_redshift']=(df['i']/(df['redshift']+1e-6)).clip(-10,10)
    df['_mag_mean']=df[['u','g','r','i','z']].mean(axis=1)
    df['_mag_range']=df[['u','g','r','i','z']].max(axis=1)-df[['u','g','r','i','z']].min(axis=1)
    df['_log1p_redshift']=np.log1p(df['redshift'].clip(lower=0))
    for col in ['alpha','delta','u','g','r','i','z','redshift']:
        df[f'{col}_cat_']=np.floor(df[col]).astype('int32')
    for n in [100,500]:
        kb=KBinsDiscretizer(n_bins=n,encode='ordinal',strategy='quantile',subsample=None)
        df[f'delta_{n}_bin_']=kb.fit_transform(df[['delta']]).ravel().astype('int32')

for col in ['spectral_type','galaxy_population']:
    train[col+'_enc']=LabelEncoder().fit_transform(train[col])
    test[col+'_enc']=LabelEncoder().fit_transform(test[col])

for df in [train,test]:
    df['_ca']=df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_cz']=df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)
for combo in ['_ca','_cz']:
    for ci in range(3):
        cn=f'{combo}_TE_{ci}'; yb=(y_all==ci).astype(int)
        te=TargetEncoder(cv=5,smooth='auto',random_state=42)
        train[cn]=te.fit_transform(train[[combo]],yb).ravel()
        te2=TargetEncoder(smooth='auto',random_state=42); te2.fit(train[[combo]],yb)
        test[cn]=te2.transform(test[[combo]]).ravel()
train.drop(['_ca','_cz'],axis=1,inplace=True); test.drop(['_ca','_cz'],axis=1,inplace=True)

# Numerical features only (FT-Transformer doesn't need categorical encoding)
num_cols = ['u','g','r','i','z','redshift','alpha','delta'] + \
           [c for c in train.columns if c.startswith('_') or '_TE_' in c]
num_cols = [c for c in num_cols if c in train.columns]
# Add derived continuous features
extra = ['u_g','g_r','r_i','i_z','u_r','g_i','r_z','u_z','g_z','alpha_sin','alpha_cos',
         'spectral_type_enc','galaxy_population_enc']
num_cols = num_cols + [c for c in extra if c in train.columns]
num_cols = list(dict.fromkeys(num_cols))  # dedupe

X_num = train[num_cols].values.astype(np.float32)
X_test_num = test[num_cols].values.astype(np.float32)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_num)
X_num = scaler.transform(X_num)
X_test_num = scaler.transform(X_test_num)

pr(f"Features: {len(num_cols)}, Train: {X_num.shape}, Test: {X_test_num.shape}")

# ========== 2. FT-Transformer Model ==========
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, F) → (B, F, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.n_heads, self.d_k = d, n_heads, d // n_heads
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1,2).contiguous().view(B, N, D)
        return self.W_o(out)

class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_token=192, n_layers=3,
                 n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.01)
        self.layers = nn.ModuleList([
            TransformerLayer(d_token, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        # x: (B, F)
        tokens = self.tokenizer(x)  # (B, F, d_token)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_token)
        x = torch.cat([cls, tokens], dim=1)  # (B, 1+F, d_token)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])  # take CLS

# ========== 3. Training ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pr(f"Device: {DEVICE}")
n_features = X_num.shape[1]
n_classes = 3

CONFIG = dict(
    d_token=192, n_layers=3, n_heads=8, d_ff=512, dropout=0.15,
    lr=3e-4, weight_decay=1e-4, batch_size=512, epochs=60,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_num), n_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test_num), n_classes), dtype=np.float32)
fold_scores = []

for fold, (tr, val) in enumerate(skf.split(X_num, y_all)):
    pr(f"\nFold {fold+1}/5")
    X_tr, X_val = X_num[tr], X_num[val]
    y_tr, y_val = y_all[tr], y_all[val]

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size']*4)

    model = FTTransformer(n_features, n_classes,
                          CONFIG['d_token'], CONFIG['n_layers'],
                          CONFIG['n_heads'], CONFIG['d_ff'], CONFIG['dropout'])
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs'])
    best_score = -np.inf
    best_state = None

    for epoch in range(CONFIG['epochs']):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in val_dl:
                val_preds.append(model(bx.to(DEVICE)).cpu().numpy())
        val_proba = np.concatenate(val_preds, axis=0)
        val_labels = np.argmax(val_proba, axis=1)
        score = balanced_accuracy_score(y_val, val_labels)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 15 == 0:
            pr(f"  epoch {epoch+1}: BA={score:.5f}")

    model.load_state_dict(best_state)
    pr(f"  Fold {fold+1} best: {best_score:.5f}")

    # OOF predictions
    model.eval()
    val_dl_all = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=CONFIG['batch_size']*4)
    with torch.no_grad():
        for i, (bx,) in enumerate(val_dl_all):
            oof[val[i*CONFIG['batch_size']*4:(i+1)*CONFIG['batch_size']*4]] = \
                F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy()

    # Test predictions
    test_dl = DataLoader(TensorDataset(torch.tensor(X_test_num)), batch_size=CONFIG['batch_size']*4)
    with torch.no_grad():
        for i, (bx,) in enumerate(test_dl):
            test_preds[i*CONFIG['batch_size']*4:(i+1)*CONFIG['batch_size']*4] += \
                F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy() / 5

    fold_scores.append(best_score)
    del model; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

oof_score = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nFT-Transformer OOF BA: {oof_score:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")

# ========== 4. Save ==========
np.save('oof_FT_Transformer.npy', oof)
np.save('test_FT_Transformer.npy', test_preds)
pr(f"Saved: oof_FT_Transformer.npy ({oof.shape}), test_FT_Transformer.npy ({test_preds.shape})")

# Quick ensemble check with existing OOFs
import glob
existing = sorted(glob.glob('oof_*.npy'))
pr(f"\nExisting OOFs: {len(existing)}")
pr("Ready for ensemble!")
