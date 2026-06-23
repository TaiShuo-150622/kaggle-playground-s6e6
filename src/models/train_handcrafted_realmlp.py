"""
Hand-crafted RealMLP — Deotte's R2-103 configuration
======================================================
Based on Chris Deotte's realmlp-v5-for-s6e6 notebook.
Key: PBLD embeddings + NTPLinear + n_ens=8 + EMA + 6 epochs
"""
import pandas as pd, numpy as np, math, time, gc, os
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; FOLDS = 5; N_CLASSES = 3
pr(f"Device: {DEVICE}")

# ============================================================
# 1. Feature Engineering (same as Deotte's notebook)
# ============================================================
pr("Loading data...")
train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

# Color features
color_pairs = [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]
for df in [train, test]:
    df['_g_div_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).clip(-10,10).astype('float32')
    df['_i_div_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).clip(-10,10).astype('float32')
    for a,b in color_pairs:
        df[f'_{a}-{b}'] = (df[a] - df[b]).astype('float32')
    mags = df[['u','g','r','i','z']].astype('float32')
    df['_mag_mean'] = mags.mean(axis=1).astype('float32')
    df['_mag_range'] = (mags.max(axis=1) - mags.min(axis=1)).astype('float32')
    df['_log1p_redshift'] = np.log1p(df['redshift'].clip(lower=0) + 1e-4).astype('float32')
    # Floor categorical views
    for col in ['alpha','delta','u','g','r','i','z','redshift']:
        df[f'{col}_cat_'] = np.floor(df[col]).astype('int32').astype('category')
    for n_bins in [100,500]:
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
        df[f'delta_{n_bins}_bin_'] = pd.Series(kb.fit_transform(df[['delta']]).ravel().astype('int32'), index=df.index).astype('category')

# Target encoding
le = LabelEncoder(); y_all = le.fit_transform(train['class'])
for df in [train,test]:
    df['_ca'] = df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_cz'] = df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)
for combo in ['_ca','_cz']:
    for ci in range(3):
        cn = f'{combo}_TE_{ci}'; yb = (y_all==ci).astype(int)
        te = TargetEncoder(cv=5, smooth='auto', random_state=42)
        train[cn] = te.fit_transform(train[[combo]], yb).ravel()
        te2 = TargetEncoder(smooth='auto', random_state=42); te2.fit(train[[combo]], yb)
        test[cn] = te2.transform(test[[combo]]).ravel()
train.drop(['_ca','_cz'], axis=1, inplace=True)
test.drop(['_ca','_cz'], axis=1, inplace=True)

for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc'] = LabelEncoder().fit_transform(test[col])

# Build feature list
num_cols = ['u','g','r','i','z','redshift','alpha','delta','u_g','g_r','r_i','i_z','u_r','g_i','r_z',
            'color_curv','u_z','g_z','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = num_cols + new_cols
feat_all = list(dict.fromkeys([c for c in feat_all if c in train.columns]))  # dedupe, keep order

cat_cols = [c for c in feat_all if str(train[c].dtype) == 'category']
num_cols_final = [c for c in feat_all if c not in cat_cols]

pr(f"Features: {len(num_cols_final)} num + {len(cat_cols)} cat")

# ============================================================
# 2. RealMLP Architecture
# ============================================================
class NumericalPreprocessor:
    """median_center + robust_scale"""
    def fit(self, X):
        self.median = np.median(X, axis=0)
        qd = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        qd[qd==0] = 0.5*(X.max(axis=0)[qd==0]-X.min(axis=0)[qd==0]); qd[qd==0]=1.0
        self.scale = 1.0/(qd+1e-30)
    def transform(self, X):
        X = X.copy().astype(np.float32)
        X -= self.median; X *= self.scale
        return X

class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))
    def forward(self, x): return x * self.scale[None,:,:]

class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_f, out_f, bias=True):
        super().__init__()
        self.in_f = in_f
        self.weight = nn.Parameter(torch.randn(n_ens, in_f, out_f))
        self.bias = nn.Parameter(torch.randn(n_ens, out_f)) if bias else None
    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_f)
        return x + self.bias if self.bias is not None else x

class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=5, freq_scale=2.33):
        super().__init__()
        self.n_features = n_features; self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim)*freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim-1)/math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim-1))
        nn.init.uniform_(self.b1, -math.pi, math.pi)
    def forward(self, x):
        periodic = torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w1.unsqueeze(0)+self.b1.unsqueeze(0)))
        transformed = torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2)+self.b2.unsqueeze(0)
        return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)

class RealMLP(nn.Module):
    def __init__(self, n_features, n_classes, cfg):
        super().__init__()
        self.n_ens = cfg['n_ens']
        self.num_embed = PBLDEmbedding(self.n_ens, n_features,
                                        hidden_dim=cfg['pbld_hidden_dim'],
                                        out_dim=cfg['pbld_out_dim'],
                                        freq_scale=cfg['pbld_freq_scale'])
        num_emb_dim = n_features * cfg['pbld_out_dim']
        hidden_dims = cfg['hidden_dims']
        act = cfg['activation']
        layers = []
        if cfg['add_front_scale']:
            layers.append(ScalingLayer(self.n_ens, num_emb_dim))
        in_dim = num_emb_dim
        self._dropout_modules = []
        for i, out_dim in enumerate(hidden_dims):
            linear = NTPLinear(self.n_ens, in_dim, out_dim)
            drop = nn.Dropout(cfg['dropout'])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(self.n_ens, in_dim, n_classes)

    def forward(self, x_num):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x = self.hidden(x_num)
        return F.softmax(self.output_layer(x), dim=2)  # (B, n_ens, C)

# ============================================================
# 3. Training Config (Deotte R2-103)
# ============================================================
CONFIG = {
    'n_ens': 8, 'hidden_dims': [512,512,512], 'dropout': 0.044,
    'p_drop_sched': 'expm4t', 'activation': nn.GELU, 'add_front_scale': True,
    'pbld_hidden_dim': 16, 'pbld_out_dim': 5, 'pbld_freq_scale': 2.33,
    'lr': 0.01, 'weight_decay': 0.0125, 'epochs': 6, 'train_bs': 256, 'eval_bs': 10240,
    'ls_eps': 0.04, 'grad_clip': 1.0, 'ema_decay': 0.997875,
    'loss_prior_power': 1.075, 'class_weight_power': 0.0,
}

def apply_schedule(init, progress, sched, flat_ratio=0.2):
    if sched == 'flat_cos':
        if progress < flat_ratio: return init
        t = (progress-flat_ratio)/(1-flat_ratio)
        return init*(math.cos(math.pi*t)+1)/2
    if sched == 'cos': return init*(math.cos(math.pi*progress)+1)/2
    if sched == 'expm4t': return init*math.exp(-4*progress)
    return init

# ============================================================
# 4. 5-Fold CV Training
# ============================================================
X_all_num = train[num_cols_final].values.astype(np.float32)
X_test_num = test[num_cols_final].values.astype(np.float32)
prep = NumericalPreprocessor(); prep.fit(X_all_num)
X_all_num = prep.transform(X_all_num); X_test_num = prep.transform(X_test_num)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_all_num), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_test_num), N_CLASSES), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all_num, y_all)):
    pr(f"\nFold {fold+1}/5")
    fold_seed = SEED + fold * 100
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)

    X_tr = torch.tensor(X_all_num[tr_idx]).to(DEVICE)
    y_tr = torch.tensor(y_all[tr_idx]).long().to(DEVICE)
    X_val = torch.tensor(X_all_num[val_idx]).to(DEVICE)
    y_val = torch.tensor(y_all[val_idx]).long().to(DEVICE)

    # Class weights
    counts = np.bincount(y_all[tr_idx], minlength=3).astype(np.float64)
    class_weights = torch.tensor(np.power(1.0/np.clip(counts,1,None), CONFIG.get('class_weight_power',0.0)),
                                  dtype=torch.float32).to(DEVICE)
    loss_mult = None
    if CONFIG['loss_prior_power'] != 0:
        class_counts = counts / np.exp(np.log(counts).mean())
        loss_mult = torch.tensor(np.power(class_counts, CONFIG['loss_prior_power']),
                                  dtype=torch.float32).to(DEVICE)

    model = RealMLP(X_all_num.shape[1], N_CLASSES, CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'],
                                   betas=(0.9, 0.98))
    total_steps = CONFIG['epochs'] * len(y_tr); step = 0
    best_score = -np.inf; best_state = None
    ema_state = {k: v.detach().clone() for k,v in model.state_dict().items()} if CONFIG['ema_decay']>0 else None

    for epoch in range(CONFIG['epochs']):
        model.train()
        perm = torch.randperm(len(y_tr))
        for start in range(0, len(y_tr), CONFIG['train_bs']):
            idx = perm[start:start+CONFIG['train_bs']]
            progress = step / total_steps; step += 1
            # LR schedule
            for g in optimizer.param_groups:
                g['lr'] = apply_schedule(CONFIG['lr'], progress, 'flat_cos')

            optimizer.zero_grad()
            y_pred = model(X_tr[idx])  # (B, n_ens, C)
            # Loss
            ls_val = apply_schedule(CONFIG['ls_eps'], progress, 'cos')
            y_smooth = torch.full_like(y_pred[:,0,:], ls_val/N_CLASSES)
            y_smooth.scatter_(1, y_tr[idx].unsqueeze(1), 1.0-ls_val+ls_val/N_CLASSES)
            y_smooth = y_smooth.unsqueeze(1).expand(-1, CONFIG['n_ens'], -1)
            loss = -(y_smooth * torch.log(y_pred.clamp(1e-15,1))).sum(dim=2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()
            if ema_state is not None:
                for k,v in model.state_dict().items():
                    if torch.is_floating_point(v):
                        ema_state[k].mul_(CONFIG['ema_decay']).add_(v.detach(),alpha=1-CONFIG['ema_decay'])
            # Dropout schedule
            for dm in model._dropout_modules:
                dm.p = apply_schedule(CONFIG['dropout'], progress, 'expm4t')

        # Validation
        model.eval()
        if ema_state is not None:
            live_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)
        with torch.no_grad():
            val_probs = model(X_val).mean(dim=1).cpu().numpy()
        if ema_state is not None:
            model.load_state_dict(live_state, strict=True)
        score = balanced_accuracy_score(y_val.cpu(), np.argmax(val_probs, axis=1))
        if score > best_score:
            best_score = score; best_epoch = epoch+1
            best_state = {k:v.detach().clone() for k,v in (ema_state if ema_state else model.state_dict()).items()}
        pr(f"  epoch {epoch+1}: BA={score:.5f} {'*' if score>best_score else ''}")

    model.load_state_dict(best_state, strict=True)
    pr(f"  Fold {fold+1} best: {best_score:.5f} (epoch {best_epoch})")

    model.eval()
    with torch.no_grad():
        oof[val_idx] = model(torch.tensor(X_all_num[val_idx]).to(DEVICE)).mean(dim=1).cpu().numpy()
        test_preds += model(torch.tensor(X_test_num).to(DEVICE)).mean(dim=1).cpu().numpy() / FOLDS

    fold_scores.append(best_score)
    del model; gc.collect(); torch.cuda.empty_cache()

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nHand-crafted RealMLP OOF BA: {oof_ba:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")
np.save('oof_RealMLP_handcrafted.npy', oof)
np.save('test_RealMLP_handcrafted.npy', test_preds)
pr("Saved OOF + test predictions")
