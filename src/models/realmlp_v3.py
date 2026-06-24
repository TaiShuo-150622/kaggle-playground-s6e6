"""
RealMLP v3 — 252 features + Deotte architecture
=================================================
Uses shared feature module (252 features) + RealMLP architecture
from deotte_realmlp.py (PBLD + NTP + EMA + n_ens=8).

Expected: OOF +0.002~0.005 over v2 (0.96742 with ~60 features)
"""

import sys, os, gc, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_prefix', type=str, default='RealMLP_v3')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = args.seed; N_CLASSES = 3
pr(f"Device: {DEVICE}  Seed: {SEED}  Save: {args.save_prefix}")


# ========== 1. Data + 252 Features ==========
pr("Loading and engineering features...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=True)
pr(f"Features: {len(feat_list)}")

# Separate numeric and categorical (categoricals already int-encoded by shared module)
num_cols = [c for c in feat_list if str(train[c].dtype) not in ('object', 'category')]
cat_cols = [c for c in feat_list if c not in num_cols]
# Convert any remaining category to int
for c in cat_cols:
    try:
        train[c] = train[c].astype('int32')
        test[c] = test[c].astype('int32')
    except:
        train[c] = train[c].astype(str).replace({'nan': '-1'}).astype('int32')
        test[c] = test[c].astype(str).replace({'nan': '-1'}).astype('int32')

X_num = train[num_cols].values.astype(np.float32)
X_num_test = test[num_cols].values.astype(np.float32)


# ========== 2. RealMLP Architecture (from deotte_realmlp.py) ==========
class NumericalPreprocessor:
    def __init__(self, tfms=None):
        self._tfms = tfms or ["median_center", "robust_scale"]

    def fit(self, X):
        self.median = np.median(X, axis=0)
        qd = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        qd[qd == 0] = 0.5 * (X.max(axis=0)[qd == 0] - X.min(axis=0)[qd == 0])
        qd[qd == 0] = 1.0
        self.scale = 1.0 / (qd + 1e-30)

    def transform(self, X):
        X = X.copy().astype(np.float32)
        if "median_center" in self._tfms:
            X -= self.median
        if "robust_scale" in self._tfms:
            X *= self.scale
        return np.clip(X, -5, 5)


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
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0)
        return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, n_features, n_classes, n_ens=8, hidden_dims=None,
                 dropout=0.044, activation=nn.GELU, pbld_hidden_dim=16,
                 pbld_out_dim=5, pbld_freq_scale=2.33):
        super().__init__()
        self.n_ens = n_ens
        hidden_dims = hidden_dims or [512, 512, 512]
        self.num_embed = PBLDEmbedding(n_ens, n_features, pbld_hidden_dim, pbld_out_dim, pbld_freq_scale)
        num_emb_dim = n_features * pbld_out_dim

        layers = []
        in_dim = num_emb_dim
        self._dropout_modules = []
        for out_dim in hidden_dims:
            linear = NTPLinear(n_ens, in_dim, out_dim)
            drop = nn.Dropout(dropout)
            self._dropout_modules.append(drop)
            layers += [linear, activation(), drop]
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens, in_dim, n_classes)

    def forward(self, x_num):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x = self.hidden(x_num)
        return F.softmax(self.output_layer(x), dim=2)


# ========== 3. Training ==========
CONFIG = dict(n_ens=2, hidden_dims=[512, 512, 512], dropout=0.044,
              pbld_hidden_dim=16, pbld_out_dim=5, pbld_freq_scale=2.33,
              lr=0.01, weight_decay=0.0125, epochs=6,
              train_bs=64, eval_bs=10240, grad_clip=1.0)

prep = NumericalPreprocessor()
prep.fit(X_num)
X_num = prep.transform(X_num)
X_num_test = prep.transform(X_num_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_num), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_num_test), N_CLASSES), dtype=np.float32)
fold_scores = []

for fold, (tr, val) in enumerate(skf.split(X_num, y_all)):
    pr(f"\nFold {fold+1}/5")
    fold_seed = SEED + fold * 100
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)

    X_tr = torch.tensor(X_num[tr]).to(DEVICE)
    y_tr = torch.tensor(y_all[tr]).long().to(DEVICE)
    X_val = torch.tensor(X_num[val]).to(DEVICE)
    y_val = torch.tensor(y_all[val]).long().to(DEVICE)

    model = RealMLP(X_num.shape[1], N_CLASSES, **{k: v for k, v in CONFIG.items()
                      if k in ['n_ens', 'hidden_dims', 'dropout', 'pbld_hidden_dim',
                                'pbld_out_dim', 'pbld_freq_scale']})
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                             weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.98))

    best_score = -np.inf; best_state = None
    total_steps = CONFIG['epochs'] * (len(y_tr) // CONFIG['train_bs'] + 1)
    step = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        perm = torch.randperm(len(y_tr))
        for start in range(0, len(y_tr), CONFIG['train_bs']):
            idx = perm[start:start + CONFIG['train_bs']]
            progress = step / total_steps; step += 1
            lr = CONFIG['lr'] * (math.cos(math.pi * progress) + 1) / 2
            for g in opt.param_groups:
                g['lr'] = lr

            opt.zero_grad()
            y_pred = model(X_tr[idx])
            loss = -(torch.log(y_pred.clamp(1e-15, 1))).sum(dim=2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val).mean(dim=1).cpu().numpy()
        score = balanced_accuracy_score(y_all[val], np.argmax(val_probs, axis=1))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 2 == 0:
            pr(f"  epoch {epoch+1}: BA={score:.5f}")

    model.load_state_dict(best_state)
    pr(f"  Fold {fold+1} best: {best_score:.5f}")

    with torch.no_grad():
        oof[val] = model(torch.tensor(X_num[val]).to(DEVICE)).mean(dim=1).cpu().numpy()
        test_preds += model(torch.tensor(X_num_test).to(DEVICE)).mean(dim=1).cpu().numpy() / 5
    fold_scores.append(best_score)
    del model; gc.collect(); torch.cuda.empty_cache()

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nRealMLP v3 OOF: {oof_ba:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")
np.save(f'oof_{args.save_prefix}.npy', oof)
np.save(f'test_{args.save_prefix}.npy', test_preds)
pr("DONE")
