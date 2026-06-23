"""
FT-Transformer v2: 5-fold CV for Playground S6E6
=================================================
v2 changes:
  - Uses shared features module (no more duplicated FE code)
  - RobustScaler instead of StandardScaler (Deotte insight: handles outliers better)
  - LR warmup (first 10% of training)
  - Label smoothing
  - Saves OOF predictions for ensemble

Original v1 OOF: 0.95981 (with StandardScaler, no warmup, no label smoothing)
"""

import sys, os, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from code.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data + Features (via shared module) ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
y_all = LabelEncoder().fit_transform(train_raw['class']).astype(np.int64)

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'])
pr(f"Features: {len(feat_list)}")

X_all = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)

# Robust scaling (better for astronomical data with outliers)
scaler = RobustScaler()
X_all = scaler.fit_transform(X_all)
X_test = scaler.transform(X_test)

pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. FT-Transformer Model ==========
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_k = d // n_heads
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
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
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])


# ========== 3. Training Config ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pr(f"Device: {DEVICE}")

CONFIG = dict(
    d_token=192, n_layers=3, n_heads=8, d_ff=512, dropout=0.15,
    lr=3e-4, weight_decay=1e-4, batch_size=512, epochs=60,
    warmup_epochs=6,  # 10% of epochs
    label_smoothing=0.05,
)

n_features = X_all.shape[1]
n_classes = 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_all), n_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)
fold_scores = []

for fold, (tr, val) in enumerate(skf.split(X_all, y_all)):
    pr(f"\nFold {fold + 1}/5")
    X_tr, X_val = X_all[tr], X_all[val]
    y_tr, y_val = y_all[tr], y_all[val]

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          drop_last=True)  # drop_last for stable BN-like behavior
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 4)

    model = FTTransformer(n_features, n_classes,
                          CONFIG['d_token'], CONFIG['n_layers'],
                          CONFIG['n_heads'], CONFIG['d_ff'], CONFIG['dropout'])
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    total_steps = CONFIG['epochs'] * len(train_dl)
    warmup_steps = CONFIG['warmup_epochs'] * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           total_steps - warmup_steps)

    best_score = -np.inf
    best_state = None
    global_step = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)

            # LR warmup
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = CONFIG['lr'] * lr_scale

            optimizer.zero_grad()

            logits = model(bx)
            # Label smoothing
            if CONFIG['label_smoothing'] > 0:
                smooth = CONFIG['label_smoothing']
                y_smooth = torch.full_like(logits, smooth / n_classes)
                y_smooth.scatter_(1, by.unsqueeze(1), 1.0 - smooth + smooth / n_classes)
                loss = -(y_smooth * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, by)

            loss.backward()
            optimizer.step()

            if global_step >= warmup_steps:
                scheduler.step()
            global_step += 1

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

        if (epoch + 1) % 15 == 0:
            pr(f"  epoch {epoch + 1}: BA={score:.5f}")

    model.load_state_dict(best_state)
    pr(f"  Fold {fold + 1} best: {best_score:.5f}")

    # OOF predictions (ordered, no shuffle)
    model.eval()
    bs = CONFIG['batch_size'] * 4
    val_dl_all = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(np.arange(len(val)))),
        batch_size=bs, shuffle=False
    )
    with torch.no_grad():
        for bx, idx in val_dl_all:
            probs = F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy()
            oof[val[idx.numpy()]] = probs

    # Test predictions
    test_dl = DataLoader(TensorDataset(torch.tensor(X_test)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        for i, (bx,) in enumerate(test_dl):
            start = i * bs
            end = min((i + 1) * bs, len(X_test))
            test_preds[start:end] += F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy()[:end - start] / 5

    fold_scores.append(best_score)
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

oof_score = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nFT-Transformer v2 OOF BA: {oof_score:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")

# ========== 4. Save ==========
np.save('oof_FT_Transformer_v2.npy', oof)
np.save('test_FT_Transformer_v2.npy', test_preds)
pr(f"Saved: oof_FT_Transformer_v2.npy, test_FT_Transformer_v2.npy")
pr("Ready for ensemble!")
