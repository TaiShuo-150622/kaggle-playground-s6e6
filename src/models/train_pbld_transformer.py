"""
PBLD-Transformer: PBLD embeddings + Transformer attention
===========================================================
Hypothesis: PBLD's Fourier-based non-linear embeddings give Transformer
attention meaningful Q·K dot products (unlike linear tokenizer = noise).

Architecture:
  1. PBLDEmbedding: each feature → 5D Fourier vector
  2. Transformer layers: self-attention on feature tokens
  3. CLS head: final classification
"""

import sys, os, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from src.features.shared import engineer_all

def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
y_all = LabelEncoder().fit_transform(train_raw['class']).astype(np.int64)
train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'], include_advanced=True)

# Only numeric features (PBLD works on continuous values)
num_cols = [c for c in feat_list if str(train[c].dtype) not in ('object', 'category')]
X_all = train[num_cols].values.astype(np.float32)
X_test = test[num_cols].values.astype(np.float32)

scaler = RobustScaler(); scaler.fit(X_all)
X_all = scaler.transform(X_all); X_test = scaler.transform(X_test)
pr(f"Features: {len(num_cols)}, Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. PBLD-Transformer Model ==========
class PBLDEmbedding(nn.Module):
    """Fourier-based feature embedding (from RealMLP)"""
    def __init__(self, n_features, d_token=32, hidden_dim=16, out_dim=5, freq_scale=2.33):
        super().__init__()
        self.n_features = n_features; self.out_dim = out_dim; self.d_token = d_token
        self.w1 = nn.Parameter(torch.randn(n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_features, out_dim - 1))
        nn.init.uniform_(self.b1, -math.pi, math.pi)
        # Project to d_token dimension
        self.proj = nn.Linear(out_dim * n_features, d_token * n_features)

    def forward(self, x):
        B, F = x.shape
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = torch.einsum("bfh,fhd->bfd", periodic, self.w2) + self.b2.unsqueeze(0)
        emb = torch.cat([x.unsqueeze(-1), transformed], dim=-1)  # (B, F, out_dim)
        return emb  # Return raw PBLD vectors, no projection needed


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d_k = d // n_heads; self.n_heads = n_heads
        self.W_q = nn.Linear(d, d); self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d); self.W_o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_ff, d), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PBLDTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_token=64, n_layers=3,
                 n_heads=4, d_ff=512, dropout=0.1, pbld_out_dim=8):
        super().__init__()
        self.pbld = PBLDEmbedding(n_features, d_token, out_dim=pbld_out_dim, freq_scale=2.33)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, pbld_out_dim))
        nn.init.normal_(self.cls_token, std=0.01)
        self.layers = nn.ModuleList([
            TransformerLayer(pbld_out_dim, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(pbld_out_dim, n_classes)

    def forward(self, x):
        tokens = self.pbld(x)  # (B, F, pbld_out_dim=5)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])


# ========== 3. Training ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pr(f"Device: {DEVICE}")

CONFIG = dict(d_token=64, n_layers=3, n_heads=4, d_ff=512, dropout=0.1,
              pbld_out_dim=8, lr=1e-3, weight_decay=1e-4, batch_size=256, epochs=50)

n_features = X_all.shape[1]; n_classes = 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_all), n_classes), dtype=np.float32)
test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)
fold_scores = []

for fold, (tr, val) in enumerate(skf.split(X_all, y_all)):
    pr(f"\nFold {fold+1}/5")
    X_tr, X_val = X_all[tr], X_all[val]; y_tr, y_val = y_all[tr], y_all[val]
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 4)

    model = PBLDTransformer(n_features, n_classes, **{k: v for k, v in CONFIG.items()
                           if k in ['d_token', 'n_layers', 'n_heads', 'd_ff', 'dropout', 'pbld_out_dim']})
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    best_score = -np.inf; best_state = None

    for epoch in range(CONFIG['epochs']):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in val_dl:
                val_preds.append(model(bx.to(DEVICE)).cpu().numpy())
        val_proba = np.concatenate(val_preds, axis=0)
        score = balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 15 == 0:
            pr(f"  epoch {epoch+1}: BA={score:.5f}")

    model.load_state_dict(best_state)
    pr(f"  Fold {fold+1} best: {best_score:.5f}")

    # OOF
    model.eval()
    bs = CONFIG['batch_size'] * 4
    val_dl_all = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        for i, (bx,) in enumerate(val_dl_all):
            oof[val[i*bs:(i+1)*bs]] = F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy()
    # Test
    test_dl = DataLoader(TensorDataset(torch.tensor(X_test)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        for i, (bx,) in enumerate(test_dl):
            start = i * bs; end = min((i+1)*bs, len(X_test))
            test_preds[start:end] += F.softmax(model(bx.to(DEVICE)), dim=-1).cpu().numpy()[:end-start] / 5

    fold_scores.append(best_score)
    del model; gc.collect(); torch.cuda.empty_cache()

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nPBLD-Transformer OOF: {oof_ba:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")
np.save('oof_PBLD_Transformer.npy', oof)
np.save('test_PBLD_Transformer.npy', test_preds)
pr("DONE")
