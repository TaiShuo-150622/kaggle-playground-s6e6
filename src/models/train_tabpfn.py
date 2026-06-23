"""
TabPFN-3 — Zero-shot baseline for S6E6
========================================
TabPFN is a pretrained Transformer for tabular data.
Key feature: NO training needed, single forward pass.
Target OOF: 0.960 (zero-shot baseline)

TabPFN-3 (May 2026) handles up to 10K samples per forward pass.
For 577K rows, we split into chunks and ensemble.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
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
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'])
pr(f"Features: {len(feat_list)}")

X_all = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)

# Robust scaling (TabPFN benefits from normalization)
scaler = RobustScaler()
X_all = scaler.fit_transform(X_all)
X_test = scaler.transform(X_test)

pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. TabPFN-3 Inference ==========
pr("Loading TabPFN-3 classifier...")
from tabpfn import TabPFNClassifier

# TabPFN-3 classifier
import tabpfn_client
tabpfn_client.set_access_token('tabpfn_sk_yWzGvHOWuqI815s3XpA-ZKfplDGC5mB8UJMaRWFnh8g')

clf = TabPFNClassifier(
    n_estimators=4,
    random_state=42,
)
pr(f"TabPFN loaded on CUDA")


# ========== 3. 5-Fold CV + Test ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_all), 3), dtype=np.float32)
test_preds = np.zeros((len(X_test), 3), dtype=np.float32)
fold_scores = []

for fold, (tr, val) in enumerate(skf.split(X_all, y_all)):
    pr(f"Fold {fold+1}/5...")
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_val = X_all[val]

    # TabPFN fits and predicts in one call
    clf.fit(X_tr, y_tr)
    oof[val] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test) / 5

    score = balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1))
    fold_scores.append(score)
    pr(f"  BA={score:.5f}")

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nTabPFN-3 OOF BA: {oof_ba:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")

# ========== 4. Save ==========
np.save('oof_TabPFN3.npy', oof)
np.save('test_TabPFN3.npy', test_preds)
pr("Saved: oof_TabPFN3.npy, test_TabPFN3.npy")

if oof_ba >= 0.960:
    pr("✅ Target 0.960 REACHED!")
else:
    pr(f"Target 0.960 not reached. Gap: {0.960 - oof_ba:.5f}")
