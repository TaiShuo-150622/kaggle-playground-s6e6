"""
XGBoost One-vs-Rest v2 — Community parameters (CV 0.96862)
============================================================
Based on ps6e6-one-vs-rest-xgb.ipynb + ps6e6-one-vs-rest-tabm.ipynb

Key differences from our v1 (0.95377):
  - 20000 estimators (not 3000)
  - max_depth=4 (not 10)
  - Proper one-vs-rest with binary:logistic
  - Raw binary probabilities (no normalization needed)
  - 60 features (same as original deotte features, not 252)
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
from datetime import datetime

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data with 60 features (same as original notebooks) ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

# Use basic features only (not 252) - matches community notebooks
train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=False)
pr(f"Features: {len(feat_list)}")

X_all = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)
pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. One-vs-Rest XGBoost (community params) ==========
CLASSES = ['GALAXY', 'QSO', 'STAR']
N_CLASSES = 3

XGB_PARAMS = dict(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=20000,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.875,
    colsample_bytree=0.5,
    max_bin=1024,
    tree_method='hist',
    device='cuda',
    random_state=42,
    early_stopping_rounds=1000,
    verbosity=0,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_all, y_all))
oof = np.zeros((len(X_all), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)

for class_idx, class_name in enumerate(CLASSES):
    pr(f"\n=== OVR Class {class_idx}: {class_name} vs Rest ===")
    y_binary = (y_all == class_idx).astype(int)

    class_oof = np.zeros(len(X_all), dtype=np.float32)
    class_test = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []

    for fold, (tr, val) in enumerate(folds):
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_all[tr], y_binary[tr],
            eval_set=[(X_all[val], y_binary[val])],
            verbose=False
        )
        class_oof[val] = model.predict_proba(X_all[val])[:, 1]
        class_test += model.predict_proba(X_test)[:, 1] / 5

        # Binary BA
        y_pred_bin = (class_oof[val] > 0.5).astype(int)
        score = balanced_accuracy_score(y_binary[val], y_pred_bin)
        fold_scores.append(score)
        pr(f"  Fold {fold+1}: BA={score:.5f}")
        del model; gc.collect()

    oof[:, class_idx] = class_oof
    test_preds[:, class_idx] = class_test
    pr(f"  {class_name} mean BA: {np.mean(fold_scores):.5f}")

# Use raw binary probabilities directly (no normalization)
oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
per_class = [balanced_accuracy_score(y_all == i, (np.argmax(oof, axis=1) == i))
             for i in range(3)]
pr(f"\nXGBoost OVR OOF: {oof_ba:.5f}")
pr(f"  Per-class: GALAXY={per_class[0]:.4f} QSO={per_class[1]:.4f} STAR={per_class[2]:.4f}")

np.save('oof_XGB_OVR_v2.npy', oof)
np.save('test_XGB_OVR_v2.npy', test_preds)
pr("DONE")
