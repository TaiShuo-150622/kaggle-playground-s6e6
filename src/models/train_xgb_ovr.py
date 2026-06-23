"""
XGBoost One-vs-Rest — Community approach (CV 0.96862)
=======================================================
Train 3 binary XGBoost classifiers instead of 1 multi-class.
Each classifier handles class imbalance independently.

Based on: ps6e6-one-vs-rest-xgb.ipynb
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data ==========
pr("Loading and engineering features...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=True)
pr(f"Features: {len(feat_list)}")

# Frequency-based rare category merging (from notebook)
for col in feat_list:
    if str(train[col].dtype) in ['object', 'category']:
        continue
    if train[col].nunique() > 50:
        freq = train[col].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[col] = train[col].map(lambda x: mapping.get(x, mapping_default))
        test[col] = test[col].map(lambda x: mapping.get(x, mapping_default))

X_all = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)
pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. One-vs-Rest Training ==========
import xgboost as xgb

CLASSES = ['GALAXY', 'QSO', 'STAR']
N_CLASSES = 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_all, y_all))

# XGBoost params (tuned for binary classification)
XGB_BINARY_PARAMS = dict(
    objective='binary:logistic',
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    gamma=0.05,
    tree_method='hist', device='cuda',
    verbosity=0,
)

oof = np.zeros((len(X_all), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)

for class_idx, class_name in enumerate(CLASSES):
    pr(f"\n=== OVR Class {class_idx}: {class_name} vs Rest ===")
    y_binary = (y_all == class_idx).astype(int)

    class_oof = np.zeros(len(X_all), dtype=np.float32)
    class_test = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []

    for fold, (tr, val) in enumerate(folds):
        model = xgb.XGBClassifier(random_state=fold * 42, **XGB_BINARY_PARAMS)
        model.fit(X_all[tr], y_binary[tr],
                  eval_set=[(X_all[val], y_binary[val])], verbose=False)
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
    mean_score = np.mean(fold_scores)
    pr(f"  {class_name} mean fold BA: {mean_score:.5f}")

# Use raw binary probabilities directly (one-vs-rest: argmax of binary probs is correct)
# No normalization needed — each classifier independently estimates P(sample ∈ class_i)
# Normalizing would distort the relative confidence between classifiers

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
per_class = [balanced_accuracy_score(y_all == i, (np.argmax(oof, axis=1) == i).astype(int)) for i in range(3)]
pr(f"\nXGBoost OVR OOF: {oof_ba:.5f}")
pr(f"  Per-class: GALAXY={per_class[0]:.4f} QSO={per_class[1]:.4f} STAR={per_class[2]:.4f}")

np.save('oof_XGB_OVR.npy', oof)
np.save('test_XGB_OVR.npy', test_preds)

target = 0.965
if oof_ba >= target:
    pr(f"✅ Target {target} REACHED!")
else:
    pr(f"Target {target} not reached. Gap: {target - oof_ba:.5f}")
