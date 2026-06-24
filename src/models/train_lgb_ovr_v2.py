"""
LightGBM One-vs-Rest v2 — Community parameters
================================================
Based on ps6e6-one-vs-rest-tabm.ipynb
Community params: 20000 estimators, max_depth=3, subsample=0.81
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from datetime import datetime

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=False)
pr(f"Features: {len(feat_list)}")

X_all = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)


# ========== 2. One-vs-Rest LGB (community params) ==========
LGB_PARAMS = dict(
    objective='binary',
    metric='auc',
    n_estimators=20000,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=247,
    min_child_samples=63,
    subsample=0.813,
    colsample_bytree=0.803,
    random_state=60,
    verbose=-1,
)

CLASSES = ['GALAXY', 'QSO', 'STAR']
N_CLASSES = 3
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
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_all[tr], y_binary[tr],
            eval_set=[(X_all[val], y_binary[val])],
            callbacks=[lgb.early_stopping(1000, verbose=False)]
        )
        class_oof[val] = model.predict_proba(X_all[val])[:, 1]
        class_test += model.predict_proba(X_test)[:, 1] / 5
        score = balanced_accuracy_score(y_binary[val], (class_oof[val] > 0.5).astype(int))
        fold_scores.append(score)
        pr(f"  Fold {fold+1}: BA={score:.5f}")
        del model; gc.collect()

    oof[:, class_idx] = class_oof
    test_preds[:, class_idx] = class_test
    pr(f"  {class_name} mean BA: {np.mean(fold_scores):.5f}")

oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
pr(f"\nLGB OVR OOF: {oof_ba:.5f}")
np.save('oof_LGB_OVR_v2.npy', oof)
np.save('test_LGB_OVR_v2.npy', test_preds)
pr("DONE")
