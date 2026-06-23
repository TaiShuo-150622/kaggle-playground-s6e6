"""
All Tree Models v3 — 252 features + community parameters
==========================================================
Uses full feature set from cat-v3 notebook.
CatBoost: 0.96897-target parameters
XGBoost: GPU hist + deeper trees
LightGBM: leaf-wise + lower lr

Expected OOF improvement: +0.005 to +0.010 over v1 (43 features)
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


# ========== 1. Data + Full Features ==========
pr("Loading and engineering features...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=True)
pr(f"Features: {len(feat_list)}")

# Prepare data — separate numeric and categorical
cat_patterns = ['_cat', '_bin_', 'COMBO_', 'PAIR_', 'TRIO_', 'mod10', 'mod100',
                'frac20', 'decimal1000', 'round']
cat_cols_cb = [c for c in feat_list if any(p in c for p in cat_patterns)]
num_cols = [c for c in feat_list if c not in cat_cols_cb]

# For LGB/XGBoost: all float32
X_num = train[num_cols].values.astype(np.float32)
X_cat = train[cat_cols_cb].fillna(-1).astype('int32').values
X_all = np.concatenate([X_num, X_cat], axis=1)
X_test_num = test[num_cols].values.astype(np.float32)
X_test_cat = test[cat_cols_cb].fillna(-1).astype('int32').values
X_test = np.concatenate([X_test_num, X_test_cat], axis=1)

# For CatBoost: keep categoricals as int32
n_num = len(num_cols)
cat_idx = list(range(n_num, n_num + len(cat_cols_cb)))
pr(f"Features: {len(num_cols)} num + {len(cat_cols_cb)} cat = {len(feat_list)} total")
pr(f"CatBoost cat indices: {len(cat_idx)} (cols {n_num} to {len(feat_list)-1})")

pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. Models ==========
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_all, y_all))
all_results = {}

# ---- CatBoost (community parameters from cat-v3) ----
CB_CONFIG = dict(
    loss_function='MultiClass',
    iterations=5000,
    depth=8,
    learning_rate=0.042,
    l2_leaf_reg=8.0,
    random_strength=1.2,
    bootstrap_type='Bayesian',
    bagging_temperature=0.2,
    one_hot_max_size=16,
    max_ctr_complexity=3,
    class_weights=[1.0, 3.25, 5.0],
    border_count=254,
    early_stopping_rounds=260,
    task_type='GPU',
    devices='0',
    thread_count=4,
    allow_writing_files=False,
    verbose=0,
)

pr("\n=== CatBoost (cat-v3 params, 252 features) ===")
oof_cb = np.zeros((len(X_all), 3), dtype=np.float32)
test_cb = np.zeros((len(X_test), 3), dtype=np.float32)
cb_scores = []

for fold, (tr, val) in enumerate(folds):
    seed = 42 + fold
    model = CatBoostClassifier(random_seed=seed, **CB_CONFIG)
    model.fit(X_all[tr], y_all[tr],
              eval_set=[(X_all[val], y_all[val])],
              cat_features=cat_idx, verbose=0)
    oof_cb[val] = model.predict_proba(X_all[val])
    test_cb += model.predict_proba(X_test) / 5
    score = balanced_accuracy_score(y_all[val], np.argmax(oof_cb[val], axis=1))
    cb_scores.append(score)
    pr(f"  Fold {fold+1}: BA={score:.5f}")
    del model; gc.collect()

cb_ba = balanced_accuracy_score(y_all, np.argmax(oof_cb, axis=1))
pr(f"CatBoost OOF: {cb_ba:.5f} (folds: {[f'{s:.5f}' for s in cb_scores]})")
all_results['CB_v3'] = cb_ba
np.save('oof_CB_v3.npy', oof_cb); np.save('test_CB_v3.npy', test_cb)

# ---- LightGBM (optimized) ----
LGB_PARAMS = dict(
    objective='multiclass', num_class=3,
    n_estimators=3000, learning_rate=0.03,
    num_leaves=255, max_depth=10,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5,
    n_jobs=-1, verbose=-1,
)

pr("\n=== LightGBM (252 features) ===")
oof_lgb = np.zeros((len(X_all), 3), dtype=np.float32)
test_lgb = np.zeros((len(X_test), 3), dtype=np.float32)
lgb_scores = []

for fold, (tr, val) in enumerate(folds):
    model = lgb.LGBMClassifier(random_state=fold * 42, **LGB_PARAMS)
    model.fit(X_all[tr], y_all[tr],
              eval_set=[(X_all[val], y_all[val])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val] = model.predict_proba(X_all[val])
    test_lgb += model.predict_proba(X_test) / 5
    score = balanced_accuracy_score(y_all[val], np.argmax(oof_lgb[val], axis=1))
    lgb_scores.append(score)
    pr(f"  Fold {fold+1}: BA={score:.5f}")
    del model; gc.collect()

lgb_ba = balanced_accuracy_score(y_all, np.argmax(oof_lgb, axis=1))
pr(f"LightGBM OOF: {lgb_ba:.5f} (folds: {[f'{s:.5f}' for s in lgb_scores]})")
all_results['LGB_v3'] = lgb_ba
np.save('oof_LGB_v3.npy', oof_lgb); np.save('test_LGB_v3.npy', test_lgb)

# ---- XGBoost (GPU) ----
XGB_PARAMS = dict(
    objective='multi:softprob', num_class=3,
    n_estimators=3000, learning_rate=0.03,
    max_depth=10, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5, gamma=0.05,
    tree_method='hist', device='cuda', n_jobs=-1, verbosity=0,
)

pr("\n=== XGBoost (GPU, 252 features) ===")
oof_xgb = np.zeros((len(X_all), 3), dtype=np.float32)
test_xgb = np.zeros((len(X_test), 3), dtype=np.float32)
xgb_scores = []

for fold, (tr, val) in enumerate(folds):
    model = xgb.XGBClassifier(random_state=fold * 42, **XGB_PARAMS)
    model.fit(X_all[tr], y_all[tr],
              eval_set=[(X_all[val], y_all[val])], verbose=False)
    oof_xgb[val] = model.predict_proba(X_all[val])
    test_xgb += model.predict_proba(X_test) / 5
    score = balanced_accuracy_score(y_all[val], np.argmax(oof_xgb[val], axis=1))
    xgb_scores.append(score)
    pr(f"  Fold {fold+1}: BA={score:.5f}")
    del model; gc.collect()

xgb_ba = balanced_accuracy_score(y_all, np.argmax(oof_xgb, axis=1))
pr(f"XGBoost OOF: {xgb_ba:.5f} (folds: {[f'{s:.5f}' for s in xgb_scores]})")
all_results['XGB_v3'] = xgb_ba
np.save('oof_XGB_v3.npy', oof_xgb); np.save('test_XGB_v3.npy', test_xgb)

# ========== 3. Summary ==========
pr(f"\n{'='*60}")
pr("RESULTS SUMMARY (252 features)")
pr(f"{'='*60}")
for name, ba in sorted(all_results.items(), key=lambda x: -x[1]):
    status = "✅" if ba >= 0.965 else "⬜"
    pr(f"  {name}: OOF={ba:.5f} {status}")
pr(f"\nTarget: CatBoost 0.965 | XGBoost 0.965 | LightGBM 0.965")
pr("ALL DONE")
