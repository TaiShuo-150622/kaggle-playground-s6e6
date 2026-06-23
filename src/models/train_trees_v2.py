"""
Tree Models v2 — Community-calibrated training
================================================
XGBoost + LightGBM with optimized parameters.

Community context (from Deotte discussion #704527):
  - CatBoost: 0.96897 (Codex) — our v1 was ~0.956 (Bernoulli bootstrap killed it)
  - XGBoost:  0.96862 (kirill0212) — our v1 was 0.95637
  - LightGBM: not listed but should be on par

Our v1 mistakes:
  - Too few iterations (2000) with too high lr (0.05)
  - Too shallow trees (depth 6-10)
  - Weak regularization
  - No GPU for XGBoost (RTX 5090 idle!)
  - Old features (no Deotte features in early runs)

v2 fixes:
  - More iterations (3000-5000) + lower lr (0.02-0.03)
  - Deeper trees (depth 10-12)
  - Stronger regularization
  - GPU for XGBoost
  - Shared features module (with Deotte features)
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


# ========== 1. Data + Features ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'])
pr(f"Features: {len(feat_list)}")

X = train[feat_list].values.astype(np.float32)
X_test = test[feat_list].values.astype(np.float32)
pr(f"Train: {X.shape}, Test: {X_test.shape}")


# ========== 2. Model Configs ==========
import lightgbm as lgb
import xgboost as xgb

COMMON = dict(n_jobs=-1, random_state=42)

LGB_CONFIGS = [
    dict(name="LGB_A_deeper", n_estimators=3000, lr=0.03, num_leaves=255,
         max_depth=10, min_child_samples=30, subsample=0.8,
         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
         desc="Deeper tree, lower lr"),

    dict(name="LGB_B_deep12", n_estimators=5000, lr=0.02, num_leaves=512,
         max_depth=12, min_child_samples=20, subsample=0.7,
         colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
         desc="Depth 12, strong regularization"),

    dict(name="LGB_C_lowlr", n_estimators=5000, lr=0.015, num_leaves=256,
         max_depth=10, min_child_samples=50, subsample=0.8,
         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.3,
         desc="Very low lr, many iterations"),
]

XGB_CONFIGS = [
    dict(name="XGB_A_deeper", n_estimators=3000, lr=0.03, max_depth=10,
         min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.1, reg_lambda=0.5, gamma=0.05,
         tree_method='gpu_hist',  # Use GPU!
         desc="GPU + deeper trees"),

    dict(name="XGB_B_deep12", n_estimators=5000, lr=0.02, max_depth=12,
         min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
         reg_alpha=0.5, reg_lambda=1.0, gamma=0.1,
         tree_method='gpu_hist',
         desc="Depth 12, strong regularization"),

    dict(name="XGB_C_lowlr", n_estimators=5000, lr=0.015, max_depth=10,
         min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.3, reg_lambda=0.5, gamma=0.05,
         tree_method='gpu_hist',
         desc="Very low lr, many iterations"),
]


# ========== 3. Training ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y_all))

for model_type, configs in [("LGB", LGB_CONFIGS), ("XGB", XGB_CONFIGS)]:
    for cfg in configs:
        pr(f"\n{'='*60}")
        pr(f"{cfg['name']} — {cfg['desc']}")
        pr(f"  n_est={cfg['n_estimators']}, lr={cfg['lr']}, depth={cfg.get('max_depth','?')}")

        oof = np.zeros((len(X), 3), dtype=np.float32)
        test_preds = np.zeros((len(X_test), 3), dtype=np.float32)
        fold_scores = []

        for fold, (tr, val) in enumerate(folds):
            seed = fold * 42

            if model_type == "LGB":
                model = lgb.LGBMClassifier(
                    objective='multiclass', num_class=3,
                    n_estimators=cfg['n_estimators'],
                    learning_rate=cfg['lr'],
                    num_leaves=cfg['num_leaves'],
                    max_depth=cfg['max_depth'],
                    min_child_samples=cfg['min_child_samples'],
                    subsample=cfg['subsample'],
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    random_state=seed, n_jobs=-1, verbose=-1,
                )
                model.fit(
                    X[tr], y_all[tr],
                    eval_set=[(X[val], y_all[val])],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
            else:  # XGB
                model = xgb.XGBClassifier(
                    objective='multi:softprob', num_class=3,
                    n_estimators=cfg['n_estimators'],
                    learning_rate=cfg['lr'],
                    max_depth=cfg['max_depth'],
                    min_child_weight=cfg['min_child_weight'],
                    subsample=cfg['subsample'],
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    gamma=cfg.get('gamma', 0),
                    random_state=seed, n_jobs=-1, verbosity=0,
                    tree_method=cfg.get('tree_method', 'hist'),
                )
                model.fit(
                    X[tr], y_all[tr],
                    eval_set=[(X[val], y_all[val])],
                    verbose=False
                )

            oof[val] = model.predict_proba(X[val])
            test_preds += model.predict_proba(X_test) / 5

            score = balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1))
            fold_scores.append(score)

        oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
        per_class = [
            balanced_accuracy_score(
                y_all == i, (np.argmax(oof, axis=1) == i).astype(int)
            ) for i in range(3)
        ]
        pr(f"  OOF BA: {oof_ba:.5f} | per-class: {[f'{s:.4f}' for s in per_class]}")
        pr(f"  folds: {[f'{s:.5f}' for s in fold_scores]}")

        np.save(f'oof_{cfg["name"]}.npy', oof)
        np.save(f'test_{cfg["name"]}.npy', test_preds)
        del model; gc.collect()

pr("\nALL TREE MODELS DONE")
pr("Target: LGB 0.965+, XGB 0.965+")
