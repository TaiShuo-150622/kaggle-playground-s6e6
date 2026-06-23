"""
CatBoost v2 — Community-calibrated training
=============================================
Our v1: Bernoulli bootstrap, depth=8, 2000 iter, lr=0.05 → OOF ~0.956
Community best: 0.96897 (Codex, from Deotte's discussion)

Key fixes:
  1. Ordered Boosting (remove bootstrap_type=Bernoulli — was killing CatBoost's main advantage)
  2. More iterations + lower lr (5000 × 0.03)
  3. Deeper trees (depth=10 → 12)
  4. Proper categorical feature handling via native CatBoost API
  5. GPU acceleration for speed
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier
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

# CatBoost handles categoricals natively — identify categorical columns
cat_cols = [c for c in feat_list
            if str(train[c].dtype) == 'category'
            or c.endswith('_cat_') or c.endswith('_bin_')
            or c in ['spectral_type', 'galaxy_population']]
# Convert category dtype columns to string for CatBoost
for c in cat_cols:
    if c in train.columns and str(train[c].dtype) == 'category':
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

cat_idx = [feat_list.index(c) for c in cat_cols if c in feat_list]
pr(f"Cat features: {len(cat_idx)} ({cat_cols[:5]}...)")

X = train[feat_list].values
X_test = test[feat_list].values
pr(f"Train: {X.shape}, Test: {X_test.shape}")


# ========== 2. Config Grid ==========
# Try multiple configs to find community-level performance
CONFIGS = [
    # Config A: Ordered Boosting restore + more iterations
    dict(name="CB_A_ordered", depth=10, lr=0.03, iters=5000, l2=1.0,
         desc="Restore Ordered Boosting, deeper trees"),

    # Config B: Even deeper
    dict(name="CB_B_deep12", depth=12, lr=0.03, iters=5000, l2=1.0,
         desc="Depth 12 symmetric trees"),

    # Config C: Lower lr, more iterations
    dict(name="CB_C_lowlr", depth=10, lr=0.02, iters=8000, l2=3.0,
         desc="Lower lr + more iterations + stronger regularization"),

    # Config D: Closer to RealMLP style
    dict(name="CB_D_deep10_lowlr", depth=10, lr=0.015, iters=10000, l2=5.0,
         desc="Very low lr, many iterations, strong L2"),
]


# ========== 3. Training ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y_all))

for cfg in CONFIGS:
    pr(f"\n{'='*60}")
    pr(f"Config: {cfg['name']} — {cfg['desc']}")
    pr(f"  depth={cfg['depth']}, lr={cfg['lr']}, iters={cfg['iters']}, l2={cfg['l2']}")

    oof = np.zeros((len(X), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_test), 3), dtype=np.float32)
    fold_scores = []

    for fold, (tr, val) in enumerate(folds):
        pr(f"  Fold {fold+1}/5...")
        seed = fold * 42

        model = CatBoostClassifier(
            iterations=cfg['iters'],
            learning_rate=cfg['lr'],
            depth=cfg['depth'],
            l2_leaf_reg=cfg['l2'],
            random_strength=1.0,
            random_seed=seed,
            task_type='GPU',
            devices='0',
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
            # Key: use default Ordered Boosting (NOT Bernoulli!)
            # bootstrap_type defaults to 'Bayesian' which is CatBoost's core advantage
            # Ordered target encoding for categoricals (auto)
            one_hot_max_size=10,  # one-hot small cardinality cats, ordered for large
            # Early stopping
            early_stopping_rounds=100,
        )

        model.fit(
            X[tr], y_all[tr],
            eval_set=[(X[val], y_all[val])],
            cat_features=cat_idx,
            verbose=0,
        )

        oof[val] = model.predict_proba(X[val])
        test_preds += model.predict_proba(X_test) / 5

        score = balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1))
        fold_scores.append(score)

    oof_ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
    pr(f"  OOF BA: {oof_ba:.5f} (folds: {[f'{s:.5f}' for s in fold_scores]})")

    # Save
    np.save(f'oof_{cfg["name"]}.npy', oof)
    np.save(f'test_{cfg["name"]}.npy', test_preds)

    del model
    gc.collect()

# ========== 4. Summary ==========
pr(f"\n{'='*60}")
pr("ALL CATBOOST CONFIGS DONE")
pr("Best OOF will be evaluated against target 0.965")
