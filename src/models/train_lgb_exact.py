"""
LGB exact replica from ps6e6-one-vs-rest-tabm.ipynb
=====================================================
1:1 copy of params + features + training strategy.
Key difference from our v2: max_bin=32000 (was 255)
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from datetime import datetime
from itertools import combinations

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ========== 1. Data (exact same feature engineering as notebook) ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=False)
pr(f"Features: {len(feat_list)}")

# Same data prep as notebook (float32 numpy)
X_all = train[feat_list].astype('float32').values
X_test = test[feat_list].astype('float32').values
pr(f"Train: {X_all.shape}, Test: {X_test.shape}")


# ========== 2. Exact LGB params from notebook ==========
PARAMS = dict(
    random_state=60,
    feature_pre_filter=False,
    verbose=-1,
    n_estimators=20000,
    learning_rate=0.05,
    max_depth=3,
    min_child_samples=63,
    subsample=0.812763123433567,
    colsample_bytree=0.8029300829885024,
    num_leaves=247,
    reg_alpha=0.07094285437903122,
    reg_lambda=0.033039097703242495,
    max_bin=32000,
    objective='multiclass',
    num_class=3,
)

SEED_LIST = [60, 0, 2809]  # 3 seeds from notebook
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_all, y_all))
all_oofs, all_tests = [], []

for seed_idx, seed in enumerate(SEED_LIST):
    pr(f"\n=== LGB seed {seed_idx+1}/{len(SEED_LIST)} (seed={seed}) ===")
    PARAMS['random_state'] = seed
    oof = np.zeros((len(X_all), 3), dtype=np.float32)
    tp = np.zeros((len(X_test), 3), dtype=np.float32)
    scores = []

    for fold, (tr, val) in enumerate(folds):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X_all[tr], y_all[tr],
            eval_set=[(X_all[val], y_all[val])],
            eval_metric='auc',
            callbacks=[lgb.log_evaluation(500), lgb.early_stopping(250)]
        )
        oof[val] = model.predict_proba(X_all[val])
        tp += model.predict_proba(X_test) / 5
        score = balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1))
        scores.append(score)
        pr(f"  Fold {fold+1}: BA={score:.5f}")
        del model; gc.collect()

    ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
    pr(f"  Seed {seed}: OOF={ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
    np.save(f'oof_LGB_exact_s{seed_idx+1}.npy', oof)
    np.save(f'test_LGB_exact_s{seed_idx+1}.npy', tp)
    all_oofs.append(oof)
    all_tests.append(tp)

# Ensemble
ens_oof = np.mean(all_oofs, axis=0)
ens_test = np.mean(all_tests, axis=0)
ens_ba = balanced_accuracy_score(y_all, np.argmax(ens_oof, axis=1))
pr(f"\nLGB exact 3-seed ensemble OOF: {ens_ba:.5f}")
np.save('oof_LGB_exact_ens.npy', ens_oof)
np.save('test_LGB_exact_ens.npy', ens_test)
pr("DONE")
