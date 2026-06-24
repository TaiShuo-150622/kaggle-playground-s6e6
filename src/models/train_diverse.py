"""
Diverse CatBoost + RealMLP variants for ensemble diversity
=============================================================
3 CatBoost configs × 3 seeds + 2 RealMLP configs × 2 seeds = 13 new models
"""

import sys, os, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from src.features.shared import engineer_all


def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cb', 'rmlp', 'both'], default='both')
args = parser.parse_args()

# ========== Data ==========
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

# 252 features for CatBoost, 60 for RealMLP
train_252, test_252, feat_list_252 = engineer_all(train_raw, test_raw, train_raw['class'], include_advanced=True)
train_60, test_60, feat_list_60 = engineer_all(train_raw, test_raw, train_raw['class'], include_advanced=False)
pr(f"Features: 252={len(feat_list_252)}, 60={len(feat_list_60)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(train_60, y_all))  # same split for all


# ========== CatBoost variants ==========
if args.model in ('cb', 'both'):
    from catboost import CatBoostClassifier

    # Prep 252-feature data for CatBoost
    cat_patterns = ['_cat', '_bin_', 'COMBO_', 'PAIR_', 'TRIO_', 'mod10', 'mod100', 'frac20', 'decimal1000', 'round']
    cat_cols_cb = [c for c in feat_list_252 if any(p in c for p in cat_patterns)]
    for c in cat_cols_cb:
        train_252[c] = train_252[c].astype(str).replace({'nan': '-1'}).astype('int32')
        test_252[c] = test_252[c].astype(str).replace({'nan': '-1'}).astype('int32')
    for c in feat_list_252:
        if c not in cat_cols_cb:
            train_252[c] = train_252[c].astype('float32')
            test_252[c] = test_252[c].astype('float32')
    cat_idx = [feat_list_252.index(c) for c in cat_cols_cb]

    # Prep 60-feature data for CatBoost
    cat_cols_60 = [c for c in feat_list_60 if any(p in c for p in cat_patterns)]
    for c in cat_cols_60:
        train_60[c] = train_60[c].astype(str).replace({'nan': '-1'}).astype('int32')
        test_60[c] = test_60[c].astype(str).replace({'nan': '-1'}).astype('int32')
    for c in feat_list_60:
        if c not in cat_cols_60:
            train_60[c] = train_60[c].astype('float32')
            test_60[c] = test_60[c].astype('float32')
    cat_idx_60 = [feat_list_60.index(c) for c in cat_cols_60]

    CB_BASE = dict(
        loss_function='MultiClass', iterations=5000, learning_rate=0.042,
        l2_leaf_reg=8.0, random_strength=1.2, bootstrap_type='Bayesian',
        bagging_temperature=0.2, one_hot_max_size=16, max_ctr_complexity=3,
        class_weights=[1.0, 3.25, 5.0], border_count=254,
        early_stopping_rounds=260, task_type='GPU', devices='0',
        thread_count=4, allow_writing_files=False, verbose=0,
    )

    CB_VARIANTS = [
        ("CB_252_d8", dict(depth=8), feat_list_252, train_252, test_252, cat_idx),
        ("CB_252_d6", dict(depth=6), feat_list_252, train_252, test_252, cat_idx),
        ("CB_252_d10", dict(depth=10, l2_leaf_reg=12.0), feat_list_252, train_252, test_252, cat_idx),
        ("CB_60_d8", dict(depth=8), feat_list_60, train_60, test_60, cat_idx_60),
        ("CB_60_d6", dict(depth=6), feat_list_60, train_60, test_60, cat_idx_60),
    ]

    SEEDS = [42, 142, 242]

    for name, extra_params, feat_list, X_df, X_test_df, cat_idx_list in CB_VARIANTS:
        oofs, tests = [], []
        for s_idx, base_seed in enumerate(SEEDS):
            pr(f"\n=== {name} seed {s_idx+1}/{len(SEEDS)} ===")
            oof = np.zeros((len(X_df), 3), dtype=np.float32)
            tp = np.zeros((len(X_test_df), 3), dtype=np.float32)
            scores = []
            for fold, (tr, val) in enumerate(folds):
                seed = base_seed + fold
                cfg = {**CB_BASE, **extra_params, 'random_seed': seed}
                model = CatBoostClassifier(**cfg)
                model.fit(X_df.iloc[tr], y_all[tr],
                          eval_set=[(X_df.iloc[val], y_all[val])],
                          cat_features=cat_idx_list, verbose=0)
                oof[val] = model.predict_proba(X_df.iloc[val])
                tp += model.predict_proba(X_test_df) / 5
                scores.append(balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1)))
                del model; gc.collect()
            ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
            pr(f"  OOF: {ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
            np.save(f'oof_{name}_s{s_idx+1}.npy', oof)
            np.save(f'test_{name}_s{s_idx+1}.npy', tp)
            oofs.append(oof); tests.append(tp)
        ens_ba = balanced_accuracy_score(y_all, np.argmax(np.mean(oofs, axis=0), axis=1))
        pr(f"  {name} 3-seed ensemble: {ens_ba:.5f}")


pr("\nALL DIVERSE VARIANTS DONE")
