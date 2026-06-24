"""
Multi-seed training for CatBoost + RealMLP
============================================
Runs N seeds per model, saves individual OOF/test files.
Then does simple average ensemble.

Usage:
  python src/models/train_multiseed.py --model cb --seeds 5
  python src/models/train_multiseed.py --model realmlp --seeds 5
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


# ========== Parse args ==========
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cb', 'realmlp', 'both'], default='both')
parser.add_argument('--seeds', type=int, default=5)
args = parser.parse_args()
SEEDS = [42 + i * 100 for i in range(args.seeds)]

# ========== Data ==========
pr("Loading data...")
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
le = LabelEncoder()
y_all = le.fit_transform(train_raw['class'])

train, test, feat_list = engineer_all(train_raw, test_raw, train_raw['class'],
                                       include_advanced=True)
pr(f"Features: {len(feat_list)}")

# ========== CatBoost multi-seed ==========
if args.model in ('cb', 'both'):
    from catboost import CatBoostClassifier

    # Prep: DataFrame with proper dtypes
    cat_patterns = ['_cat', '_bin_', 'COMBO_', 'PAIR_', 'TRIO_', 'mod10', 'mod100',
                    'frac20', 'decimal1000', 'round']
    cat_cols_cb = [c for c in feat_list if any(p in c for p in cat_patterns)]
    for c in cat_cols_cb:
        # Convert category to string first, then to int to avoid "Cannot setitem on Categorical" error
        train[c] = train[c].astype(str).replace({'nan': '-1', 'None': '-1', 'NA': '-1'}).astype('int32')
        test[c] = test[c].astype(str).replace({'nan': '-1', 'None': '-1', 'NA': '-1'}).astype('int32')
    for c in feat_list:
        if c not in cat_cols_cb:
            train[c] = train[c].astype('float32')
            test[c] = test[c].astype('float32')

    X_cb = train[feat_list]
    X_cb_test = test[feat_list]
    cat_idx = [feat_list.index(c) for c in cat_cols_cb]

    CB_CONFIG = dict(
        loss_function='MultiClass', iterations=5000, depth=8,
        learning_rate=0.042, l2_leaf_reg=8.0, random_strength=1.2,
        bootstrap_type='Bayesian', bagging_temperature=0.2,
        one_hot_max_size=16, max_ctr_complexity=3,
        class_weights=[1.0, 3.25, 5.0], border_count=254,
        early_stopping_rounds=260, task_type='GPU', devices='0',
        thread_count=4, allow_writing_files=False, verbose=0,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_cb, y_all))
    cb_oofs, cb_tests = [], []

    for s_idx, base_seed in enumerate(SEEDS):
        pr(f"\n=== CatBoost seed {s_idx+1}/{len(SEEDS)} (base={base_seed}) ===")
        oof = np.zeros((len(X_cb), 3), dtype=np.float32)
        tp = np.zeros((len(X_cb_test), 3), dtype=np.float32)
        scores = []

        for fold, (tr, val) in enumerate(folds):
            seed = base_seed + fold
            model = CatBoostClassifier(random_seed=seed, **CB_CONFIG)
            model.fit(X_cb.iloc[tr], y_all[tr],
                      eval_set=[(X_cb.iloc[val], y_all[val])],
                      cat_features=cat_idx, verbose=0)
            oof[val] = model.predict_proba(X_cb.iloc[val])
            tp += model.predict_proba(X_cb_test) / 5
            scores.append(balanced_accuracy_score(y_all[val], np.argmax(oof[val], axis=1)))
            del model; gc.collect()

        ba = balanced_accuracy_score(y_all, np.argmax(oof, axis=1))
        pr(f"  OOF: {ba:.5f} folds: {[f'{s:.5f}' for s in scores]}")
        np.save(f'oof_CB_s{s_idx+1}.npy', oof)
        np.save(f'test_CB_s{s_idx+1}.npy', tp)
        cb_oofs.append(oof)
        cb_tests.append(tp)
        gc.collect()

    # Ensemble
    cb_ens = np.mean(cb_oofs, axis=0)
    cb_ba = balanced_accuracy_score(y_all, np.argmax(cb_ens, axis=1))
    pr(f"\nCatBoost {len(SEEDS)}-seed ensemble OOF: {cb_ba:.5f}")
    np.save('oof_CB_ensemble.npy', cb_ens)
    np.save('test_CB_ensemble.npy', np.mean(cb_tests, axis=0))


# ========== RealMLP multi-seed ==========
if args.model in ('realmlp', 'both'):
    import subprocess as sp

    # Run deotte_realmlp.py via sed-pipe to avoid disk I/O issues
    for s_idx, base_seed in enumerate(SEEDS):
        pr(f"\n=== RealMLP seed {s_idx+1}/{len(SEEDS)} (SEED={base_seed}) ===")
        result = sp.run(
            f"sed 's/^SEED = 42/SEED = {base_seed}/' src/models/deotte_realmlp.py | /root/miniconda3/bin/python3 -",
            shell=True, cwd="/root/kaggle_s6e6", capture_output=True, text=True
        )
        if result.returncode != 0:
            pr(f"  ERROR: {result.stderr[-500:]}")
        else:
            for prefix in ["oof", "test"]:
                old = f"{prefix}_RealMLP_handcrafted.npy"
                new = f"{prefix}_RealMLP_s{s_idx+1}.npy"
                if os.path.exists(old):
                    os.rename(old, new)
            pr(f"  Seed {s_idx+1} done")

    # Ensemble
    rmlp_oofs = [np.load(f"oof_RealMLP_s{i+1}.npy") for i in range(len(SEEDS))]
    rmlp_tests = [np.load(f"test_RealMLP_s{i+1}.npy") for i in range(len(SEEDS))]
    rmlp_ens_oof = np.mean(rmlp_oofs, axis=0)
    rmlp_ens_test = np.mean(rmlp_tests, axis=0)
    rmlp_ba = balanced_accuracy_score(y_all, np.argmax(rmlp_ens_oof, axis=1))
    pr(f"\nRealMLP {len(SEEDS)}-seed ensemble OOF: {rmlp_ba:.5f}")
    np.save("oof_RealMLP_ensemble.npy", rmlp_ens_oof)
    np.save("test_RealMLP_ensemble.npy", rmlp_ens_test)


pr("\nALL MULTI-SEED TRAINING DONE")
