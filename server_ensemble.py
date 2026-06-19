"""
Playground S6E6 全模型集成 — 18 个模型 5-fold CV
==================================================
LightGBM ×5 + XGBoost ×5 + CatBoost ×4 + RealMLP ×4 = 18 模型
"""
import pandas as pd, numpy as np, json, time, gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from datetime import datetime
def progress(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")
feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']
for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc']  = LabelEncoder().fit_transform(test[col])
feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat] + ['u_z','g_z']
le = LabelEncoder().fit(train['class'])
train['target'] = le.transform(train['class'])
X = train[feat_all].values.astype(np.float32)
y = train['target'].values
X_test = test[feat_all].values.astype(np.float32)
progress(f"Data: {X.shape} | Test: {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 模型定义
# ============================================================
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier

# CatBoost 用原始类别列
feat_cb = feat_num + feat_pos + feat_cat + ['u_z','g_z']
X_cb = train[feat_cb].values.astype(object)
X_cb_test = test[feat_cb].values.astype(object)
cat_idx = [i for i, col in enumerate(feat_cb) if col in feat_cat]

# ---- LightGBM ×5 (不同 leaf 大小 + 深度 + 种子) ----
LGB_MODELS = [
    ("LGB_leaf64_d6",   dict(num_leaves=64,  max_depth=6,  min_child_samples=50, subsample=0.8, colsample_bytree=0.8)),
    ("LGB_leaf128_d8",  dict(num_leaves=128, max_depth=8,  min_child_samples=50, subsample=0.8, colsample_bytree=0.8)),
    ("LGB_leaf256_d10", dict(num_leaves=256, max_depth=10, min_child_samples=50, subsample=0.8, colsample_bytree=0.8)),
    ("LGB_leaf128_d8_cs9", dict(num_leaves=128, max_depth=8, min_child_samples=50, subsample=0.8, colsample_bytree=0.9)),
    ("LGB_leaf64_d8_ss7",  dict(num_leaves=64,  max_depth=8, min_child_samples=30, subsample=0.7, colsample_bytree=0.8)),
]

# ---- XGBoost ×5 (不同深度 + 正则 + 种子) ----
XGB_MODELS = [
    ("XGB_d6", dict(max_depth=6,  min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)),
    ("XGB_d8", dict(max_depth=8,  min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)),
    ("XGB_d10",dict(max_depth=10, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)),
    ("XGB_d8_reg", dict(max_depth=8, min_child_weight=100, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0)),
    ("XGB_d6_gamma", dict(max_depth=6, min_child_weight=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, gamma=0.1)),
]

# ---- CatBoost ×4 (不同深度 + 正则 + 种子) ----
CB_MODELS = [
    ("CB_d6",  dict(depth=6,  min_data_in_leaf=50, l2_leaf_reg=3, subsample=0.8)),
    ("CB_d8",  dict(depth=8,  min_data_in_leaf=50, l2_leaf_reg=3, subsample=0.8)),
    ("CB_d10", dict(depth=10, min_data_in_leaf=50, l2_leaf_reg=3, subsample=0.8)),
    ("CB_d8_reg", dict(depth=8, min_data_in_leaf=50, l2_leaf_reg=5, subsample=0.8)),
]

# ---- RealMLP ×4 ----
MLP_MODELS = [
    ("MLP_512_512_512_half", [512,512,512], 64),
    ("MLP_512_512_half",     [512,512],     64),
    ("MLP_256_128_64_half",  [256,128,64],  64),
    ("MLP_512_512_512",      [512,512,512], 128),
]

# ============================================================
# 训练 — 所有模型共用一个 5-fold split
# ============================================================
all_preds = {}  # name -> {"oof": (N,3), "test": (N_test,3)}

# 固定 fold splits（LightGBM 和 XGBoost 共用 X）
folds = list(skf.split(X, y))

# ---- LightGBM ----
for name, params in LGB_MODELS:
    progress(f"\n{name}")
    oof = np.zeros((len(X), 3)); tp = np.zeros((len(X_test), 3))
    scores = []
    for fold, (tr, val) in enumerate(folds):
        m = lgb.LGBMClassifier(
            objective='multiclass', num_class=3, n_estimators=2000,
            learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1,
            random_state=fold*42, n_jobs=-1, verbose=-1, **params)
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val] = m.predict_proba(X[val]); tp += m.predict_proba(X_test)/5
        scores.append(balanced_accuracy_score(y[val], np.argmax(oof[val], axis=1)))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    progress(f"  BA={ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
    all_preds[name] = {"oof": oof, "test": tp, "ba": ba, "scores": scores}

# ---- XGBoost ----
for name, params in XGB_MODELS:
    progress(f"\n{name}")
    oof = np.zeros((len(X), 3)); tp = np.zeros((len(X_test), 3))
    scores = []
    for fold, (tr, val) in enumerate(folds):
        m = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3,
            learning_rate=0.05, n_estimators=2000, random_state=fold*42,
            n_jobs=-1, verbosity=0, **params)
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof[val] = m.predict_proba(X[val]); tp += m.predict_proba(X_test)/5
        scores.append(balanced_accuracy_score(y[val], np.argmax(oof[val], axis=1)))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    progress(f"  BA={ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
    all_preds[name] = {"oof": oof, "test": tp, "ba": ba, "scores": scores}

# ---- CatBoost ----
for name, params in CB_MODELS:
    progress(f"\n{name}")
    oof = np.zeros((len(X), 3)); tp = np.zeros((len(X_cb_test), 3))
    scores = []
    # CatBoost 用同样的 fold split 但不同的 X
    for fold, (tr, val) in enumerate(folds):
        m = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, bootstrap_type='Bernoulli',
            random_seed=fold*42, thread_count=-1, verbose=0,
            allow_writing_files=False, **params)
        m.fit(X_cb[tr], y[tr], eval_set=[(X_cb[val], y[val])],
              cat_features=cat_idx, early_stopping_rounds=50, verbose=0)
        oof[val] = m.predict_proba(X_cb[val]); tp += m.predict_proba(X_cb_test)/5
        scores.append(balanced_accuracy_score(y[val], np.argmax(oof[val], axis=1)))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    progress(f"  BA={ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
    all_preds[name] = {"oof": oof, "test": tp, "ba": ba, "scores": scores}

# ---- RealMLP (GPU) ----
for name, hidden, ep in MLP_MODELS:
    progress(f"\n{name}")
    oof = np.zeros((len(X), 3)); tp = np.zeros((len(X_test), 3))
    scores = []
    for fold, (tr, val) in enumerate(folds):
        m = RealMLP_TD_Classifier(device='cuda', random_state=fold*42,
                                   n_epochs=ep, batch_size=8192, hidden_sizes=hidden)
        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val]); tp += m.predict_proba(X_test)/5
        scores.append(balanced_accuracy_score(y[val], np.argmax(oof[val], axis=1)))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    progress(f"  BA={ba:.5f} folds={[f'{s:.5f}' for s in scores]}")
    all_preds[name] = {"oof": oof, "test": tp, "ba": ba, "scores": scores}

# ============================================================
# 集成: 加权平均 + 阈值调优
# ============================================================
progress(f"\n{'='*50}\nENSEMBLE\n{'='*50}")

# 所有模型 OOF 加权平均（同配置的树模型平均，MLP 单独）
oof_trees = np.mean([v['oof'] for k,v in all_preds.items() if 'LGB' in k or 'XGB' in k or 'CB' in k], axis=0)
oof_mlp   = np.mean([v['oof'] for k,v in all_preds.items() if 'MLP' in k], axis=0)

# 对比不同融合比例
from scipy.optimize import minimize

def tuned_predict(probs, t):
    return np.argmax(probs / np.array(t), axis=1)
def neg_ba(t, probs, yt):
    return -balanced_accuracy_score(yt, tuned_predict(probs, t))

combos = {
    "trees_only":   oof_trees,
    "mlp_only":     oof_mlp,
    "trees+mlp_50": (oof_trees + oof_mlp) / 2,
    "trees+mlp_70": oof_trees * 0.7 + oof_mlp * 0.3,
}

for name, oof in combos.items():
    ba_base = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    res = minimize(neg_ba, [1,1,1], args=(oof, y), method='Nelder-Mead',
                   bounds=[(0.2,3),(0.2,3),(0.2,3)], options=dict(xatol=0.001, maxiter=500))
    progress(f"  {name:<16}: base={ba_base:.5f} → tuned={-res.fun:.5f} (+{-res.fun-ba_base:.5f})")

# 最佳组合生成提交
best_combo = (oof_trees + oof_mlp) / 2  # default to 50/50
test_trees = np.mean([v['test'] for k,v in all_preds.items() if 'LGB' in k or 'XGB' in k or 'CB' in k], axis=0)
test_mlp   = np.mean([v['test'] for k,v in all_preds.items() if 'MLP' in k], axis=0)
test_ens = (test_trees + test_mlp) / 2

res = minimize(neg_ba, [1,1,1], args=(best_combo, y), method='Nelder-Mead',
               bounds=[(0.2,3),(0.2,3),(0.2,3)], options=dict(xatol=0.001, maxiter=500))
pred_final = le.inverse_transform(tuned_predict(test_ens, res.x))
sub = pd.DataFrame({'id': test['id'], 'class': pred_final})
sub.to_csv('submission_ensemble.csv', index=False)
progress(f"\nSaved: submission_ensemble.csv ({len(sub)} rows)")
progress(f"Dist: {sub['class'].value_counts().to_dict()}")

# 保存所有 OOF
summary = {k: float(v['ba']) for k,v in all_preds.items()}
summary['ensemble_tuned'] = float(-neg_ba(res.x, best_combo, y))
with open('all_models_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
progress("ALL DONE!")