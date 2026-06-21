"""
Threshold Tuning for Balanced Accuracy
=======================================
argmax(prob) 只在各类概率均匀时最优。BA 优化需要调三类阈值。

方法: OOF 预测上搜索最优阈值 → 应用到 test 预测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据
# ============================================================
train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']

for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc']  = LabelEncoder().fit_transform(test[col])

feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat]
train['target'] = LabelEncoder().fit_transform(train['class'])

X = train[feat_all].values
y = train['target'].values
X_test = test[feat_all].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
le = LabelEncoder().fit(train['class'])

# ============================================================
# 2. 生成 OOF 预测（三模型）
# ============================================================
def get_oof_preds(model_class, model_kwargs, fit_kwargs, X_in, y_in, name):
    """5-fold CV → OOF 概率 + test 概率"""
    oof_probs = np.zeros((len(X_in), 3))
    test_probs = np.zeros((len(X_test), 3))

    for fold, (tr, val) in enumerate(skf.split(X_in, y_in)):
        m = model_class(**model_kwargs)
        m.fit(X_in[tr], y_in[tr], **fit_kwargs.get('fit', {}))
        oof_probs[val] = m.predict_proba(X_in[val])
        test_probs += m.predict_proba(X_test) / 5
        print(f"  {name} fold {fold+1}/5 done")

    return oof_probs, test_probs

print("Generating OOF predictions...")
print("=" * 50)

# LightGBM
oof_lgb, test_lgb = get_oof_preds(
    lgb.LGBMClassifier,
    dict(objective='multiclass', num_class=3, n_estimators=2000,
         learning_rate=0.05, num_leaves=128, max_depth=8,
         min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.1, reg_lambda=0.1,
         random_state=42, n_jobs=-1, verbose=-1),
    dict(fit=dict(callbacks=[lgb.early_stopping(50, verbose=False)])),
    X, y, "LightGBM"
)

# XGBoost
oof_xgb, test_xgb = get_oof_preds(
    xgb.XGBClassifier,
    dict(objective='multi:softprob', num_class=3, n_estimators=2000,
         learning_rate=0.05, max_depth=8, min_child_weight=50,
         subsample=0.8, colsample_bytree=0.8,
         reg_alpha=0.1, reg_lambda=0.1,
         random_state=42, n_jobs=-1, verbosity=0),
    dict(fit=dict(verbose=False)),
    X, y, "XGBoost"
)

# CatBoost (need cat features)
feat_cb = feat_num + feat_pos + feat_cat
X_cb = train[feat_cb].values
X_cb_test = test[feat_cb].values
cat_idx = [i for i, col in enumerate(feat_cb) if col in feat_cat]

oof_cb, test_cb = get_oof_preds(
    CatBoostClassifier,
    dict(iterations=2000, learning_rate=0.05, depth=8,
         min_data_in_leaf=50, bootstrap_type='Bernoulli', subsample=0.8,
         l2_leaf_reg=3, random_seed=42, thread_count=-1,
         verbose=0, allow_writing_files=False),
    dict(fit=dict(cat_features=cat_idx, early_stopping_rounds=50, verbose=0)),
    X_cb, y, "CatBoost"
)

# ============================================================
# 3. Threshold Tuning
# ============================================================
print("\n" + "=" * 50)
print("Threshold Tuning")
print("=" * 50)

# 基线: argmax
pred_argmax = np.argmax(oof_lgb, axis=1)
ba_baseline = balanced_accuracy_score(y, pred_argmax)
print(f"\n  Baseline (argmax): BA = {ba_baseline:.4f}")

# 搜索最优阈值
# 思路: 对概率做 softmax-like 变换: prob_class /= t_class, 然后 argmax
# 三个阈值 t_GALAXY, t_STAR, t_QSO，范围 [0.2, 2.0]

def tuned_predict(probs, thresholds):
    """thresholds = [t0, t1, t2], 越小=越容易被选"""
    adjusted = probs / np.array(thresholds)
    return np.argmax(adjusted, axis=1)

def neg_ba(thresholds, probs, y_true):
    pred = tuned_predict(probs, thresholds)
    return -balanced_accuracy_score(y_true, pred)

# 在 LightGBM OOF 上搜索（单模调优）
result_lgb = minimize(
    neg_ba,
    x0=[1.0, 1.0, 1.0],
    args=(oof_lgb, y),
    method='Nelder-Mead',
    bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
    options=dict(xatol=0.001, maxiter=500)
)
t_lgb = result_lgb.x
ba_lgb_tuned = -result_lgb.fun
print(f"\n  LightGBM tuned: thresholds={t_lgb} → BA = {ba_lgb_tuned:.4f}  (+{ba_lgb_tuned-ba_baseline:.4f})")

# 在三个模型各自的 OOF 上调优
for name, oof in [("XGBoost", oof_xgb), ("CatBoost", oof_cb)]:
    result = minimize(
        neg_ba, x0=[1.0, 1.0, 1.0], args=(oof, y),
        method='Nelder-Mead',
        bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
        options=dict(xatol=0.001, maxiter=500)
    )
    ba_tuned = -result.fun
    pred_base = np.argmax(oof, axis=1)
    ba_base = balanced_accuracy_score(y, pred_base)
    print(f"  {name} tuned: thresholds={result.x} → BA = {ba_tuned:.4f}  (+{ba_tuned-ba_base:.4f})")

# 在三模型 Voting 上调优
oof_vote = (oof_lgb + oof_xgb + oof_cb) / 3
result_vote = minimize(
    neg_ba, x0=[1.0, 1.0, 1.0], args=(oof_vote, y),
    method='Nelder-Mead',
    bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
    options=dict(xatol=0.001, maxiter=500)
)
ba_vote_base = balanced_accuracy_score(y, np.argmax(oof_vote, axis=1))
ba_vote_tuned = -result_vote.fun
print(f"\n  Voting tuned: thresholds={result_vote.x} → BA = {ba_vote_tuned:.4f}  (+{ba_vote_tuned-ba_vote_base:.4f})")

# ============================================================
# 4. 生成提交
# ============================================================
print("\n" + "=" * 50)
print("Generating Submissions")
print("=" * 50)

# 提交1: Voting + argmax
test_vote = (test_lgb + test_xgb + test_cb) / 3
pred1 = le.inverse_transform(np.argmax(test_vote, axis=1))
sub1 = pd.DataFrame({'id': test['id'], 'class': pred1})
sub1.to_csv('submission_voting_argmax.csv', index=False)
print(f"  voting_argmax: {sub1['class'].value_counts().to_dict()}")

# 提交2: Voting + tuned thresholds
t_opt = result_vote.x
pred2 = le.inverse_transform(tuned_predict(test_vote, t_opt))
sub2 = pd.DataFrame({'id': test['id'], 'class': pred2})
sub2.to_csv('submission_voting_tuned.csv', index=False)
print(f"  voting_tuned:   {sub2['class'].value_counts().to_dict()}")

# 提交3: LightGBM + tuned (通常单模调优效果好)
pred3 = le.inverse_transform(tuned_predict(test_lgb, t_lgb))
sub3 = pd.DataFrame({'id': test['id'], 'class': pred3})
sub3.to_csv('submission_lgb_tuned.csv', index=False)
print(f"  lgb_tuned:      {sub3['class'].value_counts().to_dict()}")

print("\nDone! 3 submissions ready:")
print("  submission_voting_argmax.csv")
print("  submission_voting_tuned.csv")
print("  submission_lgb_tuned.csv")
