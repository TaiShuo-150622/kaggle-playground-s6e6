"""
完整 Pipeline: 5 个方向逐一执行
=================================
1. Adversarial Validation  → 确认 train/test 分布一致
2. Class-weighted 采样     → 提升 minority class recall
3. Threshold Tuning        → 搜索最优三分类阈值
4. Pseudo-Labeling         → 高置信度 test 样本回训练集
5. Soft Voting + Stacking  → 三模型融合
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from scipy.optimize import minimize
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import time, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 数据加载
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
le = LabelEncoder().fit(train['class'])

X = train[feat_all].values.astype(np.float32)
y = train['target'].values
X_test = test[feat_all].values.astype(np.float32)

# CatBoost features
feat_cb = feat_num + feat_pos + feat_cat
X_cb = train[feat_cb].values.astype(object)
X_cb_test = test[feat_cb].values.astype(object)
cat_idx = [i for i, col in enumerate(feat_cb) if col in feat_cat]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Class dist: {dict(zip(le.classes_, np.bincount(y)))}")

# ============================================================
# 步骤 1: Adversarial Validation
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Adversarial Validation (train vs test 分布差异检测)")
print("=" * 60)

combined = np.vstack([X, X_test])
labels_av = np.array([0]*len(X) + [1]*len(X_test))

skf_av = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
av_scores = []
for fold, (tr, val) in enumerate(skf_av.split(combined, labels_av)):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                           num_leaves=31, random_state=fold,
                           n_jobs=-1, verbose=-1)
    m.fit(combined[tr], labels_av[tr],
          eval_set=[(combined[val], labels_av[val])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = m.predict_proba(combined[val])[:,1]
    av_scores.append(roc_auc_score(labels_av[val], pred))

av_auc = np.mean(av_scores)
print(f"  AV AUC: {av_auc:.4f}  (5-fold CV)")
if av_auc > 0.60:
    print("  ⚠️  train/test 有明显分布差异! 查看重要特征...")
    # 看哪些特征导致差异
    m_full = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    m_full.fit(combined, labels_av,
               callbacks=[lgb.early_stopping(30, verbose=False)])
    imp = pd.DataFrame({'feature': feat_all, 'importance': m_full.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(f"  Top 5 导致差异的特征: {imp.head(5)['feature'].tolist()}")
else:
    print("  ✅ train/test 分布基本一致，无需修正特征")

# ============================================================
# 步骤 2: Class-Weighted Sampling
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Class-Weighted Sampling (三类等权)")
print("=" * 60)

# 方案: 欠采样多数类 + sample_weight
class_counts = np.bincount(y)
print(f"  原始: GAL={class_counts[0]:,}, STAR={class_counts[1]:,}, QSO={class_counts[2]:,}")

# 等权采样 (STAR 最少 → 以此为标准)
target_count = min(class_counts)  # ~82k
rng = np.random.RandomState(42)
idx_balanced = []
for c in range(3):
    c_idx = np.where(y == c)[0]
    if len(c_idx) <= target_count:
        idx_balanced.extend(c_idx.tolist())
    else:
        idx_balanced.extend(rng.choice(c_idx, target_count, replace=False).tolist())
X_bal = X[idx_balanced]
y_bal = y[idx_balanced]
print(f"  等权采样后: {X_bal.shape[0]:,} rows ({np.bincount(y_bal)})")

# 训练等权 LightGBM
oof_bal = np.zeros((len(X), 3))
test_bal = np.zeros((len(X_test), 3))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    # 在平衡数据上训练
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, n_estimators=2000,
        learning_rate=0.05, num_leaves=128, max_depth=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=fold, n_jobs=-1, verbose=-1
    )
    m.fit(X_bal, y_bal,  # 在平衡数据上训练
          eval_set=[(X[val], y[val])],  # 在原分布上验证
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_bal[val] = m.predict_proba(X[val])
    test_bal += m.predict_proba(X_test) / 5

pred_bal = np.argmax(oof_bal, axis=1)
rec_bal = recall_score(y, pred_bal, average=None)
ba_bal = balanced_accuracy_score(y, pred_bal)
print(f"  Balanced LGB: BA={ba_bal:.4f}  GAL={rec_bal[0]:.4f}  STAR={rec_bal[1]:.4f}  QSO={rec_bal[2]:.4f}")

# ============================================================
# 步骤 3: Threshold Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Threshold Tuning")
print("=" * 60)

# 先生成标准 OOF (不平衡采样)
print("  生成标准 LGB OOF...")
oof_lgb = np.zeros((len(X), 3))
test_lgb = np.zeros((len(X_test), 3))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, n_estimators=2000,
        learning_rate=0.05, num_leaves=128, max_depth=8,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=fold, n_jobs=-1, verbose=-1
    )
    m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val] = m.predict_proba(X[val])
    test_lgb += m.predict_proba(X_test) / 5
    print(f"    fold {fold+1}/5")

# XGBoost OOF
print("  生成 XGBoost OOF...")
oof_xgb = np.zeros((len(X), 3))
test_xgb = np.zeros((len(X_test), 3))
for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                          n_estimators=2000, learning_rate=0.05,
                          max_depth=8, min_child_weight=50,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=0.1,
                          random_state=fold, n_jobs=-1, verbosity=0)
    m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
    oof_xgb[val] = m.predict_proba(X[val])
    test_xgb += m.predict_proba(X_test) / 5
    print(f"    fold {fold+1}/5")

# CatBoost OOF
print("  生成 CatBoost OOF...")
oof_cb = np.zeros((len(X), 3))
test_cb = np.zeros((len(X_test), 3))
for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                           min_data_in_leaf=50, bootstrap_type='Bernoulli',
                           subsample=0.8, l2_leaf_reg=3,
                           random_seed=fold, thread_count=-1,
                           verbose=0, allow_writing_files=False)
    m.fit(X_cb[tr], y[tr], eval_set=[(X_cb[val], y[val])],
          cat_features=cat_idx, early_stopping_rounds=50, verbose=0)
    oof_cb[val] = m.predict_proba(X_cb[val])
    test_cb += m.predict_proba(X_cb_test) / 5
    print(f"    fold {fold+1}/5")

# 调阈值
def tuned_predict(probs, t):
    return np.argmax(probs / np.array(t), axis=1)

def neg_ba(t, probs, y_true):
    return -balanced_accuracy_score(y_true, tuned_predict(probs, t))

# 在 Voting OOF 上调优
oof_vote_raw = (oof_lgb + oof_xgb + oof_cb) / 3
baseline_ba = balanced_accuracy_score(y, np.argmax(oof_vote_raw, axis=1))
print(f"\n  Voting baseline BA: {baseline_ba:.4f}")

result = minimize(neg_ba, x0=[1.0, 1.0, 1.0], args=(oof_vote_raw, y),
                  method='Nelder-Mead',
                  bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
                  options=dict(xatol=0.001, maxiter=500))
t_opt = result.x
tuned_ba = -result.fun
print(f"  Tuned thresholds: {t_opt} → BA = {tuned_ba:.4f} (+{tuned_ba-baseline_ba:.4f})")

# 也调平衡版的阈值
result_bal = minimize(neg_ba, x0=[1.0, 1.0, 1.0], args=(oof_bal, y),
                      method='Nelder-Mead',
                      bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
                      options=dict(xatol=0.001, maxiter=500))
t_bal = result_bal.x
print(f"  Balanced LGB thresholds: {t_bal} → BA = {-result_bal.fun:.4f}")

# ============================================================
# 步骤 4: Pseudo-Labeling
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Pseudo-Labeling")
print("=" * 60)

# 融合所有模型的 test 预测
test_vote = (test_lgb + test_xgb + test_cb) / 3
test_pred = tuned_predict(test_vote, t_opt)
test_conf = np.max(test_vote / np.array(t_opt), axis=1) / np.sum(test_vote / np.array(t_opt), axis=1)

# 三模型一致 + 高置信度 → 伪标签
# 三模型各自预测
test_pred_lgb = np.argmax(test_lgb, axis=1)
test_pred_xgb = np.argmax(test_xgb, axis=1)
test_pred_cb  = np.argmax(test_cb, axis=1)
agreement = (test_pred_lgb == test_pred_xgb) & (test_pred_xgb == test_pred_cb)

pseudo_mask = agreement & (test_conf > 0.5)  # 三模型一致 + 高置信
n_pseudo = pseudo_mask.sum()
print(f"  伪标签样本: {n_pseudo:,} / {len(test):,} ({100*n_pseudo/len(test):.1f}%)")

if n_pseudo > 50000:
    # 加入训练集
    X_pseudo = np.vstack([X, X_test[pseudo_mask]])
    y_pseudo = np.concatenate([y, test_pred[pseudo_mask]])
    print(f"  扩展训练集: {X_pseudo.shape[0]:,} rows")

    # 用扩展数据训练
    oof_pl = np.zeros((len(X_pseudo), 3))
    test_pl = np.zeros((len(X_test), 3))
    skf_pl = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr, val) in enumerate(skf_pl.split(X_pseudo, y_pseudo)):
        m = lgb.LGBMClassifier(
            objective='multiclass', num_class=3, n_estimators=2000,
            learning_rate=0.05, num_leaves=128, max_depth=8,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=fold, n_jobs=-1, verbose=-1
        )
        m.fit(X_pseudo[tr], y_pseudo[tr],
              eval_set=[(X_pseudo[val], y_pseudo[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_pl[val] = m.predict_proba(X_pseudo[val])
        test_pl += m.predict_proba(X_test) / 5
        print(f"    fold {fold+1}/5")

    # 调阈值
    result_pl = minimize(neg_ba, x0=t_opt, args=(oof_pl, y_pseudo),
                         method='Nelder-Mead',
                         bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
                         options=dict(xatol=0.001, maxiter=500))
    t_pl = result_pl.x
    ba_pl_base = balanced_accuracy_score(y_pseudo, np.argmax(oof_pl, axis=1))
    ba_pl_tuned = -result_pl.fun
    print(f"  Pseudo-Label LGB: BA={ba_pl_base:.4f} → tuned BA={ba_pl_tuned:.4f}")
else:
    test_pl = test_lgb
    t_pl = t_opt
    print("  伪标签不足 50000，跳过")

# ============================================================
# 步骤 5: 生成最终提交
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: 生成提交文件")
print("=" * 60)

test_vote = (test_lgb + test_xgb + test_cb) / 3

# S1: Voting + argmax
pred = le.inverse_transform(np.argmax(test_vote, axis=1))
pd.DataFrame({'id': test['id'], 'class': pred}).to_csv('submission_01_voting_argmax.csv', index=False)
print("  S1: voting_argmax → submission_01_voting_argmax.csv")

# S2: Voting + tuned thresholds
pred = le.inverse_transform(tuned_predict(test_vote, t_opt))
pd.DataFrame({'id': test['id'], 'class': pred}).to_csv('submission_02_voting_tuned.csv', index=False)
print(f"  S2: voting_tuned (t={[f'{x:.3f}' for x in t_opt]}) → submission_02_voting_tuned.csv")

# S3: Balanced LGB + tuned
test_vote_bal = (test_bal + test_xgb + test_cb) / 3
pred = le.inverse_transform(tuned_predict(test_vote_bal, t_bal))
pd.DataFrame({'id': test['id'], 'class': pred}).to_csv('submission_03_balanced.csv', index=False)
print(f"  S3: balanced_LGB (t={[f'{x:.3f}' for x in t_bal]}) → submission_03_balanced.csv")

# S4: Pseudo-Label
if n_pseudo > 50000:
    pred = le.inverse_transform(tuned_predict(test_pl, t_pl))
    pd.DataFrame({'id': test['id'], 'class': pred}).to_csv('submission_04_pseudo_label.csv', index=False)
    print(f"  S4: pseudo_label (t={[f'{x:.3f}' for x in t_pl]}) → submission_04_pseudo_label.csv")
else:
    print("  S4: pseudo_label → skipped (insufficient pseudo labels)")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
