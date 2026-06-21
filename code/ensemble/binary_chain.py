"""
Binary Chain: 三分类 → 两个二分类链式求解
=========================================
Step 1: STAR vs 非STAR     (z≈0 的天然容易分)
Step 2: GALAXY vs QSO      (在非STAR上精细分类)
        输入加了 Step1 的 p_star 作为额外特征

最终: P(STAR) = p1_star
      P(GAL)  = (1-p1_star) * p2_galaxy
      P(QSO)  = (1-p1_star) * p2_qso
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import time, warnings, sys
from datetime import datetime
warnings.filterwarnings('ignore')

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time(); progress(f"{self.name}..."); return self
    def __exit__(self, *args):
        progress(f"{self.name} done ({time.time()-self.t0:.0f}s)")

# ============================================================
# 0. 数据
# ============================================================
progress("Loading data...")
train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']

for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc']  = LabelEncoder().fit_transform(test[col])

feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat]
le_class = LabelEncoder().fit(train['class'])
train['target'] = le_class.transform(train['class'])

X = train[feat_all].values.astype(np.float32)
y = train['target'].values
X_test = test[feat_all].values.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

progress(f"Train: {X.shape}, Test: {X_test.shape}")
progress(f"Classes: {dict(zip(le_class.classes_, np.bincount(y)))}")

# ============================================================
# 1. Step 1: STAR vs 非STAR
# ============================================================
progress("=" * 60)
progress("STEP 1: STAR vs 非STAR (binary)")

y1 = (y == 2).astype(int)  # STAR=1, 非STAR=0
progress(f"  STAR={y1.sum():,}  非STAR={(1-y1).sum():,}")

oof_p1 = np.zeros(len(X))      # OOF P(STAR)
test_p1 = np.zeros(len(X_test))

with Timer("Step 1 (LGB 5-fold)"):
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective='binary', n_estimators=2000, learning_rate=0.05,
            num_leaves=128, max_depth=8, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=fold, n_jobs=-1, verbose=-1
        )
        m.fit(X[tr], y1[tr], eval_set=[(X[val], y1[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_p1[val] = m.predict_proba(X[val])[:, 1]
        test_p1 += m.predict_proba(X_test)[:, 1] / 5
        progress(f"  Fold {fold+1}/5")

ba1 = balanced_accuracy_score(y1, (oof_p1 > 0.5).astype(int))
progress(f"  Step1 BA: {ba1:.4f}")

# ============================================================
# 2. Step 2: GALAXY vs QSO（只在非STAR样本上）
# ============================================================
progress("=" * 60)
progress("STEP 2: GALAXY vs QSO (binary, on non-STAR samples)")

# 非STAR 子集
mask_ns = (y != 2)  # non-STAR
X_ns = X[mask_ns]
y_ns = y[mask_ns]  # 0=GALAXY, 1=QSO
y2 = (y_ns == 1).astype(int)  # QSO=1, GALAXY=0
progress(f"  GALAXY={(y2==0).sum():,}  QSO={y2.sum():,}")

# 把 oof_p1 (P_STAR) 作为额外特征拼回去
X2 = np.column_stack([X_ns, oof_p1[mask_ns]])       # 20 维
X2_test = np.column_stack([X_test, test_p1])         # 20 维

oof_p2_qso = np.zeros(len(y_ns))    # OOF P(QSO | 非STAR)
test_p2_qso = np.zeros(len(X_test))

with Timer("Step 2 (LGB 5-fold)"):
    # 用 GALAXY vs QSO 分层
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, val) in enumerate(skf2.split(X2, y2)):
        m = lgb.LGBMClassifier(
            objective='binary', n_estimators=2000, learning_rate=0.05,
            num_leaves=128, max_depth=8, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=fold, n_jobs=-1, verbose=-1
        )
        m.fit(X2[tr], y2[tr], eval_set=[(X2[val], y2[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_p2_qso[val] = m.predict_proba(X2[val])[:, 1]
        test_p2_qso += m.predict_proba(X2_test)[:, 1] / 5
        progress(f"  Fold {fold+1}/5")

ba2 = balanced_accuracy_score(y2, (oof_p2_qso > 0.5).astype(int))
progress(f"  Step2 BA: {ba2:.4f}")

# ============================================================
# 3. 组合概率: 链式法则
# ============================================================
progress("=" * 60)
progress("STEP 3: 组合概率")

# OOF: 重建全量 OOF
oof_probs = np.zeros((len(X), 3))      # [GAL, QSO, STAR]
oof_probs[:, 2] = oof_p1                # P(STAR)
oof_probs[mask_ns, 1] = oof_p2_qso * (1 - oof_p1[mask_ns])     # P(QSO) = P(QSO|非STAR)*(1-P(STAR))
oof_probs[mask_ns, 0] = (1 - oof_p2_qso) * (1 - oof_p1[mask_ns])  # P(GAL)

# Test
test_probs = np.zeros((len(X_test), 3))
test_probs[:, 2] = test_p1
test_probs[:, 1] = test_p2_qso * (1 - test_p1)
test_probs[:, 0] = (1 - test_p2_qso) * (1 - test_p1)

# 评估
base_pred = np.argmax(oof_probs, axis=1)
base_ba = balanced_accuracy_score(y, base_pred)
base_rec = recall_score(y, base_pred, average=None)
progress(f"  Binary Chain (argmax): BA={base_ba:.4f}  "
         f"GAL={base_rec[0]:.4f}  STAR={base_rec[1]:.4f}  QSO={base_rec[2]:.4f}")

# ============================================================
# 4. Threshold Tuning (在 chain OOF 上调)
# ============================================================
progress("Threshold tuning on Binary Chain...")
from scipy.optimize import minimize

def tuned_predict(probs, t):
    return np.argmax(probs / np.array(t), axis=1)

def neg_ba(t, probs, y_true):
    return -balanced_accuracy_score(y_true, tuned_predict(probs, t))

result = minimize(neg_ba, x0=[1.0, 1.0, 1.0], args=(oof_probs, y),
                  method='Nelder-Mead',
                  bounds=[(0.2, 3.0), (0.2, 3.0), (0.2, 3.0)],
                  options=dict(xatol=0.001, maxiter=500))

t_chain = result.x
tuned_ba = -result.fun
tuned_pred = tuned_predict(oof_probs, t_chain)
tuned_rec = recall_score(y, tuned_pred, average=None)
progress(f"  Tuned: BA={tuned_ba:.4f} (+{tuned_ba-base_ba:.4f})  thresholds={t_chain}")
progress(f"  GAL={tuned_rec[0]:.4f}  STAR={tuned_rec[1]:.4f}  QSO={tuned_rec[2]:.4f}")

# ============================================================
# 5. 生成提交
# ============================================================
progress("=" * 60)
progress("Generating submissions...")

# S5: Binary Chain argmax
test_pred5 = np.argmax(test_probs, axis=1)
pred5 = le_class.inverse_transform(test_pred5)
pd.DataFrame({'id': test['id'], 'class': pred5}).to_csv('submission_05_chain_argmax.csv', index=False)

# S6: Binary Chain tuned
test_pred6 = tuned_predict(test_probs, t_chain)
pred6 = le_class.inverse_transform(test_pred6)
pd.DataFrame({'id': test['id'], 'class': pred6}).to_csv('submission_06_chain_tuned.csv', index=False)

# S7: Chain + voting(S2) 融合
s2 = pd.read_csv('submission_02_voting_tuned.csv')
chain_prob = pd.DataFrame({'id': test['id'], 'chain_gal': test_probs[:,0],
                           'chain_qso': test_probs[:,1], 'chain_star': test_probs[:,2]})
# 简单平均（如果有 S2 的概率更好，这里用 label 做 majority vote）
# 实际可以用 50/50 blend

progress(f"  S5: chain_argmax → submission_05_chain_argmax.csv")
progress(f"  S6: chain_tuned  → submission_06_chain_tuned.csv")
progress(f"  OOF: base={base_ba:.4f} → tuned={tuned_ba:.4f} (+{tuned_ba-base_ba:.4f})")

progress("=" * 60)
progress("ALL DONE")
