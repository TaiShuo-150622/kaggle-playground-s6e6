"""
Quick test: u_z + g_z 特征是否有提升？
只跑 LightGBM 5-fold，对比新旧特征集的 OOF BA
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
import lightgbm as lgb
from datetime import datetime
def progress(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc']  = LabelEncoder().fit_transform(test[col])

train['target'] = LabelEncoder().fit_transform(train['class'])

# 旧特征（不加 u_z, g_z）
feat_old = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv',
            'alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc']

# 新特征（加 u_z, g_z）
feat_new = feat_old + ['u_z', 'g_z']

progress(f"Old: {len(feat_old)} features, New: {len(feat_new)} features")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_features(feats, name):
    X = train[feats].values.astype(np.float32)
    y = train['target'].values
    scores = []
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
        pred = m.predict(X[val])
        scores.append(balanced_accuracy_score(y[val], pred))
    return np.mean(scores)

ba_old = eval_features(feat_old, "old")
ba_new = eval_features(feat_new, "new")
delta = ba_new - ba_old

progress(f"OLD (no u_z/g_z):  BA={ba_old:.6f}")
progress(f"NEW (with u_z/g_z): BA={ba_new:.6f}")
progress(f"DELTA={'+' if delta >= 0 else ''}{delta:.6f}")

# 保存结果
with open("quick_test_result.txt", "w") as f:
    f.write(f"OLD={ba_old:.6f}\nNEW={ba_new:.6f}\nDELTA={'+' if delta>=0 else ''}{delta:.6f}\n")

# 查看新特征的 feature importance
if delta > 0:
    X_new = train[feat_new].values.astype(np.float32)
    m = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, n_estimators=2000,
        learning_rate=0.05, num_leaves=128, max_depth=8,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1
    )
    X_tr, X_val = X_new[:400000], X_new[400000:]
    y_tr, y_val = train['target'].values[:400000], train['target'].values[400000:]
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    imp = pd.DataFrame({'feature': feat_new, 'importance': m.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    progress(f"\nTop 10 features:")
    for _, r in imp.head(10).iterrows():
        marker = " NEW!" if r['feature'] in ['u_z','g_z'] else ""
        progress(f"  {r['feature']:>15}: {r['importance']}{marker}")
EOF
