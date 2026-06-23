"""
TabICL 三类分别测试: 每类单独做 one-vs-rest 二分类
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from datetime import datetime
def progress(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

train = pd.read_csv("data/train_fe.csv")
feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']
for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat]
le = LabelEncoder().fit(train['class'])
train['target'] = le.transform(train['class'])

X = train[feat_all].values.astype(np.float32)
y = train['target'].values

from tabicl import TabICLClassifier
rng = np.random.RandomState(42)

results = {}
for cls_name, cls_idx in [('GALAXY', 0), ('QSO', 1), ('STAR', 2)]:
    progress(f"\n{'='*50}")
    progress(f"{cls_name} vs 非{cls_name}")

    # 二分类: 该类 vs 其他，每类采样 15K
    pos_idx = np.where(y == cls_idx)[0]
    neg_idx = np.where(y != cls_idx)[0]

    n_each = 15000
    pos_sample = rng.choice(pos_idx, min(n_each, len(pos_idx)), replace=False)
    neg_sample = rng.choice(neg_idx, min(n_each, len(neg_idx)), replace=False)

    X_s = np.vstack([X[pos_sample], X[neg_sample]])
    y_s = np.array([1]*len(pos_sample) + [0]*len(neg_sample))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
    )
    progress(f"  Train: {X_tr.shape[0]:,}  Val: {X_val.shape[0]:,}  "
             f"Pos={y_tr.sum()}+{y_val.sum()}")

    model = TabICLClassifier(device='cpu', n_estimators=3, verbose=False)
    progress(f"  Fitting...")
    model.fit(X_tr, y_tr)
    progress(f"  Predicting...")
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    ba = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, proba)
    rec = recall_score(y_val, pred)

    progress(f"  → BA={ba:.4f}  AUC={auc:.4f}  Recall={rec:.4f}")
    results[cls_name] = {'ba': ba, 'auc': auc, 'recall': rec}

progress(f"\n{'='*50}")
progress("三类 TabICL 汇总:")
progress(f"  {'':>8}  {'BA':>8}  {'AUC':>8}  {'Recall':>8}")
for cls_name in ['GALAXY', 'QSO', 'STAR']:
    r = results[cls_name]
    progress(f"  {cls_name:>8}: {r['ba']:.4f}  {r['auc']:.4f}  {r['recall']:.4f}")
