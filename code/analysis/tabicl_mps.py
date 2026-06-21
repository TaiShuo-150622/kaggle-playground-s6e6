"""
TabICL MPS GPU 测试: 逐步放大样本量
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from datetime import datetime
def progress(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

train = pd.read_csv("data/train_fe.csv")
feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']
for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat]
le = LabelEncoder().fit(train['class']); train['target'] = le.transform(train['class'])

X = train[feat_all].values.astype(np.float32); y = train['target'].values
rng = np.random.RandomState(42)

from tabicl import TabICLClassifier

# 逐步放大: 50K → 100K → 150K
for n_total in [50000, 100000, 150000]:
    n_per = n_total // 3
    progress(f"\n{'='*50}")
    progress(f"Testing {n_total:,} samples (MPS GPU)")

    idx = []
    for c in range(3):
        c_idx = np.where(y==c)[0]
        idx.extend(rng.choice(c_idx, min(n_per, len(c_idx)), replace=False))
    X_s = X[idx]; y_s = y[idx]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
    )
    progress(f"  Train: {X_tr.shape[0]:,}  Val: {X_val.shape[0]:,}")

    try:
        model = TabICLClassifier(device='mps', n_estimators=3, verbose=False)
        progress(f"  Fitting...")
        model.fit(X_tr, y_tr)
        progress(f"  Predicting...")
        pred = model.predict(X_val)
        ba = balanced_accuracy_score(y_val, pred)
        rec = recall_score(y_val, pred, average=None)
        progress(f"  ✅ BA={ba:.4f}  GAL={rec[0]:.4f}  STAR={rec[1]:.4f}  QSO={rec[2]:.4f}")
        if ba > 0.96:
            progress(f"  🎉 Above 0.96! Good enough for ensemble!")
    except Exception as e:
        progress(f"  ❌ {type(e).__name__}: {str(e)[:150]}")
        break

progress("\nDone!")
