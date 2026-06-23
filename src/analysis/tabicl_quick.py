"""TabICL: 降采样快速测试"""
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

# 降采样: 每类 15000，总共 45000
rng = np.random.RandomState(42)
idx = []
for c in range(3):
    c_idx = np.where(y==c)[0]
    idx.extend(rng.choice(c_idx, min(15000,len(c_idx)), replace=False))
X_s = X[idx]; y_s = y[idx]
progress(f"Sampled: {X_s.shape}")

X_tr, X_val, y_tr, y_val = train_test_split(X_s, y_s, test_size=0.2, stratify=y_s, random_state=42)
progress(f"Train: {X_tr.shape}, Val: {X_val.shape}")

from tabicl import TabICLClassifier
model = TabICLClassifier(device='cpu', n_estimators=3, verbose=False)
progress("Fitting...")
model.fit(X_tr, y_tr)
progress("Predicting...")
pred = model.predict(X_val)
ba = balanced_accuracy_score(y_val, pred)
rec = recall_score(y_val, pred, average=None)
progress(f"TabICL (45K sample): BA={ba:.4f}  GAL={rec[0]:.4f}  STAR={rec[1]:.4f}  QSO={rec[2]:.4f}")
