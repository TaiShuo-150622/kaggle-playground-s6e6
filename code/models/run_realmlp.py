"""Playground S6E6: RealMLP 快速测试 (MPS GPU)"""
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
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

progress(f"Train: {X.shape}, Test: {X_test.shape}")

from pytabkit import RealMLP_TD_Classifier

configs = [
    ("[512,512] 64ep", [512,512], 64),
    ("[512,512,512] 64ep", [512,512,512], 64),
    ("[256,512,256] 32ep", [256,512,256], 32),
]

for name, hidden, ep in configs:
    progress(f"\n{'='*50}")
    progress(f"{name}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    t0 = time.time()
    model = RealMLP_TD_Classifier(device='mps', random_state=42, n_epochs=ep, batch_size=8192, hidden_sizes=hidden)
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    pred = model.predict(X_val)
    ba = balanced_accuracy_score(y_val, pred)
    rec = recall_score(y_val, pred, average=None)
    progress(f"  BA={ba:.5f}  GAL={rec[0]:.4f}  STAR={rec[1]:.4f}  QSO={rec[2]:.4f}  {elapsed:.0f}s")
