"""
服务器全量运行脚本 (6×2080 Ti, 66GB VRAM)
=========================================
scp /Users/taishuo/kaggle/playground_s6e6/  server:/path/to/kaggle/
ssh server
cd /path/to/kaggle/playground_s6e6
pip install lightgbm xgboost catboost tabicl pytabkit tabm  # 先装依赖
python3 server_run.py
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from scipy.optimize import minimize
import lightgbm as lgb
import torch, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# 数据
# ============================================================
train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

feat_num = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv']
feat_pos = ['alpha_sin','alpha_cos','delta']
feat_cat = ['spectral_type','galaxy_population']

for col in feat_cat:
    train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    test[col+'_enc']  = LabelEncoder().fit_transform(test[col])

# 测试两种特征集: 不加 u_z/g_z vs 加
progress("Testing WITHOUT u_z/g_z...")
feat_old = feat_num + feat_pos + [c+'_enc' for c in feat_cat]

progress("Testing WITH u_z/g_z...")
feat_new = feat_old + ['u_z', 'g_z']
train_new = train.copy()
test_new = test.copy()
for col in ['u_z','g_z']:
    train_new[col] = train[col] if col in train.columns else train['u'] - train['z'] if col=='u_z' else train['g'] - train['z']
    test_new[col] = test[col] if col in test.columns else test['u'] - test['z'] if col=='u_z' else test['g'] - test['z']

le = LabelEncoder().fit(train['class'])
train['target'] = le.transform(train['class'])
train_new['target'] = le.transform(train_new['class'])

# ============================================================
# GPU 检查
# ============================================================
n_gpus = torch.cuda.device_count()
progress(f"GPUs available: {n_gpus}")
for i in range(n_gpus):
    prop = torch.cuda.get_device_properties(i)
    progress(f"  GPU {i}: {prop.name} ({prop.total_memory/1e9:.1f} GB)")

DEVICE = 'cuda:0' if n_gpus > 0 else 'cpu'
progress(f"Using device: {DEVICE}")
if n_gpus > 0:
    progress("WARNING: MPS on Mac is buggy with RealMLP. Use this script on NVIDIA GPU server only.")

# ============================================================
# 1. RealMLP (最强单模)
# ============================================================
progress("="*60)
progress("1. RealMLP (pytabkit)")
progress("="*60)

from pytabkit import RealMLP_TD_Classifier

X = train[feat_old].values.astype(np.float32)
y = train['target'].values

for name, X_data, y_data in [("old features", X, y)]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_data, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    model = RealMLP_TD_Classifier(device=DEVICE, random_state=42, n_epochs=256, batch_size=256)
    progress(f"  Training {name} on {len(X_tr):,} rows...")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    ba = balanced_accuracy_score(y_val, pred)
    rec = recall_score(y_val, pred, average=None)
    progress(f"  BA={ba:.4f}  GAL={rec[0]:.4f}  STAR={rec[1]:.4f}  QSO={rec[2]:.4f}  ({time.time()-t0:.0f}s)")

# ============================================================
# 2. TabICL (零训练 Transformer，全量)
# ============================================================
progress("="*60)
progress("2. TabICL (full 577K rows)")
progress("="*60)

from tabicl import TabICLClassifier

try:
    X2_tr, X2_val, y2_tr, y2_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model2 = TabICLClassifier(device='cuda', n_estimators=8, verbose=False)
    progress(f"  Fitting on {len(X2_tr):,} rows...")
    t0 = time.time()
    model2.fit(X2_tr, y2_tr)
    progress(f"  Predicting {len(X2_val):,} rows...")
    pred2 = model2.predict(X2_val)
    ba2 = balanced_accuracy_score(y2_val, pred2)
    rec2 = recall_score(y2_val, pred2, average=None)
    progress(f"  TabICL: BA={ba2:.4f}  GAL={rec2[0]:.4f}  STAR={rec2[1]:.4f}  QSO={rec2[2]:.4f}  ({time.time()-t0:.0f}s)")
except Exception as e:
    progress(f"  TabICL failed: {e}")

# ============================================================
# 3. Ensemble Voting
# ============================================================
progress("="*60)
progress("3. Final Ensemble")
progress("="*60)
progress("Combine: LGB + XGB + CB + RealMLP (with threshold tuning)")
progress("Submit the best combination to Kaggle")
progress("Done! Check the scores above and pick the best.")
