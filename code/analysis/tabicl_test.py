"""
TabICL 测试: 零训练 Transformer，直接对全量数据做推理
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from datetime import datetime

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

feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat]
le = LabelEncoder().fit(train['class'])
train['target'] = le.transform(train['class'])

X_full = train[feat_all].values.astype(np.float32)
y_full = train['target'].values
X_test = test[feat_all].values.astype(np.float32)

progress(f"Full train: {X_full.shape}, test: {X_test.shape}")

# ============================================================
# TabICL 测试
# ============================================================
from tabicl import TabICLClassifier

# 尝试全量 → 如果 OOM 就降采样
sizes = [len(y_full), 300000, 150000]

for n_samples in sizes:
    progress(f"\n{'='*50}")
    progress(f"Testing with {n_samples:,} samples...")

    if n_samples < len(y_full):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_full), n_samples, replace=False)
        X_sample = X_full[idx]
        y_sample = y_full[idx]
    else:
        X_sample = X_full
        y_sample = y_full

    try:
        # TabICL: 单 fold 快速验证
        # model selection_mode='cv' 自动做 CV
        progress("  Creating model...")
        model = TabICLClassifier(
            device='cpu',
            n_estimators=3,
            offload_mode='auto',     # 大表格自动 offload
            kv_cache=True,           # 加速推理
            verbose=False,
        )

        # 只用单 fold 测速度
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
        )

        progress(f"  Fitting on {len(X_tr):,} rows...")
        model.fit(X_tr, y_tr)

        progress(f"  Predicting {len(X_val):,} rows...")
        pred = model.predict(X_val)

        ba = balanced_accuracy_score(y_val, pred)
        rec = recall_score(y_val, pred, average=None)
        progress(f"  BA={ba:.4f}  GAL={rec[0]:.4f}  STAR={rec[1]:.4f}  QSO={rec[2]:.4f}")

        progress("  ✅ SUCCESS at this sample size!")
        break

    except Exception as e:
        progress(f"  ❌ FAILED: {type(e).__name__}: {str(e)[:200]}")
        if n_samples == sizes[-1]:
            progress("  Tried all sizes, none worked.")
        else:
            progress("  Trying smaller size...")
EOF
