"""
服务器全量运行 — 6×2080 Ti 多 GPU 并行
=========================================
用法:
  git clone https://github.com/TaiShuo-150622/kaggle-playground-s6e6
  cd kaggle-playground-s6e6
  # 拷贝 data/train_fe.csv + test_fe.csv 到 data/
  pip install lightgbm xgboost catboost pytabkit tabicl tabm
  python server_run.py
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from scipy.optimize import minimize
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import torch, time, warnings, os, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# GPU 检测
# ============================================================
n_gpus = torch.cuda.device_count()
progress(f"GPUs: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    progress(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

if n_gpus < 1:
    progress("ERROR: No CUDA GPU found!")
    exit(1)

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

# 带 u_z/g_z 的新特征
feat_all = feat_num + feat_pos + [c+'_enc' for c in feat_cat] + ['u_z','g_z']
le = LabelEncoder().fit(train['class'])
train['target'] = le.transform(train['class'])

X = train[feat_all].values.astype(np.float32)
y = train['target'].values
X_test = test[feat_all].values.astype(np.float32)

progress(f"Features: {len(feat_all)}, Train: {X.shape}, Test: {X_test.shape}")

# ============================================================
# 多 GPU 并行: 三个模型同时训练
# ============================================================
results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_realmlp(gpu_id):
    """GPU i: RealMLP"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from pytabkit import RealMLP_TD_Classifier

    progress(f"[GPU {gpu_id}] RealMLP starting...")
    oof = np.zeros(len(X))
    test_prob = np.zeros((len(X_test), 3))

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = RealMLP_TD_Classifier(device='cuda', random_state=fold,
                                   n_epochs=256, batch_size=256)
        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val])
        test_prob += m.predict_proba(X_test) / 5
        progress(f"[GPU {gpu_id}] RealMLP fold {fold+1}/5 done")

    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None)
    progress(f"[GPU {gpu_id}] RealMLP: BA={ba:.4f} GAL={rec[0]:.4f} STAR={rec[1]:.4f} QSO={rec[2]:.4f}")
    return {'oof': oof, 'test': test_prob, 'ba': ba}

def train_tabm(gpu_id):
    """GPU i: TabM"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from tabm import TabMClassifier

    progress(f"[GPU {gpu_id}] TabM starting...")
    oof = np.zeros((len(X), 3))
    test_prob = np.zeros((len(X_test), 3))

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = TabMClassifier(device='cuda', n_estimators=8, random_state=fold)
        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val])
        test_prob += m.predict_proba(X_test) / 5
        progress(f"[GPU {gpu_id}] TabM fold {fold+1}/5 done")

    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None)
    progress(f"[GPU {gpu_id}] TabM: BA={ba:.4f} GAL={rec[0]:.4f} STAR={rec[1]:.4f} QSO={rec[2]:.4f}")
    return {'oof': oof, 'test': test_prob, 'ba': ba}

def train_tree_ensemble(gpu_id):
    """GPU i: 三棵树 (串行，不抢GPU)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    progress(f"[GPU {gpu_id}] Tree ensemble (LGB+XGB+CB) starting...")

    oof_lgb = np.zeros((len(X), 3)); test_lgb = np.zeros((len(X_test), 3))
    oof_xgb = np.zeros((len(X), 3)); test_xgb = np.zeros((len(X_test), 3))
    oof_cb  = np.zeros((len(X), 3)); test_cb  = np.zeros((len(X_test), 3))

    # LightGBM (CPU only, 快)
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=2000,
                               learning_rate=0.05, num_leaves=128, max_depth=8,
                               min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.1, reg_lambda=0.1,
                               random_state=fold, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val] = m.predict_proba(X[val])
        test_lgb += m.predict_proba(X_test) / 5
    progress(f"[GPU {gpu_id}] LGB done")

    # XGBoost
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=2000,
                              learning_rate=0.05, max_depth=8, min_child_weight=50,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=0.1,
                              random_state=fold, n_jobs=-1, verbosity=0)
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof_xgb[val] = m.predict_proba(X[val])
        test_xgb += m.predict_proba(X_test) / 5
    progress(f"[GPU {gpu_id}] XGB done")

    # CatBoost
    feat_cb = feat_all
    X_cb = train[feat_cb].values.astype(object)
    X_cb_test = test[feat_cb].values.astype(object)
    cat_idx = [i for i, col in enumerate(feat_cb) if col in feat_cat]
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
    progress(f"[GPU {gpu_id}] CB done")

    # Voting
    oof_vote = (oof_lgb + oof_xgb + oof_cb) / 3
    test_vote = (test_lgb + test_xgb + test_cb) / 3
    ba = balanced_accuracy_score(y, np.argmax(oof_vote, axis=1))
    progress(f"[GPU {gpu_id}] Tree ensemble: BA={ba:.4f}")
    return {'oof': oof_vote, 'test': test_vote, 'ba': ba}

# ============================================================
# 主流程: 并行训练
# ============================================================
progress("=" * 60)
progress("PARALLEL TRAINING — 3 models on 3 GPUs")
progress("=" * 60)

t0 = time.time()
model_results = {}

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {
        'realmlp': pool.submit(train_realmlp, 0),
        'tabm':    pool.submit(train_tabm, 1),
        'trees':   pool.submit(train_tree_ensemble, 2),
    }
    for name, f in futures.items():
        model_results[name] = f.result()

progress(f"All models done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# Final Ensemble
# ============================================================
progress("=" * 60)
progress("FINAL ENSEMBLE")
progress("=" * 60)

# 收集 OOF
all_oof = {}
for name, r in model_results.items():
    all_oof[name] = r['oof']
    progress(f"  {name}: BA={r['ba']:.4f}")

# Threshold tuning on combined OOF
def tuned_predict(probs, t):
    return np.argmax(probs / np.array(t), axis=1)
def neg_ba(t, probs, yt):
    return -balanced_accuracy_score(yt, tuned_predict(probs, t))

# 尝试不同组合
combos = [
    ('trees', model_results['trees']['oof']),
    ('trees+realmlp', (model_results['trees']['oof'] + model_results['realmlp']['oof']) / 2),
    ('trees+tabm', (model_results['trees']['oof'] + model_results['tabm']['oof']) / 2),
    ('all3', (model_results['trees']['oof'] + model_results['realmlp']['oof'] + model_results['tabm']['oof']) / 3),
]

best_ba, best_combo, best_t = 0, None, None
for name, oof in combos:
    ba_base = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    result = minimize(neg_ba, x0=[1.,1.,1.], args=(oof, y),
                      method='Nelder-Mead',
                      bounds=[(0.2, 3.), (0.2, 3.), (0.2, 3.)],
                      options=dict(xatol=0.001, maxiter=500))
    ba_tuned = -result.fun
    progress(f"  {name:>16}: base={ba_base:.4f} → tuned={ba_tuned:.4f} (+{ba_tuned-ba_base:.4f})")
    if ba_tuned > best_ba:
        best_ba, best_combo, best_t = ba_tuned, name, result.x

# 生成提交
progress(f"\n  🏆 Best: {best_combo} BA={best_ba:.4f} thresholds={best_t}")

test_probs = None
if best_combo == 'trees':
    test_probs = model_results['trees']['test']
elif best_combo == 'trees+realmlp':
    test_probs = (model_results['trees']['test'] + model_results['realmlp']['test']) / 2
elif best_combo == 'trees+tabm':
    test_probs = (model_results['trees']['test'] + model_results['tabm']['test']) / 2
elif best_combo == 'all3':
    test_probs = (model_results['trees']['test'] + model_results['realmlp']['test'] + model_results['tabm']['test']) / 3

pred = le.inverse_transform(tuned_predict(test_probs, best_t))
sub = pd.DataFrame({'id': test['id'], 'class': pred})
sub.to_csv('submission_final.csv', index=False)
progress(f"  Submission: submission_final.csv ({len(sub)} rows)")
progress(f"  Distribution: {sub['class'].value_counts().to_dict()}")
progress("DONE!")
