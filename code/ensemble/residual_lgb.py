"""
Residual LGB: train on Deotte's prediction residuals
Each LGB learns to correct where Deotte is wrong
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")
y = pd.factorize(train['class'])[0]

for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = pd.factorize(train[col])[0]
    test[col+'_enc']  = pd.factorize(test[col])[0]

feat_base = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv',
             'alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = [c for c in feat_base+new_cols if c in train.columns]
X = train[feat_all].values.astype(np.float32)
X_test = test[feat_all].values.astype(np.float32)
print(f"Features: {len(feat_all)}, Samples: {len(y)}")

# Load Deotte's OOF predictions
import glob
oof_files = sorted(glob.glob('oof_Deotte_*.npy'))
test_files = sorted(glob.glob('test_Deotte_*.npy'))
deotte_oof = sum(np.load(f) for f in oof_files) / len(oof_files)
deotte_test = sum(np.load(f) for f in test_files) / len(test_files)
print(f"Deotte files: {len(oof_files)} OOF + {len(test_files)} test")

# Train residual LGB: target = y - Deotte_prob
# Convert y to one-hot
y_onehot = np.zeros((len(y), 3))
y_onehot[np.arange(len(y)), y] = 1.0
y_residual = y_onehot - deotte_oof  # (N, 3) — where Deotte is wrong

# Train per-class LGB on residual
oof_correction = np.zeros((len(y), 3))
test_correction = np.zeros((len(X_test), 3))
residual_models = []

for cls in range(3):
    target = y_residual[:, cls]  # Deotte's error for this class
    oof_c = np.zeros(len(y)); test_c = np.zeros(len(X_test))

    for fold,(tr,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=128, max_depth=6,
                               min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
                               reg_alpha=0.3, reg_lambda=0.3, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X[tr], target[tr], eval_set=[(X[val], target[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_c[val] = m.predict(X[val])
        test_c += m.predict(X_test) / 5
        residual_models.append(m)

    oof_correction[:, cls] = oof_c
    test_correction[:, cls] = test_c
    print(f"Class {cls} residual LGB: mean_abs_error={np.abs(oof_c).mean():.4f}")

# Corrected predictions
oof_corrected = deotte_oof + oof_correction
# Clip and normalize
oof_corrected = np.clip(oof_corrected, 0, 1)
oof_corrected = oof_corrected / oof_corrected.sum(axis=1, keepdims=True)
ba = balanced_accuracy_score(y, np.argmax(oof_corrected, axis=1))
print(f"\nResidual LGB corrected OOF: {ba:.5f}")
print(f"Pure Deotte OOF: {balanced_accuracy_score(y, np.argmax(deotte_oof, axis=1)):.5f}")
print(f"Improvement: {ba - balanced_accuracy_score(y, np.argmax(deotte_oof, axis=1)):+.5f}")

# Final ensemble: Deotte + corrected
test_corrected = deotte_test + test_correction
test_corrected = np.clip(test_corrected, 0, 1)
test_corrected = test_corrected / test_corrected.sum(axis=1, keepdims=True)

np.save('oof_residual_lgb.npy', oof_corrected)
np.save('test_residual_lgb.npy', test_corrected)
print("Saved!")
