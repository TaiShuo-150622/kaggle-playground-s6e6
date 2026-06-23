"""Add Deotte's features: ratios, floor bins, quantile bins, target encoding"""
import pandas as pd, numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

train = pd.read_csv("data/train_fe.csv")
test  = pd.read_csv("data/test_fe.csv")

# ---- Step 1: Deotte's features ----
for df in [train, test]:
    # redshift ratios (new!)
    df['_g_div_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).clip(-10, 10)
    df['_i_div_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).clip(-10, 10)

    # mag stats
    mags = df[['u','g','r','i','z']]
    df['_mag_mean'] = mags.mean(axis=1)
    df['_mag_range'] = mags.max(axis=1) - mags.min(axis=1)

    # log1p redshift
    df['_log1p_redshift'] = np.log1p(df['redshift'].clip(lower=0))

    # floor bins (every numeric feature → category)
    base_num = ['alpha','delta','u','g','r','i','z','redshift']
    for col in base_num:
        if col in df.columns:
            df[f'{col}_cat_'] = np.floor(df[col])

    # delta quantile bins
    for n_bins in [100, 500]:
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
        df[f'delta_{n_bins}_bin_'] = kb.fit_transform(df[['delta']]).ravel().astype(int)

print(f"Train: {train.shape}  Test: {test.shape}")

# ---- Step 2: Fold-safe target encoding ----
print("Target encoding (5-fold)...")
le = LabelEncoder()
le.fit(train['class'])
y_enc = le.transform(train['class'])
n_classes = 3

# Build interaction features
train['_combo_alpha_delta'] = train['alpha_cat_'].astype(str) + '|' + train['delta_cat_'].astype(str)
train['_combo_u_z'] = train['u_cat_'].astype(str) + '|' + train['z_cat_'].astype(str)

test['_combo_alpha_delta'] = test['alpha_cat_'].astype(str) + '|' + test['delta_cat_'].astype(str)
test['_combo_u_z'] = test['u_cat_'].astype(str) + '|' + test['z_cat_'].astype(str)

from sklearn.preprocessing import TargetEncoder

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for combo_name in ['_combo_alpha_delta', '_combo_u_z']:
    # Fold-safe: for each fold, encode using TRAIN ONLY from other folds
    for cls_idx, cls_name in enumerate(le.classes_):
        col_name = f'{combo_name}_TE_{cls_name}'
        y_binary = (y_enc == cls_idx).astype(int)

        train[col_name] = 0.0
        te = TargetEncoder(cv=5, smooth='auto', random_state=42)
        train[col_name] = te.fit_transform(train[[combo_name]], y_binary).ravel()

        # For test: fit on all train data
        te_full = TargetEncoder(smooth='auto', random_state=42)
        te_full.fit(train[[combo_name]], y_binary)
        test[col_name] = te_full.transform(test[[combo_name]]).ravel()

# Drop intermediate columns
train.drop(['_combo_alpha_delta','_combo_u_z'], axis=1, inplace=True)
test.drop(['_combo_alpha_delta','_combo_u_z'], axis=1, inplace=True)

print(f"With new features: Train {train.shape}  Test {test.shape}")

# ---- Step 3: Quick test - LightGBM 5-fold old vs new ----
print("\n=== Quick test: LGB with new features ===")
le = LabelEncoder()
train['target'] = le.fit_transform(train['class'])
y = train['target'].values

# Old features
feat_old = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv',
            'alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']

# New features only (the ones we added)
feat_new_extra = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]

# Combined
feat_combined = feat_old + feat_new_extra

for col in ['spectral_type','galaxy_population']:
    if col in train.columns:
        train[col+'_enc'] = LabelEncoder().fit_transform(train[col])
    if col in test.columns:
        test[col+'_enc'] = LabelEncoder().fit_transform(test[col])

X = train[feat_combined].values.astype(np.float32)
X_test = test[feat_combined].values.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=2000,
                            learning_rate=0.05, num_leaves=128, max_depth=8,
                            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.1, random_state=fold, n_jobs=-1, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = m.predict(X[val])
    scores.append(balanced_accuracy_score(y[val], pred))

ba_new = np.mean(scores)
print(f"\nOLD best LGB: 0.95670")
print(f"NEW best LGB: {ba_new:.5f}")
print(f"Delta: {'+' if ba_new > 0.95670 else ''}{ba_new - 0.95670:.5f}")
