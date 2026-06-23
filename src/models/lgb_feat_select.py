"""LGB feature selection: drop low-importance features"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb

train = pd.read_csv("train_fe.csv")
for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = pd.factorize(train[col])[0]

feat_base = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = [c for c in feat_base+new_cols if c in train.columns]
X = train[feat_all].values.astype(np.float32)
y = pd.factorize(train['class'])[0]
print(f"Features: {len(feat_all)}")

m = lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=2000,learning_rate=0.05,
                        num_leaves=256,max_depth=10,min_child_samples=50,subsample=0.8,
                        colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,random_state=42,n_jobs=-1,verbose=-1)

skf = StratifiedKFold(5,shuffle=True,random_state=42)
scores_full = []
for tr,val in skf.split(X,y):
    m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],callbacks=[lgb.early_stopping(50,verbose=False)])
    scores_full.append(balanced_accuracy_score(y[val],m.predict(X[val])))
ba_full = np.mean(scores_full)
print(f"Full ({len(feat_all)} feats): {ba_full:.5f}")

# Permutation importance
perm = permutation_importance(m, X, y, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({'feat':feat_all,'imp':perm.importances_mean}).sort_values('imp')
bottom = list(imp.head(10)['feat'])
print(f"Bottom 10: {bottom}")

# Test dropping
for drop_n in [5,10,15,20]:
    keep_feats = imp.iloc[drop_n:]['feat'].tolist()
    keep_idx = [feat_all.index(f) for f in keep_feats]
    X_k = X[:, keep_idx]
    scores = []
    for tr,val in skf.split(X,y):
        m.fit(X_k[tr],y[tr],eval_set=[(X_k[val],y[val])],callbacks=[lgb.early_stopping(50,verbose=False)])
        scores.append(balanced_accuracy_score(y[val],m.predict(X_k[val])))
    ba_k = np.mean(scores)
    up = "UP" if ba_k > ba_full else "down"
    print(f"Drop {drop_n} ({len(keep_feats)} left): {ba_k:.5f} ({up} {abs(ba_k-ba_full):.5f})")
