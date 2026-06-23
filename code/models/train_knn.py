"""KNN for diversity in ensemble"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv("train_fe.csv"); test = pd.read_csv("test_fe.csv")
y = pd.factorize(train['class'])[0]
for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = pd.factorize(train[col])[0]; test[col+'_enc'] = pd.factorize(test[col])[0]

feat_base = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = [c for c in feat_base+new_cols if c in train.columns]
X = train[feat_all].values.astype(np.float32); Xt = test[feat_all].values.astype(np.float32)
print(f"Features: {len(feat_all)}")

# Full KNN
oof = np.zeros((len(y), 3)); tp = np.zeros((len(Xt), 3))
for fold,(tr,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    m = KNeighborsClassifier(n_neighbors=51, n_jobs=-1)
    m.fit(X[tr], y[tr])
    oof[val] = m.predict_proba(X[val]); tp += m.predict_proba(Xt)/5
    print(f"  Fold {fold+1}: {balanced_accuracy_score(y[val], np.argmax(oof[val], axis=1)):.5f}")
ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
print(f"KNN OOF: {ba:.5f}")
np.save("oof_KNN.npy", oof); np.save("test_KNN.npy", tp)
print("Saved!")
