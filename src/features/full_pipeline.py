"""
Full pipeline: raw data → features → 18-model ensemble → submission
"""
import pandas as pd, numpy as np, json, time, gc, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from datetime import datetime
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# 1. Load raw data
# ============================================================
pr("Loading raw data...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
pr(f"Train: {train.shape}  Test: {test.shape}")

# ============================================================
# 2. Feature engineering (basic + Deotte)
# ============================================================
pr("Feature engineering...")

# 2a. Basic color indices
train['u_g']=train['u']-train['g']; test['u_g']=test['u']-test['g']
train['g_r']=train['g']-train['r']; test['g_r']=test['g']-test['r']
train['r_i']=train['r']-train['i']; test['r_i']=test['r']-test['i']
train['i_z']=train['i']-train['z']; test['i_z']=test['i']-test['z']
train['u_r']=train['u']-train['r']; test['u_r']=test['u']-test['r']
train['g_i']=train['g']-train['i']; test['g_i']=test['g']-test['i']
train['r_z']=train['r']-train['z']; test['r_z']=test['r']-test['z']
train['color_curv']=train['u_g']-train['g_r']; test['color_curv']=test['u_g']-test['g_r']
train['u_z']=train['u']-train['z']; test['u_z']=test['u']-test['z']
train['g_z']=train['g']-train['z']; test['g_z']=test['g']-test['z']

# 2b. Alpha cyclic encoding
alpha_rad = np.deg2rad(train['alpha'])
train['alpha_sin']=np.sin(alpha_rad); train['alpha_cos']=np.cos(alpha_rad)
alpha_rad = np.deg2rad(test['alpha'])
test['alpha_sin']=np.sin(alpha_rad); test['alpha_cos']=np.cos(alpha_rad)

# 2c. Encode categoricals
for col in ['spectral_type','galaxy_population']:
    train[col+'_enc']=LabelEncoder().fit_transform(train[col])
    test[col+'_enc']=LabelEncoder().fit_transform(test[col])

# 2d. Deotte features
for df in [train, test]:
    df['_g_div_redshift']=(df['g']/(df['redshift']+1e-6)).clip(-10,10)
    df['_i_div_redshift']=(df['i']/(df['redshift']+1e-6)).clip(-10,10)
    mags=df[['u','g','r','i','z']]
    df['_mag_mean']=mags.mean(axis=1)
    df['_mag_range']=mags.max(axis=1)-mags.min(axis=1)
    df['_log1p_redshift']=np.log1p(df['redshift'].clip(lower=0))
    for col in ['alpha','delta','u','g','r','i','z','redshift']:
        df[f'{col}_cat_']=np.floor(df[col]).astype('int32')
    for n in [100,500]:
        kb=KBinsDiscretizer(n_bins=n,encode='ordinal',strategy='quantile',subsample=None)
        df[f'delta_{n}_bin_']=kb.fit_transform(df[['delta']]).ravel().astype('int32')

# 2e. Target encoding
le=LabelEncoder(); le.fit(train['class']); y_enc=le.transform(train['class'])
for df in [train,test]:
    df['_combo_ad']=df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_combo_uz']=df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)

y_cls=le.transform(train['class'])
for combo in ['_combo_ad','_combo_uz']:
    for ci,cn in enumerate(le.classes_):
        cn=f'{combo}_TE_{ci}'
        yb=(y_cls==ci).astype(int)
        te=TargetEncoder(cv=5,smooth='auto',random_state=42)
        train[cn]=te.fit_transform(train[[combo]],yb).ravel()
        te2=TargetEncoder(smooth='auto',random_state=42)
        te2.fit(train[[combo]],yb)
        test[cn]=te2.transform(test[[combo]]).ravel()

train.drop(['_combo_ad','_combo_uz'],axis=1,inplace=True)
test.drop(['_combo_ad','_combo_uz'],axis=1,inplace=True)

pr(f"Features: {train.shape[1]} cols")

# ============================================================
# 3. Train ensemble
# ============================================================
feat_base=['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols=[c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all=feat_base+new_cols; feat_all=[c for c in feat_all if c in train.columns]
train['target']=le.transform(train['class'])

X=train[feat_all].values.astype(np.float32)
y=train['target'].values
X_test=test[[c for c in feat_all if c in test.columns]].values.astype(np.float32)
pr(f"Train: {X.shape}  Test: {X_test.shape}  Features: {len(feat_all)}")

import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
folds=list(skf.split(X,y))
all_preds={}

LGB_MODELS=[
    ("LGB_64_d6",dict(num_leaves=64,max_depth=6,min_child_samples=50,subsample=0.8,colsample_bytree=0.8)),
    ("LGB_128_d8",dict(num_leaves=128,max_depth=8,min_child_samples=50,subsample=0.8,colsample_bytree=0.8)),
    ("LGB_256_d10",dict(num_leaves=256,max_depth=10,min_child_samples=50,subsample=0.8,colsample_bytree=0.8)),
    ("LGB_128_cs9",dict(num_leaves=128,max_depth=8,min_child_samples=50,subsample=0.8,colsample_bytree=0.9)),
    ("LGB_64_ss7",dict(num_leaves=64,max_depth=8,min_child_samples=30,subsample=0.7,colsample_bytree=0.8)),
]
XGB_MODELS=[
    ("XGB_d6",dict(max_depth=6,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1)),
    ("XGB_d8",dict(max_depth=8,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1)),
    ("XGB_d10",dict(max_depth=10,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1)),
    ("XGB_d8_reg",dict(max_depth=8,min_child_weight=100,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.5,reg_lambda=1.0)),
    ("XGB_d6_gamma",dict(max_depth=6,min_child_weight=30,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,gamma=0.1)),
]

for name,kw in LGB_MODELS+XGB_MODELS:
    pr(f"\n{name}")
    oof=np.zeros((len(X),3)); tp=np.zeros((len(X_test),3)); sc=[]
    is_lgb='LGB' in name
    for fold,(tr,val) in enumerate(folds):
        if is_lgb:
            m=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=2000,learning_rate=0.05,reg_alpha=0.1,reg_lambda=0.1,random_state=fold*42,n_jobs=-1,verbose=-1,**kw)
            m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],callbacks=[lgb.early_stopping(50,verbose=False)])
        else:
            m=xgb.XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=2000,learning_rate=0.05,random_state=fold*42,n_jobs=-1,verbosity=0,**kw)
            m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],verbose=False)
        oof[val]=m.predict_proba(X[val]); tp+=m.predict_proba(X_test)/5
        sc.append(balanced_accuracy_score(y[val],np.argmax(oof[val],axis=1)))
    ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
    pr(f"  BA={ba:.5f} folds={[f'{s:.5f}' for s in sc]}")
    all_preds[name]={'oof':oof,'test':tp,'ba':ba}; np.save(f'oof_{name}.npy',oof); np.save(f'test_{name}.npy',tp)

# CatBoost (skip for speed, focus on trees + MLP)
# RealMLP
MLP_MODELS=[
    ("MLP_512_512_512",[512,512,512],64),
    ("MLP_512_512",[512,512],64),
]
feat_cb=[c for c in feat_all if c in train.columns]
X_cb=train[feat_cb].values.astype(object)
X_cb_test=test[[c for c in feat_cb if c in test.columns]].values.astype(object)
cat_idx=[i for i,col in enumerate(feat_cb) if col in ['spectral_type','galaxy_population']]

CB_MODELS=[
    ("CB_d8",dict(depth=8,min_data_in_leaf=50,l2_leaf_reg=3,subsample=0.8)),
    ("CB_d10",dict(depth=10,min_data_in_leaf=50,l2_leaf_reg=3,subsample=0.8)),
]
for name,kw in CB_MODELS:
    pr(f"\n{name}")
    oof=np.zeros((len(X),3)); tp=np.zeros((len(X_cb_test),3)); sc=[]
    for fold,(tr,val) in enumerate(folds):
        m=CatBoostClassifier(iterations=2000,learning_rate=0.05,bootstrap_type='Bernoulli',random_seed=fold*42,thread_count=-1,verbose=0,allow_writing_files=False,**kw)
        m.fit(X_cb[tr],y[tr],eval_set=[(X_cb[val],y[val])],cat_features=cat_idx,early_stopping_rounds=50,verbose=0)
        oof[val]=m.predict_proba(X_cb[val]); tp+=m.predict_proba(X_cb_test)/5
        sc.append(balanced_accuracy_score(y[val],np.argmax(oof[val],axis=1)))
    ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
    pr(f"  BA={ba:.5f}")
    all_preds[name]={'oof':oof,'test':tp,'ba':ba}; np.save(f'oof_{name}.npy',oof); np.save(f'test_{name}.npy',tp)

for name,hidden,ep in MLP_MODELS:
    pr(f"\n{name}")
    oof=np.zeros((len(X),3)); tp=np.zeros((len(X_test),3)); sc=[]
    for fold,(tr,val) in enumerate(folds):
        m=RealMLP_TD_Classifier(device='cuda',random_state=fold*42,n_epochs=ep,batch_size=8192,hidden_sizes=hidden)
        m.fit(X[tr],y[tr])
        oof[val]=m.predict_proba(X[val]); tp+=m.predict_proba(X_test)/5
        sc.append(balanced_accuracy_score(y[val],np.argmax(oof[val],axis=1)))
    ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
    pr(f"  BA={ba:.5f}")
    all_preds[name]={'oof':oof,'test':tp,'ba':ba}; np.save(f'oof_{name}.npy',oof); np.save(f'test_{name}.npy',tp)

# ============================================================
# 4. Ensemble
# ============================================================
pr("\n=== ENSEMBLE ===")
from scipy.optimize import minimize
def tp(probs,t): return np.argmax(probs/np.array(t),axis=1)
def nba(t,probs,yt): return -balanced_accuracy_score(yt,tp(probs,t))

oof_trees=np.mean([v['oof'] for k,v in all_preds.items() if 'LGB' in k or 'XGB' in k or 'CB' in k],axis=0)
oof_mlp=np.mean([v['oof'] for k,v in all_preds.items() if 'MLP' in k],axis=0)

for name,oof in [('trees_only',oof_trees),('trees+mlp',(oof_trees+oof_mlp)/2)]:
    bb=balanced_accuracy_score(y,np.argmax(oof,axis=1))
    r=minimize(nba,[1,1,1],args=(oof,y),method='Nelder-Mead',bounds=[(0.2,3),(0.2,3),(0.2,3)],options=dict(xatol=0.001,maxiter=500))
    pr(f"  {name}: base={bb:.5f} → tuned={-r.fun:.5f} (+{-r.fun-bb:.5f})")

# Best submission
test_trees=np.mean([v['test'] for k,v in all_preds.items() if 'LGB' in k or 'XGB' in k or 'CB' in k],axis=0)
test_mlp=np.mean([v['test'] for k,v in all_preds.items() if 'MLP' in k],axis=0)

# trees_only
res=minimize(nba,[1,1,1],args=(oof_trees,y),method='Nelder-Mead',bounds=[(0.2,3),(0.2,3),(0.2,3)],options=dict(xatol=0.001,maxiter=500))
pred_trees=le.inverse_transform(tp(test_trees,res.x))
pd.DataFrame({'id':test['id'],'class':pred_trees}).to_csv('sub_trees.csv',index=False)
pr(f"sub_trees: {pd.Series(pred_trees).value_counts().to_dict()}")

# trees+mlp
test_both=(test_trees+test_mlp)/2; oof_both=(oof_trees+oof_mlp)/2
r2=minimize(nba,[1,1,1],args=(oof_both,y),method='Nelder-Mead',bounds=[(0.2,3),(0.2,3),(0.2,3)],options=dict(xatol=0.001,maxiter=500))
pred_both=le.inverse_transform(tp(test_both,r2.x))
pd.DataFrame({'id':test['id'],'class':pred_both}).to_csv('sub_both.csv',index=False)
pr(f"sub_both (trees+mlp): tuned={-r2.fun:.5f} {pd.Series(pred_both).value_counts().to_dict()}")

# also save final (trees_only for compat)
pred_final=pred_trees; sub=pd.DataFrame({'id':test['id'],'class':pred_final})
sub.to_csv('submission_final.csv',index=False)
pr("ALL DONE!")