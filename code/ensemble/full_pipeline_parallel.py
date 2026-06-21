"""
Parallel pipeline: CPU trees + GPU models simultaneously
"""
import pandas as pd, numpy as np, json, time, gc, sys, os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import balanced_accuracy_score, recall_score
from datetime import datetime
import subprocess, pickle
def pr(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ========== 1. Load + Features (same as full_pipeline.py) ==========
pr("Loading raw data...")
train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
pr(f"Train: {train.shape}  Test: {test.shape}")

pr("Feature engineering...")
for df in [train, test]:
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('r','z')]:
        df[f'{a}_{b}']=df[a]-df[b]
    df['color_curv']=df.get('u_g',df['u']-df['g'])-df.get('g_r',df['g']-df['r'])
    df['u_z']=df['u']-df['z']; df['g_z']=df['g']-df['z']
    alpha_rad=np.deg2rad(df['alpha'])
    df['alpha_sin']=np.sin(alpha_rad); df['alpha_cos']=np.cos(alpha_rad)
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

for col in ['spectral_type','galaxy_population']:
    train[col+'_enc']=LabelEncoder().fit_transform(train[col])
    test[col+'_enc']=LabelEncoder().fit_transform(test[col])

le=LabelEncoder(); le.fit(train['class']); yc=le.transform(train['class'])
for df in [train,test]:
    df['_ca']=df['alpha_cat_'].astype(str)+'|'+df['delta_cat_'].astype(str)
    df['_cz']=df['u_cat_'].astype(str)+'|'+df['z_cat_'].astype(str)
for combo in ['_ca','_cz']:
    for ci in range(3):
        cn=f'{combo}_TE_{ci}'; yb=(yc==ci).astype(int)
        te=TargetEncoder(cv=5,smooth='auto',random_state=42)
        train[cn]=te.fit_transform(train[[combo]],yb).ravel()
        te2=TargetEncoder(smooth='auto',random_state=42); te2.fit(train[[combo]],yb)
        test[cn]=te2.transform(test[[combo]]).ravel()
train.drop(['_ca','_cz'],axis=1,inplace=True)
test.drop(['_ca','_cz'],axis=1,inplace=True)

feat_base=['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z',
           'color_curv','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols=[c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all=feat_base+new_cols; feat_all=[c for c in feat_all if c in train.columns]
train['target']=le.transform(train['class'])

X=train[feat_all].values.astype(np.float32); y=train['target'].values
X_test=test[[c for c in feat_all if c in test.columns]].values.astype(np.float32)
pr(f"Features: {len(feat_all)} | Train: {X.shape} | Test: {X_test.shape}")

folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y))

import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier

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
CB_MODELS=[
    ("CB_d8",dict(depth=8,min_data_in_leaf=50,l2_leaf_reg=3,subsample=0.8)),
    ("CB_d10",dict(depth=10,min_data_in_leaf=50,l2_leaf_reg=3,subsample=0.8)),
]
MLP_MODELS=[
    ("MLP_512_512_512",[512,512,512],64),
    ("MLP_512_512",[512,512],64),
]

all_preds={}

def train_trees():
    """LGB×5 + XGB×5 on CPU"""
    t0=time.time()
    for name,kw in LGB_MODELS+XGB_MODELS:
        pr(f"[CPU] {name}")
        oof=np.zeros((len(X),3)); tp=np.zeros((len(X_test),3))
        is_lgb='LGB' in name
        for fold,(tr,val) in enumerate(folds):
            if is_lgb:
                m=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=2000,learning_rate=0.05,reg_alpha=0.1,reg_lambda=0.1,random_state=fold*42,n_jobs=-1,verbose=-1,**kw)
                m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],callbacks=[lgb.early_stopping(50,verbose=False)])
            else:
                m=xgb.XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=2000,learning_rate=0.05,random_state=fold*42,n_jobs=-1,verbosity=0,**kw)
                m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],verbose=False)
            oof[val]=m.predict_proba(X[val]); tp+=m.predict_proba(X_test)/5
        ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
        pr(f"[CPU] {name}: BA={ba:.5f}")
        all_preds[name]={'oof':oof,'test':tp,'ba':ba}
    pr(f"[CPU] Trees done in {time.time()-t0:.0f}s")
    return all_preds

def train_gpu():
    """CB×2 + MLP×2 on GPU"""
    t0=time.time()
    feat_cb=[c for c in feat_all if c in train.columns]
    X_cb=train[feat_cb].values.astype(object)
    X_cb_test=test[[c for c in feat_cb if c in test.columns]].values.astype(object)
    cat_idx=[i for i,col in enumerate(feat_cb) if col in ['spectral_type','galaxy_population']]

    results={}
    for name,kw in CB_MODELS:
        pr(f"[GPU] {name}")
        oof=np.zeros((len(X),3)); tp=np.zeros((len(X_cb_test),3))
        for fold,(tr,val) in enumerate(folds):
            m=CatBoostClassifier(iterations=2000,learning_rate=0.05,bootstrap_type='Bernoulli',random_seed=fold*42,thread_count=-1,verbose=0,allow_writing_files=False,**kw)
            m.fit(X_cb[tr],y[tr],eval_set=[(X_cb[val],y[val])],cat_features=cat_idx,early_stopping_rounds=50,verbose=0)
            oof[val]=m.predict_proba(X_cb[val]); tp+=m.predict_proba(X_cb_test)/5
        ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
        pr(f"[GPU] {name}: BA={ba:.5f}")
        results[name]={'oof':oof,'test':tp,'ba':ba}

    for name,hidden,ep in MLP_MODELS:
        pr(f"[GPU] {name}")
        oof=np.zeros((len(X),3)); tp=np.zeros((len(X_test),3))
        for fold,(tr,val) in enumerate(folds):
            m=RealMLP_TD_Classifier(device='cuda',random_state=fold*42,n_epochs=ep,batch_size=8192,hidden_sizes=hidden)
            m.fit(X[tr],y[tr])
            oof[val]=m.predict_proba(X[val]); tp+=m.predict_proba(X_test)/5
        ba=balanced_accuracy_score(y,np.argmax(oof,axis=1))
        pr(f"[GPU] {name}: BA={ba:.5f}")
        results[name]={'oof':oof,'test':tp,'ba':ba}
    pr(f"[GPU] GPU done in {time.time()-t0:.0f}s")
    return results

# ========== Parallel Execution ==========
pr("\nStarting parallel: CPU trees + GPU models...")
t0=time.time()
with ThreadPoolExecutor(max_workers=2) as pool:
    f_cpu=pool.submit(train_trees)
    f_gpu=pool.submit(train_gpu)
    cpu_results=f_cpu.result()
    gpu_results=f_gpu.result()
all_preds.update(cpu_results)
all_preds.update(gpu_results)
pr(f"All models done in {time.time()-t0:.0f}s")

# ========== Ensemble ==========
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

test_trees=np.mean([v['test'] for k,v in all_preds.items() if 'LGB' in k or 'XGB' in k or 'CB' in k],axis=0)
res=minimize(nba,[1,1,1],args=(oof_trees,y),method='Nelder-Mead',bounds=[(0.2,3),(0.2,3),(0.2,3)],options=dict(xatol=0.001,maxiter=500))
pred_final=le.inverse_transform(tp(test_trees,res.x))
sub=pd.DataFrame({'id':test['id'],'class':pred_final})
sub.to_csv('submission_final.csv',index=False)
pr(f"\nSubmission: {len(sub)} rows | {sub['class'].value_counts().to_dict()}")
pr("ALL DONE!")
