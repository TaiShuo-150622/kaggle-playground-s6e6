"""Optuna hyperparameter search for LightGBM"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb, optuna, warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("data/train_fe.csv")
for col in ['spectral_type','galaxy_population']:
    train[col+'_enc'] = pd.factorize(train[col])[0]

feat_base = ['u','g','r','i','z','redshift','u_g','g_r','r_i','i_z','u_r','g_i','r_z','color_curv','alpha_sin','alpha_cos','delta','spectral_type_enc','galaxy_population_enc','u_z','g_z']
new_cols = [c for c in train.columns if c.startswith('_') or '_TE_' in c or '_cat_' in c or '_bin_' in c]
feat_all = [c for c in feat_base+new_cols if c in train.columns]
X = train[feat_all].values.astype(np.float32)
y = pd.factorize(train['class'])[0]
print(f"Features: {len(feat_all)}, Samples: {len(y)}")

def objective(trial):
    params = {
        'objective': 'multiclass', 'num_class': 3,
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 32, 512),
        'max_depth': trial.suggest_int('max_depth', 4, 14),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    scores = []
    for fold,(tr,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr],y[tr],eval_set=[(X[val],y[val])],
              callbacks=[lgb.early_stopping(50,verbose=False)])
        scores.append(balanced_accuracy_score(y[val],m.predict(X[val])))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best BA: {study.best_value:.5f}")
print("Best params:")
for k,v in study.best_params.items():
    print(f"  {k}: {v}")
