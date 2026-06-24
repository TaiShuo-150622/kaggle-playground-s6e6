#!/bin/bash
# Overnight batch: runs ~8 hours, covers all remaining experiments
# Launched at ~01:30, finishes by ~09:00
set -e
cd /root/kaggle_s6e6
PY=/root/miniconda3/bin/python3
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ===== Phase 1: CatBoost multi-seed (already running) =====
log "Waiting for cb_seeds to finish..."
# cb_seeds should finish ~50 min from start (01:25 → 02:15)

# ===== Phase 2: RealMLP v3 (252 features) ×3 seeds =====
log "=== Phase 2: RealMLP v3 (252 features) ==="
for s in 42 142 242; do
    log "RealMLP v3 seed=$s"
    # Run realmlp_v3.py with seed override
    sed "s/^SEED = 42/SEED = $s/" src/models/realmlp_v3.py > /tmp/realmlp_v3_s${s}.py
    $PY /tmp/realmlp_v3_s${s}.py > realmlp_v3_s${s}.log 2>&1
    # Rename output
    [ -f oof_RealMLP_v3.npy ] && mv oof_RealMLP_v3.npy oof_RealMLP_v3_s${s}.npy
    [ -f test_RealMLP_v3.npy ] && mv test_RealMLP_v3.npy test_RealMLP_v3_s${s}.npy
    log "RealMLP v3 seed=$s done"
done

# Ensemble RealMLP v3 seeds
log "Ensembling RealMLP v3 seeds..."
$PY -c "
import numpy as np
oofs = [np.load(f'oof_RealMLP_v3_s{s}.npy') for s in [42,142,242]]
tests = [np.load(f'test_RealMLP_v3_s{s}.npy') for s in [42,142,242]]
np.save('oof_RealMLP_v3_ens.npy', np.mean(oofs, axis=0))
np.save('test_RealMLP_v3_ens.npy', np.mean(tests, axis=0))
print('RealMLP v3 ensemble saved')
"

# ===== Phase 3: Final ensemble =====
log "=== Phase 3: Final Ensemble ==="
log "Waiting for all experiments to complete..."

$PY -c "
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

train = pd.read_csv('data/train.csv')
le = LabelEncoder(); y = le.fit_transform(train['class'])
test = pd.read_csv('data/test.csv')

def tune(probs, t): return np.argmax(probs / np.array(t), axis=1)
def nba(t, probs, yt): return -balanced_accuracy_score(yt, tune(probs, t))

# Collect all OOF/test predictions
models = {}
for name, oof_f, test_f in [
    ('CB_s1', 'oof_CB_s1.npy', 'test_CB_s1.npy'),
    ('CB_s2', 'oof_CB_s2.npy', 'test_CB_s2.npy'),
    ('CB_s3', 'oof_CB_s3.npy', 'test_CB_s3.npy'),
    ('CB_s4', 'oof_CB_s4.npy', 'test_CB_s4.npy'),
    ('CB_s5', 'oof_CB_s5.npy', 'test_CB_s5.npy'),
    ('RMLP_old_s1', 'oof_RealMLP_s1.npy', 'test_RealMLP_s1.npy'),
    ('RMLP_old_s2', 'oof_RealMLP_s2.npy', 'test_RealMLP_s2.npy'),
    ('RMLP_old_s3', 'oof_RealMLP_s3.npy', 'test_RealMLP_s3.npy'),
    ('RMLP_old_s4', 'oof_RealMLP_s4.npy', 'test_RealMLP_s4.npy'),
    ('RMLP_old_s5', 'oof_RealMLP_s5.npy', 'test_RealMLP_s5.npy'),
    ('RMLP_v3_ens', 'oof_RealMLP_v3_ens.npy', 'test_RealMLP_v3_ens.npy'),
    ('FT_v3', 'oof_FT_Transformer_v2.npy', 'test_FT_Transformer_v2.npy'),
]:
    try:
        oof = np.load(oof_f)
        tst = np.load(test_f)
        if oof.shape[0] == len(y):
            models[name] = {'oof': oof, 'test': tst}
            print(f'  Loaded {name}: OOF shape={oof.shape}')
    except:
        print(f'  Skipping {name} (not found)')

if len(models) < 2:
    print('Not enough models for ensemble, skipping')
    exit()

# Weighted average ensemble
oof_ens = np.mean([m['oof'] for m in models.values()], axis=0)
test_ens = np.mean([m['test'] for m in models.values()], axis=0)
ba_base = balanced_accuracy_score(y, np.argmax(oof_ens, axis=1))
res = minimize(nba, [1,1,1], args=(oof_ens, y), method='Nelder-Mead',
               bounds=[(0.2,3),(0.2,3),(0.2,3)])
ba_tuned = -res.fun
print(f'Ensemble ({len(models)} models): base={ba_base:.5f} tuned={ba_tuned:.5f}')

# Submission
preds = le.inverse_transform(tune(test_ens, res.x))
sub = pd.DataFrame({'id': test['id'], 'class': preds})
fname = 'submission_overnight_final.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}')
print(sub['class'].value_counts().to_dict())
"

log "=== OVERNIGHT BATCH COMPLETE ==="
log "Submission: submission_overnight_final.csv"
