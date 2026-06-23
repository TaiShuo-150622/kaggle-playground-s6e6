"""
GPU LR Stacker — adapted from Chris Deotte's notebook
Uses OUR OOF predictions instead of his pre-computed files
"""
import numpy as np, pandas as pd, glob, torch, torch.nn as nn, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# Load data
train = pd.read_csv("data/train.csv", index_col='id')
test  = pd.read_csv("data/test.csv", index_col='id')
TARGET = 'class'; CLASSES = ['GALAXY','QSO','STAR']
target_map = {c:i for i,c in enumerate(CLASSES)}
inv_map = {v:k for k,v in target_map.items()}
y = train[TARGET].map(target_map).astype(int).values
N, M = len(y), len(test)

# Load all OOF predictions
print("Loading OOF predictions...")
oof_files = sorted(glob.glob("oof_*.npy"))
test_files = sorted(glob.glob("test_*.npy"))
print(f"Found {len(oof_files)} OOF files, {len(test_files)} test files")

loaded_oofs = {}
loaded_tests = {}
for f in oof_files:
    name = f.replace("oof_","").replace(".npy","")
    loaded_oofs[name] = np.load(f).astype(np.float32)

for f in test_files:
    name = f.replace("test_","").replace(".npy","")
    loaded_tests[name] = np.load(f).astype(np.float32)

# Use only models that have both OOF and test
model_names = sorted(set(loaded_oofs.keys()) & set(loaded_tests.keys()))
n_models = len(model_names)
print(f"Using {n_models} models", flush=True)

# ====== GPU LR Stacker ======
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = list(range(42, 47))  # 5 seeds
FOLDS = 5
print(f"Starting LR Stacker: {len(SEEDS)} seeds × {FOLDS} folds × 200 epochs", flush=True)
print(f"Device: {DEVICE}")

oof_sum = np.zeros((N, 3), dtype=np.float64)
test_sum = np.zeros((M, 3), dtype=np.float64)

for si, seed in enumerate(SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    print(f"\nSeed {si+1}/{len(SEEDS)} (seed={seed})")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(N), y), start=1):
        t0 = time.time()

        # Build meta-features: OOF predictions → X_meta (N, n_models*3)
        X_meta_tr = np.column_stack([loaded_oofs[name][tr_idx] for name in model_names])
        X_meta_va = np.column_stack([loaded_oofs[name][val_idx] for name in model_names])
        X_meta_te = np.column_stack([loaded_tests[name] for name in model_names])

        y_tr = y[tr_idx]; y_va = y[val_idx]

        # LR in PyTorch
        class LogisticModel(nn.Module):
            def __init__(self, in_features, n_classes):
                super().__init__()
                self.linear = nn.Linear(in_features, n_classes)
            def forward(self, x):
                return self.linear(x)

        model = LogisticModel(n_models*3, 3).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

        Xt_tr = torch.tensor(X_meta_tr).float().to(DEVICE)
        yt_tr = torch.tensor(y_tr).long().to(DEVICE)
        Xt_va = torch.tensor(X_meta_va).float().to(DEVICE)
        Xt_te = torch.tensor(X_meta_te).float().to(DEVICE)

        # Train LR
        for epoch in range(200):
            model.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xt_tr), yt_tr)
            loss.backward(); opt.step()

        # Predict
        model.eval()
        with torch.no_grad():
            va_p = nn.functional.softmax(model(Xt_va), dim=-1).cpu().numpy()
            te_p = nn.functional.softmax(model(Xt_te), dim=-1).cpu().numpy()

        oof_sum[val_idx] += va_p.astype(np.float64)
        test_sum += te_p.astype(np.float64) / (FOLDS * len(SEEDS))

        ba = balanced_accuracy_score(y_va, np.argmax(va_p, axis=1))
        elapsed = time.time() - t0
        print(f"  Fold {fold}: BA={ba:.5f} ({elapsed:.0f}s)")

    oof_this = oof_sum / ((fold) * (si+1))  # running avg
    ba_seed = balanced_accuracy_score(y, np.argmax(oof_this, axis=1))
    print(f"  Seed {si+1} cumulative OOF: {ba_seed:.5f}")

oof_final = oof_sum / (FOLDS * len(SEEDS))
ba_final = balanced_accuracy_score(y, np.argmax(oof_final, axis=1))

# Threshold tuning
from scipy.optimize import minimize
def tp(p,t): return np.argmax(p/np.array(t), axis=1)
def nba(t,p,yt): return -balanced_accuracy_score(yt,tp(p,t))
r = minimize(nba, [1,1,1], args=(oof_final,y), method='Nelder-Mead',
             bounds=[(0.2,3),(0.2,3),(0.2,3)], options=dict(xatol=0.001,maxiter=500))

print(f"\n{'='*50}")
print(f"LR Stacker OOF: {ba_final:.5f}")
print(f"Tuned: {-r.fun:.5f}")
print(f"vs simple avg (0.96786): {ba_final - 0.96786:+.5f}")

# Submission
pred = inv_map[tp(test_sum / (FOLDS * len(SEEDS)), r.x)]
sub = pd.DataFrame({'id':test.index,'class':pred})
sub.to_csv('submission_lr_stacker.csv',index=False)
print(f"\nsubmission_lr_stacker.csv: {len(sub)} rows")
np.save('oof_lr_stacker.npy', oof_final)
np.save('test_lr_stacker.npy', test_sum)
print("Done!")
