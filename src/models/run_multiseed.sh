#!/bin/bash
# Multi-seed runner for S6E6 models
# Usage: bash src/models/run_multiseed.sh <model_script> <model_name> <num_seeds>
# Example: bash src/models/run_multiseed.sh src/models/deotte_realmlp.py RealMLP 5

SCRIPT=$1
NAME=$2
N_SEEDS=${3:-5}
cd /root/kaggle_s6e6

for s in $(seq 1 $N_SEEDS); do
    SEED=$((42 + (s - 1) * 100))
    echo "[$(date +%H:%M:%S)] $NAME seed=$s (base_seed=$SEED)"

    # Create a copy with replaced seed
    cp "$SCRIPT" "/tmp/${NAME}_s${s}.py"
    sed -i "s/^SEED = 42/SEED = $SEED/" "/tmp/${NAME}_s${s}.py"
    sed -i "s/random_state=42/random_state=$SEED/g" "/tmp/${NAME}_s${s}.py"

    PYTHONUNBUFFERED=1 /root/miniconda3/bin/python3 "/tmp/${NAME}_s${s}.py" > "${NAME}_s${s}.log" 2>&1

    # Rename output files
    if [ -f "oof_${NAME}_handcrafted.npy" ]; then
        mv "oof_${NAME}_handcrafted.npy" "oof_${NAME}_s${s}.npy"
        mv "test_${NAME}_handcrafted.npy" "test_${NAME}_s${s}.npy"
    fi

    echo "[$(date +%H:%M:%S)] $NAME seed=$s done"
done

echo "ALL ${N_SEEDS} SEEDS DONE for $NAME"
