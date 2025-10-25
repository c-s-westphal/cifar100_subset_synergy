#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N resnet92_cifar100
#$ -t 1-10
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_resnet92.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
  source /share/apps/source_files/cuda/cuda-11.8.source || true
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  if [[ -f /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate ]]; then
    source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
  fi
fi

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
mkdir -p data
mkdir -p plots

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line: <arch> <seed>
arch=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

if [[ -z "$arch" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running job: arch=$arch, seed=$seed"

# Define expected checkpoint path for conditional training
checkpoint_file="checkpoints/${arch}_seed${seed}_final.pt"
results_file="results/${arch}_seed${seed}_results.npz"

# ---------------------------------------------------------------------
# 5.  Train (conditional on checkpoint and results existence)
# ---------------------------------------------------------------------
if [ -f "$checkpoint_file" ] && [ -f "$results_file" ]; then
    echo "Checkpoint and results already exist; skipping training."
    echo "  Checkpoint: $checkpoint_file"
    echo "  Results: $results_file"
else
    echo "Starting training..."
    python3.9 -u train_resnet.py \
        --arch "$arch" \
        --seed $seed \
        --epochs 201 \
        --max_epochs 500 \
        --batch_size 256 \
        --lr 0.1 \
        --momentum 0.9 \
        --weight_decay 1e-4 \
        --target_train_acc 99.0 \
        --eval_interval 10 \
        --n_masks_train 20 \
        --n_masks_final 80 \
        --max_eval_batches_train 20 \
        --max_eval_batches_final 80 \
        --checkpoint_dir checkpoints \
        --results_dir results \
        --data_dir ./data \
        --grad_clip 1.0 \
        --device cuda

    date
    echo "Training completed: arch=$arch seed=$seed"
fi

date
echo "Job completed: arch=$arch seed=$seed"
