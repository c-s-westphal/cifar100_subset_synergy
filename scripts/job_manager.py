#!/usr/bin/env python3
"""
SLURM Job Manager for VGG CIFAR-100 Training

Reads job files and generates SLURM submission scripts for each architecture.
Uses the same virtual environment as the comparing_bn_adam repository.
"""

import os
import argparse
from pathlib import Path


def create_slurm_script(arch: str, num_jobs: int) -> str:
    """Create SLURM submission script for a given architecture.

    Args:
        arch: Architecture name (vgg9, vgg11, vgg13, vgg16, vgg19)
        num_jobs: Number of jobs (seeds) to run

    Returns:
        SLURM script content as string
    """
    script = f"""#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N {arch}_cifar100
#$ -t 1-{num_jobs}
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_{arch}.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
  source /share/apps/source_files/cuda/cuda-11.8.source || true
fi
if [[ -n "${{VIRTUAL_ENV:-}}" ]]; then
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
arch=$(sed -n ${{number}}p "$paramfile" | awk '{{print $1}}')
seed=$(sed -n ${{number}}p "$paramfile" | awk '{{print $2}}')

if [[ -z "$arch" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running job: arch=$arch, seed=$seed"

# Define expected checkpoint path for conditional training
checkpoint_file="checkpoints/${{arch}}_seed${{seed}}_final.pt"
results_file="results/${{arch}}_seed${{seed}}_results.npz"

# ---------------------------------------------------------------------
# 5.  Train (conditional on checkpoint and results existence)
# ---------------------------------------------------------------------
if [ -f "$checkpoint_file" ] && [ -f "$results_file" ]; then
    echo "Checkpoint and results already exist; skipping training."
    echo "  Checkpoint: $checkpoint_file"
    echo "  Results: $results_file"
else
    echo "Starting training..."
    python3.9 -u train_vgg.py \\
        --arch "$arch" \\
        --seed $seed \\
        --epochs 300 \\
        --batch_size 256 \\
        --lr 0.001 \\
        --weight_decay 0.0 \\
        --eval_interval 10 \\
        --n_masks_train 20 \\
        --n_masks_final 40 \\
        --max_eval_batches_train 20 \\
        --max_eval_batches_final 40 \\
        --checkpoint_dir checkpoints \\
        --results_dir results \\
        --data_dir ./data \\
        --device cuda

    date
    echo "Training completed: arch=$arch seed=$seed"
fi

date
echo "Job completed: arch=$arch seed=$seed"
"""
    return script


def main():
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts for VGG training')
    parser.add_argument('--output_dir', type=str, default='scripts',
                        help='Directory to save job scripts')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate scripts, do not submit')

    args = parser.parse_args()

    # Architectures to process
    architectures = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_scripts = []

    for arch in architectures:
        # Count jobs from job file
        job_file = output_dir / f'jobs_{arch}.txt'

        if not job_file.exists():
            print(f"Warning: Job file {job_file} not found, skipping {arch}")
            continue

        with open(job_file, 'r') as f:
            num_jobs = sum(1 for line in f if line.strip())

        # Create SLURM script
        script_content = create_slurm_script(arch, num_jobs)
        script_path = output_dir / f'job_manager_{arch}.sh'

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        generated_scripts.append(script_path)
        print(f"Generated: {script_path} ({num_jobs} jobs)")

    print(f"\nGenerated {len(generated_scripts)} job scripts.")

    if not args.generate_only:
        print("\nTo submit jobs, run:")
        for script_path in generated_scripts:
            print(f"  qsub {script_path}")
        print("\nOr submit all at once:")
        print(f"  for script in {output_dir}/job_manager_*.sh; do qsub $script; done")
    else:
        print("\nScripts generated but not submitted (--generate_only flag used)")


if __name__ == '__main__':
    main()
