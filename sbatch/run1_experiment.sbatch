#!/bin/bash
#SBATCH --job-name=memrot_run1
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run1_%j.out
#SBATCH --error=logs/run1_%j.err
# Discovered cluster facts (2026-06-09): account/partition notchpeak-gpu exists for
# this user; bf16-capable GPUs in that partition are 3090 (notch293/notch328) and
# a100 (notch293) -> the constraint. V100/2080Ti/P40 lack bf16 and are excluded.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"
export HF_HOME="$SCRATCH/hf_cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"     # required by torch deterministic mode
export TOKENIZERS_PARALLELISM=false
export MEMROT_MODEL_ID=2b                      # namespace -> artifacts/2b (scaling layout)

bash env/setup_env.sh                         # idempotent bootstrap
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

# Legacy 2B entry point; the model-generic path is `bash sbatch/submit_model.sh 2b`.
python -u src/run_experiment.py --config config/config_2b.yaml --smoke-test
python -u src/run_experiment.py --config config/config_2b.yaml --full
