#!/bin/bash
# Generic Run-2 (analysis) stage. Loads the model for H3 ablation/intervention, so
# it needs the same GPU class as the experiment stage. Resources come from
# sbatch/submit_model.sh; config is $1, namespace is $MEMROT_MODEL_ID.
#SBATCH --job-name=memrot_ana
#SBATCH --output=logs/ana_%j.out
#SBATCH --error=logs/ana_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"
export HF_HOME="$SCRATCH/hf_cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false

CFG="${1:-config/config.yaml}"
: "${MEMROT_MODEL_ID:?set MEMROT_MODEL_ID (2b|9b|27b) so artifacts are namespaced}"
echo "STAGE analyze: config=$CFG MEMROT_MODEL_ID=$MEMROT_MODEL_ID"

bash env/setup_env.sh
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

python -u src/run_analysis.py --config "$CFG"
