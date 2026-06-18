#!/bin/bash
# Generic Run-3 (report) stage. CPU-only. Resources from sbatch/submit_model.sh;
# config is $1, namespace is $MEMROT_MODEL_ID.
#SBATCH --job-name=memrot_rep
#SBATCH --output=logs/rep_%j.out
#SBATCH --error=logs/rep_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"

CFG="${1:-config/config.yaml}"
: "${MEMROT_MODEL_ID:?set MEMROT_MODEL_ID (2b|9b|27b) so artifacts are namespaced}"
echo "STAGE report: config=$CFG MEMROT_MODEL_ID=$MEMROT_MODEL_ID"

bash env/setup_env.sh
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

python -u src/build_report.py --config "$CFG"
