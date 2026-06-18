#!/bin/bash
# Cross-model comparison (CPU). Reads analysis/<id>/results_summary.json and
# artifacts/<id>/ for every model_id present and writes the scaling summary +
# figures. Run after the per-model chains finish.
#SBATCH --job-name=memrot_compare
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"

bash env/setup_env.sh
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

# Models default to all that have a results_summary.json; override with args.
python -u src/compare_models.py --models "${@:-2b 9b 27b}"
