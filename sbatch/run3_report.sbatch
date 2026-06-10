#!/bin/bash
#SBATCH --job-name=memrot_run3
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/run3_%j.out
#SBATCH --error=logs/run3_%j.err
# notchpeak-shared-short: CPU-friendly short queue; this account/partition pair
# was verified in sacctmgr for this user (2026-06-09)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"

bash env/setup_env.sh
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

python -u src/build_report.py --config config/config.yaml
command -v pandoc >/dev/null \
    && pandoc paper/paper.md -o paper/paper.pdf \
    || echo "pandoc absent; paper.md only"
