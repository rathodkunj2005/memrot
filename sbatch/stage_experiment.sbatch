#!/bin/bash
# Generic Run-1 (experiment) stage. Model-agnostic: the model is selected by the
# config passed as $1 and the namespace by $MEMROT_MODEL_ID. Resource directives
# (account/partition/gres/constraint/mem/time) are supplied on the `sbatch`
# command line by sbatch/submit_model.sh, so this file carries none.
#SBATCH --job-name=memrot_exp
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export MEMROT_ROOT="$SLURM_SUBMIT_DIR"
export HF_HOME="$SCRATCH/hf_cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"     # required by torch deterministic mode
export TOKENIZERS_PARALLELISM=false

CFG="${1:-config/config.yaml}"
: "${MEMROT_MODEL_ID:?set MEMROT_MODEL_ID (2b|9b|27b) so artifacts are namespaced}"
echo "STAGE experiment: config=$CFG MEMROT_MODEL_ID=$MEMROT_MODEL_ID"

bash env/setup_env.sh                         # idempotent bootstrap
source "$SCRATCH/memrot/miniforge3/etc/profile.d/conda.sh"
conda activate memrot

python -u src/run_experiment.py --config "$CFG" --smoke-test
python -u src/run_experiment.py --config "$CFG" --full
