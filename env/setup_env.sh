#!/bin/bash
# Idempotent environment bootstrap. The user's ~/miniconda3 base install was
# scratch-purged (symlink with no bin/), so we install a fresh Miniforge under
# $SCRATCH/memrot and create the 'memrot' env from environment.yml.
# Called automatically at the top of every sbatch script; safe to re-run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMROT_SCRATCH="${SCRATCH:?SCRATCH not set}/memrot"
MF="$MEMROT_SCRATCH/miniforge3"

if [ ! -x "$MF/bin/conda" ]; then
    echo "[setup_env] installing Miniforge3 -> $MF"
    mkdir -p "$MEMROT_SCRATCH"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O "$MEMROT_SCRATCH/miniforge_installer.sh"
    bash "$MEMROT_SCRATCH/miniforge_installer.sh" -b -p "$MF"
fi
source "$MF/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx memrot; then
    echo "[setup_env] creating conda env 'memrot'"
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi
conda activate memrot

python - <<'EOF'
import torch, transformers, h5py, statsmodels, pandas, scipy, yaml, matplotlib
print(f"[setup_env] torch {torch.__version__}, transformers {transformers.__version__}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f"[setup_env] CUDA OK: {name}, bf16={torch.cuda.is_bf16_supported()}")
else:
    print("[setup_env] no GPU visible (fine on login/CPU nodes)")
print("STATUS: setup_env PASS imports+cuda verified")
EOF
