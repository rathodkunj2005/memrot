#!/bin/bash
# Submit one model's full chain (experiment -> analyze -> report), namespaced by
# MEMROT_MODEL_ID and chained with afterok. Per-model GPU resources live here so
# the stage_*.sbatch files stay model-agnostic.
#
#   bash sbatch/submit_model.sh 9b
#   bash sbatch/submit_model.sh 27b
#
# Prereq: accept the HF license for the chosen google/gemma-2-<M>-it model.
set -euo pipefail
M="${1:-}"

case "$M" in
  2b)
    GPU=(--account=notchpeak-gpu --partition=notchpeak-gpu --gres=gpu:1
         --constraint=3090\|a100 --cpus-per-task=8 --mem=48G)
    EXP_T=10:00:00; ANA_T=03:00:00 ;;
  9b)
    # 9B bf16 ~18GB weights + activations -> needs a 40GB A100 (24GB 3090 OOMs at high k).
    GPU=(--account=notchpeak-gpu --partition=notchpeak-gpu --gres=gpu:1
         --constraint=a100 --cpus-per-task=8 --mem=64G)
    EXP_T=12:00:00; ANA_T=03:00:00 ;;
  27b)
    # 27B bf16 ~54GB weights -> needs an 80GB A100; only on the preemptible guest
    # partition. --requeue is safe because the sweep resumes per (qid,k).
    GPU=(--account=notchpeak-gpu-guest --partition=notchpeak-gpu-guest --gres=gpu:1
         --constraint=a100_80gb_pcie --requeue --cpus-per-task=12 --mem=96G)
    EXP_T=24:00:00; ANA_T=06:00:00 ;;
  *)
    echo "usage: bash sbatch/submit_model.sh <2b|9b|27b>" >&2; exit 1 ;;
esac

CFG="config/config_${M}.yaml"
[ -f "$CFG" ] || { echo "missing $CFG" >&2; exit 1; }
EXP=(--export=ALL,MEMROT_MODEL_ID="$M")

J1=$(sbatch --parsable "${GPU[@]}" --time="$EXP_T" --job-name="memrot_${M}_exp" \
        "${EXP[@]}" sbatch/stage_experiment.sbatch "$CFG")
echo "experiment ($M): $J1"

J2=$(sbatch --parsable --dependency=afterok:"$J1" "${GPU[@]}" --time="$ANA_T" \
        --job-name="memrot_${M}_ana" "${EXP[@]}" sbatch/stage_analyze.sbatch "$CFG")
echo "analyze    ($M): $J2  (afterok:$J1)"

J3=$(sbatch --parsable --dependency=afterok:"$J2" \
        --account=notchpeak-shared-short --partition=notchpeak-shared-short \
        --cpus-per-task=4 --mem=8G --time=00:30:00 \
        --job-name="memrot_${M}_rep" "${EXP[@]}" sbatch/stage_report.sbatch "$CFG")
echo "report     ($M): $J3  (afterok:$J2)"

echo "Submitted $M chain. Watch: sacct -j $J1,$J2,$J3 --format=JobID,JobName%16,State,Elapsed"
