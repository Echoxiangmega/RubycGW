#!/bin/bash
#SBATCH --job-name=ruby_quarter_competing
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/quarter_competing_%A_%a.out
#SBATCH --error=logs/quarter_competing_%A_%a.err

set -euo pipefail

# Share OUT with the ordinary Raw-Slater V scan. The collector then sees
# normal/CO/LC and these four restricted Slater/Pfaffian candidate branches.
# Example: sbatch --export=ALL,L1=2,L2=2,OUT=results/vscan_quarter \
#   ed_benchmark/vmc_zeroT/slurm_vscan_quarter_competing_cpu.sh

SEEDS=(sdw pf_s pf_dplus pf_dminus)
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}"
if (( SLURM_ARRAY_TASK_ID < ${#SEEDS[@]} )); then
  DIRECTION=up
else
  DIRECTION=down
fi

REPO="${REPO:-$PWD}"
OUT="${OUT:-$REPO/results/vscan_quarter}"
L1="${L1:-2}"
L2="${L2:-2}"
T="${T:-1}"
T1="${T1:--0.5}"
T2="${T2:-0.5}"
V_MIN="${V_MIN:-0}"
V_MAX="${V_MAX:-4}"
V_STEP="${V_STEP:-0.5}"

mkdir -p "$REPO/logs" "$OUT"
cd "$REPO"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u ed_benchmark/vmc_zeroT/run_supercell_vscan.py \
  --L1 "$L1" --L2 "$L2" --filling 1/4 \
  --t "$T" --t1 "$T1" --t2 "$T2" \
  --seed "$SEED" --mode continuation --direction "$DIRECTION" \
  --V-min "$V_MIN" --V-max "$V_MAX" --V-step "$V_STEP" \
  --competing-train-samples "${TRAIN_SAMPLES:-512}" \
  --competing-eval-samples "${EVAL_SAMPLES:-4096}" \
  --competing-refine-steps "${REFINE_STEPS:-9}" \
  --competing-pair-samples "${PAIR_SAMPLES:-64}" \
  --rng-seed "${RNG_SEED:-2026}" --output "$OUT"
