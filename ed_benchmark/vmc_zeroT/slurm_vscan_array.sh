#!/bin/bash
#SBATCH --job-name=ruby_vscan
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vscan_%A_%a.out
#SBATCH --error=logs/vscan_%A_%a.err

set -euo pipefail

# Submit one array per supercell size, for example:
#   sbatch --export=ALL,L1=4,L2=4,V_MIN=0,V_MAX=8,V_STEP=0.5,OUT=results/vscan #     ed_benchmark/vmc_zeroT/slurm_vscan_array.sh
#
# Array tasks 0..5  : six seed branches, V increasing.
# Array tasks 6..11 : the same six seed branches, V decreasing.
#
# Adapt the GPU/partition/module/conda lines to your cluster.

mkdir -p logs

SEEDS=(normal sd isd lc_same lc_opposite random)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
DIR_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
SEED="${SEEDS[${SEED_INDEX}]}"
if [[ "${DIR_INDEX}" -eq 0 ]]; then
  DIRECTION="up"
else
  DIRECTION="down"
fi

L1="${L1:-2}"
L2="${L2:-2}"
V_MIN="${V_MIN:-0.0}"
V_MAX="${V_MAX:-8.0}"
V_STEP="${V_STEP:-0.5}"
OUT="${OUT:-results/vscan}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"
NCHAINS="${NCHAINS:-32}"

# Avoid JAX reserving essentially all device memory at startup.  The scan
# driver controls the dominant allocations through --chunk-size and
# --qgt-chunk-size.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python ed_benchmark/vmc_zeroT/run_supercell_vscan.py \
  --L1 "${L1}" --L2 "${L2}" \
  --seed "${SEED}" \
  --mode continuation \
  --direction "${DIRECTION}" \
  --V-min "${V_MIN}" --V-max "${V_MAX}" --V-step "${V_STEP}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --n-chains "${NCHAINS}" \
  --output "${OUT}"
