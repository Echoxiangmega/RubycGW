#!/bin/bash
#SBATCH --job-name=ruby3x3_V
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ruby3x3_%A_%a.out
#SBATCH --error=logs/ruby3x3_%A_%a.err
#
# Add cluster-specific lines if required, for example:
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

REPO="${REPO:-$HOME/RubycGW}"
VENV="${VENV:-$HOME/venvs/rubycgw}"
OUT="${OUT:-$REPO/results/vscan_3x3}"
V_MIN="${V_MIN:-0.0}"
V_MAX="${V_MAX:-8.0}"
V_STEP="${V_STEP:-0.5}"

SEEDS=(
  normal
  sd
  isd
  lc_same
  lc_opposite
  random
  ps_co
  mix_co_lc_opposite
)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p "$REPO/logs" "$OUT"
cd "$REPO"

source "$VENV/bin/activate"

# JAX: do not reserve most GPU memory at process startup.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname)"
echo "seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import jax, netket
print("JAX devices:", jax.devices())
print("NetKet:", netket.__version__)
PY

srun python ed_benchmark/vmc_zeroT/run_supercell_vscan.py \
  --L1 3 --L2 3 \
  --seed "$SEED" \
  --mode continuation \
  --direction up \
  --V-min "$V_MIN" --V-max "$V_MAX" --V-step "$V_STEP" \
  --eval-samples 8192 \
  --n-chains 32 \
  --n-discard 16 \
  --output "$OUT"
