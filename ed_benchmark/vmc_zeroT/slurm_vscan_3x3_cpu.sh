#!/bin/bash
#SBATCH --job-name=ruby3x3_V
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/ruby3x3_%A_%a.log
#
# Add cluster-specific directives if required:
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ACCOUNT
#
# Submit from the repository root after creating logs/:
#   mkdir -p logs
#   sbatch ed_benchmark/vmc_zeroT/slurm_vscan_3x3_cpu.sh
#
# Limit concurrency, for example at most four seed branches at once:
#   sbatch --array=0-7%4 ed_benchmark/vmc_zeroT/slurm_vscan_3x3_cpu.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# User-editable paths / environment
# ---------------------------------------------------------------------------
REPO="${REPO:-$HOME/RubycGW}"
CONDA_ENV="${CONDA_ENV:-rubycgw}"
OUT="${OUT:-$REPO/results/vscan_3x3}"

# If 'conda' is not already visible in non-interactive batch shells, set
# CONDA_BASE explicitly when submitting, for example:
#   sbatch --export=ALL,CONDA_BASE=$HOME/miniconda3 ...
CONDA_BASE="${CONDA_BASE:-}"

# ---------------------------------------------------------------------------
# Physics / scan parameters
# ---------------------------------------------------------------------------
L1="${L1:-3}"
L2="${L2:-3}"
T="${T:-1.0}"
T1="${T1:-0.5}"
T2="${T2:--0.5}"
SEED_AMPLITUDE="${SEED_AMPLITUDE:-0.08}"

V_MIN="${V_MIN:-0.0}"
V_MAX="${V_MAX:-8.0}"
V_STEP="${V_STEP:-0.5}"

# continuation = each V warm-starts from the previous V of the same seed.
# independent  = every V starts from the named seed.
MODE="${MODE:-continuation}"
DIRECTION="${DIRECTION:-up}"

# ---------------------------------------------------------------------------
# First-V anchor optimization schedule
# ---------------------------------------------------------------------------
ANCHOR_ITER="${ANCHOR_ITER:-230}"
ANCHOR_SWITCH1="${ANCHOR_SWITCH1:-80}"
ANCHOR_SWITCH2="${ANCHOR_SWITCH2:-180}"
ANCHOR_LR="${ANCHOR_LR:-0.03}"
ANCHOR_LR_MID="${ANCHOR_LR_MID:-0.015}"
ANCHOR_LR_REFINE="${ANCHOR_LR_REFINE:-0.005}"
ANCHOR_DIAG="${ANCHOR_DIAG:-0.05}"
ANCHOR_DIAG_MID="${ANCHOR_DIAG_MID:-0.05}"
ANCHOR_DIAG_REFINE="${ANCHOR_DIAG_REFINE:-0.02}"

# ---------------------------------------------------------------------------
# Warm-started continuation-point schedule
# ---------------------------------------------------------------------------
CONT_ITER="${CONT_ITER:-160}"
CONT_SWITCH1="${CONT_SWITCH1:-50}"
CONT_SWITCH2="${CONT_SWITCH2:-120}"
CONT_LR="${CONT_LR:-0.015}"
CONT_LR_MID="${CONT_LR_MID:-0.0075}"
CONT_LR_REFINE="${CONT_LR_REFINE:-0.003}"
CONT_DIAG="${CONT_DIAG:-0.05}"
CONT_DIAG_MID="${CONT_DIAG_MID:-0.03}"
CONT_DIAG_REFINE="${CONT_DIAG_REFINE:-0.02}"

# ---------------------------------------------------------------------------
# Monte Carlo / SR parameters
# ---------------------------------------------------------------------------
TRAIN_SAMPLES_EARLY="${TRAIN_SAMPLES_EARLY:-1024}"
TRAIN_SAMPLES_LATE="${TRAIN_SAMPLES_LATE:-2048}"
TRAIN_SAMPLES_REFINE="${TRAIN_SAMPLES_REFINE:-4096}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8192}"

NCHAINS="${NCHAINS:-32}"
NDISCARD="${NDISCARD:-16}"

# For 3x3 the driver auto value is 36.  Set empty to keep auto.
SWEEP_SIZE="${SWEEP_SIZE:-36}"

# 3x3 defaults.  Lower these if memory is tight; larger values may be faster.
CHUNK_SIZE="${CHUNK_SIZE:-256}"
QGT_CHUNK_SIZE="${QGT_CHUNK_SIZE:-128}"

CG_TOL="${CG_TOL:-1e-5}"
CG_MAXITER="${CG_MAXITER:-100}"
JASTROW_SHELLS="${JASTROW_SHELLS:-4}"
RNG_SEED="${RNG_SEED:-2026}"

# ---------------------------------------------------------------------------
# Eight independent seed branches
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Conda activation
# ---------------------------------------------------------------------------
if [[ -n "$CONDA_BASE" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not visible in this batch shell."
    echo "Set CONDA_BASE, e.g. --export=ALL,CONDA_BASE=\$HOME/miniconda3"
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# CPU threading
# ---------------------------------------------------------------------------
# These control common BLAS/OpenMP libraries.  JAX/XLA also has its own CPU
# runtime/threading, so wall-time scaling with core count should be benchmarked.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "================================================================"
echo "start_time        = $(date -Is)"
echo "host              = $(hostname)"
echo "job_id            = ${SLURM_JOB_ID:-none}"
echo "array_task        = ${SLURM_ARRAY_TASK_ID:-none}"
echo "seed              = $SEED"
echo "L                 = ${L1} x ${L2}"
echo "V scan            = $V_MIN : $V_STEP : $V_MAX"
echo "mode/direction    = $MODE / $DIRECTION"
echo "cpus_per_task     = ${SLURM_CPUS_PER_TASK:-unknown}"
echo "conda_env         = $CONDA_ENV"
echo "output            = $OUT"
echo "================================================================"

python - <<'PY'
import os, sys
import jax, netket
print("Python:", sys.version.replace("\n", " "))
print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("NetKet:", netket.__version__)
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

CMD=(
  python -u ed_benchmark/vmc_zeroT/run_supercell_vscan.py
  --L1 "$L1" --L2 "$L2"
  --seed "$SEED"
  --mode "$MODE"
  --direction "$DIRECTION"
  --V-min "$V_MIN" --V-max "$V_MAX" --V-step "$V_STEP"
  --t "$T" --t1 "$T1" --t2 "$T2"
  --seed-amplitude "$SEED_AMPLITUDE"
  --jastrow-shells "$JASTROW_SHELLS"

  --anchor-iter "$ANCHOR_ITER"
  --anchor-switch1 "$ANCHOR_SWITCH1"
  --anchor-switch2 "$ANCHOR_SWITCH2"
  --anchor-lr "$ANCHOR_LR"
  --anchor-lr-mid "$ANCHOR_LR_MID"
  --anchor-lr-refine "$ANCHOR_LR_REFINE"
  --anchor-diag "$ANCHOR_DIAG"
  --anchor-diag-mid "$ANCHOR_DIAG_MID"
  --anchor-diag-refine "$ANCHOR_DIAG_REFINE"

  --cont-iter "$CONT_ITER"
  --cont-switch1 "$CONT_SWITCH1"
  --cont-switch2 "$CONT_SWITCH2"
  --cont-lr "$CONT_LR"
  --cont-lr-mid "$CONT_LR_MID"
  --cont-lr-refine "$CONT_LR_REFINE"
  --cont-diag "$CONT_DIAG"
  --cont-diag-mid "$CONT_DIAG_MID"
  --cont-diag-refine "$CONT_DIAG_REFINE"

  --train-samples-early "$TRAIN_SAMPLES_EARLY"
  --train-samples-late "$TRAIN_SAMPLES_LATE"
  --train-samples-refine "$TRAIN_SAMPLES_REFINE"
  --eval-samples "$EVAL_SAMPLES"
  --n-chains "$NCHAINS"
  --n-discard "$NDISCARD"
  --sweep-size "$SWEEP_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --qgt-chunk-size "$QGT_CHUNK_SIZE"
  --cg-tol "$CG_TOL"
  --cg-maxiter "$CG_MAXITER"
  --rng-seed "$RNG_SEED"
  --output "$OUT"
)

echo "[command]"
printf ' %q' "${CMD[@]}"
echo
echo "================================================================"

# -u makes Python stdout unbuffered, so scan-stage messages appear promptly.
srun "${CMD[@]}"

echo "================================================================"
echo "finish_time = $(date -Is)"
echo "seed        = $SEED"
echo "================================================================"
