#!/bin/bash
#SBATCH -J ruby3x3_VAP
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --partition=C064M0256G
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --output=logs/ruby3x3_vap_%A_%a.log

# Submit from the repository root:
#   cd "$HOME/RubycGW"; mkdir -p logs
#   sbatch ed_benchmark/vmc_zeroT/slurm_vap_3x3_seed_array.sh
#
# One array task = ONE seed, sequential V values per task.
# To select fewer seeds, edit SEEDS below and set --array=0-(nseeds-1),
# e.g. sbatch --array=0-2 ed_benchmark/vmc_zeroT/slurm_vap_3x3_seed_array.sh.
set -euo pipefail

REPO="${REPO:-$HOME/RubycGW}"
CONDA_ENV="${CONDA_ENV:-rubycgw}"
CONDA_BASE="${CONDA_BASE:-}"
RAW_ROOT="${RAW_ROOT:-$REPO/results/vscan_3x3_down}"
VAP_ROOT="${VAP_ROOT:-$REPO/results/vap_3x3_down_array}"

L1="${L1:-3}"
L2="${L2:-3}"
NF="${NF:-54}"
T="${T:-1.0}"
T1="${T1:--0.5}"
T2="${T2:--0.5}"
SEED_AMPLITUDE="${SEED_AMPLITUDE:-0.09}"

# Start with V=3.6; expand to 0.8..4.0 after one-parameter testing.
V_MIN="${V_MIN:-3.6}"
V_MAX="${V_MAX:-3.6}"
V_STEP="${V_STEP:-0.1}"
MODE="${MODE:-continuation}"
DIRECTION="${DIRECTION:-down}"

# Optional override, e.g. VAP_SEEDS='sd isd lc_same lc_opposite'.
# With an override, submit --array=0-(number of selected seeds - 1).
if [[ -n "${VAP_SEEDS:-}" ]]; then
    read -r -a SEEDS <<< "$VAP_SEEDS"
else
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
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM array index required}"
if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then
    echo "ERROR: array index $TASK_ID outside seed list" >&2
    exit 2
fi

TINT_ITER="${TINT_ITER:-20}"
SPATIAL_ITER="${SPATIAL_ITER:-15}"
TR_ITER="${TR_ITER:-15}"
TINT_LR="${TINT_LR:-0.01}"
SPATIAL_LR="${SPATIAL_LR:-0.01}"
TR_LR="${TR_LR:-0.01}"

SAMPLER="${SAMPLER:-standard}"
NCHAINS="${NCHAINS:-32}"
NDISCARD="${NDISCARD:-16}"
TINT_SWEEP="${TINT_SWEEP:-54}"
SPATIAL_SWEEP="${SPATIAL_SWEEP:-27}"
SAMPLER_CHECK="${SAMPLER_CHECK:-3}"
TINT_SAMPLES="${TINT_SAMPLES:-1024}"
SPATIAL_SAMPLES="${SPATIAL_SAMPLES:-512}"
EVAL_INTERMEDIATE="${EVAL_INTERMEDIATE:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4096}"

DIAG_SHIFT="${DIAG_SHIFT:-0.05}"
CG_TOL="${CG_TOL:-1e-5}"
CG_MAXITER="${CG_MAXITER:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
QGT_CHUNK="${QGT_CHUNK:-32}"
OBS_CHUNK="${OBS_CHUNK:-8}"
RNG_SEED="${RNG_SEED:-2026}"
JASTROW_SHELLS="${JASTROW_SHELLS:-4}"

mkdir -p "$REPO/logs" "$VAP_ROOT"
cd "$REPO"

if [[ -n "$CONDA_BASE" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found; set CONDA_BASE" >&2
        exit 2
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "================================================================="
echo "task=$TASK_ID seed=${SEEDS[$TASK_ID]} time=$(date -Is)"
echo "V=$V_MIN:$V_STEP:$V_MAX mode=$MODE direction=$DIRECTION"
echo "Raw=$RAW_ROOT"
echo "VAP=$VAP_ROOT"
echo "sampler=$SAMPLER sweep(Tint/Spatial)=$TINT_SWEEP/$SPATIAL_SWEEP"
echo "samples(Tint/Spatial)=$TINT_SAMPLES/$SPATIAL_SAMPLES"
echo "iterations(Tint/Spatial/TR)=$TINT_ITER/$SPATIAL_ITER/$TR_ITER"
echo "================================================================="

CMD=(
  python -u ed_benchmark/vmc_zeroT/run_supercell_vap_vscan.py
  --L1 "$L1" --L2 "$L2" --nf "$NF"
  --t "$T" --t1 "$T1" --t2 "$T2"
  --V-min "$V_MIN" --V-max "$V_MAX" --V-step "$V_STEP"
  --seeds "${SEEDS[@]}" --array-index "$TASK_ID"
  --mode "$MODE" --direction "$DIRECTION"
  --scan-root "$RAW_ROOT" --output-root "$VAP_ROOT"
  --missing-raw skip
  --seed-amplitude "$SEED_AMPLITUDE"
  --random-seed "$RNG_SEED"
  --jastrow-shells "$JASTROW_SHELLS"
  --irreps A1 A2 E --no-screen-irreps
  --restore-tr --stages tint spatial tr
  --tint-iter "$TINT_ITER" --spatial-iter "$SPATIAL_ITER" --tr-iter "$TR_ITER"
  --tint-lr "$TINT_LR" --spatial-lr "$SPATIAL_LR" --tr-lr "$TR_LR"
  --sampler "$SAMPLER"
  --sampler-check-proposals "$SAMPLER_CHECK"
  --n-chains "$NCHAINS" --n-discard "$NDISCARD"
  --sweep-size "$TINT_SWEEP" --spatial-sweep-size "$SPATIAL_SWEEP"
  --samples "$TINT_SAMPLES" --spatial-samples "$SPATIAL_SAMPLES"
  --eval-samples-intermediate "$EVAL_INTERMEDIATE"
  --eval-samples "$EVAL_SAMPLES"
  --trainer shared_sr
  --diag-shift "$DIAG_SHIFT" --cg-tol "$CG_TOL" --cg-maxiter "$CG_MAXITER"
  --chunk-size "$CHUNK_SIZE" --qgt-chunk "$QGT_CHUNK"
  --observable-chunk "$OBS_CHUNK"
  --fast-local --local-kernel fused --lc-estimator onehop --observables
  --no-check-gradient --no-check-currents
  --checkpoint-every 5 --resume --strict
)

echo "[command]"
printf ' %q' "${CMD[@]}"
echo
srun "${CMD[@]}"
echo "FINISH task=$TASK_ID seed=${SEEDS[$TASK_ID]} time=$(date -Is)"
