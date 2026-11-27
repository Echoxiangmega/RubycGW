#!/bin/bash
#SBATCH -J ruby_quarter
#SBATCH --array=0-13
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/ruby_quarter_%A_%a.log
#SBATCH --partition=C064M0256G
#SBATCH --nodes=1
#SBATCH --qos=low

# Submit from the repository root after `mkdir -p logs`.
# One array task follows one seed across the entire V sequence.
set -euo pipefail

REPO="${REPO:-$HOME/RubycGW}"
CONDA_ENV="${CONDA_ENV:-rubycgw}"
CONDA_BASE="${CONDA_BASE:-}"
OUT="${OUT:-$REPO/results/vscan_quarter_L2}"

L1="${L1:-2}"
L2="${L2:-2}"
T="${T:-1.0}"
T1="${T1:--0.5}"
T2="${T2:-0.5}"
SEED_AMPLITUDE="${SEED_AMPLITUDE:-0.09}"
V_MIN="${V_MIN:-0.9}"
V_MAX="${V_MAX:-4.5}"
V_STEP="${V_STEP:-0.2}"
MODE="${MODE:-continuation}"
DIRECTION="${DIRECTION:-up}"

ANCHOR_ITER="${ANCHOR_ITER:-300}"
ANCHOR_SWITCH1="${ANCHOR_SWITCH1:-130}"
ANCHOR_SWITCH2="${ANCHOR_SWITCH2:-240}"
ANCHOR_LR="${ANCHOR_LR:-0.03}"
ANCHOR_LR_MID="${ANCHOR_LR_MID:-0.02}"
ANCHOR_LR_REFINE="${ANCHOR_LR_REFINE:-0.008}"
ANCHOR_DIAG="${ANCHOR_DIAG:-0.05}"
ANCHOR_DIAG_MID="${ANCHOR_DIAG_MID:-0.05}"
ANCHOR_DIAG_REFINE="${ANCHOR_DIAG_REFINE:-0.02}"
CONT_ITER="${CONT_ITER:-170}"
CONT_SWITCH1="${CONT_SWITCH1:-80}"
CONT_SWITCH2="${CONT_SWITCH2:-135}"
CONT_LR="${CONT_LR:-0.015}"
CONT_LR_MID="${CONT_LR_MID:-0.008}"
CONT_LR_REFINE="${CONT_LR_REFINE:-0.003}"
CONT_DIAG="${CONT_DIAG:-0.05}"
CONT_DIAG_MID="${CONT_DIAG_MID:-0.03}"
CONT_DIAG_REFINE="${CONT_DIAG_REFINE:-0.02}"

TRAIN_SAMPLES_EARLY="${TRAIN_SAMPLES_EARLY:-1024}"
TRAIN_SAMPLES_LATE="${TRAIN_SAMPLES_LATE:-2048}"
TRAIN_SAMPLES_REFINE="${TRAIN_SAMPLES_REFINE:-8192}"
EVAL_SAMPLES="${EVAL_SAMPLES:-16384}"
MAPS_EVERY="${MAPS_EVERY:-6}"
NCHAINS="${NCHAINS:-32}"
NDISCARD="${NDISCARD:-16}"
SWEEP_SIZE="${SWEEP_SIZE:-24}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
QGT_CHUNK_SIZE="${QGT_CHUNK_SIZE:-128}"
CG_TOL="${CG_TOL:-1e-5}"
CG_MAXITER="${CG_MAXITER:-100}"
JASTROW_SHELLS="${JASTROW_SHELLS:-4}"
RNG_SEED="${RNG_SEED:-2026}"

# These control only the SDW/Pfaffian CPU branches. The original eight
# Slater/backflow branches use the optimizer and sample counts above.
COMPETING_TRAIN_SAMPLES="${COMPETING_TRAIN_SAMPLES:-512}"
COMPETING_EVAL_SAMPLES="${COMPETING_EVAL_SAMPLES:-4096}"
COMPETING_REFINE_STEPS="${COMPETING_REFINE_STEPS:-9}"
COMPETING_PAIR_SAMPLES="${COMPETING_PAIR_SAMPLES:-128}"
COMPETING_WORKERS="${COMPETING_WORKERS:-1}"

SEEDS=(normal sd isd lc_same lc_opposite random ps_co mix_co_lc_opposite
       sdw pf_s pf_dplus pf_dminus pf_f_ab pf_f_nnn)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then
  echo "ERROR: array task $TASK_ID outside 0..$((${#SEEDS[@]}-1))" >&2
  exit 2
fi
SEED="${SEEDS[TASK_ID]}"

if (( L1 % 2 != 0 || L2 % 2 != 0 )); then
  echo 'ERROR: both L1 and L2 must be even for 1/4 filling and all three M points' >&2
  exit 2
fi
if [[ "$T1" != "-$T2" ]]; then
  python - "$T1" "$T2" <<'PY'
import math,sys
if not math.isclose(float(sys.argv[1]),-float(sys.argv[2]),abs_tol=1e-12):
    raise SystemExit('ERROR: the current Pfaffian pseudospin seeds require t1=-t2')
PY
fi

mkdir -p "$REPO/logs" "$OUT"
cd "$REPO"
if [[ -n "$CONDA_BASE" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo 'ERROR: conda not visible; set CONDA_BASE' >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "seed=$SEED L=${L1}x${L2} Ns=$((18*L1*L2)) Nf=$((18*L1*L2/4))"
echo "t1=$T1 t2=$T2 V=$V_MIN:$V_STEP:$V_MAX $MODE/$DIRECTION OUT=$OUT"
echo "start=$(date -Is)"

CMD=(python -u ed_benchmark/vmc_zeroT/run_supercell_vscan.py
  --L1 "$L1" --L2 "$L2" --filling 1/4 --seed "$SEED"
  --mode "$MODE" --direction "$DIRECTION"
  --V-min "$V_MIN" --V-max "$V_MAX" --V-step "$V_STEP"
  --t "$T" --t1 "$T1" --t2 "$T2"
  --seed-amplitude "$SEED_AMPLITUDE" --jastrow-shells "$JASTROW_SHELLS"
  --anchor-iter "$ANCHOR_ITER" --anchor-switch1 "$ANCHOR_SWITCH1"
  --anchor-switch2 "$ANCHOR_SWITCH2" --anchor-lr "$ANCHOR_LR"
  --anchor-lr-mid "$ANCHOR_LR_MID" --anchor-lr-refine "$ANCHOR_LR_REFINE"
  --anchor-diag "$ANCHOR_DIAG" --anchor-diag-mid "$ANCHOR_DIAG_MID"
  --anchor-diag-refine "$ANCHOR_DIAG_REFINE"
  --cont-iter "$CONT_ITER" --cont-switch1 "$CONT_SWITCH1"
  --cont-switch2 "$CONT_SWITCH2" --cont-lr "$CONT_LR"
  --cont-lr-mid "$CONT_LR_MID" --cont-lr-refine "$CONT_LR_REFINE"
  --cont-diag "$CONT_DIAG" --cont-diag-mid "$CONT_DIAG_MID"
  --cont-diag-refine "$CONT_DIAG_REFINE"
  --train-samples-early "$TRAIN_SAMPLES_EARLY"
  --train-samples-late "$TRAIN_SAMPLES_LATE"
  --train-samples-refine "$TRAIN_SAMPLES_REFINE" --eval-samples "$EVAL_SAMPLES"
  --maps-every "$MAPS_EVERY" --n-chains "$NCHAINS" --n-discard "$NDISCARD"
  --sweep-size "$SWEEP_SIZE" --chunk-size "$CHUNK_SIZE"
  --qgt-chunk-size "$QGT_CHUNK_SIZE" --cg-tol "$CG_TOL"
  --cg-maxiter "$CG_MAXITER" --rng-seed "$RNG_SEED"
  --competing-train-samples "$COMPETING_TRAIN_SAMPLES"
  --competing-eval-samples "$COMPETING_EVAL_SAMPLES"
  --competing-refine-steps "$COMPETING_REFINE_STEPS"
  --competing-workers "$COMPETING_WORKERS"
  --competing-pair-samples "$COMPETING_PAIR_SAMPLES"
  --output "$OUT")
printf '[command]'; printf ' %q' "${CMD[@]}"; echo
srun "${CMD[@]}"
echo "finish=$(date -Is) seed=$SEED"
