#!/bin/bash
#SBATCH --job-name=ruby-fast-vap
#SBATCH --output=fast_vap_%j.out
#SBATCH --error=fast_vap_%j.err
#SBATCH --time=12:00:00
# Adjust partition, GPU, memory and Python environment for your cluster.
set -euo pipefail
: "${RAW_STAGE:?Set RAW_STAGE to a completed large_raw_backflow__SEED directory}"
: "${OUT_DIR:?Set OUT_DIR to a new result directory}"
cd /path/to/RubycGW
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python ed_benchmark/vmc_zeroT/run_supercell_vap_fast.py \
  --L1 2 --L2 2 --nf 24 --t 1 --t1 0.5 --t2 -0.5 --V 5.5 \
  --seed lc_opposite --raw-stage "$RAW_STAGE" --output "$OUT_DIR" \
  --irreps A1 A2 E --no-screen-irreps --restore-tr \
  --tint-iter 50 --spatial-iter 30 --tr-iter 30 \
  --samples 2048 --spatial-samples 512 --eval-samples 8192 \
  --checkpoint-every 10 --sampler projected-cache --fast-local \
  --trainer shared_sr --strict --check-gradient
