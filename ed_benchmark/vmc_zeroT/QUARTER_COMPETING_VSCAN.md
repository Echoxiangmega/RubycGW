# Quarter-filled SDW and pairing candidates in the ring3 V scan

The established `run_supercell_vscan.py` still runs its eight Slater/backflow
branches unchanged. It also accepts four new seeds: `sdw`, `pf_s`, `pf_dplus`,
and `pf_dminus`. Each seed has its own continuation branch and writes the same
`branch_scan.csv` layout consumed by the existing collector and analyzer.
The analyzer groups the two time-reversal partner d branches under the same
candidate-phase label while retaining their separate seed energies.

## Geometry

The ring3 torus has `Ns=18*L1*L2` microscopic sites. One-quarter filling has
`Nf=9*L1*L2/2`; an integer particle number only requires an even product.
The three primitive-lattice M wavevectors all fit when **both** L1 and L2 are
even. This also makes Nf even, which the fixed-N Pfaffian singlet needs. For a
square `LxL` torus, choose an even L. The common 3x3 ring3 torus does not
have one-quarter filling with an integer particle number or all three M points.
The 2x2 ring3 torus has 72 sites and 18 particles. It is a different finite
system from the 24-site primitive 2x2 benchmark, which has 6 particles.

## Wavefunctions

The SDW branch uses the chiral two-orbital E basis on each primitive triangle,
with the three sign functions `(-1)^x`, `(-1)^y`, `(-1)^(x+y)` as the M-point
sources for x/y/z pseudospin. Its auxiliary Slater determinant optimizes
`mx`, `my`, and `mz` independently and tries both B-sublattice orientations.
Initial states cover single-M, double-M and noncoplanar three-M patterns. The scan
measures all three M-order sources on the final state; an inconclusive
orientation is reported as `SDW_M_orientation_uncertain`.

The saved `M_pseudospin_x/y/z`, their block errors, and the combined
`M_pseudospin_xy_amplitude` and `M_pseudospin_z_amplitude` are **measured
one-point source expectations per triangle**. The optimized auxiliary fields
`mx/my/mz` are separate columns and are not substituted for observables.

The `pf_*` branches form spinless, fixed-particle-number, bordered Pfaffian
BCS states with the effective A-B orbital-singlet pair matrix. Their bond
phases are `(1,1,1)` for s, `(1,omega,omega²)` for d+id and its complex
conjugate for d-id. Each branch optimizes the gap amplitude and chemical
potential. All four new branches use the same seven bond-aware density
Jastrow channels, optimized stochastically; the SDW and Pf branches do **not**
have the fast Slater-specific backflow correlator. They are restricted
variational candidates. A lower energy is useful evidence, but the phase
diagram must account for Monte Carlo errors, optimization quality, and
unequal variational flexibility. A Pfaffian candidate lowering energy alone
does not establish off-diagonal long-range order.

At fixed particle number the anomalous mean `<cc>` vanishes exactly.
Consequently, the final chain also measures the equal-time, maximally
separated pair correlator `<Delta_R^dag Delta_R'>` on a specified pair of
triangle cells, projected into s, d+id and d-id. The result columns are
`pair_far_s/dplus/dminus_{real,imag,abs,error}` together with
`pair_far_cells`, `pair_far_distance`, and `pair_far_samples`. These are
single-pair finite-size diagnostics, not a proof of superconducting long-range
order; do finite-size scaling and distance averaging before assigning a
thermodynamic phase. Adjust `--competing-pair-samples`, or set it to zero to
disable the additional estimator. The new analyzer writes
`candidate_order_parameters.csv`, `competing_order_diagnostics.png`, and the
winner observables in `phase_table.csv` alongside its older energy plots.

The new restricted optimizer is a CPU reference implementation: it scans
multiple initial structural parameters, optimizes Jastrow via stochastic
gradient/SR, refines structural parameters, and measures final energies on
independent samples. Defaults (512 training and 4096 final samples) are for
exploratory scans. Tight energy comparisons require larger
sample counts, multiple RNG seeds, and converged parameter search. Its
original Slater and paired counterparts are evaluated with the same native
Hamiltonian; `tests/check_quarter_competing_vscan.py` verifies representative
local energies against the exact 24-site Fock matrix.

At 4x4 the first V has 12 independent SDW initial states (six for each Pf
branch), then `refine_steps` rounds of several further sample calls, before
the first energy summary is saved. This work is sampling and optimization,
not a long Hamiltonian construction. The CPU sampler caches the inverse and
the seven Jastrow density fields for every state, uses an exact O(7) Jastrow
ratio for each proposed hop, and incrementally updates those fields after an
accepted hop. It rebuilds the occupied inverse after accepted hops, so this
is still distinct from the JAX/Woodbury sampler of the original eight seeds.
The job log prints initial-state and refinement progress before the first V
summary. Updating the code cannot speed up a Python job that is already
running; stop and resubmit only the four new array tasks after pulling the
new code. Completed V-point summaries are resumed with the same OUT and
unchanged scan settings; an incomplete V point starts again.

## Run

From the repository root, for a single branch:

```bash
OPENBLAS_NUM_THREADS=1 python ed_benchmark/vmc_zeroT/run_supercell_vscan.py \
  --L1 2 --L2 2 --filling 1/4 --t1 -0.5 --t2 0.5 \
  --seed sdw --V-values 1 1.5 2.5 4 \
  --competing-train-samples 256 --competing-eval-samples 4096 \
  --competing-refine-steps 10 --competing-pair-samples 128 \
  --output results/vscan_quarter
```

Change `--seed` to `pf_s`, `pf_dplus`, or `pf_dminus`. Run the eight ordinary
branches with the same filling, hopping, and output root. The old GPU array
script takes `FILLING=1/4,T1=-0.5,T2=0.5,OUT=...` as exported environment
variables; the new CPU array script is `slurm_vscan_quarter_competing_cpu.sh`.
Their `--output` roots must agree. Mixing Hamiltonians in one master file is
rejected by the collector.

SLURM arrays, submitted separately from the repository root after
`mkdir -p logs`:

```bash
sbatch --export=ALL,L1=2,L2=2,FILLING=1/4,T1=-0.5,T2=0.5,OUT=results/vscan_quarter \
  ed_benchmark/vmc_zeroT/slurm_vscan_array.sh
sbatch --export=ALL,L1=2,L2=2,T1=-0.5,T2=0.5,OUT=results/vscan_quarter,PAIR_SAMPLES=128 \
  ed_benchmark/vmc_zeroT/slurm_vscan_quarter_competing_cpu.sh
```

The first script retains the original eight Slater/backflow seeds on GPU;
the second launches four Slater/Pfaffian candidate seeds in both scan
directions (eight CPU tasks). Their stochastic optimizers and correlators
differ; energies compare valid variational candidates but not equal ansatz
flexibility. Run `collect_supercell_vscan.py` only after both arrays finish,
then `analyze_supercell_vscan.py`. Inspect `candidate_order_parameters.csv`
even when a different seed wins, since `phase_table.csv` includes only
the winning row's extra diagnostics.

```bash
python ed_benchmark/vmc_zeroT/collect_supercell_vscan.py \
  --root results/vscan_quarter
python ed_benchmark/vmc_zeroT/analyze_supercell_vscan.py \
  --root results/vscan_quarter --energy-reference normal
```

The current pairing matrix assumes `t1=-t2`. A new derivation of its orbital
channel is needed before applying these `pf_*` seeds to `t1=t2`.

## One CPU array matching the C064M0256G cluster script

For the previous `3x3, NF=81, t1=t2=-0.5` script, change to `L1=L2=2`
and `--filling 1/4` (`Ns=72, Nf=18`), and use `t1=-0.5,t2=+0.5` for the
implemented pairing channel. `slurm_vscan_quarter_all_cpu.sh` keeps the
user's partition, qos, three CPUs, conda initialization and `srun`. It has
`#SBATCH --array=0-11`: eight original Slater/backflow branches, then SDW
and the three Pfaffian branches. The old Slater optimizer receives the old
anchor/continuation schedules. The four new branches receive separate
`COMPETING_*` controls because their stochastic optimizer is different.

From the repository root, first `mkdir -p logs`, then:

```bash
sbatch ed_benchmark/vmc_zeroT/slurm_vscan_quarter_all_cpu.sh
```

Override `REPO`, `OUT`, `L1`, `L2`, `V_MIN`, `V_MAX`, `V_STEP`, or
`DIRECTION` using `sbatch --export=ALL,...` as in the old script. Reuse the
same OUT for all 12 tasks. Run a second array with `DIRECTION=down` to
cross-check continuation hysteresis.

After all array tasks finish, one command both collects and analyzes:

```bash
python ed_benchmark/vmc_zeroT/analyze_quarter_vscan.py \
  --root results/vscan_quarter_L2 \
  --V-min 0.9 --V-max 4.5 --V-step 0.2
```

The wrapper checks that all 12 branches exist at every requested V for the
selected scan direction, calls `collect_supercell_vscan.py` and then
`analyze_supercell_vscan.py`. It rejects incomplete scans by default;
`--allow-incomplete` produces an explicitly partial preview. Default analysis
is `continuation/up`; use `--direction down` for a down scan. Outputs include
`vscan_master.csv`, `analysis/L2x2/phase_table.csv`,
`candidate_order_parameters.csv`, and `competing_order_diagnostics.png`.
