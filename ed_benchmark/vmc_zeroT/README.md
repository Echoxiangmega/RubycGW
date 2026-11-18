# Hex18 ED vs Varma-style VMC benchmark

This directory benchmarks the existing 18-site \`ring3\` PBC Ruby cluster against a Varma-style complex Slater and Slater × density-Jastrow variational wavefunction.

## Model

The physical Hamiltonian is exactly the one used by \`ed_benchmark/native_ed\`:

\[
H=\sum_{\langle ij\rangle}t_{ij}(c_i^\dagger c_j+c_j^\dagger c_i)
+V\sum_{\langle ij\rangle\in\triangle}n_i n_j .
\]

The hopping convention is the native-ED convention: positive \`t,t1,t2\` enter as \(+t(c_i^\dagger c_j+h.c.)\).

## Ansatz

\[
\Psi(\mathbf n)=\det M_{\rm occ}(\mathbf n)
\exp\left[\sum_s v_s\frac12\delta n^T W_s\delta n\right].
\]

The full complex orbital matrix \(M\) is optimized. The density Jastrow \(W_s\) is grouped by PBC distance shell. Default physical seeds are

- \`normal\`
- \`sd\`
- \`isd\`
- \`lc_same\`
- \`lc_opposite\`

The SD/ISD seeds are opposite signs of the same Ruby real-bond form factor. Therefore \(S=\langle O^2\rangle\) alone cannot distinguish SD from ISD; the output also records the signed \`B_star_mean\`.

## Compared observables

The same exact operator code is applied to ED and to the final enumerated VMC state:

- total energy and energy per site;
- \`CO_qmax\` and its momentum;
- \`B_star_mean\`, \`S_BOW_star\`;
- \`J_same_mean\`, \`J_opposite_mean\`;
- \`S_LC_same\`, \`S_LC_opposite\`;
- connected versions of the BOW/LC diagnostics;
- VMC overlap with the complete ED ground-state manifold.

For ED, exactly degenerate ground vectors are treated as an equal incoherent ensemble for structure factors.

## Important benchmark design

VMC optimization is Monte Carlo. After optimization, Hex18 is small enough that the trained wavefunction is evaluated on every fixed-\(N\) configuration. Final benchmark energies and structure factors therefore contain ansatz/optimization error but no Monte-Carlo estimator noise.

The stochastic VMC estimate and its error bar are saved separately as \`mc_E\` and \`mc_E_error\`.

Before the first VMC run, the NetKet sparse Hamiltonian is reordered into the native-ED basis and compared element-by-element. A basis/sign mismatch stops the calculation.

## Install

From the repository root:

\`\`\`bash
python -m pip install -r ed_benchmark/vmc_zeroT/requirements.txt
\`\`\`

## Core ED-side regression

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_benchmark_core.py
\`\`\`

For the repository reference point

\`\`\`text
ring3, Ns=18, Nf=6, t=1, t1=t2=0.5, V=1
E0 = -7.9938493221581854
\`\`\`

the regression checks the same native-ED value and the finite-size ground manifold.

## First VMC benchmark

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_benchmark.py \
  --nf 6 \
  --t 1 --t1 0.5 --t2 0.5 --V 1 \
  --ansatz both \
  --seeds normal sd isd lc_same lc_opposite \
  --n-samples 4096 \
  --n-iter 300 \
  --output results/vmc_hex18_N6_V1
\`\`\`

Hex18 contains three primitive cells, so the filling per six-site primitive cell is

\[
n=N_f/3.
\]

For example, \(N_f=6\) corresponds to \(n=2\).

## Output

\`\`\`text
OUTPUT/
  settings.json
  benchmark_all.csv
  benchmark_best.csv
  benchmark_best.json
  slater__normal/
  slater__sd/
  ...
  slater_jastrow__lc_opposite/
      summary.json
      history.json
      final_state.npz
\`\`\`

The main comparison file is \`benchmark_best.csv\`, containing ED, best Slater, and best Slater-Jastrow rows.

Key timing columns are

- \`time_build_s\`
- \`time_jit_warmup_s\`
- \`time_solve_or_opt_s\`
- \`time_mc_final_eval_s\`
- \`time_postprocess_s\`
- \`time_total_s\`
- \`time_ratio_vs_ED\`

JAX compilation is kept separate so that a cold first VMC run is not confused with the steady optimization cost.

## Recommended first use

Start with selected \((N_f,V)\) points where HF predicts different competitors. The purpose of this Hex18 stage is to measure how much Slater and Slater-Jastrow recover the exact energy and the CO/BOW/LC channel structure before scaling VMC to clusters that ED cannot solve.


## Projection-after-variation against the exact ED symmetry sector

For a **nondegenerate** Hex18 ED ground state, an already completed VMC run can
be projected into the exact ordinary finite-torus space-group sector without
rerunning VMC. The first implementation uses the verified 3 translations
times ordinary \(C_{3v}\) point group (18 many-body symmetry operations).

First test the second-quantized permutations and their fermionic signs:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_symmetry_projection.py
\`\`\`

Then postprocess an existing run, for example:

\`\`\`bash
python ed_benchmark/vmc_zeroT/postprocess_symmetry.py \
  --input results/vmc_hex18_N6_V2_jastrow
\`\`\`

This writes:

\`\`\`text
ed_symmetry_sector.json
symmetry_projection_all.csv
symmetry_projection_best.csv
*/projected_state.npz
*/projected_summary.json
\`\`\`

For every VMC state it reports the target-sector weight

\[
w_\Gamma=\langle\Psi|P_\Gamma|\Psi\rangle,
\]

then exactly reevaluates \(E\), ED-ground-space fidelity, and the same
CO/BOW/LC diagnostics for \(P_\Gamma|\Psi\rangle\).

Interpretation:

- small \(w_\Gamma\) but high projected fidelity means much of the original
  error was symmetry-sector contamination;
- \(w_\Gamma\approx1\) with little improvement after projection means the
  Slater/Jastrow variational manifold itself is missing important correlations.

This PAV step restores finite-cluster symmetry for a fair ED comparison. It
does **not** remove ordinary finite-size effects of the 18-site torus.

For \(t_1=-t_2\), the model also has gauge-compensated point operations such as
the signed \(C_6\) defined in \`native_ed.geometry\`. Those are deliberately
not mixed into this first ordinary-space-group PAV and can be added after this
baseline is validated.
