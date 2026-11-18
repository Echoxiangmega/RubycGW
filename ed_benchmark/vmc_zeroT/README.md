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


## Variation after projection (VAP)

After the PAV diagnostic, the projected wavefunction can be optimized directly:

\[
\Psi_\Gamma(x;\theta)
=
\sum_{g\in G}\chi_\Gamma(g)^*(U_g\Psi_\theta)(x),
\]

where all 18 symmetry images share the same Slater/Jastrow parameters. This is
a symmetry-restored multi-determinant state generated by projection, not 18
independently parameterized determinants.

For the current reference point, first run only the PAV-best seed:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_vap.py \
  --nf 6 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2 \
  --ansatz slater_jastrow \
  --seeds lc_same \
  --n-samples 4096 \
  --n-iter 300 \
  --output results/vap_hex18_N6_V2
\`\`\`

The first run performs an independent full-basis consistency check: the
NetKet projected initial state must agree with taking the same initial
unprojected Slater state and applying the already validated PAV projector.
If the fermion sign, group-action direction, or character conjugation is
wrong, the run stops before optimization.

Outputs include \`vap_all.csv\`, \`vap_best.csv\`, exact final \(E\), fidelity,
CO/BOW/LC diagnostics, projected-sector residual, stochastic energy error,
and timing. Projected amplitude evaluation is about \(|G|=18\) times heavier
than the unprojected ansatz before JAX vectorization/compilation effects, so
start with one seed.

If the first \`lc_same\` run is stable, add the other competitive starts:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_vap.py \
  --nf 6 --t 1 --t1 0.5 --t2 -0.5 --V 2 \
  --ansatz slater_jastrow \
  --seeds lc_same sd normal isd lc_opposite \
  --n-samples 4096 --n-iter 300 \
  --output results/vap_hex18_N6_V2_multiseed
\`\`\`

The first VAP implementation uses the same ordinary translations x \(C_{3v}\)
sector as PAV. Gauge-compensated \(C_6\) for \(t_1=-t_2\) remains a separate
extension.


## Symmetry-tied reduced one-body parameterization

To test a scalable Varma-style parameterization before moving beyond Hex18,
use the translation+C3 independent auxiliary Hamiltonian:

\[
h_{\rm var}=h_0+\sum_a\lambda_a D_a+
\sum_b\left(x_b K_b^{R}+y_b K_b^{I}\right).
\]

Each \(b\) is one translation+C3 inequivalent bond orbit. The imaginary
coefficient \(y_b\) is retained, so loop-current states are allowed. Mirrors
and time reversal are not imposed. One uniform onsite-energy gauge direction
is removed.

First inspect and test the actual independent parameter count:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_parametric_ansatz.py
\`\`\`

Then run the three reduced families on the same Hex18 benchmark:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_parametric.py \
  --nf 6 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2 \
  --families param_slater param_sj param_vap_sj \
  --seeds normal sd isd lc_same lc_opposite \
  --n-samples 4096 \
  --n-iter 300 \
  --output results/parametric_hex18_N6_V2
\`\`\`

The three families are:

- \`Param-Slater\`: reduced complex one-body Hamiltonian only;
- \`Param-Slater-Jastrow\`: same reduced one-body parameters plus density Jastrow;
- \`Param-VAP-Slater-Jastrow\`: the same reduced parameters optimized after
  projection into the exact Hex18 ED ordinary-space-group sector.

All final energies and observables are again reevaluated by full fixed-\(N\)
enumeration. The main output is:

\`\`\`text
results/parametric_hex18_N6_V2/parametric_best.csv
\`\`\`

To compare the exact ED result, the previous full-orbital VAP result, and all
three reduced families in one five-row table:

\`\`\`bash
python ed_benchmark/vmc_zeroT/compare_reduced.py \
  --full-vap results/vap_hex18_N6_V2_multiseed \
  --parametric results/parametric_hex18_N6_V2
\`\`\`

This writes:

\`\`\`text
results/parametric_hex18_N6_V2/comparison_ed_fullvap_parametric.csv
results/parametric_hex18_N6_V2/comparison_ed_fullvap_parametric.json
\`\`\`

The comparison includes \(E\), \(E-E_{\rm ED}\), fidelity, CO/BOW/LC
structure factors, opposite/same LC ratio, target-sector weight, variational
parameter count, and wall time.


### Degenerate ED ground multiplets

The reduced-parameter benchmark now supports exactly degenerate Hex18 ED
ground spaces. For a unique ground state, VAP keeps the previous ordinary
translations x \(C_{3v}\) projector. For a degenerate ground multiplet, the
code automatically decomposes the complete ED ground space into the abelian
ordinary translations x \(C_3\) one-dimensional character sectors and runs
\`Param-VAP-Slater-Jastrow\` separately for every sector present in the ED
ground manifold. The reported fidelity is always the overlap with the entire
exact ED ground subspace.

Before running a degenerate point such as \(N_f=3,V=2.5,t_1=0.5,t_2=-0.5\),
test the decomposition:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_degenerate_ground_sectors.py
\`\`\`

Then \`run_parametric.py\` can be used normally. If the ED ground multiplet
fills all states requested by \`--ed-k\`, the script stops and asks for a
larger \`--ed-k\` rather than silently using an incomplete multiplet.

Because VAP runs once per ED ground symmetry sector, the VAP wall time scales
approximately with the number of such sectors in addition to the requested
seed count.


### Stable reduced-orbital optimization at single-particle degeneracies

The first reduced implementation differentiated through
\`eigh(h_var)\` at every VMC step. This is ill-defined when the auxiliary
single-particle spectrum is degenerate (for example the Hex18 \(N_f=3\)
benchmark), and can produce NaN eigenvector gradients.

The current implementation uses the same translation+C3 independent one-body
generators only to construct the occupied-virtual tangent space once:

\[
\delta Q_a=(1-P_0)G_aQ_0 .
\]

Real-linearly dependent directions are removed by SVD. VMC then optimizes only
the remaining reduced orbital-rotation parameters through

\[
Q(\theta)=\mathrm{QR}\!\left[Q_0+\sum_a\theta_a\delta Q_a\right],
\]

so the differentiable path contains no eigendecomposition. The program prints
both the original symmetry-generator count and the active tangent rank.
The SR Jacobian is explicitly evaluated in complex mode because the parameters
are real but the Slater wavefunction can be complex.


### Integrated full-orbital VAP baseline on Hex18

\`run_parametric.py\` also accepts:

\`\`\`text
full_vap_sj
\`\`\`

This is the original high-flexibility Hex18 ansatz: a full complex
\(N_s\times N_f\) orbital matrix, real density Jastrow, and variation after
projection. It is not intended as the scalable large-cluster parameterization;
it is retained as the Hex18 accuracy baseline.

For a unique ED ground state it uses the ordinary \(T\times C_{3v}\) projector.
For a degenerate ED ground multiplet it runs separately in every
\(T\times C_3\) sector present in the exact ground space, just like the reduced
VAP branch. Therefore the number of full-VAP jobs is
(number of seeds) x (number of ED ground symmetry sectors).

For example, on the \(N_f=3,V=2.5\) point, start with one seed:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_parametric.py \
  --nf 3 --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --families full_vap_sj \
  --seeds normal \
  --n-samples 4096 --n-iter 300 --ed-k 12 \
  --output results/fullvap_hex18_N3_V2p5
\`\`\`

To run all four comparison families in a single output directory:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_parametric.py \
  --nf 3 --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --families param_slater param_sj param_vap_sj full_vap_sj \
  --seeds normal sd isd lc_same lc_opposite \
  --n-samples 4096 --n-iter 300 --ed-k 12 \
  --output results/hex18_N3_V2p5_allfamilies
\`\`\`

The best full-VAP row is written together with the reduced families to
\`parametric_best.csv\`.


### Resuming interrupted reduced/VAP scans and unreachable sectors

For degenerate ED ground manifolds, a fixed reduced orbital tangent manifold
can be exactly orthogonal to one of the target symmetry sectors. In that case
increasing the initialization jitter cannot create a nonzero projected state.
Such a seed-sector branch is now recorded as

\`\`\`text
status = unreachable_target_sector
\`\`\`

and skipped instead of aborting the whole scan. This is a property of the
reduced variational manifold, not a Monte Carlo failure.

Long interrupted jobs can be resumed without recomputing completed branches:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_parametric.py ... --resume
\`\`\`

Existing per-run \`summary.json\` files are reused. Skipped unreachable branches
are also remembered on subsequent resumes.


## ED / PAV / full-VAP / C3-constrained PAV/VAP comparison

Use \`run_projection_comparison.py\` to compare five methods on exactly the same
Hex18 Hamiltonian and Jastrow-shell definition:

1. \`ED\`
2. \`PAV-Slater-Jastrow\`: full unrestricted complex Slater x Jastrow,
   optimized without symmetry restoration and projected afterward.
3. \`Full-VAP-Slater-Jastrow\`: the same full orbital matrix optimized after
   projection.
4. \`C3-PAV-Slater-Jastrow\`: the base occupied subspace is constrained to
   preserve ordinary C3 but primitive translations are free to break; it is
   optimized first and then projected into the full ED target sector.
5. \`C3-VAP-Slater-Jastrow\`: the same C3-constrained, translation-breaking
   base ansatz optimized directly after the full target-sector projection.

The C3-constrained ansatz does not use an auxiliary-Hamiltonian eigendecomposition
during optimization and does not use the fixed-tangent approximation. The
occupied orbitals are arbitrary complex subspaces inside the one-particle C3
eigenspaces. The code prints the discrete C3 occupation counts and the physical
orbital degrees of freedom for every branch.

The full target symmetry sector is:
- unique ED ground: ordinary translations x C3v;
- degenerate ED ground multiplet: the ordinary T x C3 sectors actually present
  in the exact ground space.

Every ED full sector is printed with an explicit finite-torus momentum label
\`Gamma\`, \`K\`, or \`K'\`, its reciprocal coordinates, and the C3 character.

First test the C3 orbital construction:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_c3_constrained_ansatz.py
\`\`\`

For the current Nf=3, V=2.5 point, start with one seed because the VAP branches
multiply by the number of ED ground symmetry sectors:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_projection_comparison.py \
  --nf 3 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --seeds normal \
  --n-samples 4096 \
  --n-iter 300 \
  --ed-k 12 \
  --output results/projection_compare_hex18_N3_V2p5
\`\`\`

Then add more seeds with \`--resume\`:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_projection_comparison.py \
  --nf 3 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --seeds normal sd isd lc_same lc_opposite \
  --n-samples 4096 \
  --n-iter 300 \
  --ed-k 12 \
  --output results/projection_compare_hex18_N3_V2p5 \
  --resume
\`\`\`

Main outputs:

\`\`\`text
ed_projection_sectors.json
projection_comparison_all.csv
projection_comparison_best.csv
projection_comparison_best.json
\`\`\`

The best table contains \(E\), \(E-E_{\rm ED}\), ground-subspace fidelity,
exact post-enumeration energy variance, CO/BOW/LC observables, target momentum,
and C3 character.

For this first comparison all methods deliberately use the same distance-shell
density Jastrow. Bond-aware \(v_t,v_{t1},v_{t2}\) parameters should be tested
as a separate controlled extension rather than mixed into the C3 constraint
comparison.


## Full PAV / Full VAP comparison with bond-aware Jastrow

The current recommended Hex18 benchmark deliberately drops the C3-constrained
reference ansatz and compares only:

\`\`\`text
ED
PAV-Slater-Jastrow
Full-VAP-Slater-Jastrow
\`\`\`

using the same unrestricted full complex Slater matrix in both variational
methods.

The recommended Jastrow mode is now \`bond_aware\`. Its first three independent
density-correlation parameters are

\[
v_t,\qquad v_{t1},\qquad v_{t2},
\]

acting only on the corresponding Ruby graph-bond pairs. Graph bonds are then
excluded from all longer-range channels. The remaining non-graph pairs are
grouped by minimum-image Euclidean distance.

Thus, with

\`\`\`text
--jastrow-mode bond_aware --jastrow-shells 4
\`\`\`

there are \(3+4=7\) real Jastrow parameters: three bond-type channels plus
four non-bond distance shells. The old pure-distance parameterization remains
available with \`--jastrow-mode distance\`.

Test the channel partition first:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_bond_aware_jastrow.py
\`\`\`

For example:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_projection_comparison.py \
  --nf 3 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --seeds normal sd isd lc_same lc_opposite \
  --jastrow-mode bond_aware \
  --jastrow-shells 4 \
  --n-samples 4096 \
  --n-iter 300 \
  --ed-k 12 \
  --output results/full_pav_vap_bondjastrow_N3_V2p5
\`\`\`

The script prints and saves the ED momentum sector labels and writes:

\`\`\`text
jastrow_channels.json
ed_projection_sectors.json
comparison_all.csv
comparison_best.csv
comparison_best.json
\`\`\`

For \`bond_aware\`, \`--jastrow-shells\` means the number of additional
non-bond distance shells, not the total number of Jastrow parameters.

Use \`--resume\` to reuse completed raw-SJ and full-VAP runs. Resume refuses to
reuse a run whose saved \`jastrow_mode\` differs from the requested mode.


## Six-way Hex18 comparison with exact one-Lanczos postprocessing

The recommended projection benchmark now compares exactly six rows:

\`\`\`text
ED
Raw-Slater-Jastrow
PAV-Slater-Jastrow
PAV-Slater-Jastrow+1LS
Full-VAP-Slater-Jastrow
Full-VAP-Slater-Jastrow+1LS
\`\`\`

\`Raw-Slater-Jastrow\` is the optimized unrestricted full complex Slater x
Jastrow state before any symmetry projection.

For PAV and Full VAP, the optional one-Lanczos result is not another Monte
Carlo optimization. On Hex18 the complete fixed-N wavefunction is already
enumerated, so the code constructs

\[
r=(H-E)\Psi
\]

and diagonalizes the exact \(2\times2\) Hamiltonian in

\[
\mathrm{span}\{\Psi,r\}=\mathrm{span}\{\Psi,H\Psi\}.
\]

This gives the variationally optimal one-step Krylov/Lanczos state with no
sampling error in the postprocessing. Because \(H\) commutes with the target
symmetry projector, PAV+1LS and VAP+1LS remain in the same \(k\)/point-group
sector as their parent states.

Test the Lanczos implementation with:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_one_lanczos_step.py
\`\`\`

The main output \`comparison_best.csv\` contains the best-energy representative
of all six methods across the requested seeds and ED-ground sectors.


## Time-reversal-restored finite-size benchmark

The recommended Hex18 projection benchmark now restores spinless time reversal
in addition to the ordinary spatial symmetry target. The six reported methods
are

\`\`\`text
ED
Raw-Slater-Jastrow
PAV-TR-Slater-Jastrow
PAV-TR-Slater-Jastrow+1LS
Full-VAP-TR-Slater-Jastrow
Full-VAP-TR-Slater-Jastrow+1LS
\`\`\`

The raw state remains intentionally unprojected because it is the reference
for future large-cluster broken-symmetry VMC.

For spinless fermions in the occupation basis,

\[
\mathcal T = K.
\]

For a TR-self-conjugate spatial irrep, PAV/VAP restore a real representative
(up to an overall phase). For a complex spatial irrep, the code first combines
the conjugate spatial pair

\[
\Gamma\oplus\Gamma^*
\]

(for example \(K/K'\)), then restores time reversal. Such a target is reported
with paired quantum-number labels rather than incorrectly assigning a
TR-invariant state to only one complex sector.

PAV-TR performs exact post-variation spatial projection followed by an exact
minimum-energy real combination in
\(\mathrm{span}_{\mathbb R}\{\Re\Psi,\Im\Psi\}\).

Full-VAP-TR includes the same restoration inside the NetKet variational model.
A trainable real phase spans the possible TR-invariant combinations during
optimization.

The output includes

\`\`\`text
time_reversal_residual
J_same_mean
J_opposite_mean
target_support_weight
projected_sector_residual
\`\`\`

For the projected finite-size benchmark, both current means should vanish to
numerical precision, while \(S_{\rm LC}\) can remain finite.

Regression test:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_time_reversal_benchmark.py
\`\`\`


## Bond-aware complex backflow

The recommended next correlator is a scalable three-channel backflow on top of
the unrestricted full Slater determinant and bond-aware density Jastrow:

\[
Q^{\rm BF}_{i\alpha}(x)
=
Q_{i\alpha}
+
\sum_{a\in\{t,t_1,t_2\}}\eta_a
\sum_j B^{(a)}_{ij}(n_j-\nu)Q_{j\alpha}.
\]

The three \(\eta_a\) are independent complex variational parameters. The
backflow therefore changes the configuration-dependent determinant amplitude
and phase while adding only six real variational degrees of freedom,
independent of system size.

The first implementation deliberately does not include triangle/chiral
backflow channels, so the effect of ordinary bond backflow can be isolated.

Regression test:

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_backflow_ansatz.py
\`\`\`

It verifies the channel counts \(18,9,9\) for \(t,t_1,t_2\) on Hex18 and
checks that \(\eta=0\) reproduces the previous Slater-Jastrow wavefunction.

The projection benchmark now supports

\`\`\`text
--correlator sj
--correlator backflow
\`\`\`

and defaults to \`backflow\`. For example:

\`\`\`bash
python ed_benchmark/vmc_zeroT/run_projection_comparison.py \
  --nf 3 \
  --t 1 --t1 0.5 --t2 -0.5 --V 2.5 \
  --seeds normal sd isd lc_same lc_opposite \
  --correlator backflow \
  --jastrow-mode bond_aware \
  --jastrow-shells 4 \
  --n-samples 4096 \
  --n-iter 300 \
  --ed-k 12 \
  --output results/backflow_hex18_N3_V2p5
\`\`\`

Backflow runs use separate directories and method names, for example
\`Raw-Slater-Jastrow-Backflow\` and
\`Full-VAP-TR-Slater-Jastrow-Backflow\`, so \`--resume\` cannot silently reuse
an old density-Jastrow-only run. Optimized values of
\(\eta_t,\eta_{t1},\eta_{t2}\) are stored as real/imaginary columns in the
summary CSV/JSON.
