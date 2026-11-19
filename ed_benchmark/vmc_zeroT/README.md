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


## Plotting the projection benchmark

After a projection scan finishes, generate the standard five comparison figures with

\`\`\`bash
python ed_benchmark/vmc_zeroT/plot_projection_comparison.py \
  results/backflow_hex18_N3_V2p5/comparison_all.csv
\`\`\`

By default the figures are written to

\`\`\`text
<comparison_all.csv parent>/plots/
\`\`\`

The script automatically detects whether the CSV contains the density-Jastrow
or Backflow method names. It selects the lowest-energy seed for each method
for the best-result plots.

The five standard figures are:

\`\`\`text
best_energy_vs_ed.png
best_observables_vs_ed.png
best_fidelity.png
pav_vap_rank_consistency.png
pav_vap_energy_fidelity.png
\`\`\`

The observable comparison contains

\[
CO_{q,\max},\qquad S_{\rm BOW},\qquad
S_{\rm LC+},\qquad S_{\rm LC-}.
\]

The script also writes

\`\`\`text
best_results_for_plotting.csv
pav_vap_seed_ranking.csv
\`\`\`

for later plotting in Origin or other software.

A different output directory can be specified with

\`\`\`bash
python ed_benchmark/vmc_zeroT/plot_projection_comparison.py \
  path/to/comparison_all.csv \
  --output path/to/my_plots
\`\`\`


## Staged VAP warm start

The projection benchmark now follows the staged optimization strategy

\[
\mathrm{Raw}
\rightarrow
\mathrm{translations}
\rightarrow
\mathrm{translations+rotations}
\rightarrow
[\mathrm{spinless\ time\ reversal}],
\]

with the optimized \(M\), density-Jastrow \(v\), and backflow \(\eta\)
parameters transferred from one stage to the next.

The default iteration budget after the Raw optimization is

\`\`\`text
--vap-translation-iter 100
--vap-rotation-iter 100
--vap-tr-iter 100
\`\`\`

so the three projected stages together use 300 iterations by default.
They can be changed independently.

Time reversal is restored by default. To stop after the spatial stages use

\`\`\`text
--no-restore-time-reversal
\`\`\`

and to request it explicitly use

\`\`\`text
--restore-time-reversal
\`\`\`

When time reversal is disabled, conjugate momentum/point-group sectors are
kept as separate spatial targets rather than being merged into a TR pair.

Because staged VAP requires the optimized Raw parameters, older result
directories created before parameter checkpointing cannot be used as Raw
warm starts. Run each Raw seed once without \`--resume\` to create
\`variational_M\`, \`variational_v\`, and \`variational_eta\` in
\`final_state.npz\`. Subsequent runs can use \`--resume\`.

A light regression test for parameter transfer is

\`\`\`bash
python ed_benchmark/vmc_zeroT/test_staged_vap_warmstart.py
\`\`\`


## Larger Ruby supercells for finite-size scaling

The native geometry module now contains two families of larger periodic Ruby
clusters.

The preferred **same-shape high-symmetry sequence** is

\[
M_L=L I,\qquad N_{\rm cell}=L^2,\qquad N_s=6L^2,
\]

with the named clusters

\`\`\`text
2x2   :  24 sites
3x3   :  54 sites
4x4   :  96 sites
5x5   : 150 sites
6x6   : 216 sites
\`\`\`

For generic \(t_1,t_2\), these finite tori retain the ordinary \(C_3\) and
mirror symmetries of the existing 2x2 cluster. When \(t_1=t_2\), ordinary
\(C_6\) is also retained. This same-shape family should be the primary sequence
for thermodynamic finite-size fits.

A second **compact tilted C3-compatible family** uses

\[
M(a,b)=
\begin{pmatrix}
a&b\\
-b&a+b
\end{pmatrix},
\qquad
N_{\rm cell}=a^2+ab+b^2.
\]

Named examples are

\`\`\`text
ring3  :  18 sites   (a,b)=(1,1)
ring7  :  42 sites   (2,1)
ring12 :  72 sites   (2,2)
ring13 :  78 sites   (3,1)
ring19 : 114 sites   (3,2)
ring21 : 126 sites   (4,1)
ring28 : 168 sites   (4,2)
ring31 : 186 sites   (5,1)
\`\`\`

These tilted clusters are useful for checking boundary-shape and shell effects.
They should not be mixed blindly with the \(L\times L\) sequence in a single
finite-size fit.

Geometry and symmetry checks can be run with

\`\`\`bash
python ed_benchmark/native_ed/test_large_supercells.py
\`\`\`

The test verifies the site count, quotient-cell count, permitted-momentum
count, PBC bond consistency, and finite-torus point-group compatibility.


## 24-site 2x2 ED/VMC benchmark

The projection comparison runner also supports the 24-site `2x2` Ruby torus.
At the same one-third filling as Hex18, use `Nf=8`.  The fixed-N Hilbert
space has 735471 states, so `--ed-backend auto` uses the native NumPy/SciPy
symmetry solver in `ed_benchmark/native_ed/symmetry.py`: translations are
blocked first and compatible little-group rotation/mirror sectors are added.
The resulting ground vector(s) are expanded back to the complete fixed-N basis
for exactly the same VMC fidelity and observable definitions used on Hex18.

A direct V=5.5 comparison is

```powershell
python ed_benchmark/vmc_zeroT/run_projection_comparison.py `
  --cluster 2x2 `
  --nf 8 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal sd isd lc_same lc_opposite `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware `
  --jastrow-shells 4 `
  --n-samples 4096 `
  --n-iter 300 `
  --vap-translation-iter 100 `
  --vap-rotation-iter 100 `
  --vap-tr-iter 100 `
  --restore-time-reversal `
  --ed-backend auto `
  --ed-k 12 `
  --ed-sector-low 4 `
  --output results/backflow_staged_2x2_N8_V5p5
```

For a first smoke test, add `--skip-vap` and reduce `--n-iter`; the ED,
Raw VMC, PAV and exact post-processing paths will still be exercised.  The
24-site benchmark uses a matrix-free full-basis Hamiltonian after the
symmetry-blocked ED, avoiding construction of the full sparse matrix solely
for post-processing.

Run the non-NetKet plumbing regression with

```bash
python ed_benchmark/vmc_zeroT/test_2x2_benchmark_plumbing.py
```

The benchmark now uses vectorized fixed-N operator application, vectorized
second-quantized symmetry maps, and vectorized native symmetry projectors.
No QuSpin binary extension is required for the 2x2 benchmark.


For the native symmetry ED, `--ed-k` is the number of globally retained low
states, while `--ed-sector-low` is the number of eigenpairs computed in each
symmetry block. The default `--ed-sector-low 4` keeps the 2x2 scan much
cheaper than requesting 12 states in every block. If the benchmark reports
that the ground manifold is truncated inside a symmetry block, increase
`--ed-sector-low`.


## Plot saved ED/VMC site, bond, and current maps

`plot_saved_state_maps.py` reads the state files written by
`run_projection_comparison.py` and produces real-space maps analogous to the
GW site/bond/current plots.

For a saved ED ground manifold:

```powershell
python ed_benchmark/vmc_zeroT/plot_saved_state_maps.py `
  --result-root results/backflow_staged_2x2_N8_V5p5 `
  --state-file results/backflow_staged_2x2_N8_V5p5/ed/final_state.npz `
  --mode average `
  --output-dir results/backflow_staged_2x2_N8_V5p5/ed_maps
```

If `--state-file` is omitted, the script defaults to
`<result-root>/ed/final_state.npz`.

The outputs are

```text
site_density.png / .svg
bond_order.png / .svg
bond_current.png / .svg
site_density.csv
bond_observables.csv
summary.json
```

The plotted quantities are

[
n_i=\langle n_i\rangle,
qquad
B_{ij}=2\,\mathrm{Re}\langle c_i^\dagger c_j\rangle,
qquad
J_{i\to j}
=i t_{ij}\left(
\langle c_i^\dagger c_j\rangle-
\langle c_j^\dagger c_i\rangle
\right).
]

For a degenerate ED ground manifold, `--mode average` evaluates the
basis-independent equal-weight incoherent mixture over all saved ground
vectors.  `--mode vector --column N` instead plots one saved basis vector.
A unique spinless time-reversal-even ED ground state has vanishing one-point
bond currents up to numerical noise; this does not by itself exclude
loop-current correlations.

The same script can plot any Raw/PAV/VAP `final_state.npz` containing
`psi_native` and `native_states`.

A lightweight convention check is

```bash
python ed_benchmark/vmc_zeroT/test_saved_state_maps.py
```

which verifies that time reversal leaves density/bond order unchanged, flips
every bond current, and cancels one-point current in an equal-weight
time-reversal pair.


### Ground-manifold branch-selection modes for saved-state maps

For an exactly degenerate ED ground manifold, the plotting tool can now select
physical pure states *inside the saved ground subspace* before drawing the
site/bond/current maps:

```text
--mode average
    Basis-independent equal-weight ground-manifold mixture.

--mode vector --column N
    One raw saved ED basis vector. Useful for debugging, but not
    basis-independent inside an exactly degenerate manifold.

--mode lc_same
    Diagonalize the benchmark-normalized equal-orientation triangle
    loop-current operator inside the ground manifold and choose its largest
    signed eigen-current branch.

--mode lc_opposite
    Same procedure for the opposite A/B triangle loop-current channel.

--mode charge_leading
    Compute the basis-independent connected density covariance of the
    equal-weight ground manifold, find the leading allowed-q six-orbital
    charge mode, form its real Hermitian quadratures, and select the pure
    ground-manifold state with the largest static density shift along that
    mode. The plotted site density is therefore an expectation value of an
    actual ground-manifold state, not a structure-factor template.

--mode charge_ab
    Extremize the q=Gamma A-versus-B triangle density imbalance inside the
    ground manifold.

--mode bow_star
    Extremize the existing Ruby star bond-order operator inside the ground
    manifold.
```

Examples:

```powershell
python ed_benchmark/vmc_zeroT/plot_saved_state_maps.py `
  --result-root results/backflow_staged_2x2_N8_V5p5 `
  --mode lc_opposite
```

```powershell
python ed_benchmark/vmc_zeroT/plot_saved_state_maps.py `
  --result-root results/backflow_staged_2x2_N8_V5p5 `
  --mode charge_leading
```

```powershell
python ed_benchmark/vmc_zeroT/plot_saved_state_maps.py `
  --result-root results/backflow_staged_2x2_N8_V5p5 `
  --mode charge_ab
```

Mode-specific default output directories are used, for example
`ed_maps_average`, `ed_maps_lc_opposite`, and
`ed_maps_charge_leading`, so repeated plots do not overwrite one another.

For every pure selected branch, `selected_state.npz` is also saved. The
`summary.json` records the projected order-operator eigenvalues, selected
ground-manifold coefficients, selected eigenvalue/shift, and, for
`charge_leading`, the leading momentum and structure-factor eigenvalue.

If the exact ground state is unique (`ground_dim=1`), the order-selecting
modes do not manufacture symmetry breaking. They return the unique state and
record `selection_available=false` with the reason in `summary.json`.


## Translation-built-in 18-site ring3 supercell ansatz

A separate scalable ansatz path is implemented in
`supercell_ansatz.py` and `run_supercell_benchmark.py`.  The historical
Hex18/2x2 benchmark code is unchanged.

The variational unit cell is the 18-site `ring3` cluster. A total
(L_1\times L_2) torus therefore contains

[
N_s=18L_1L_2.
]

The one-body orbitals are represented in the reciprocal lattice of this
18-site supercell. For each allowed superlattice momentum (k), a trainable
complex (18\times n_{\rm occ}(k)) matrix is QR-orthonormalized and Bloch
expanded over the complete system. This makes translation by one ring3
supercell exact by construction; no explicit translation projector is used.

The one-body parameter count is

[
18N_f=O(N),
]

instead of the unrestricted real-space (N_sN_f=O(N^2)). The three complex
nearest-neighbor backflow amplitudes and bond-aware density Jastrow are retained.

The SD/ISD seed form factor is explicitly tiled with the ring3 periodicity,
rather than placing only one star in a larger torus.

### 1x1 supercell benchmark against historical Hex18

For (L_1=L_2=1), there is only one superlattice momentum. Therefore the
new raw one-body parameterization is exactly a full complex (18\times N_f)
matrix and has the same one-body expressivity as the old unrestricted Hex18
ansatz. This makes the 1x1 calculation a direct implementation/cost check.

Run, for example, the same V=5.5 backflow benchmark with

```powershell
python ed_benchmark/vmc_zeroT/run_supercell_benchmark.py `
  --L1 1 --L2 1 `
  --nf 6 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal sd isd lc_same lc_opposite `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware `
  --jastrow-shells 4 `
  --n-samples 4096 `
  --n-iter 300 `
  --rotation-iter 100 `
  --learning-rate 0.03 `
  --rotation-learning-rate 0.01 `
  --output results/supercell_1x1_N6_V5p5
```

The runner evaluates exact ED on 1x1 and writes

```text
comparison_supercell.csv
best_supercell.csv
settings.json
geometry.json
ed/final_state.npz
supercell_raw_backflow__<seed>/
supercell_c3v_backflow__A1__<seed>/
supercell_c3v_backflow__A1__<seed>_1ls/
supercell_c3vtr_backflow__A1__<seed>/
```

The relevant comparison columns include

```text
E
delta_E_vs_ED
fidelity_ground_subspace
exact_energy_variance
CO_qmax
S_BOW_star
S_LC_same
S_LC_opposite
n_parameters
explicit_projection_images
time_per_iteration_s
```

The default supercell VAP now restores the full ordinary C3v point group:
E, C3, C3^2 and the three mirrors, for exactly six explicit spatial images.
The historical full ring3 spatial projector contains 3 primitive translations
times 6 C3v point operations = 18 images. The 1x1 benchmark therefore measures
directly how much speed is gained when primitive translations are built into
the 18-site supercell ansatz while the complete residual point group is still
restored explicitly.

No spinless time-reversal restoration is imposed in this new path by default,
because the intended thermodynamic LC branch must be allowed to carry a
nonzero one-point current.

Before running the full VMC benchmark, check the geometry/orbital construction:

```bash
python ed_benchmark/vmc_zeroT/test_supercell_ansatz.py
```

The test verifies that the 1x1 seed occupied subspaces for normal, SD, ISD,
LC-same and LC-opposite are identical to the historical Hex18 seeds, and that
a 2x2 ring3 superlattice (72 sites) is exactly invariant under the two ring3
supercell translations.


After the 1x1 run, compare it directly with an existing historical Hex18
`comparison_all.csv`:

```powershell
python ed_benchmark/vmc_zeroT/compare_supercell_vs_hex18.py `
  --old results/backflow_staged_N6_V5p5/comparison_all.csv `
  --new results/supercell_1x1_N6_V5p5/comparison_supercell.csv `
  --output results/supercell_1x1_N6_V5p5/comparison_vs_old_hex18.csv
```

The comparison utility selects the lowest-energy seed in each family and
reports energy error, ground-subspace fidelity, exact variance, CO/BOW/LC
observables and time per optimization iteration. It also records the explicit projection-image counts
(old full spatial Hex18: 18; new residual C3v: 6).


For a fast smoke test of the new implementation before the full 1x1 run:

```powershell
python ed_benchmark/vmc_zeroT/run_supercell_benchmark.py `
  --L1 1 --L2 1 --nf 6 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware --jastrow-shells 4 `
  --n-samples 1024 `
  --n-iter 20 --rotation-iter 20 `
  --output results/supercell_1x1_smoke
```

For a unique ED ground state the 1x1 CSV also reports
`ed_primitive_translation_sector_weight` and
`ed_full_spatial_sector_weight`. These are diagnostics only: the new ansatz
does not enforce primitive-cell translations, while C3v explicitly restores
the three mirrors. These diagnostics show whether the C3v-optimized state also
returns to the exact primitive-translation sector.
The CSV also reports the gauge-invariant spinless time-reversal residual.


### Optional spinless time-reversal restoration for the supercell path

The supercell runner now supports a third optimization stage

```text
Raw supercell -> C3v VAP -> C3v+TR VAP
```

without explicitly restoring primitive-cell translations; the full residual
C3v point group, including all three mirrors, is restored. Enable it with

```powershell
python ed_benchmark/vmc_zeroT/run_supercell_benchmark.py `
  --L1 1 --L2 1 --nf 6 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal sd isd lc_same lc_opposite `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware --jastrow-shells 4 `
  --n-samples 4096 `
  --n-iter 300 `
  --rotation-iter 300 `
  --restore-time-reversal `
  --tr-iter 100 `
  --rotation-learning-rate 0.01 `
  --tr-learning-rate 0.01 `
  --output results/supercell_1x1_N6_V5p5_tr
```

For the 1x1 exact benchmark, the initial TR phase is chosen from the preceding
C3v-VAP wavefunction to maximize the norm of its TR-even real ray before SR
optimization. The TR phase is then trainable. Spinless time reversal is
(T=K) in the occupation basis, so the projected amplitude is proportional to

[
e^{i\phi}\Psi_{C_{3v}}(x)+
\left[e^{i\phi}\Psi_{C_{3v}}(x)\right]^*.
]

This adds no new spatial images: the explicit spatial image count remains 6.
The default C3v implementation supports the one-dimensional A1 and A2 irreps.
The two-dimensional E irrep is not used by the runner until a matrix-valued
irrep projector is added. Legacy C3-only runs remain available through
`--point-group c3`.

The symmetry diagnostic columns have explicit names:
`ed_primitive_translation_sector_weight` and
`ed_full_spatial_sector_weight`. Here "translation" is unrelated to time
reversal. They are projector weights into the symmetry sector of the unique ED
ground state, not order parameters and not fidelities.


If the Raw and C3v stages were already completed in the same output directory,
the TR refinement can be continued without rerunning them:

```powershell
python ed_benchmark/vmc_zeroT/run_supercell_benchmark.py `
  --L1 1 --L2 1 --nf 6 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal sd isd lc_same lc_opposite `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware --jastrow-shells 4 `
  --n-samples 4096 `
  --skip-raw --reuse-point-group `
  --restore-time-reversal `
  --tr-iter 100 --tr-learning-rate 0.01 `
  --output results/supercell_1x1_N6_V5p5
```

`--reuse-point-group` loads the saved C3v parameters and, on the 1x1
benchmark, the saved exact projected wavefunction to choose a non-cancelling
initial TR phase.


### Staged residual-space VAP: internal translation -> C3v -> TR

The recommended supercell benchmark path now restores the residual spatial
symmetry in stages:

```text
Raw
  -> Tint VAP
  -> Tint x C3v VAP
  -> Tint x C3v + TR VAP
```

Here

[
T_{\rm int}=T_{\rm primitive}/T_{\rm ring3}\cong Z_3
]

contains exactly three primitive-translation cosets inside the 18-site ring3
variational supercell.  The full residual spatial projector therefore has

[
3\times 6=18
]

images, independent of the total (L_1\times L_2) number of ring3
supercells.  Supercell translations themselves are never summed explicitly.

For the current 1x1, Nf=6, V=5.5 benchmark, reuse the already completed Raw
states and run the new staged refinement with

```powershell
python ed_benchmark/vmc_zeroT/run_supercell_benchmark.py `
  --L1 1 --L2 1 --nf 6 `
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 `
  --seeds normal sd isd lc_same lc_opposite `
  --correlator backflow `
  --point-group c3v `
  --jastrow-mode bond_aware --jastrow-shells 4 `
  --n-samples 4096 `
  --skip-raw `
  --internal-translation-iter 100 `
  --internal-translation-learning-rate 0.01 `
  --full-spatial-iter 100 `
  --full-spatial-learning-rate 0.01 `
  --restore-time-reversal `
  --tr-iter 100 --tr-learning-rate 0.01 `
  --output results/supercell_1x1_N6_V5p5
```

The stage directories are

```text
supercell_tint_backflow__Gamma__<seed>/
supercell_tint_c3v_backflow__A2__<seed>/
supercell_tint_c3vtr_backflow__A2__<seed>/
```

for the current unique A2 exact ground state.  Each stage records its explicit
image count and timing, so the optimization effect of the 3-image internal
translation stage can be separated from the 18-image full residual spatial
stage.

Use `--reuse-internal-translation` to reuse an existing Tint stage and
`--reuse-point-group` to reuse an existing Tint x C3v stage.

### Per-seed site/bond/current triptychs

For 1x1 exact benchmarks the runner automatically creates

```text
seed_final_maps/<seed>/final/site_bond_current_triptych.png
seed_final_maps/<seed>/final/site_bond_current_triptych.svg
```

for every seed.  The three panels are site density, bond order and directed
bond current.

If TR restoration is enabled, the final TR-even current panel should vanish
up to numerical noise.  Therefore the default also saves the preceding
TR-broken residual-spatial branch:

```text
seed_final_maps/<seed>/pre_tr/site_bond_current_triptych.png
```

together with `site_density.csv`, `bond_observables.csv` and
`summary.json`.

The plots can also be regenerated without rerunning VMC:

```powershell
python ed_benchmark/vmc_zeroT/plot_supercell_seed_triptychs.py `
  --result-root results/supercell_1x1_N6_V5p5
```
