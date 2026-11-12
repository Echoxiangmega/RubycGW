# Three-torus Ruby benchmark: ED / GW / cGW / post-GW / cluster-ED+GW

## Run

Run from the repository root. The script is research/benchmark_three_tori.py.

    python research/benchmark_three_tori.py --geometry hex18 --V 1.0 --filling 2 \
      --K 0.333333333333 0.333333333333 --Q 0.333333333333 0.333333333333 \
      --T 0.125 --out results/hex18_V1_n2

Defaults are t (intratriangle ti) = 1, t1 = -0.5, t2 = +0.5,
and channels x_even, y_even, z_same, z_opposite. Add charge_ab and/or
charge_total with --channels to check density order. Filling means the
number of spinless electrons per six-site primitive cell; filling times
the number of cells must be an integer for canonical ED.

## Matched finite PBC geometries

- 3x1: 18 sites, allowed K = (0,0), (+1/3,0), (-1/3,0).
- 2x2: 24 sites, allowed K = (0,0), (1/2,0), (0,1/2), (1/2,1/2).
- hex18: 18-site periodic skew torus, T1=a1-a2 and T2=a1+2a2,
  allowed K = (0,0), (+1/3,+1/3), (-1/3,-1/3).

The hex18 option is NOT the separate OBC six-triangle ring of
rubycgw/hex18_ed.py. The 3 or 4 primitive-cell momenta are folded to
the 18x18 or 24x24 one-point supercell representation for GW, cGW and
post-GW. The user-selected K projects the one-particle Green matrix to
the 6x6 Bloch block. The independently selected Q defines the external
pseudospin/charge modulation. Both are checked for commensurability.

## Observables

One-particle Green error:
  ||G_method(K,iw)-G_ED(K,iw)||_F / ||G_ED(K,iw)||_F.

The plotted scalar is Tr[G(K,tau)]/6, or a selected orbital diagonal
(--plot-orbital 0..5). The finite Matsubara transform uses an exact
noninteracting reference tail to reduce spurious endpoint errors.

The triangle pseudospin x and y are time-reversal-even intra-triangle
charge/orbital polarizations. The z component measures a physical
time-reversal-odd loop current. Physical same/opposite A-B circulation
are z_same and z_opposite. More general density probes:
  charge_ab = (n_A-n_B)/sqrt(6);
  charge_total = sum of six site densities per primitive cell.

At nonzero Q, the susceptibility operator is

  O(Q) = (1/sqrt(Ncell)) sum_R exp(-2 pi i Q.R) c_R^dag K c_R,

and the comparison is the connected static Kubo response
  chi(Q,0) = integral_0^beta <delta O^dag(Q,tau) delta O(Q,0)> d tau.

The implementation uses real Hermitian cosine/sine sources for Q not
equivalent to -Q. Their diagonal response sum is the complex-Q diagonal.
For uniform charge_total Q=0, canonical ED has exactly zero response
while a grand-canonical fixed-mu GW derivative measures compressibility.
Do not interpret those two quantities as the same physical observable.

ED susceptibility: fixed-N canonical thermal typicality with
--ed-samples trace vectors and --ed-tau-points Simpson points.
This is a stochastic approximation, not a full 18/24-site finite-T
spectral trace.

ED Green: truncated-sector grand-canonical thermal Lanczos, with
--ed-sector-padding, --ed-thermal-eigs and --ed-krylov-steps. Use
--ed-green-symmetry momentum for true N+/-1 translation-block Krylov,
or the default legacy mode for the former full-sector algorithm.
Inspect thermal_kept_weight, sector_edge_weight, translation_residual,
and Krylov diagnostics in summary.json before trusting ED as a reference.
The 2x2 case can be costly and is protected by --ed-max-dim.

GW: split H+F+(W-V) self-consistent Green and bare GG response.
cGW: same G and energy as GW, but solve the tail-consistent static
H+F+MT+AL vertex and recalculate susceptibility. There is no new
single-particle Dyson loop for cGW.
post-GW: full covariant density response updates W and then one-shot
Dyson updates G; only G is compared. --post-mmax gives an explicitly
windowed diagnostic, not full post-GW.
Cluster-ED+GW: one six-site primitive-cell impurity replicated over
the torus, with V only inside the two triangles and the matched
cluster-GW double-counting subtraction. For the full finite-Q
cluster response, a separate finite-source solver permits Ncell
inequivalent six-site impurities with complex bath. This is expensive;
--skip-cluster-chi skips it, and does not fill it in with a bare bubble.

ED E0: lowest energy at strictly fixed N (T=0). GW energy: a
finite-temperature Galitskii-Migdal estimate plus GW Helmholtz free
energy. Cluster-ED+GW energy: nonvariational GM-type proxy. These are
NOT interchangeable ground-state energies. Any numerical comparison
of a finite-T energy estimate against ED E0 is marked
mixed-temperature, not a controlled ground-state energy relative error.


## Three different GW interaction-channel approximations

The command-line default is now

    --gw-channels density pseudospin bj

These options select the GW **bosonic interaction representation**, not
the external pseudospin response observable. All three still report the
SAME physical external channels from --channels.

* density: original site-density matrix, with V on exactly six
  intra-triangle bonds per primitive cell. This is the historical
  production lattice-GW/cGW/post-GW reference.
* pseudospin: the **exact** A1+E unitary triangle rewriting. The
  E-doublet Pauli components x,y,z have g_E=-V/3, while n_A1 and
  n_E=n_++n_- remain as auxiliary density channels with
  g_(A1,E)=g_(E,A1)=V. The full operator identity is

      V sum_{i<j} n_i*n_j = -(V/6)(sigma_x^2+sigma_y^2+sigma_z^2)
                              + V*n_A1*n_E + (V/2)*n_E.

  The last term is the single-particle normal-ordering shift of the
  operator-square representation. The code retains the physically exact
  original-V first-order Hartree-Fock map and does not add this shift
  a second time. All A1/E residual bosonic screening channels are
  included; this is no longer the earlier E-only truncation.
  Exact bare-channel equivalence does NOT guarantee Fierz-invariant
  approximate GW screening at higher order.
* bj: for each physical interacting bond define B_ij=c_i^dag c_j+
  c_j^dag c_i and J_ij=i(c_i^dag c_j-c_j^dag c_i). The symmetric
  Fierz representation uses g_B=g_J=-V/2, retaining the full
  B/J bosonic cross-channel response. Both B and J weights are 0.5.
  The static Hartree-Fock map is the same physical original V.

Each scheme solves its own self-consistent GW background. Within one
scheme, the *same* converged GW G/W feeds the cGW static external
susceptibilities and dynamic full-channel covariant post-GW W update.
A cGW calculation of a few physical O(Q) susceptibilities is not itself
enough for post-GW: the latter needs the full internal channel response
at every bosonic transfer (unless --post-mmax explicitly truncates).
No second ordinary GW fixed-point calculation is done for cGW or post.

Output labels are GW[density], cGW[density], post-GW[density],
GW[pseudospin], etc. Cumulative time is reported per scheme:

    T_GW(mode)  = GW solve time
    T_cGW(mode) = T_GW(mode) + cGW static response time
    T_post(mode)= T_GW(mode) + cGW static time (if requested)
                               + full dynamic post update time

This cumulative value is independent for the three schemes; do not sum
one variant's GW into the next variant's total. A common initial ED
reference is computed only once. The primitive-cell ED+GW embedding
continues to use the *density* GW lattice background, and is not
implicitly converted to the alternative E/BJ bosonic Hamiltonian.

## Finite-temperature ED symmetry scope

The default --ed-symmetry momentum now exactly block-diagonalizes
the fixed-N ED Hamiltonian by primitive many-body translation momentum
on 3x1, 2x2, and skew hex18 PBC. Its finite-temperature canonical
stochastic trace is evaluated separately within each momentum sector;
the pseudospin operator O(Q) connects sectors K to K-Q, with proper
fermionic translation signs and full normalization. The solver
constructs small sparse H_K matrices, not a projected full-Hilbert
Hamiltonian matvec each iteration. --ed-symmetry legacy restores the
older unblocked canonical thermal Krylov solver.

The finite-temperature ED Green is also available with a dedicated
--ed-green-symmetry momentum option: it uses selected low-lying N,K
states, and particle addition/removal resolvents in the corresponding
N+/-1 momentum sectors. This path returns only the user-selected
G(K,iw) 6x6 matrix; --ed-green-symmetry legacy (currently the
default while further low-T benchmarking continues) retains the
former full-sector thermal Lanczos implementation. Both Green
variants still truncate low thermal states and Lanczos depth and
require convergence checks. Neither uses point-group irreducible
representations yet.


## Optimized post-GW dynamic covariant response

One-shot post-GW is **not** an extra nonlinear GW self-consistency loop.
For each internal bosonic transfer m, the code solves the linear covariant
vertex equation on the converged channel GW G and W. The expensive part
is constructing the full internal n/B/J/A1+E response matrix, not the
final one-shot Sigma and Dyson update.

The default post solver now applies three performance improvements:

1. Precompute/caches the frequency-specific out-of-box W lookup and
   reference-tail eigensystem once per transfer rather than once for every
   GMRES operator application and source.
2. For a translation-invariant periodic torus, solve only the source
   operators located in one reference primitive cell, and reconstruct
   the other columns using the exact group identity
   chi[(R,a),(R',b)](iOmega)=chi[(R-R',a),(0,b)](iOmega).
   The code checks translation invariance of G, W, bare coupling,
   and all translated source form factors before enabling this shortcut.
   It automatically falls back to the full source solve if any check fails.
3. Share a bounded GCROT(m,k) Krylov subspace across the different
   right-hand-source operators at fixed external bosonic frequency.
   Because the m=0 static HF response includes a Hermitian projection,
   the matrix-free operator is real-linear and is solved using a realified
   representation. The final true complex max-norm residual is checked;
   failures fall back to the original per-source GMRES solver. At most
   four recycled vectors are stored (bounded memory).

For hex18, the number of independent solves per nonnegative transfer
drops from density 18 to 6, exact A1+E pseudospin 30 to 10, and
symmetric BJ 36 to 12. For 2x2 the source count is reduced by four.
Actual wall-time improvement depends on kernel matvec cost, Krylov
convergence and machine BLAS performance.

The existing *static-window post-GW* remains intact:

    --post-mmax 0

This computes chi_cov only at Omega=0 and retains W_GW for all nonzero
bosonic transfer frequencies. It is a diagnostic approximation, not the
same output as full dynamical post-GW (default: --post-mmax omitted).

For comparisons or difficult parameter regimes, the original paths
remain available:

    --post-no-translation   # solve every source explicitly
    --post-no-recycle       # GMRES without recycled Krylov basis

The benchmark summary.json now records each post channel's
translation_reduced status, number of solved source vertices,
Krylov fallbacks, transfer_wall_seconds and phase_seconds split into
covariant, screening, self-energy and Dyson stages.


## Cluster-ED+GW can start from a nonconverged preliminary density-GW iterate

The preliminary **density-channel GW** is used only as an *initial guess*
for the six-site primitive-cell ED+GW embedding.  Unlike independent
GW, cGW and post-GW comparisons, that preliminary GW need not have
reached its own fixed point: every cluster embedding iteration
recomputes lattice polarization P, screened W, lattice GW self-energy,
cluster double counting, impurity ED self-energy, and the embedded
Dyson Green function. The embedding has its *own* convergence residual.

Default behavior now:

* If preliminary GW[density] reaches its iteration cap but returns
  finite Sigma_H, Sigma_GW, mu, G, run cluster-ED+GW anyway.
* The GW[density] result retains status `unconverged`; it is not
  reported as a valid standalone Green function or source for cGW
  and post-GW.
* Because the preliminary channel-GW solver returns its last
  *evaluated* Sigma along with a Green function from the preceding
  mixed Dyson step, a nonconverged seed is reconciled by **one
  fixed-filling Dyson solve** (not another self-consistent GW loop).
* The embedding receives the explicit target filling, rather than
  silently using the preliminary GW iterate's approximate density.
* Only the embedding's final convergence status determines whether
  its Green function and energy proxy are reported. The convergence
  flag of the preliminary GW is independent.
* `summary.json` records `diagnostics.cluster_seed`, including
  source residual, source convergence, Dyson rebuild and density error.
  The CSV includes `initial_density_GW_residual` marked
  `initial-only-unconverged` when relevant.

To require an already converged density-GW initializer, pass
`--cluster-require-converged-gw`. Without that flag the preliminary GW
is only a seed and does *not* need to converge.

If the preliminary GW solver raises an exception or returns nonfinite
Sigma/G, there is no valid seed; cluster ED+GW is marked skipped.
If a GW *continuation* stops at a V below the requested target, its
lower-V state is not silently used as a target-V embedding seed.

To save time on runs where the cluster embedding is the primary goal,
one may set `--methods ed cluster --gw-max-iter 10` to limit
preliminary density-GW iterations without altering the subsequent
cluster embedding iteration limit `--cluster-max-iter`.
The shorter preliminary solve can produce a less favorable seed and
is not guaranteed to speed overall convergence.

The separate primitive-basis `solve_cluster_ed_gw` function now also
permits finite unconverged preliminary GW states by default, with a
`require_converged_background=True` opt-in to enforce its former gate.

## Large-V GW convergence and ED checkpointing

The original benchmark directly started each Fierz GW solver from zero
self-energy at the requested V (150 Pulay iterations and mixing 0.22 by
default). At V=5 a nonzero final residual such as 1.708e-3 **does not**
satisfy the default tolerance 1e-7. Do not run cGW/post-GW or compare
its energy as a converged solution by adding --allow-unconverged.

Use a damped direct GW diagnostic without repeating ED:

    python research/benchmark_three_tori.py --geometry hex18 --V 5 \
      --filling 2 --K 0.333333333333 0.333333333333 \
      --Q 0.333333333333 0.333333333333 --T 0.125 \
      --methods gw --gw-channels density --gw-mixing 0.05 \
      --gw-max-iter 600 --verbose --out results/hex18_gw_debug

For a default fast benchmark, continuation is OFF: the program solves
each GW representation directly at the requested V with
--gw-ramp-steps 0 (the default). It does NOT run through intermediate V
unless the user explicitly requests it. A failed method is marked
"unconverged" or "failed" in comparison.csv and summary.json, and
the program continues with the other independent interaction channels.
The corresponding dependent cGW/post-GW (and cluster-ED+GW when the
density GW background fails) are marked "skipped"; no invalid Green
function, energy or susceptibility is compared against ED. If the
covariant static vertex fails but GW succeeds, the independent bare
GG bubble is still reported and post-GW can be attempted separately.
A failed ED reference does not prevent GW-only diagnostics.

--strict-method-failures restores the old fail-fast behavior for
debugging. --skip-failed-gw-channel is kept as a compatibility alias,
but is no longer needed. --allow-unconverged is deprecated for
comparative calculations and does not authorize using failed results.

To systematically approach strong coupling, set --gw-ramp-steps 10.
This starts from a converged V=0 state, advances through intermediate
interactions, uses the previous solution's static/dynamic self-energy
and chemical potential as a warm start, but always rebuilds P and W
at the *new* V. A failed ramp step retries with half the mixing and
then bisects the step. --gw-ramp-min-step 0.02 and
--gw-ramp-max-stages 100 limit this process. You can also select
--gw-mixing-method linear for more conservative damping.

    python research/benchmark_three_tori.py --geometry hex18 --V 5 \
      --filling 2 --K 0.333333333333 0.333333333333 \
      --Q 0.333333333333 0.333333333333 --T 0.125 \
      --gw-channels density --gw-ramp-steps 12 \
      --gw-ramp-min-step 0.01 --gw-max-iter 400 \
      --gw-mixing 0.08 --verbose --post-mmax 0 \
      --skip-cluster-chi --out results/hex18_V5

A ramp **does not guarantee convergence** beyond a screening
instability or within an artificially constrained translation-symmetric
phase. Every attempt records V, actual residual, mixing, time, and
minimum singular value of I-gP. If the target is not reached, the
program stops and writes gw_failure_<mode>.json with the *last
converged* lower V. It does not silently use that lower-V G as the
requested target. With --skip-failed-gw-channel it safely proceeds
to other interaction decompositions, leaving the failed mode
explicitly absent from cGW/post-GW.

To avoid losing ED work after a GW failure, the script writes
out/ed_reference.npz **immediately after successful ED E0, canonical
thermal susceptibility and selected-K Green calculations**, before
starting GW. A rerun to the same --out path with identical ED
parameters automatically reloads this checkpoint. GW mixing, ramp
parameters, cGW settings and which GW channels are requested are not
part of the ED cache signature; changing geometry, V, filling,
temperature, orbital momenta, ED truncation or RNG seed invalidates it.
--refresh-ed-cache forces a recalculation, and --no-ed-cache disables
it. This does not cache GW continuation states across *processes*:
only within one ramp run are intermediate GW results reused.

The summary.json method_status map provides status and reason for
each requested method; comparison.csv contains rows with empty
numeric values for unconverged/failed/skipped methods. The
gw_failure_<mode>.json file retains numerical residual, smin, and
ramp history. A normal process exit means the benchmark collected all
possible outputs, NOT that all methods converged; always check status.

The fixed-point symmetry assumption still matters: a symmetric
continuation from V=0 can reach an unstable symmetric branch or fail
at an interaction-driven finite-Q transition. If the screening
smin becomes small, compare symmetry-broken supercell solutions or
use targeted finite-Q source fields; changing damping alone is not a
physical remedy.

## Human-readable scorecards (automatically generated)

The original long-format comparison.csv is retained for complete numerical
provenance. Four additional files are automatically written at the end of
every benchmark, including runs where one or more methods do not converge:

- **comparison_wide.csv**: one row per method, ED-relative Matsubara G error,
  imaginary-time G error, four-channel mean chi error, charge (x/y) mean,
  loop-current (z_same/z_opposite) mean, each individual chi value/error,
  actual runtime and separate finite-T-vs-E0 energy diagnostic.
- **ranking_green.csv**: independent Green G(iw) ranking excluding cGW's
  duplicate GW Green; includes the tau error and time.
- **ranking_susceptibility.csv**: independent ranking by the arithmetic mean
  of four physical susceptibility relative errors; a missing, unconverged
  or skipped channel prevents that method from being ranked.
- **comparison_summary.md**: two readable ranked tables and physical caveats.

No single all-purpose score is reported, since good G, good susceptibility
and accurate zero-T energy are not equivalent criteria. Covariant GW
does not recalculate the one-particle G and therefore reuses the parent
GW value in comparison_wide.csv (marked shared). post-GW has no chi
in the current benchmark, and cluster chi can be skipped. The energy
columns are explicitly *not* ranked because ED E0 and finite-T GW or
nonvariational cluster energy estimates are not thermodynamically matched.

A particularly important additional diagnostic is
chi(z_same)-chi(z_opposite). If the ED result distinguishes the two current
patterns but a GW/cGW variant gives identical values, inspect the
inter-triangle susceptibility, the physical current vertices and the
lattice symmetry at Q before asserting that the method selects the
right current ground state.

## Fractional finite-temperature ED filling

The old benchmark required `Ncell*filling` to be an integer **even
when ED was not requested**. This restriction was only necessary for
canonical fixed-particle ED, and has been removed for GW and the
cluster embedding.

With `--ed-ensemble auto` (default):
- If `Ncell*filling` is integer, the existing **canonical** ED
  ground E0 and static thermal susceptibility remain unchanged.
- Otherwise, finite-temperature **grand-canonical** ED uses several
  adjacent integer particle sectors, estimates each sector's
  canonical partition function and connected C_N(Q,tau), and solves
  for the chemical potential reproducing the requested noninteger
  total mean particle number. The grand-canonical connected
  correlation includes the extra inter-sector covariance of <O>_N,
  not just a weighted average of connected canonical χ_N.
- The Green's function already uses a separate low-energy
  sector-truncated grand-canonical thermal Lanczos treatment, and
  can represent fractional target <N>. Its spectral-state
  truncation is not identical to the thermal trace used for χ.

The grand-canonical result is a **selected-charge-sector and
stochastic thermal trace approximation** to the exact finite-T ED
thermal sum, not an exact full Fock-space trace. The CLI option
`--ed-charge-sector-padding 1` includes one extra integer sector
on both sides of floor/ceil(Ntarget) (default 1); 0 uses only the
two adjacent sectors and can be useful for a fast diagnostic.
Increase padding and `--ed-samples` and check the recorded
`grand_edge_probability` before treating the result as converged.
The grand-canonical calculation can cost several times the
canonical fixed-N ED susceptibility.

For instance, hex18 has 3 primitive cells. Filling = 5/3 implies
fixed Ntotal=5, so canonical ED is valid even though the per-cell
filling is fractional. Filling = 1.5 implies target <Ntotal>=4.5,
so the default switches to grand canonical. Use
`--ed-ensemble canonical` to demand strict fixed-N (and produce
a clear error if total N is noninteger), or `--ed-ensemble grand`
to explicitly demand the grand-canonical ensemble even at integer
mean filling (requires >=2 charge sectors).

At fractional mean filling there is **no unique canonical T=0
ground energy E0** for that noninteger N. Thus the benchmark
does not fabricate an ED E0 or a GW-vs-ED E0-relative error.
It does report the ED thermal chemical potential and mean N.

## Faster cluster ED and finite-Q cluster susceptibility

When `--skip-cluster-chi` is omitted, the code computes the
full finite-source derivative of the *multicell* cluster-ED+GW
fixed point using two Hermitian source components cos/sin and
positive/negative fields. It still requires
`2 * Nchannels * N_HermitianComponents` fully self-consistent
source calculations.

The old implementation **cold-started each source from a new
folded-lattice GW solve**, which itself could take >=200 iterations
or fail at large V. By default, the new implementation converts the
already converged unperturbed one-impurity cluster-ED+GW state into
a multicell restart: replicate its 6-orbital impurity self-energy
and converged finite-bath energies/couplings to every cell, and
reuse the converged full-lattice G, Sigma_emb and mu. The +h and -h
source solves each receive independent copies of this SAME
zero-field state. They still solve their own full nonlinear
multicell embedding including feedback from the perturbation.
The old cold path is retained as a diagnostic via
`--cluster-chi-coldstart`.

The uniform real-bath fit and the multicell complex-bath
fit now use **analytic Jacobians** by default for the nonlinear
least-squares bath optimization. This replaces the numerical
finite-difference Jacobian which required O(n_bath*n_orb)
extra residual evaluations per Jacobian. Both preserve the
same bath-fit loss, fit parameter constraints and self-consistency
conditions. The old numerical Jacobians can still be selected by
calling the fit routines with `jacobian="2-point"` for a direct
research comparison.

Each physical channel's cluster susceptibility is saved as soon
as its field solves converge. If another channel fails, it is
marked failed without discarding already completed channels.
The new `summary.json` cluster_source_chi diagnostics contain
per-channel maximum residual, bath error, source count, wall time,
and the warm/cold initial-state selection.

Unperturbed uniform cluster-ED+GW and the source calculation use
different symmetry Ansätze (one equivalent impurity vs one
impurity per physical primitive cell). Warm-started finite
differences must still be checked for h -> 0, bath discretization,
convergence thresholds and possible branch/hysteresis effects.
Smaller h is not always better because noise in bath fitting
and residual errors is divided by h. Testing 0.0005, 0.001,
0.002 is a sensible finite-difference convergence check.

## Optional fast cluster-only initialization

The cluster embedding self-consistency recomputes lattice GW screening and
cluster corrections every iteration. Its preliminary lattice-GW solve is
therefore only an initial guess, not a required separate fixed point.

For a fast exploratory cluster-only run, use:

    python research/benchmark_three_tori.py --geometry hex18 --V 5 \
      --filling 2 --T 0.125 --methods ed cluster \
      --cluster-initial-state free --cluster-max-iter 60 \
      --skip-cluster-chi --out results/hex18_cluster_free

With `--cluster-initial-state free`, a fixed-filling noninteracting
Dyson G with zero self-energy initializes the embedding; when no
standalone GW/cGW/post-GW method is selected, **no preliminary GW
self-consistency is run at all**. The same full embedding loop and
final convergence residual still apply. Default `gw` retains the
existing preliminary density-GW initializer, whether converged or
not, with a consistent Dyson reconstruction when necessary.
An initially noninteracting seed can cost fewer setup operations
but may require more cluster iterations or reach a different branch
in a multistable strong-coupling regime. Check both initializations
when studying phase competition.

The two largest safe numerical speedups in the current code are
(1) warm-starting every cos/sin, +h/-h multicell susceptibility
calculation from the converged uniform cluster state, and
(2) analytic Jacobians of real and complex bath fits.
They do NOT remove the essential impurity exact diagonalizations
at every embedding iteration. An increased bath size still raises
the impurity Fock-space cost rapidly; a full cluster susceptibility
at nonzero Q remains substantially more expensive than its
unperturbed Green function.

## Second audit: exact-physics cluster speedups and extreme ED filling

The `FinBathImpurityED` constructor is recreated on every bath update,
but its fixed-particle-number Fock basis, occupation masks, and hopping
destination/sign topology depend ONLY on orbitals and N, not the fitted
bath energies, hybridization or V. Those combinatorial arrays are now
reused from a small read-only LRU cache for the typical 6 correlated +
up to 6 bath orbitals. Larger 13-16 orbital ED clusters retain the
uncached implementation to avoid large resident topology arrays.
The Hamiltonian's coefficients are still rebuilt for each new bath,
and the physical finite-bath eigensystem is still freshly diagonalized.
A real-symmetric impurity Hamiltonian now uses the cheaper real LAPACK
eigensolver. A complex bond/current field or complex bath still uses
the full complex Hermitian solver. Tests reconstruct the Fock Hamiltonian
independently with fermion operator signs and compare eigenvalues.

Cluster finite-Q `chi` source solves (cos/sin and +h/-h) are mutually
independent once the zero-field uniform cluster seed has been copied.
Use `--cluster-chi-workers 2` (or 3-4 with ample RAM) to run them in
separate Windows-safe Python worker processes. Default is 1 worker.
The workers request a single BLAS thread using threadpoolctl when
available, avoiding CPU oversubscription. Multiple processes can use
substantial memory because each impurity holds dense many-body
Hamiltonians/eigenvectors; for the 12-orbital 6+6 impurity, start with
2 workers and monitor RAM before increasing. Each source restarts from
the same zero-field state and performs the same complete finite-field
self-consistency; parallel execution does not share mutable baths.

The benchmark saves more useful timings:
- `diagnostics.cluster-ED+GW.timing_seconds`: unperturbed lattice,
  bath_fit, impurity_ed, Dyson, other and total time.
- `diagnostics.cluster_source_chi._execution`: number of source jobs,
  worker count, elapsed wall time and aggregate source work.
- Each physical channel's source `phase_seconds` combines time spent
  inside the source lattice, bath-fit, ED and Dyson stages.
  `work_seconds` is the sum of individual source runtimes and must NOT
  be confused with the concurrently elapsed wall time.

The grand-canonical ED charge-sector audit also includes N=0 and
N=Nsites, which are exact one-state sectors. Their connected canonical
response is zero, but their particle-number weights are essential near
vacuum or full filling. The momentum-resolved ED Green now skips the
nonexistent removal transition below N=0 and addition transition above
N=Nsites. This fixes prior selected-sector bracketing failures for
0 < mean total N < 1 and Nsites-1 < mean N < Nsites. In these extreme
regimes, use `--ed-green-symmetry momentum`; legacy full-sector
thermal Lanczos still has stricter interior-sector limits.

The finite-field sources also have optional **independent** convergence
controls: `--cluster-chi-tol 1e-7` and
`--cluster-chi-max-iter 100`. By default both inherit the ordinary
uniform cluster `--cluster-tol` and `--cluster-max-iter`. A source
solver residual divided by the small field h can be amplified in a
susceptibility derivative, so the code reports `residual_over_h` and
warns if it exceeds 0.01. This is a *diagnostic scale*, not a rigorous
bound on the error in chi. For publication-quality static χ, check
h, tolerances and bath convergence independently, rather than trusting
a small unperturbed-cluster residual.

None of these changes removes the physical cost of diagonalizing each
new impurity Hamiltonian or the O(h^2) finite-field differentiation
error in static chi. A small `--source-step` also amplifies source
solver residuals by 1/h; always compare source steps and convergence
tolerances. Wall-time improvements depend on bath size, CPU BLAS
configuration, worker count and whether EVD or bath fitting dominates.

## Outputs

- comparison.csv: method, observable, value, ED reference, relative
  error where meaningful, incremental stage run time, status, and caveats. A dedicated\n  total_method_runtime_seconds row also includes any shared GW background\n  and the cluster source-response stage where applicable.
- summary.json: geometry, allowed momenta, solver diagnostics, failures,
  and all stage timings.
- raw_arrays.npz: full Matsubara/imaginary-time matrices and ED
  pseudospin correlation data.
- green_tau.png and green_tau.pdf: Li-Hui-style imaginary-time Green
  comparison of ED, GW, post-GW and primitive cluster-ED+GW.
  No separate redundant cGW G curve.

## Useful examples

    python research/benchmark_three_tori.py --geometry 3x1 \
      --V 1 --filling 2 --K 0.333333333333 0 \
      --Q 0.333333333333 0 --T 0.125 \
      --channels x_even y_even z_same z_opposite charge_total \
      --out results/3x1_V1

    python research/benchmark_three_tori.py --geometry 2x2 \
      --V 1 --filling 2 --K 0.5 0 --Q 0.5 0 --T 0.125 \
      --methods ed gw cgw cluster --skip-cluster-chi \
      --ed-max-dim 3000000 --out results/2x2_V1

    python research/benchmark_three_tori.py --geometry hex18 \
      --V 1 --filling 2 --K 0 0 --Q 0 0 --T 0.125 \
      --channels x_even y_even z_same z_opposite charge_ab \
      --skip-cluster-chi --out results/hex18_gamma

Increase Matsubara sizes, thermal samples, Lanczos steps, and bath
numbers after a first successful run. Note that 24-site ED memory is
controlled by the largest adjacent fixed-N particle sector, not
only by the Hilbert dimension at the target N.
