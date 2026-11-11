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
