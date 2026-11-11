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
--ed-sector-padding, --ed-thermal-eigs and --ed-krylov-steps.
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

## Outputs

- comparison.csv: method, observable, value, ED reference, relative
  error where meaningful, incremental run time, status, and caveats.
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
