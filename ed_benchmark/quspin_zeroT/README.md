# QuSpin zero-temperature ED prototype (Ruby lattice)

This is a first **source-code implementation**, not a completed many-body benchmark. The model and PBCs are exactly those in the parent directory. All bond amplitudes are **positive** (`t=+1`, `t1=t2=+0.5`), interactions `V` occur **only** on triangle edges, and fermions are spinless.

## Quick start

Create a Python environment with a supported version of QuSpin, NumPy and SciPy (see [QuSpin installation](https://quspin.github.io/QuSpin/Installation.html)). QuSpin is **not present** in the authoring environment, so the many-body `eigsh` and QuSpin symmetry-projector paths have not been runtime-validated. The geometry/sign tests below **have** passed independently of QuSpin.

```bash
python test_geometry.py
python run_zeroT.py --cluster ring3 --nf 6 --V 1.0 --pair --chi-lanczos 80 --check-signs --save-psi --out results/ring3_N6_V1
python response_iw.py --input results/ring3_N6_V1 --orbitals 0 1 2 3 4 5 --nfreq 30
python run_zeroT.py --cluster 3x1 --nf 6 --V 1.0 --no-point --save-psi --out results/3x1_N6_V1
```

**Validation before scans:** run the 18-site case with `--no-point` and compare ground energies and the exact number of symmetry-state dimensions with the point-group-enabled mode. The code checks the latter. Run `--check-signs` to compare a full fixed-`Nf` QuSpin Hamiltonian to an independently constructed fermion-action matrix-vector product. Raise/disable `--max-full` if postprocessing a very large full fixed-N basis would exceed available RAM.

## Symmetry policy: ordinary C3 / mirrors / gauge-twisted C6

We do **not** presume `t1==t2` to identify valid symmetries. All candidate
permutations are checked against every PBC hopping amplitude and every
intra-triangle `V` edge. The following is for the specific 18/24-site PBCs:

| PBC cluster | generic `t1,t2` | `t1=t2` | `t1=-t2` (both nonzero) |
|---|---|---|---|
| `2x2` | ordinary `C3` and three mirrors | ordinary `C6` and six mirrors | ordinary `C3`/three mirrors plus gauge-twisted `C6` |
| `ring3` | ordinary `C3` and three mirrors | ordinary `C6` and six mirrors | ordinary `C3`/three mirrors plus gauge-twisted `C6` |
| `3x1` | one ordinary mirror | ordinary `C2` and two mirrors | one ordinary mirror plus gauge-twisted `C2` (no `C3`/`C6`) |

The **3x1 finite torus does not preserve C3**, even though the infinite ruby
lattice preserves C3 for arbitrary t1/t2. A generic mirror for the `3x1`
cluster is at 120 degrees in our real-space embedding; for `2x2`/`ring3`,
ordinary generic mirror axes are 0, 60, 120 degrees, measured from x.

The `run_zeroT.py` solver automatically chooses the **full little group of each
momentum sector**, and only combines commuting point-group eigenvalues:
`C6` when ordinary, otherwise `C3` when available, otherwise `C2`, together
with a compatible reflection. Mirrors cannot generally be imposed on
non-real rotation eigenvalue sectors. All sectors are still included when
searching for the global ground state. `--no-point` switches this off for
comparison. This sector implementation needs verification with an installed
QuSpin, in particular the sum of projected dimensions and energies.

**Gauge-twisted C6 at `t1=-t2`:** Let `G` flip the sign of fermion operators
on the B triangle (orbitals 3,4,5) but not A (0,1,2); let `R6` rotate 60
degrees around the physical hexagon center. `G R6` preserves H when
`t1=-t2`; its sixth power is `(-1)^Nf` on the Fock space. Negative indices
in QuSpin site-permutation maps correspond to **particle-hole transforms**,
not these on-site phase changes. We therefore use an **exact local U(1)
gauge change**, `H' = D H D^dagger` with `D` diagonal in the occupation
basis, so the 60-degree site permutation is ordinary for H'. QuSpin can
then use `C6` blocks directly. The physical eigenvalue of `G R6` in a
fixed-Nf block whose QuSpin C6 label is q is
`exp(i*pi*Nf/6) * exp(-2*pi*i*q/6)` for the gauge convention used in
`geometry.twisted_c6_gauge`.

This mode is **automatic** for `2x2`/`ring3` at `t1=-t2` (unless `--no-point`
or `--no-twisted-c6` is passed). The resulting **wavefunction is restored
to the ORIGINAL hopping gauge before any rho/current/flux/Green-function
calculation or saved psi**. In the C6-adapted gauge, plain mirror blocks
are disabled: the original mirror is generally gauge-dressed and cannot
be represented by a site permutation alone. We instead use C6; a future
custom signed-map basis could combine both. For `3x1`, no sixfold
rotation of the PBC exists; its gauge-twisted C2 is **verified but not yet
used to reduce the QuSpin blocks**. Ordinary mirror blocks still work.

**Independent tests:** `python test_point_symmetry.py` verifies all ordinary
and signed symmetries for `(t1,t2)=(0.5,0.5),(0.5,-0.5),(0.6,0.2)` and
all three PBCs. It also verifies fermionic signs, the relation
`(G R6)^6=(-1)^Nf`, the gauge-transformed Hamiltonian, and the reverse
many-body wavefunction gauge change without QuSpin.

Examples:

```bash
python test_geometry.py
python test_point_symmetry.py
python run_zeroT.py --cluster ring3 --nf 6 --t1 0.6 --t2 0.2 --V 1 --out results/generic
python run_zeroT.py --cluster ring3 --nf 6 --t1 0.5 --t2 -0.5 --V 1 --save-psi --out results/twisted_C6
python run_zeroT.py --cluster 3x1 --nf 6 --t1 0.6 --t2 0.2 --V 1 --out results/stripe_mirror
```

## Outputs (each parameter point)

- `geometry.json`: all sites, edge list, PBC winding, symmetry permutations, model parameters and conventions.
- `sectors.json`: dimensions, labels and low eigenvalues for every symmetry block.
- `summary.json`: global ground energy, sector, near-degenerate manifold count; leading equal-time charge/current/pair strength when enabled.
- `equal_time.npz`: site densities and connected charge covariance, one-body density matrix `rho[i,j]=<c_i^dag c_j>`, three-/four-/six-site contractible plaquette loops, current correlations, gauge-invariant correlation-loop phase `arg(prod rho_ij)` (not electromagnetic flux), pairing Gram matrix on NN bonds if `--pair`; optionally static linear susceptibilities in three selected channels (`--chi-lanczos`).
- `ground_state.npz`: full fixed-N basis ordering and ground vector if `--save-psi` (can be large).
- `green_iw.npz` (from `response_iw.py`): zero-temperature cluster Matsubara Green matrix for a chosen orbital subset; includes `omega`, all matrix entries, chemical potential choice and N±1 addition/removal energies. The reference `beta` sets **only the imaginary-frequency grid**, not a finite temperature.

## Physics cautions

- **Maximize J:** `--max-current` diagonalizes `P J_loop P` in a numerically near-degenerate low-energy manifold selected by `--degeneracy-tol`. It records the maximum possible current **and the energy window**, not a spontaneous current expectation in a unique finite-size time-reversal-symmetric eigenstate. For a reliable order comparison, also compute size scaling and/or source-field response.
- **Flux:** bare hopping (Peierls) flux, `arg(prod <c_i^dag c_j>)`, and the flux of an effective mean-field hopping are **not identical**. Gauge-invariant correlation flux is undefined when the complex loop product vanishes. A real correlation product with phase `pi` does not imply broken time reversal.
- **Pairing:** for spinless fermions onsite s-wave pairing is forbidden. Only intersite anti-symmetric bond pairs are included. At fixed Nf, `<c_i c_j>=0`; the pairing Gram matrix is an equal-time indicator, not proof of superconductivity. A true pairing susceptibility needs N±2 sectors and a specified pair field.
- **Susceptibility:** static `chi(0)` is extracted from a zero-temperature Krylov spectral approximation in the **complete fixed-N basis**. Quasi-zero energy poles are flagged because a susceptibility can diverge for exact degeneracy. `--chi-lanczos` determines the maximum Krylov depth; convergence must be checked.
- **Physical current:** `I_{i->j}=i*t_ij*(c_i^dag c_j-c_j^dag c_i)` in units with `hbar=1` (particle current); electrical charge factors can be restored separately.
- **Sector-projected operators:** a local loop current or bond-pair field need not preserve the symmetry labels of the ground state. Consequently correlations and resolvents are computed in full fixed-`Nf` (or adjacent `Nf`) spaces rather than squaring the operator restricted to a single momentum sector.

## Next steps

Once QuSpin is available, verify every case with and without point-group reduction, and compare first at `V=0` against the filled single-particle spectrum. Then scan `V` and integer `Nf`, extract charge/current/pair eigenmodes and dynamic response data; only after these tests interpret a finite-size correlation-dominance map as an incipient phase diagram.

## Batch scans

```bash
python scan_phase.py --cluster ring3 --nf 6 7 8 9 --V 0 0.5 1 1.5 2 3 --pair --chi-lanczos 80 --output results/ring3_scan
```

The scan resumes by skipping folders with an existing `summary.json`; it writes a compact `summary.csv` and preserves each individual spectral/correlation archive. No absolute phase assignment is made by comparing differently normalized channels.