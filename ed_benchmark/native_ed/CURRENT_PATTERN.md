# Ruby native ED: real-space loop-current maps

This visualization is **independent of QuSpin** and uses the same `geometry.py`,
PBC hopping, time-reversal convention and physical triangle chirality as the
native zero-temperature ED susceptibility scan.

## Windows PowerShell / CMD

From `RubycGW\ed_benchmark\native_ed` (after `git pull origin main`):

```powershell
python test_current_pattern.py
python plot_current_pattern.py --input results\chi_ring3_n2 --V 1 2 3 --channel opposite
```

Change the directory to **your actual scan directory**. For an earlier scan of
`n=2.66667` at `t1=-0.6`, `t2=+0.6`, for example:

```powershell
python plot_current_pattern.py --input results\chi_n2667 --V 1 2 3 --channel opposite
```

The tool reads the `scan_metadata.json` and each point's `responses.json`.
If `V_+...\ground_states.npz` exists (from `--save-state`), it uses that
**without re-running ED**. If not, it solves **only the specified V points**
using `VScanSolver` and the scan's hopping/filling/symmetry convention. The
original scan data are not changed. For a single point directory, omit `--V`:

```powershell
python plot_current_pattern.py --input results\chi_ring3_n2\V_+1.000000000
```

Optional flags: `--channel same`, `--output results\my_current_maps`,
`--arrow-threshold 0.01`, `--no-check-H`, `--low 8`, `--max-basis 70000`.
If re-solving returns a different ground-manifold dimension from the original
scan, stop and check the degeneracy tolerance/eigensolver rather than silently
accept an inconsistent optimized state.

## What is plotted?

For each **independent undirected hopping bond**, with the representative
orientation `(i,j)` from the existing `geometry.py`:

`J[i -> j] = i * tij * (<c_i^dag c_j> - <c_j^dag c_i>) = -2*tij*Im(rho_ij)`,

where `rho_ij=<c_i^dag c_j>` and hbar = 1. The arrow points in the **actual
physical particle-current direction**, not necessarily the direction of the
stored hopping representative. Thickness scales with absolute current.
The faint underlying bond color encodes the hopping type (`t`, `t1`, `t2`).
A dashed bond and hollow endpoint mean that the destination is shown as a
periodic image. The edge is **not omitted**; the reverse term is included
by Hermitian conjugation. For the `ring3` cluster we select periodic images
such that **six complete triangles surround a physical hexagonal plaquette**.

All triangle loops are oriented **geometrically counterclockwise**. We compute
`I_p = (J_01+J_12+J_20)/3` around each triangle, and

`O_opposite = (sum_{A triangles} I_A - sum_{B triangles} I_B) / sqrt(N_triangles)`,

`O_same = (sum_{A triangles} I_A + sum_{B triangles} I_B) / sqrt(N_triangles)`.

This is the **physical** same/opposite convention, not the old algebraic
Gamma-plus/minus vertex labels whose triangle orientations differ.

The code diagonalizes the exact finite-size ground-space matrix
`J0 = P0 O_channel P0`, selects the eigenvector of the **largest signed**
eigenvalue (as selected by an infinitesimal positive source `h`), and forms
that actual many-body superposition before computing any bond current.
The time-reversed wavefunction is simply its complex conjugate in the
original real-hopping Fock basis, and its currents are checked to reverse.

At every site we check the current-conservation law
`divJ[i] = sum_outgoing J - sum_incoming J = 0` for a stationary eigenstate.
We also record the net PBC winding current `sum_b J_b * winding_b` in the
canonical supercell basis. This helps distinguish a local orbital current
from a persistent transport current around the periodic torus.

## Output (per chosen V)

At `SCAN/current_patterns/V_+1.000000000/`:

- `bond_current.png`, `bond_current.svg`: **all sites and hoppings**, with arrows
  in the physical direction, PBC images, and numeric summary;
- `triangle_current.png`, `triangle_current.svg`: actual signed current in
  each of the six triangular plaquettes, with the same CCW convention;
- `bond_current.csv`: every microscopic bond current, type, density-matrix
  element, and canonical/display PBC winding;
- `plaquette_current.csv`: every contractible triangle, square, and hexagon
  loop with signed circulation, bare Peierls phase, and the **correlation
  loop phase** `arg(product <c_i^dag c_j>)` (not a real electromagnetic flux);
- `current_pattern.npz`: arrays for downstream comparison, including the
  optimized wavefunction, its time-reversal partner, site coordinates, edge
  currents, PBC windings and current-conservation residuals;
- `current_summary.json`: source parameters, selected ground-space current,
  eigenvalues, current per triangle, maximal current-conservation error,
  time-reversal error and net PBC winding current.

The scan-wide `current_patterns/overview.csv` contains one row per V.

## Physical interpretation

A nonzero optimized `O_opposite` **within a genuinely degenerate ground
manifold** proves that this finite cluster admits an exact ground eigenstate
with an opposite-triangle-current expectation. It does **not** by itself
establish a thermodynamic spontaneous LC phase. Check (a) symmetry-sector
energies and the actual numerical splitting, (b) complete spatial current
patterns and current conservation, (c) absence of unwanted torus winding
current, and (d) larger/differently shaped clusters and source-field response.
If `ground_dim=1`, the optimized current should vanish for this real-hopping
spinless model; do not artificially mix excited states to create one.

## Validation

```powershell
python test_current_pattern.py
```

This tests all three PBC supercells (including negative intertriangle
hopping), hopping length and image winding, matrix-current sign against the
existing fermionic operator implementation, time-reversal reversal,
plaquette current agreement, continuity and the figure/CSV pipeline.
A saved twofold-degenerate `ring3, n=2, V=1, t1=t2=0.5` example gave
`O_opposite=+0.5987032793`, triangle A/B currents `+/-0.24442`, maximum
current divergence `2e-16` and net torus winding below `1e-16` in this
example. These are **a demonstration of the plotting code**, not new
results for another choice of `n,t1,t2`.