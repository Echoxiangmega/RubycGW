# Ruby native ED: all-momentum loop-current structure factor and V=0 benchmark

This standalone NumPy/SciPy module works alongside `scan_susceptibility.py`.
It does **not** need QuSpin and does **not** modify existing scan outputs.

## Commands (Windows CMD or PowerShell, single lines)

From `RubycGW\ed_benchmark\native_ed`, after `git pull origin main`:

```powershell
python test_lc_structure.py
python plot_lc_structure.py --input results\chi_ring3_n2 --V 0 1 2 3 4
```

Or for a full existing scan, using all completed `V_*/responses.json` points:

```powershell
python plot_lc_structure.py --input results\chi_ring3_n2
```

Custom hoppings and filling are read **from the scan metadata**. The V=0
reference is always computed for the **same** PBC cluster, total particle
number, and hopping parameters. It is added automatically even when the
original V scan omitted V=0. To avoid new ED runs when the full results were
already processed:

```powershell
python plot_lc_structure.py --input results\chi_ring3_n2 --cache-only
```

`--cache-only` requires files previously generated in `lc_structure/`,
including the V=0 baseline. It does not reconstruct an exact full
triangle-triangle correlation matrix from a scalar `susceptibility.csv`.

If the original scan used `--save-state`, each `ground_states.npz` is
reused, so no ED solve is needed at that V. Otherwise only selected V points
and the needed V=0 reference are diagonalized. For free systems with larger
ground degeneracy, `--low 12` (or larger) may be necessary to fully capture
the low-energy manifold. `--max-basis` controls the projector memory guard.

Example for previously discussed signed hopping:

```powershell
python plot_lc_structure.py --input results\chi_t-06_t06_n3 --V 0 1 2 3 4 5 --low 12
```

## Definitions and normalization

Each triangle has a **physical CCW** loop current (same convention as the
existing ED susceptibility scanner):

\[
I_{\mathbf R,a}=\frac{1}{3}\sum_{(i\to j)\in\partial\triangle_{\mathbf R,a}}
 i\left(t_{ij}c_i^\dagger c_j-t_{ij}^*c_j^\dagger c_i\right),\quad
 a\in\{A,B\},\quad\hbar=1.
\]

For an exactly degenerate multiplet of dimension `d`, use the unbiased
canonical ground mixture `rho_0=P_0/d`. Define

\[
C_{pq}=\operatorname{Tr}(\rho_0 I_p I_q)-
\operatorname{Tr}(\rho_0 I_p)\operatorname{Tr}(\rho_0 I_q),
\]

where `p,q` range over all `2*Nc` triangles, including different cells.
This `C` is Hermitian and positive semidefinite. A pure state selected to
maximize `Jmax` should **not** be substituted into this multiplet-averaged
correlation without an explicit symmetry-breaking prescription.

The momentum-resolved `2 x 2` matrix is

\[
S_{ab}(\mathbf q)=\frac{1}{N_c}\sum_{\mathbf R,\mathbf R'}
 e^{i\mathbf q\cdot(\mathbf R'-\mathbf R)}
 C_{(\mathbf R,a),(\mathbf R',b)},\qquad a,b\in\{A,B\}.
\]

Only allowed **finite-PBC momenta** are used: Gamma/K/K' for `ring3`,
Gamma plus three M points for `2x2`, and three momenta for `3x1`.
The complete 2x2 matrices and their eigenmodes are archived in `lc_structure.npz`.
At Gamma, `u_same=(1,1)/sqrt(2)` and `u_opposite=(1,-1)/sqrt(2)`:

\[
S_{\mathrm{opp}}(\Gamma)=u_{\mathrm{opp}}^\dagger
 S(\Gamma)u_{\mathrm{opp}}=\langle O_{\mathrm{opp}}^2\rangle_c,
\quad O_{\mathrm{opp}}=\frac{1}{\sqrt{2N_c}}
 \sum_\mathbf R(I_{\mathbf R,A}-I_{\mathbf R,B}).
\]

For comparisons across system sizes, the *squared order-density*
diagnostic is `m2_opp_conn=S_opp(Gamma)/(2*Nc)`; nonzero finite-cluster
values alone do not imply thermodynamic LC order.

## Three complementary decompositions

1. **Full connected** `C`: includes on-triangle `p=q` self-variance and
   all `p!=q` correlations. The full `S(q)` is PSD.
2. **Inter-triangle** `C_nonlocal`: remove only diagonal `C_pp`. Its
   `S_nonlocal(q)` can have negative eigenvalues: it is *not* a PSD
   structure factor, only a diagnostic for coherent off-site correlations.
   Per-distance, opposite-staggered pair values are also exported.
3. **Ground zero-mode / finite-energy**: write projected matrices
   `M_p=P0 I_p P0`; then
   `C_zero[p,q]=Tr(M_p M_q)/d - Tr(M_p)Tr(M_q)/d^2`,
   `C_finite=C-C_zero`. This decomposition is basis-independent inside
   the degenerate ground multiplet. It is particularly useful when
   the total static Kubo susceptibility is `NaN` because of a zero mode.

`Jmax` is computed by diagonalizing `P0 O_same/opp P0`. It is supplied as
**auxiliary ground-manifold current capacity**, not treated as an order
parameter. It can be nonzero at V=0 because of one-particle chirality and
ground-state degeneracy.

`Delta_S(V)=S(V)-S(V=0)` and the normalized inter-triangle contributions are
additional diagnostics, **not proof** of interaction-driven long-range order.
A convincing LC phase requires size dependence at compatible q, spectral
information and/or a controlled small symmetry-breaking field.

## Output

Root: `<scan>/lc_structure` unless `--output` is provided.

- `lc_comparison.png`: side-by-side full Gamma correlations, V=0-subtracted
  correlations, ground-zero/finite-energy pieces and Jmax per triangle.
- `lc_Sq_vs_V.png`: largest eigenvalue of the complete 2x2 S(q) for every allowed q.
- `lc_nonlocal_vs_V.png`: onsite-subtracted Gamma same/opposite correlations.
- `lc_structure_vs_V.csv`: precise numbers at each V, baseline differences,
  ground-manifold dimension and size-normalized order-density diagnostics.
- `V_.../lc_structure.npz`: full `C`, `C_zero`, `C_finite`, all 2x2 `S(q)`
  matrices and zero-mode matrices, eigenmodes, Jmax and indexing arrays.
- `V_.../lc_correlations.png`: matrix of all connected triangle-triangle
  correlations, with A/B staggered signs.
- `V_.../lc_correlations_distance.png`: pair-distance resolved staggered
  nonlocal correlations including zero/finite decomposition.
- `V_.../triangle_pair_correlations.csv`: every p!=q pair, minimum-image
  separation and direct/staggered correlations.
- `V_.../triangle_distance_shells.csv`: distance-shell averages.

No existing `susceptibility.csv`, `responses.json`, `sectors.json`, or previous
charge/current plots are deleted or changed.

## Checks

```powershell
python test_lc_structure.py
```

The tests compare fast triangle-current actions with the existing exact Fock
operator implementation, check q sum rules, `S(q)` positivity, the
zero/finite decomposition, `Jmax` zero-mode normalization and invariance of
all full-ground-multiplet results under an arbitrary unitary rotation among
ground vectors on all three PBC geometries.