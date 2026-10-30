# Fixed-filling zero-temperature ED susceptibility scan

This is a **standalone NumPy/SciPy** workflow in `ed_benchmark/native_ed`. No QuSpin or Numba.
It scans the intra-triangle density interaction `V` at **fixed integer total particle number** `Nf`, using the primitive-cell filling `n=Nf/Nc` and fixed `t=1`, `t1`, `t2`. The underlying Hamiltonian and all PBC hoppings are unchanged.

## Windows PowerShell / CMD quick start

From your local `RubycGW/ed_benchmark/native_ed` directory (after `git pull origin main`):

```powershell
python -m pip install -r requirements.txt
python -m pip install matplotlib
python test_susceptibility.py
```

A first **fast CO/LC** scan at the default hopping:

```powershell
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --Vmin -1 --Vmax 2 --Vstep 0.5 --depth 60 --no-extra-loops --output results\chi_ring3_n2
```

To calculate the full set of **triangle, square, hexagon current** responses (slower), simply omit `--no-extra-loops`:

```powershell
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --Vmin -1 --Vmax 2 --Vstep 0.5 --depth 80 --output results\chi_ring3_full
```

To test another fixed pair of inter-triangle hoppings:

```powershell
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.6 --t2 0.2 --Vmin -2 --Vmax 3 --Vstep 0.25 --depth 80 --output results\chi_ring3_t06_t02
```

Or choose exact V values (`--V` overrides the range):

```powershell
python scan_susceptibility.py --cluster 3x1 --nf 6 --t1 0.5 --t2 -0.5 --V -1 0 0.5 1 2 --depth 60 --output results\chi_3x1_n2
```

Generate all plots again **without running ED**:

```powershell
python plot_susceptibility.py --input results\chi_ring3_full
```

The main figures are `chi_overview.png`, `chi_charge.png`, `chi_loop_current.png`, and (when selected) `chi_pairing.png`. The singular/near-degenerate response diagnostic is `chi_zero_mode_weight.png`.

**Optional pairing response** (more costly, may become undefined if the fixed-N sector is not locally stable in the grand canonical ensemble):

```powershell
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --V 0.5 1 1.5 --depth 60 --pair --no-extra-loops --output results\chi_ring3_n2_pair
```

To retain the many-body ground manifold, compatible `geometry.json` and one representative `ground_state.npz` for later use in `response_iw.py`, add `--save-state` to the initial scan. A multiple-degenerate-state manifold is saved in `ground_states.npz`; the single-vector `ground_state.npz` is **not** a unique or canonical physical ground state at degeneracy.

## Zero-T response definition

For a Hermitian number-conserving observable `O` and **nondegenerate** ground state:

\[
\chi_O(0)=2\sum_{m\ne0}\frac{|\langle m|O|0\rangle|^2}{E_m-E_0}.
\]

This is the static derivative of the source-induced order parameter for `H_h=H-h O`. It is **not** `beta*variance` and has units of inverse hopping energy when `O` is dimensionless, or current squared / energy for a particle-current operator.

The solver searches **all symmetry sectors** for its ground manifold, constructs the low-energy vectors in the complete fixed-`Nf` Fock basis, and uses a **fully reorthogonalized Lanczos response calculation** on the full fixed-`Nf` Hamiltonian. At exact degeneracy the code evaluates the maximally mixed ground-manifold average (rather than selecting an arbitrary single eigenvector), deflates the **whole** ground manifold, and measures the zero-mode weight of the operator inside it. A nonzero zero-mode weight means the `T=0`, `h→0` response is nonanalytic/undefined without an explicit selection prescription. The program sets `chi_static=NaN` and records `status=zero_mode`; `chi_regular` is the finite part *excluding* the degenerate subspace, **not the full static susceptibility**. Very low unresolved poles and incomplete ground manifolds are also flagged. The `chi_zero_mode_weight.png` figure is the correct additional diagnostic.

`--low` is the number of Lanczos eigenvectors sought in **each symmetry block**, default 5; increase it if `ground_incomplete=True`. `--depth` is the number of resolvent Krylov steps. `*_krylov_difference` compares the result to one from half the Krylov depth as a rough convergence indicator; rerun selected V points with more steps for quantitative precision. `--pole-tol` controls the low-pole detection threshold.

## Operator normalization and channels

Each elementary **plaquette loop** has a geometrically CCW orientation and its current operator is the average directed particle current over its sides:

\[
I_p=\frac{1}{|\partial p|}\sum_{(i\to j)\in\partial p}
 i\,[t_{ij}c_i^\dagger c_j-t^*_{ij}c_j^\dagger c_i],\quad\hbar=1.
\]

For the two triangular sublattices (A = 0,1,2 and B = 3,4,5), both CCW:

\[
O_{\rm same}=\frac{1}{\sqrt{2N_c}}\sum_{p\in\triangle}I_p,\qquad
O_{\rm opposite}=\frac{1}{\sqrt{2N_c}}\sum_{p\in\triangle}s_pI_p,\quad s_{A/B}=\pm1.
\]

This is *physical* same/opposite circulation, not the earlier raw Gamma-plus/minus naming. `LC_square` and `LC_hexagon` are the similarly normalized sums of geometrically CCW current operators around each elementary 4/6-site plaquette. For all plaquette channels, `sum_p |w_p|^2=1`; the magnitudes are meaningful within the same operator definition across V and must not be treated as phase labels by comparing against a density response with different operator units.

Charge:

* `CO_AB` uses the **fixed** q=Gamma A-minus-B density contrast `O=sum_i w_i n_i`, with `w_{A/B}=±1/sqrt(Ns)`.
* `CO_leading` is **adaptive**: at each V, diagonalize the charge connected covariance matrix in the **allowed supercell momenta**. For the largest eigenvalue, use normalized real/imaginary components of its site-space charge vector as Hermitian density sources. Their responses are summed. The mode/q can change along V, and the selected `q` and orbital coefficients are saved. It is **not**, in general, the maximum eigenvalue of the full dynamical susceptibility matrix.

Pairing (optional `--pair`):

* The leading spinless NN bond pair mode `Δ=sum_b v_b c_i c_j`, with `||v||=1`, comes from the equal-time pair Gram matrix at each V; its coefficients are saved. It need not have the same p/f representation for different V.
* The Hermitian pairing source `Δ+Δ†` connects `Nf` to `Nf±2`. The code computes its static susceptibility from `H-μN` using the `Nf-2` and `Nf+2` **resolvents** and a chosen `μ` (default the midpoint of the `Nf±1` addition/removal chemical potentials). It checks that both `Nf±2` excitation gaps in `H-μN` are positive. Otherwise its status is `pair_source_unstable_grand_canonical` and no finite χ is reported. This is different from merely examining a fixed-N equal-time pair correlation eigenvalue.

## Output files

Each `V_+...` parameter directory contains:

* `responses.json`: parameters, E0, ground-manifold dimension, selected charge momentum, every χ value **and status**, loop definitions, and optional pair-mode form factors.
* `spectra.npz`: Lehmann/Lanczos poles and weights **for each channel and each independent source**. These are needed to compare frequency-dependent response and to audit low-energy singularities.
* `sectors.json`: per-symmetry-sector low energies.
* `geometry.json`: PBC and hopping conventions. Optionally `ground_states.npz` and `ground_state.npz` with `--save-state`.

The root contains `susceptibility.csv`, individual plot PNGs, `scan_metadata.json`, and `failures.json`. A scan is resumable and `--V` can be extended later while preserving all already computed points. To prevent silent numerical miscomparisons, **different t1/t2, n, symmetry or response-accuracy parameters require separate output folders**.

## Validation and limitations

Run `python test_susceptibility.py`: tests dense exact Lehmann sums for small sectors, fermion-current/charge operators, symmetry-reduced ground energies, zero-mode flags, optional `Nf±2` pairing response against independent direct sparse resolvents, CSV/plot generation, resume and noninteger filling rejection.

The expensive full fixed-`Nf` resolvent and projector implementation targets **18-site** `ring3` and `3x1` clusters first. For 24-site half-filling, D=2,704,156, and the current projectors/resolvents are not memory-optimized. For the 18-site clusters `n` is quantized in steps `1/3` (because `Nf=3n` must be integer); for 2×2 it is quantized in steps `1/4`. This script is zero-T fixed-N ED, not finite-temperature or grand-canonical interpolation.

The CO/LC responses can identify a *leading instability tendency* and are especially useful for comparison against HF/GW computed with the same source, sign, normalization, k-grid, and fixed-N finite torus. They do not automatically certify a thermodynamic broken-symmetry phase, and large χ can be induced by low gaps or pre-existing degeneracy. Finite-size cross-checks remain necessary.