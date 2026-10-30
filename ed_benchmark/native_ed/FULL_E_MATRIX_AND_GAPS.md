# Zero-temperature Ruby ED: full E-charge response matrix, gaps, and mode tracking

This is an extension of `scan_susceptibility.py` in `native_ed/`. It uses **only NumPy/SciPy**, the existing spinless Hamiltonian, the same PBC edge list and the same fixed-`Nf` ensemble. No QuSpin required.

## Source operator convention: physical HF pseudospin frame

In each primitive cell, use `A=(0,1,2)` and `B=(3,4,5)`:

```
tauAx = 2*n0 - n1 - n2                 tauBx = 2*n3 - n4 - n5
tauAy = sqrt(3)*(n2 - n1)             tauBy = sqrt(3)*(n4 - n5)
```

These microscopic density operators reduce to local `tau_x,tau_y` Pauli pseudospins only in the strong-coupling, one-fermion-per-triangle E-doublet subspace. For all fillings, they remain valid number-conserving, Hermitian charge sources.

Define four **normalized** uniform (`q=Gamma`) fields in the fixed order:

```
E_CHANNEL_NAMES = (CO_E_even_x, CO_E_even_y, CO_E_odd_x, CO_E_odd_y)
O_even_x = sum_R (tauAx(R) + tauBx(R)) / sqrt(12*Nc)
O_even_y = sum_R (tauAy(R) + tauBy(R)) / sqrt(12*Nc)
O_odd_x  = sum_R (tauAx(R) - tauBx(R)) / sqrt(12*Nc)
O_odd_y  = sum_R (tauAy(R) - tauBy(R)) / sqrt(12*Nc)
```

`CO_E_even`/`CO_E_odd` are the corresponding half-traces. They are not independent eigenmodes when non-diagonal E responses are present. **Do not** compare raw CO, LC and pairing susceptibilities as if these operators had identical units.

## Full 4x4 response, including off-diagonal matrix elements

For a **nondegenerate** ground state, the response to `H(h)=H-sum_a h_a O_a` is

```
chi_ab = 2 * Re sum_{m>0} [ <0|O_a|m> <m|O_b|0> / (Em-E0) ]
```

This is the real symmetric static source-field Hessian, not the general-frequency dynamical matrix. The implementation reuses each source's fully reorthogonalized, ground-deflated Lanczos Krylov resolvent:

```
f_a = (O_a - <O_a>)|0>
chi_ab = 2*Re <f_a|(H-E0)^(-1)_excited|f_b>
```

At exact ground-state degeneracy, the code instead averages over the **complete** ground manifold and also saves the ground-subspace zero-mode Gram matrix. If a zero mode is present (or the ground manifold is incomplete, or a low Lanczos pole unresolved), the full static `chi`/eigenvalues are **NaN**. The finite `chi_regular` excludes the ground manifold and **must not** be silently interpreted as the full static response. In this case `CO_E_matrix_status` and `CO_E_matrix_zero_eigenmax` identify the problem.

For every V, the full `chi_regular[4,4]`, `chi_static[4,4]`, `zero_mode_matrix[4,4]`, four eigenvalues and eigenvectors are saved to `V_.../co_e_matrix.npz`. The `responses.json` also contains the 4x4 matrices.

**C3 degeneracy:** the largest eigenvalue may be exactly two-fold degenerate. Then individual leading eigenvectors are arbitrary within this subspace. The code saves the gauge-invariant **leading eigenspace projector** (`leading_projector`) and its even/odd weights, and reports the eigenspace rank. Use these to identify the candidate CO pattern, not an arbitrary one-dimensional eigenvector orientation.

## Energy gaps and source-active gaps

From all symmetry-sector low energies (including exact degeneracies), the scan writes:

- `gap_E1`, `gap_E2`, `gap_E3`, `gap_E4`: ordered `E_k-E0`, with repeated E0 for exact ground-state degeneracies.
- `gap_above_manifold`: the smallest resolved energy above the whole ground manifold, using `--degeneracy-tol`.
- `CO_leading_active_gap`, `CO_E_even_x_active_gap`, etc. and the associated `*_active_weight`: the smallest **positive** Krylov pole whose weight exceeds `max(1e-12, --active-weight-fraction * (sum of source weights))`. Default relative threshold is `0.001`. These source-active gaps depend on Krylov depth and threshold; increase both for convergence checks. At degeneracy they refer to the **regular part only**.

The adaptive `CO_leading` mode can select a finite `q` not included in the 4x4 *Gamma* E basis. Its momentum `CO_leading_q1/q2`, projection `CO_leading_E_fraction`, and normalized overlap with the previous-V mode `CO_leading_overlap_prev` are saved. The checker treats `q` and `-q` as conjugate representations of the **same real charge modulation**, and records `CO_leading_q_conjugated_to_prev=True` instead of falsely diagnosing a mode switch.

## Ground-state degeneracy line convention

For each figure, **solid segments + solid circles** connect unique ground states. **Dashed segments + hollow circles** mark a segment touching a degenerate ground manifold; x markers mean an incomplete manifold. The choice is descriptive only, not itself evidence of symmetry breaking.

For susceptibility, non-`ok` points are **not** connected as valid finite responses. The flagged **regular** contribution may be shown separately as an x marker, never as the full static susceptibility.

## Main generated files

- `susceptibility.csv`: every V's energy, multiplet dimension, gap, full E-channel eigen-response strength, leading eigenspace even/odd weights, charge momentum/overlap, individual CO/LC responses and zero-mode status.
- `gap_levels.png`: `E1-E0`, `E2-E0`, `E3-E0` and `gap_above_manifold`.
- `gap_active.png`: lowest sufficiently weighted CO/LC pole gaps.
- `ground_degeneracy.png`: dimension of the ground manifold.
- `chi_E_matrix_leading.png`, `chi_E_matrix_eigenvalues.png`, `chi_E_matrix_character.png`: full E-matrix eigen-response, full spectrum, even/odd character.
- `charge_mode_identity.png`, `charge_mode_momentum.png`: track adaptive charge eigenmode.
- `chi_charge.png`, `chi_charge_components.png`, `chi_loop_current.png`, `chi_overview.png`, `chi_zero_mode_weight.png`.
- `V_.../co_e_matrix.npz`, `spectra.npz`, `sectors.json`, `responses.json`: full numerical data for reuse.

## Windows CMD / PowerShell: run a fresh scan

From `RubycGW/ed_benchmark/native_ed`:

```powershell
git pull origin main
python test_susceptibility.py
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --Vmin 1 --Vmax 5 --Vstep 0.25 --depth 80 --no-extra-loops --output results\chi_E_matrix_gaps_n2
```

If other hopping parameters are needed, pass different `--t1` and `--t2`, and use a separate output folder. A previous `CO_AB`-only scan is intentionally incompatible with the new channel schema: recompute the ED data in a fresh folder. The script caches the fixed-`Nf` symmetry basis and reuses it for different V within one scan.

Plot again, **without ED reruns**:

```powershell
python plot_susceptibility.py --input results\chi_E_matrix_gaps_n2
```

Plot the full 4x4 E charge susceptibility at one measured V:

```powershell
python plot_e_matrix.py --input results\chi_E_matrix_gaps_n2 --V 3
```

If that V has a flagged zero mode, explicitly ask for **only the regular part**:

```powershell
python plot_e_matrix.py --input results\chi_E_matrix_gaps_n2 --V 3 --regular
```

## Interpretation limits

A susceptibility peak is evidence of a strongly softening **finite-size charge response**, **not** proof of a thermodynamic phase transition. A convincing CO claim additionally needs stable source identity, matching symmetry of the low-energy excited states, and checks across clusters or weak source fields. When `CO_leading` occurs at K/K' and the E matrix is defined at Gamma, they are **different momentum sectors**: do not identify the former as the HF uniform E charge mode. The 18-site ring supercell samples K/K'; 2x2 samples a different momentum set and its 24-site half-filled benchmark remains resource-intensive.