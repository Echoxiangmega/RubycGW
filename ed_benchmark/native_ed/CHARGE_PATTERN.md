# Ruby native ED: real-space site density and charge-order fluctuations

This adds **zero-temperature, fixed-Nf, symmetry-consistent charge-pattern maps** alongside `plot_current_pattern.py`. Uses only NumPy, SciPy, Matplotlib. Works for `ring3`, `3x1`, `2x2` periodic clusters and arbitrary real `t,t1,t2` supported by the native ED. It does not change the Hamiltonian or any existing susceptibility result.

## Quick start (Windows PowerShell or CMD, one-line commands)

From `RubycGW\ed_benchmark\native_ed`:

```powershell
git pull origin main
python test_charge_pattern.py
python plot_charge_pattern.py --input results\chi_ring3_n2 --V 1 2 3 --kind triangle
```

Use **your own scan folder** instead of `results\chi_ring3_n2`. For example, if the scan with `n=2.66667` and `V=1...8` is stored in `results\my_n8_scan`:

```powershell
python plot_charge_pattern.py --input results\my_n8_scan --V 1 2 3 4 --kind triangle
```

If the scan used `--save-state`, the full ground manifold is loaded from each `V_...\ground_states.npz` and **no ED rerun** occurs. Otherwise the script solves **only selected V points**, retaining the original scan results. Keep the original `scan_metadata.json` alongside the `V_...` folders. The script checks `ground_dim` against any previously saved `responses.json`. For near-degenerate manifolds, increase `--low` if necessary.

Options:

- `--kind triangle` (default): maximize **intra-triangle** density disproportionation, directly suited to the HF pseudospin `2+1` CO question.
- `--kind global`: maximize the total spatial variance among all sites, which can also include A–B triangle charge imbalance.
- `--source CO_E_odd_x --polarity positive`: select the **exact** largest eigenvalue state of the specified projected HF charge operator `P0 O P0` instead of maximizing a quartic variance. Available fixed sources: `CO_E_even_x/y`, `CO_E_odd_x/y`, and adaptive `CO_leading_real/imag`. `--polarity negative` selects the smallest eigenvalue, useful for comparing two-small/one-large versus one-small/two-large patterns. By default `--source auto` selects the multi-start maximum spatial variance. Use a different `--output` for different selection prescriptions.
- `--restarts 20`: increase random initial conditions for the nonlinear optimization; results are a numerical maximum, not a mathematically certified global maximum.
- `--output PATH`: choose output directory (default: `<scan>/charge_patterns`).
- `--no-check-H`: skip the final many-body eigenstate-residual check (can save time/memory).

## Three different quantities—do not confuse them

For spinless fermions, the local site number operator is a projector:

\[
\hat n_i^2=\hat n_i,\qquad
\operatorname{Var}_{\Psi}(\hat n_i)=\langle\hat n_i\rangle_\Psi
                (1-\langle\hat n_i\rangle_\Psi).
\]

This is **local quantum occupation fluctuation**. It can be large for a perfectly spatially uniform, non-charge-ordered state. It does **not** independently diagnose charge order.

The **spatial variance of site expectation values** is a different observable:

\[
D_{\rm global}(\Psi)=\frac1{N_s}\sum_{i=1}^{N_s}
  (\langle n_i\rangle_\Psi-N_f/N_s)^2.
\]

For the triangle-specific `2+1` CO:

\[
D_{\triangle}(\Psi)=\frac1{N_s}\sum_{p\in\triangle}
 \sum_{i\in p}\left(\langle n_i\rangle_\Psi-
 \frac13\sum_{j\in p}\langle n_j\rangle_\Psi\right)^2.
\]

These diagnose **static charge disproportionation in the selected state**. They vanish for uniform site-density expectation values even if quantum fluctuations are large. A vanishing value in the unique symmetry-preserving ED ground state does **not** rule out a charge-ordered thermodynamic limit.

To retain correlations invisible in the local mean, we also save and plot the connected charge covariance

\[
 C_{ij}^{nn}=\overline{\langle n_i n_j\rangle}
          -\overline{\langle n_i\rangle}\;\overline{\langle n_j\rangle},
\]

where the bar is the **equal mixture over all exactly/near-degenerate numerical ground eigenvectors** with the solver's specified tolerance. The leading eigenvalue and momentum-sector charge structure-factor information are preserved.

## How degenerate grounds are treated

Let `P0` project onto the complete ED ground manifold with dimension `d`.

1. A **symmetry-unbiased equal mixture** `rho0=P0/d` produces `n_ensemble[i]=Tr(P0 n_i)/d`, `Var(n_i)=n_ensemble[i]*(1-n_ensemble[i])`, and its charge covariance. These results do not depend on the arbitrary basis of the ground manifold.
2. Inside that same manifold, form the **projected site operators** `M_i=P0 n_i P0`. For the physical HF pseudospin `CO_E_even_x/y`, `CO_E_odd_x/y`, and the adaptive leading charge mode (both real quadratures), separately diagonalize `sum_i w_i M_i` and save its min/max exact eigenvalues and trial patterns in `projected_charge_sources.csv`.
3. To display a **maximally charge-disproportionated pure state**, numerically maximize `D_triangle` (or `D_global`) over complex normalized coefficients `c_a` of `sum_a c_a |Psi_a>`. This is a quartic optimization, **not** the same as the linear eigenproblem for current `P0 J P0`. It uses all fixed-source eigenvector candidates plus multiple deterministic random initial conditions. The best attained variance and restart spread are saved; it is not guaranteed to be the global quartic optimum.
4. The selected state's local site densities, triangle pseudospins, particle-number sum, time-reversal-related density invariance and eigen-residual are checked. Near-degenerate states within a finite tolerance have an eigen-residual; their combination is not mathematically an *exact* eigenstate if energies are distinct.

With a **unique** ground state (`d=1`), there is no optimization freedom. The chosen pure state equals the symmetric ground-state density. CO may still show up in `C_ij`, susceptibility or near-degenerate excited states. Artificial density patterns are never added.

Triangle physical-frame pseudospins agree with our HF convention:

\[
\begin{aligned}
\tau_A^x&=2n_0-n_1-n_2, & \tau_A^y&=\sqrt3(n_2-n_1),\\
\tau_B^x&=2n_3-n_4-n_5, & \tau_B^y&=\sqrt3(n_4-n_5).
\end{aligned}
\]

The B-triangle y sign is reversed because of opposite geometric handedness. At a general filling, these are microscopic density contrasts; interpreting them as Pauli matrices requires the local one-particle `E` projection.

## Outputs

Each `<scan>/charge_patterns/V_+.../` contains:

- `site_density.png/.svg`: two real-space panels, **ensemble mean** vs **selected pure state** site density; identical `[0,1]` color scale.
- `density_contrast.png/.svg`: signed `n_selected-n_ensemble` at every site (dynamic diverging scale with real numeric values).
- `onsite_quantum_variance.png/.svg`: `n_i(1-n_i)`, identical `[0,0.25]` scale in both panels. **This is not a direct CO order parameter.**
- `triangle_charge.png/.svg`: per-triangle physical `(tau_x,tau_y)` in the charge-polarized state.
- `charge_covariance.png/.svg`: full `N_s x N_s` connected density covariance of the ground-manifold ensemble.
- `site_density.csv`, `triangle_charge.csv`: site-by-site and triangle-by-triangle numbers, including orbital labels and PBC display coordinates.
- `projected_charge_sources.csv`: eigenstate polarization for the **fixed HF** and adaptive leading CO source directions.
- `charge_pattern.npz`: full ensemble covariance, projected site-density matrices, chosen many-body wavefunction and raw site densities/variances.
- `charge_summary.json`: ground dimension, residual, local/spatial variances, leading charge eigenvalues and momenta, physical `(tau_x,tau_y)`.

Scan root `<scan>/charge_patterns/` also includes `overview.csv`, `charge_variance_vs_V.png`, and `onsite_variance_vs_V.png`. On the V graphs, solid segments/filled points denote unique finite-cluster ground states and dashed segments/hollow points denote ground-state multiplets.

The code **does not label a thermodynamic CO phase** from a finite cluster. Compare density correlations, the `CO_leading` wave vector and susceptibility, energy splitting, and cluster-size dependence. For GW/HF comparison, match the operator normalization and source momentum exactly.