# Hex18 PBC: Fierz-consistent six-site cluster ED+GW benchmark

Run this comparison from the repository root:

```powershell
python research/compare_fierz_cluster_ed_gw.py `
  --V 0.5 --n 2 --T 0.125 `
  --nbath 4 --nw 19 --nomega 5 `
  --K 0.333333333333 0.333333333333 `
  --Q 0.333333333333 0.333333333333 `
  --modes density pseudospin bj `
  --channels x_even y_even z_same z_opposite `
  --out results/hex18_fierz_cluster_V0p5_n2
```

The default cluster susceptibility is **analytic**, with no +/-h
self-consistency. It differentiates the full finite-bath Fierz-matched
fixed-point map for all three modes, including the static HF matrix,
channel polarization, screening, local GW subtraction, bath fit and
impurity ED Green function. It is **not** a bare GG bubble.

For the original finite-source calculation, add
`--cluster-chi-finite-source --source-step 0.001 --source-workers 2`.
At finite Q, this uses two Hermitian cos/sin source components and
both signs; four channels and three embeddings can trigger up to
**48** complete three-impurity source self-consistencies.

Increase `--nbath 6 --nw 49 --nomega 10` for main numerical
results only after checking bath-size and frequency-grid dependence,
bath stationarity and the analytic GMRES residual. Cross-check
representative points with the finite-source route, including a
scan of field step and source tolerance. For a quick Green-only check,
use `--skip-chi`; start with only `--channels z_same z_opposite`.

The benchmark's exact 18-site ED reference is computed **once** for
all Fierz choices. Exact PBC geometry, the physical Hamiltonian,
filling n per six-site primitive cell, frequency grid and target
K/Q are shared by ED and all cluster results.

## Three distinct GW embeddings, one unchanged interacting impurity

For each choice `f=density,pseudospin,bj`, the physical embedding is

\[
\Sigma_{\rm emb}^{(f)} =
\Sigma_{\rm GW,lat}^{(f)}
+ \bigoplus_{C=0}^2
\bigl(\Sigma_{\rm ED}^{C}-
\Sigma_{\rm GW,cluster}^{(f),C}\bigr).
\]

The impurity Hamiltonian is the **same physical density-density model**
for all three choices, with V on the 3 A-triangle and 3 B-triangle
bonds. The impurity is not Fierz decoupled. Each choice is instead a
different approximate GW resummation of the same physical interaction,
and its own local GW skeleton is subtracted to prevent double counting.

The `density` mode retains the historical code path exactly:
the solver's `Sigma_H` stores the Hartree tadpole and `Sigma_emb`
stores Fock+dynamic GW and the local ED-minus-GW correction.

For `pseudospin` and `bj`:
- lattice GW uses the full exact A1+E pseudospin vertices
  (not a discarded A1 sector) or the symmetric bond/current vertices;
- the exact first-order physical Hartree–Fock static self-energy is
  put in `Sigma_H` (a historic field name, now including Fock);
- dynamic screened `W-g` GW is put in `Sigma_emb`, together with
  the local ED-minus-**same-channel** cluster GW subtraction;
- the six-site subtracted GW includes that channel's complete
  physical Hartree–Fock and `W-g` dynamic part.

A `FierzEmbeddingChannels` instance validates that the full lattice
bare interaction contains exactly the six physical intratriangle
density bonds per primitive cell; a nonrepresented intercell V is
rejected. The present Fierz embedding comparison deliberately does
not silently omit extra physical interactions.

At fixed V the three approximate Green functions and susceptibilities
can differ. These are physically useful manifestations of the
approximate GW Fierz ambiguity, not distinct physical Hamiltonians.

## Common zero-field background for each chi

For each Fierz channel:
1. Start from `--initial-state free` (default, no standalone GW)
   or `--initial-state gw` (a GW seed in the **same** Fierz mode).
   A nonconverged finite GW seed may initialize the cluster map,
   but the standalone GW is never reported converged without a valid
   residual.
2. Converge a uniform single six-site impurity with **real** bath.
3. Replicate it to three equivalent primitive cells and relax a
   zero-source **complex**-bath three-impurity fixed point in that
   same Fierz mode.
4. Save that converged zero-source `G(K,iw)` as the cluster reference.
   By default, build the exact linear response of the same finite-bath
   self-consistency equations and solve the matrix-free GMRES response
   for each physical cos/sin component. The static HF response includes
   the off-diagonal elements in the pseudospin and BJ modes.
5. With `--cluster-chi-finite-source`, instead run separate
   fully self-consistent `+h/-h` sources from identical copies of
   that mode's converged zero-source restart. All impurity baths are
   reoptimized. Failed source or analytic solves are reported, with
   **no** fallback to a bare bubble or another approximation.

The analytic response assumes a smooth interior stationary complex-bath
fit. Near fit bounds, nonstationary fits, or ill-conditioned soft modes,
the analytic method may fail its stationarity or GMRES checks.
Its consistency with finite-source responses should be checked
numerically at representative interacting points.

## ED Green reference and numerical reporting

The finite-temperature ED susceptibility uses the thermal momentum
sector trace, while the ED Green reference uses a low-energy
grand-canonical thermal Lanczos truncation. The latter may require
a larger `--ed-sector-padding`, `--ed-thermal-eigs` and
`--ed-krylov-steps` for convergence. For noninteger total particles,
the ED chi uses selected integer charge sectors with the
`--ed-charge-sector-padding` control.

Outputs are written incrementally:
- `green_comparison.csv`: full complex 6x6 G(K,iw) Frobenius error,
  lowest Matsubara frequencies error, trace G(K,tau) error,
  cluster convergence, bath fit and impurity/lattice mismatch.
- `chi_comparison.csv`: ED and three Fierz-matched static chi
  results for each channel. The `chi_method` field identifies
  `analytic` or `finite-source`; it also reports relevant solver
  residuals and elapsed times. Failed values are not filled in.
- `green_curves.npz`: complex 6x6 G(K,iw) and real G(K,tau)
  matrices for ED and all completed embeddings.
- `green_iomega_trace.csv` and `green_tau_trace.csv`:
  Origin-readable real/imaginary traces versus frequency and imaginary
  time for ED and all completed Fierz embeddings.
- `green_tau_comparison.png`, `chi_comparison.png`:
  plots of the three Fierz cluster results against ED, plus
  `green_iomega_comparison.png` for the imaginary-frequency trace.
- `diagnostics.json`: seed choice, time costs, mode-specific GW and
  local impurity fit residuals, and any error messages.

To assess Fierz ambiguity, compare the three methods' relative
Green errors *and* mode ordering of
`chi[z_same]` vs `chi[z_opposite]`. Do not use a small bath fitting
error by itself as proof of accurate susceptibility.
