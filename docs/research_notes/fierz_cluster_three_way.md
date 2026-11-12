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
  --source-workers 2 `
  --out results/hex18_fierz_cluster_V0p5_n2
```

Increase `--nbath 6 --nw 49 --nomega 10` for main numerical
results **after** checking the bath size, Matsubara cutoff, field step
and source self-consistency. At finite Q the finite-source method uses
two Hermitian cos/sin fields and two field signs; four channels and
three Fierz embeddings can trigger up to **48** separate, complete,
three-impurity cluster ED+GW self-consistency calculations.
For a quick Green-only check use `--skip-chi`, then start full
susceptibility with only `--channels z_same z_opposite`.

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
   Run separate fully self-consistent finite-source `+h/-h` sources
   (cos and sin components) from identical copies of that mode's
   zero-source restart. This derivative includes feedback of
   `G, P, W, Sigma_GW, Sigma_ED` and reoptimizes every impurity bath.
   Unconverged source solves return an explicit failed status.

The previous analytic cluster susceptibility linearizer is derived
for the density-channel GW self-energy. It has NOT been generalized
to BJ/pseudospin in this change. The matched three-channel comparison
therefore uses the same **finite-source** susceptibility definition
for all three, rather than silently using a density kernel with
a different Fierz Green function.

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
- `chi_comparison.csv`: ED and three full finite-source static chi
  results for each requested physical channel, relative errors and
  source solver residuals; failed/unavailable values are NOT filled in.
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
