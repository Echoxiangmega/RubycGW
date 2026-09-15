# Bulk orbital magnetization

This module evaluates the interacting bulk orbital-magnetization formula of R. Nourafkan, G. Kotliar, and A.-M. S. Tremblay, *Phys. Rev. B* **90**, 125132 (2014), Eq. (2), for the 18-site Ruby GW checkpoint.

For a line-by-line derivation of the second term, including the exact role of Nourafkan Eq. (A13), the repository-defined Jacobian-vector operator `C_GW`, the Hartree/Fock/MT/AL decomposition, the self-consistent `G_B`--`Sigma_B` loop, and the matrix-free GMRES implementation, see [uniform_B_self_energy_derivation.md](uniform_B_self_energy_derivation.md). That note also distinguishes notation taken from the paper from notation introduced only for this implementation and lists the relevant references.

## 1. Complete Eq. (2)

For a two-dimensional system and the `z` component,

\[
M_z=M_z^{(1)}+M_z^{(2)}.
\]

The first term is

\[
M_z^{(1)}=
\frac{i e}{2\hbar}\frac{T}{N_k}
\sum_{\mathbf k,n}
\mathrm{Tr}\left\{
\left[H_0-\mu+\frac{\Sigma}{2}\right]
G J_x G J_y G
-(x\leftrightarrow y)
\right\},
\]

with

\[
J_\alpha=-\frac{\partial G^{-1}}{\partial k_\alpha}
=D_\alpha H_0+D_\alpha\Sigma,
\qquad
\Sigma=\Sigma_H+\Sigma_{GW}.
\]

The second term is

\[
M_z^{(2)}=
\frac{T}{2N_k}\sum_{\mathbf k,n}
\mathrm{Tr}\left\{
\left[H_0+(i\omega_n-\mu)I\right]
G
\left.\frac{\partial\widetilde\Sigma^{(B)}}{\partial B_z}\right|_{B=0}
G
\right\}.
\]

The local-DMFT example in Nourafkan et al. has `partial_k Sigma=0` and `partial_B Sigma_tilde^(B)=0`. Those simplifications do not hold automatically for nonlocal GW.

The repository now evaluates both terms for self-consistent GW. For `V=0`, `M2=0` exactly. For `V != 0`, the gauge-invariant uniform-field derivative `Sigma_B` is solved with a matrix-free linear response described below.

## 2. Physical k derivative and Ruby orbital embedding

The repository cell-gauge convention is

\[
X_{IJ}(\mathbf k)=\sum_{\mathbf S}X_{IJ}(\mathbf S)
 e^{2\pi i\mathbf k\cdot\mathbf S}.
\]

A physical current vertex must use the complete displacement

\[
\mathbf d_{IJ}(\mathbf S)=
S_1\mathbf T_1+S_2\mathbf T_2+\mathbf r_J-\mathbf r_I.
\]

Therefore

\[
[D_\alpha X]_{IJ}(\mathbf k)=
 i\sum_{\mathbf S}d_{IJ,\alpha}(\mathbf S)
 X_{IJ}(\mathbf S)e^{2\pi i\mathbf k\cdot\mathbf S}.
\]

This embedding term is essential. Differentiating only the cell phase would incorrectly make every intra-triangle `S=(0,0)` hopping contribute zero current vertex.

For `H0`, the derivative is evaluated analytically from `supercell_hoppings`. For `Sigma(k,iw)`, the sampled k mesh is Fourier transformed to the represented real-space harmonics, multiplied by `i d_alpha`, and transformed back. This is a spectral derivative rather than a nearest-neighbor finite difference.

## 3. Uniform-B gauge-invariant self-energy derivative

Nourafkan Appendix A factors the Peierls phase from the Green function and self-energy. In terms of the dimensionless field

\[
b=\frac{ea^2}{\hbar}B_z,
\]

the gauge-invariant Green-function derivative obeys

\[
G_b=Y_b+G\Sigma_bG,
\qquad
\Sigma_b=\frac{\partial\widetilde\Sigma^{(B)}}{\partial b}\bigg|_{b=0}.
\]

Using Eq. (A13) and `J_alpha=-D_alpha G^{-1}`, the explicit geometric source is

\[
Y_b=-\frac{i}{2}
\left[
GJ_xGJ_yG-GJ_yGJ_xG
\right].
\]

For the self-consistent split-GW functional

\[
\Sigma=\Sigma_H+\Sigma_F+\Sigma_c,
\qquad
\Sigma_c=-G*(W-V),
\]

the linearized self-energy response to an arbitrary Green-function tangent `X` is

\[
\mathcal C_{GW}[X]
=\delta\Sigma_H[X]+\delta\Sigma_F[X]
+\delta\Sigma_{MT}[X]+\delta\Sigma_{AL1}[X]+\delta\Sigma_{AL2}[X].
\]

Here `C_GW` is repository notation for the Jacobian-vector product `(delta Sigma_GW / delta G) X`; it is not notation used by Nourafkan et al. The detailed derivation and code mapping are given in [uniform_B_self_energy_derivation.md](uniform_B_self_energy_derivation.md).

The magnetic self-energy derivative therefore satisfies

\[
\boxed{
(I-L_{GW})\Sigma_b=\mathcal C_{GW}[Y_b]
}
\]

with

\[
L_{GW}[S]=\mathcal C_{GW}[GSG].
\]

The repository solves this equation with the same restarted matrix-free GMRES strategy used by supercell cGW. The final equation residual is evaluated directly from

\[
\Sigma_b-\mathcal C_{GW}[Y_b+G\Sigma_bG].
\]

No finite magnetic supercell and no finite-B subtraction are needed.

For the density-density interaction used here there is no explicit linear coupling of the interaction to `B`; the gauge-invariant linear field dependence of GW therefore enters through `G_b`. The precise GW-level identification of the gauge-invariant tilde self-energy derivative is an implementation assumption that should continue to be checked with independent uniform-field or Ward-identity benchmarks; see the caveat section of the detailed derivation note.

## 4. Finite mesh caveat

A `3x3` k mesh can represent only a very short-ranged periodic real-space self-energy. The spectral derivative is the exact derivative of that finite Fourier representation, but it is not a substitute for k-mesh convergence. Production bulk-M calculations should compare several meshes, for example `nk=3,4,6,8`, and also increase `nw`.

The second term uses the same finite Matsubara box as cGW. The geometric source falls rapidly at large frequency, but `nw` convergence should still be checked explicitly.

## 5. Units

The code reports `M_code` such that, when the hopping-energy unit is `E0` and the lattice-length unit is `a`,

\[
m=\frac{e}{\hbar}E_0a^2 M_{\rm code}.
\]

Supply

```bash
--energy-unit-ev E0 --lattice-constant-angstrom a
```

to obtain Bohr magnetons. The 18-site supercell contains three primitive Ruby cells, so the CLI reports both the supercell value and `M/3` per primitive cell.

## 6. Command line

The default command now computes `M1`, solves `Sigma_B`, evaluates `M2`, and reports the complete Eq. (2):

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --npz bulk_M.npz \
  --json bulk_M.json
```

Uniform-B response solver controls:

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --field-tol 1e-8 \
  --field-max-iter 150 \
  --field-gmres-restart 12
```

Use `--field-verbose` to print the GMRES progress. Diagnostic switches `--no-field-hartree`, `--no-field-fock`, `--no-field-mt`, and `--no-field-al` remove individual pieces of the GW magnetic self-energy response.

With physical units:

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --energy-unit-ev 1.0 \
  --lattice-constant-angstrom 5.0
```

A precomputed `Sigma_B_code` can still be supplied explicitly:

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --sigma-b-npz sigma_B.npz
```

Use `--no-auto-sigma-b` only when intentionally requesting the first term alone.

The output NPZ stores `Sigma_B_code`, `G_B_code`, the geometric source `geometric_G_source_code`, and the Hartree/Fock/MT/AL decomposition whenever the automatic uniform-B solver is used.

## 7. Validation tests

The test suite checks:

1. analytic `D H0` against a centered finite difference of the physical-gauge Hamiltonian;
2. the spectral derivative against a known Bravais Fourier harmonic;
3. the intra-cell embedding derivative for a k-independent off-diagonal matrix;
4. zero derivative for a local diagonal self-energy;
5. zero bulk orbital magnetization for the noninteracting time-reversal-symmetric Ruby model;
6. the algebraic rewriting of Nourafkan A13 into the implemented geometric `Y_b` source;
7. exact `Sigma_B=0` for a noninteracting self-energy functional;
8. the direct fixed-point residual of the uniform-B self-energy equation in a Hartree-only test.

## 8. References

See the full annotated reference list in [uniform_B_self_energy_derivation.md](uniform_B_self_energy_derivation.md). The central sources are:

1. R. Nourafkan, G. Kotliar, and A.-M. S. Tremblay, *Phys. Rev. B* **90**, 125132 (2014), DOI: 10.1103/PhysRevB.90.125132, arXiv:1404.3673.
2. L. Hedin, *Phys. Rev.* **139**, A796 (1965), DOI: 10.1103/PhysRev.139.A796.
3. H. Li, Z. Sun, Y. Su, H. Lin, H. Huang, and D. Li, *Phys. Rev. B* **107**, 085106 (2023), DOI: 10.1103/PhysRevB.107.085106, arXiv:2208.10401.
