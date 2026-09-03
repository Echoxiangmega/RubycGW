# Bulk orbital magnetization

This module implements the checkpoint evaluation of the interacting bulk orbital-magnetization formula of R. Nourafkan, G. Kotliar, and A.-M. S. Tremblay, *Phys. Rev. B* **90**, 125132 (2014), Eq. (2).

## 1. Exact formula and what is currently available

For a two-dimensional system and the `z` component, the first term of Eq. (2) is

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

The *full* interacting Eq. (2) also contains

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

`analyze_bulk_magnetization.py` always evaluates `M1`. It reports the total only when the second term is known. For `V=0`, `M2=0` exactly. For a nonlocal interacting GW self-energy, `M2` is **not** assumed to vanish.

The local-DMFT example in Nourafkan et al. has `partial_k Sigma=0` and also `partial_B Sigma^(B)=0`; those simplifications do not automatically apply to GW.

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

This embedding term is essential. If one differentiated only the cell phase, all intra-triangle hoppings with `S=(0,0)` would incorrectly contribute zero current vertex.

For `H0`, the derivative is evaluated analytically from `supercell_hoppings`. For `Sigma(k,iw)`, the sampled k mesh is Fourier transformed to the represented real-space harmonics, multiplied by `i d_alpha`, and transformed back. This is a spectral derivative rather than a nearest-neighbor finite difference on the k mesh.

## 3. Finite mesh caveat

A `3x3` k mesh can represent only a very short-ranged periodic real-space self-energy. The spectral derivative is the exact derivative of that finite Fourier representation, but it is not a substitute for k-mesh convergence. Production bulk-M calculations should compare several meshes, for example `nk=3,4,6,8`, and also increase `nw`.

## 4. Units

The code reports a dimensionless coefficient `M_code` such that, when the hopping-energy unit is `E0` and the lattice-length unit is `a`,

\[
m=\frac{e}{\hbar}E_0a^2 M_{\rm code}.
\]

Supply

```bash
--energy-unit-ev E0 --lattice-constant-angstrom a
```

to obtain Bohr magnetons. The 18-site supercell contains three primitive Ruby cells, so the CLI reports both the supercell value and `M/3` per primitive cell.

## 5. Command line

First term and diagnostics:

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --npz bulk_M.npz \
  --json bulk_M.json
```

With physical units:

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --energy-unit-ev 1.0 \
  --lattice-constant-angstrom 5.0
```

If a future uniform-field calculation provides

\[
\Sigma_b=\frac{\partial\widetilde\Sigma}{\partial b},
\qquad
b=\frac{ea^2}{\hbar}B_z,
\]

save it as `Sigma_B_code` in an NPZ and use

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --sigma-b-npz sigma_B.npz
```

Then both terms and the complete Eq. (2) total are reported.

## 6. Validation tests

The test suite checks:

1. the analytic `D H0` against a centered finite difference of the physical-gauge Hamiltonian;
2. the spectral derivative against a known Bravais Fourier harmonic;
3. the intra-cell embedding derivative for a k-independent off-diagonal matrix;
4. zero derivative for a local diagonal self-energy;
5. zero bulk orbital magnetization for the noninteracting time-reversal-symmetric Ruby model;
6. explicit incomplete/complete status of the second Eq. (2) term.
