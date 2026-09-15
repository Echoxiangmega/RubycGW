# Effective pseudospin model and finite-size ED

## Leading strong-coupling model

At filling two fermions per six-site primitive cell, the strong-coupling construction keeps one low-energy E doublet on each triangle.  The resulting triangle-center model uses Pauli pseudospins with eigenvalues ±1:

\[
H_{\rm eff}=\sum_{\langle ij\rangle_\gamma}
\left[
J_n\tau_i^{n_\gamma}\tau_j^{n_\gamma}
+J_m\tau_i^{m_\gamma}\tau_j^{m_\gamma}
+J_z\tau_i^z\tau_j^z
\right].
\]

The three bond directions use

\[
\theta_\gamma=2\pi\gamma/3,\qquad
n_\gamma=(\cos\theta_\gamma,\sin\theta_\gamma),\qquad
m_\gamma=(-\sin\theta_\gamma,\cos\theta_\gamma).
\]

For `t1=t2=t`, `RubyModel.effective_couplings()` returns

\[
J_n=\frac{5}{9}\frac{t^2}{V}+\frac{V'+V_\times}{18},
\]

\[
J_m=-\frac{1}{3}\frac{t^2}{V}+\frac{-V'+V_\times}{6},
\]

\[
J_z=-\frac{1}{3}\frac{t^2}{V}.
\]

These are leading-order strong-coupling coefficients.  The microscopic fermion model contains charge fluctuations and higher-order processes that are not represented here.

## Finite lattice

There are two pseudospins, A and B, per primitive cell.  The three A-to-B bond offsets are

```text
(0,0), (0,1), (-1,0)
```

so an `Lx x Ly` effective-model torus contains

\[
N_{spin}=2L_xL_y
\]

quantum pseudospins.

The exact symmetry

\[
P=\prod_i\tau_i^z
\]

splits the Hilbert space into even and odd parity sectors because the x/y terms flip two pseudospins.  The production ED solver diagonalizes the two parity blocks independently with SciPy sparse `eigsh` and combines their lowest levels.

A 3x3 torus has 18 pseudospins and 131072 basis states per parity sector.  This is the current practical production size for direct sparse ED.  The solver is guarded against impractically large direct constructions.

## Public workflow

```python
from rubycgw.api import RubyModel, EffectiveEDConfig, run_effective_ed

model = RubyModel(
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.1,
    Vcross=-0.05,
)

result = run_effective_ed(
    model,
    EffectiveEDConfig(
        Lx=3,
        Ly=3,
        nev=4,
        degeneracy_tol=1e-7,
    ),
)
```

The result includes ground-state energy, gap above the detected ground manifold, ground-state parity information, the allowed q mesh, and the full 6x6 equal-time structure-factor matrix in the `Ax Ay Az Bx By Bz` basis.

## Structure-factor diagnostics

For every allowed q, the code diagonalizes

\[
S_{ab}(q)=\langle O_a^\dagger(q)O_b(q)\rangle.
\]

It stores:

```text
lambda_max(q)     largest eigenvalue of S(q)
z_same(q)         same-circulation current projection
z_opposite(q)     opposite-circulation current projection
xy_max(q)         largest eigenvalue of the x/y block
```

With the implemented normalization, an ideal uniform same-current cat state has approximately

\[
S_{z,\mathrm{same}}(\Gamma)/N_{spin}=1.
\]

## Finite-size momentum coverage

Small clusters sample different competing wave vectors:

- 2x2 resolves Gamma and M-type momenta;
- 3x3 resolves Gamma and K/K' but not M.

For the present direct solver, comparing 2x2 and 3x3 is therefore more informative than pretending either cluster contains the complete Gamma/M/K competition.

## Interpretation

The effective ED directly gives the finite-size quantum ground state of the **leading projected pseudospin Hamiltonian**.  It does not prove the zero-temperature ground state of the original interacting fermion model.  A discrepancy between effective ED and microscopic cluster-ED+GW/JF is useful information: it can signal higher-order strong-coupling terms, finite-temperature effects, charge-amplitude relaxation, or approximations in the embedding/response calculation.
