# Finite-size quantum ED of the effective pseudospin model

This is a direct exact-diagonalization check of the strong-coupling triangle-pseudospin Hamiltonian, separate from the microscopic cluster-ED+GW/JF calculation.

The quantum model is

\[
H=\sum_{\langle ij\rangle_\gamma}
\left[J_n\tau_i^{n_\gamma}\tau_j^{n_\gamma}
+J_m\tau_i^{m_\gamma}\tau_j^{m_\gamma}
+J_z\tau_i^z\tau_j^z\right],
\]

with Pauli pseudospins (eigenvalues `+/-1`), three bond orientations `gamma=0,1,2`, and

\[
\theta_\gamma=2\pi\gamma/3,\qquad
n_\gamma=(\cos\theta_\gamma,\sin\theta_\gamma),\qquad
m_\gamma=(-\sin\theta_\gamma,\cos\theta_\gamma).
\]

For the straight `Vprime` and crossed `Vcross` microscopic interactions, the leading strong-coupling coefficients are

\[
J_n=\frac{5t^2}{9V}+\frac{V'+V_\times}{18},\qquad
J_m=-\frac{t^2}{3V}+\frac{-V'+V_\times}{6},\qquad
J_z=-\frac{t^2}{3V}.
\]

The triangle centers form a two-sublattice honeycomb-like network. For each A triangle the three B neighbors use primitive-cell offsets `(0,0)`, `(0,1)`, and `(-1,0)`, matching the conventions used in the q-space analysis.

## What the ED computes

The Hamiltonian is diagonalized in the two exact sectors of `prod_i tau_i^z`. The code then constructs the full six-channel equal-time structure-factor matrix at every momentum allowed by the finite torus,

`Ax Ay Az Bx By Bz`,

and diagonalizes that matrix. The dominant structure factor therefore determines the finite-size quantum ordering tendency without preselecting current or charge order. The saved diagnostics include

- the lowest energies in both parity sectors and the many-body gap;
- the ground-state degeneracy within a numerical tolerance;
- the full 6x6 structure-factor matrix at every allowed q;
- the leading structure-factor eigenvalue/eigenvector;
- `z_same`, `z_opposite`, and the largest full x/y-sector eigenvalue.

`lambda/Nspin` is especially useful: an ideal uniform same-current cat state has `z_same(Gamma)/Nspin = 1`.

## Recommended sizes

A `2x2` torus has 8 pseudospins and contains Gamma plus the three M points, but no K point. A `3x3` torus has 18 pseudospins and contains K/K' but not M. Direct ED therefore cannot contain Gamma, M and K simultaneously at these small rectangular sizes. Comparing `2x2` and `3x3` is the practical first check.

A `3x3` calculation uses 131072 basis states per parity sector and is the recommended production size. Sizes above 20 pseudospins become much more expensive with direct sparse ED.

## Single point

```bat
python run_effective_pseudospin_ed.py ^
    --Lx 3 --Ly 3 ^
    --t 0.2 --V 1.8 --Vp -0.1 --Vx -0.05
```

For the M-point check:

```bat
python run_effective_pseudospin_ed.py ^
    --Lx 2 --Ly 2 ^
    --t 0.2 --V 1.8 --Vp -0.1 --Vx -0.05
```

## Vx scan

```bat
python scan_effective_pseudospin_ed_vcross.py ^
    --Lx 3 --Ly 3 ^
    --t 0.2 --V 1.8 --Vp -0.1 ^
    --Vx-values "0,-0.03,-0.04,-0.05,-0.0555556,-0.06,-0.07"
```

Repeat with `--Lx 2 --Ly 2` to follow the M-point sector.

## Interpretation

The q-space exchange-kernel analysis is a classical/soft-mode diagnostic. The finite-size ED goes one step further: it diagonalizes the full quantum pseudospin Hamiltonian and asks which correlations dominate its actual finite-system ground state. A strong `z_same(Gamma)/Nspin` together with a small competing `xy_max/Nspin` is direct evidence that the effective quantum model lies in the uniform loop-current regime.

This still does not by itself prove that the original microscopic fermion model has the same zero-temperature phase: the effective Hamiltonian is a leading strong-coupling truncation, while the cluster-ED+GW/JF result is a finite-temperature normal-state response calculation.
