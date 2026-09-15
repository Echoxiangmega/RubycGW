# Uniform-B self-energy derivative for the second bulk orbital-magnetization term

This note explains the theory and implementation behind the second term of the interacting bulk orbital magnetization used in `RubycGW`.

The goal is to make the logic reproducible for future users of the code: where the uniform magnetic-field Green-function derivative comes from, what the code-specific operator `C_GW` means, why `G_B` and `Sigma_B` must be solved self-consistently, and how the resulting `Sigma_B` enters the second term of the orbital magnetization.

Throughout, zero-field quantities are assumed to be a converged self-consistent GW solution.

---

## 1. References and notation

The bulk orbital-magnetization formula and the gauge-invariant uniform-field Green-function expansion follow Ref. [1], especially Eq. (2) and Appendix A, including Eq. (A13).

The covariant-GW viewpoint used to linearize a self-consistent GW functional and solve the resulting response equation is closely related to Ref. [3]. The underlying GW approximation itself goes back to Hedin, Ref. [2].

Important notation distinction:

- `G_B`, `Sigma_B`, and the gauge-invariant tilde quantities originate from the magnetic-field expansion in Ref. [1].
- `C_GW` is **not** notation used in Ref. [1]. It is a compact code/theory notation introduced in this repository for the Jacobian-vector product of the self-consistent GW self-energy functional.

We use the dimensionless magnetic variable

\[
b = \frac{e a^2}{\hbar} B_z,
\]

where `a` is the real-space length unit used by the Ruby-lattice embedding. Thus

\[
G_b \equiv \left.\frac{\partial \widetilde G}{\partial b}\right|_{b=0},
\qquad
\Sigma_b \equiv \left.\frac{\partial \widetilde\Sigma^{(B)}}{\partial b}\right|_{b=0}.
\]

The code names these arrays `G_B_code` and `Sigma_B_code`.

---

## 2. The second term in the bulk orbital magnetization

For the two-dimensional `z` component, the interacting orbital-magnetization expression of Ref. [1] contains two pieces,

\[
M_z=M_z^{(1)}+M_z^{(2)}.
\]

The first term is evaluated from the zero-field full Green function and physical momentum derivatives. The second term is

\[
M_z^{(2)} =
\frac{T}{2N_k}
\sum_{\mathbf k,n}
\mathrm{Tr}\left\{
\left[H_0(\mathbf k)+(i\omega_n-\mu)I\right]
G(\mathbf k,i\omega_n)
\Sigma_b(\mathbf k,i\omega_n)
G(\mathbf k,i\omega_n)
\right\},
\]

in the repository normalization using `b=(e a^2/\hbar)B_z`.

Therefore the only genuinely new quantity required for the second term is

\[
\boxed{\Sigma_b}.
\]

The rest is available from the converged zero-field checkpoint.

---

## 3. What Eq. (A13) gives us

Ref. [1], Appendix A, Eq. (A13), gives the gauge-invariant Green function in a weak uniform magnetic field. For a field along `z`, its first derivative can be written schematically as

\[
\frac{\partial\widetilde G}{\partial B_z}
=
G\frac{\partial\widetilde\Sigma^{(B)}}{\partial B_z}G
+
\frac{ie}{2\hbar}
\left[
G(\partial_{k_x}G^{-1})(\partial_{k_y}G)
-
G(\partial_{k_y}G^{-1})(\partial_{k_x}G)
\right].
\]

Using

\[
\partial_{k_\alpha}G
=-G(\partial_{k_\alpha}G^{-1})G,
\]

and defining the dressed physical current vertices

\[
J_\alpha\equiv-D_\alpha G^{-1},
\]

we obtain, in the repository dimensionless field variable `b`,

\[
\boxed{
G_b = Y_b + G\Sigma_b G
}
\]

with

\[
\boxed{
Y_b =
-\frac{i}{2}
\left(
GJ_xGJ_yG-GJ_yGJ_xG
\right).
}
\]

This decomposition has a direct interpretation:

\[
\underbrace{G_b}_{\text{full magnetic Green-function derivative}}
=
\underbrace{Y_b}_{\text{explicit geometric / Peierls part}}
+
\underbrace{G\Sigma_bG}_{\text{interaction-feedback part}}.
\]

`Y_b` is completely known from the zero-field solution. `Sigma_b` is the unknown quantity that must be solved self-consistently.

In the code, `Y_b` is constructed by

```python
geometric_uniform_B_green_source(G, Jx, Jy)
```

in `rubycgw/magnetic_self_energy.py`.

---

## 4. Physical momentum derivatives and orbital embedding

Ref. [1] uses a Bloch convention in which the orbital position is included in the Fourier phase. The Ruby code instead stores matrices in a cell gauge,

\[
X_{IJ}(\mathbf k)
=
\sum_{\mathbf S}
X_{IJ}(\mathbf S)e^{2\pi i\mathbf k\cdot\mathbf S}.
\]

Therefore an ordinary derivative with respect to the reduced cell momentum is not the physical derivative needed in Eq. (A13).

For a matrix element connecting orbital `I` to orbital `J`, the physical displacement is

\[
\mathbf d_{IJ}(\mathbf S)
=
S_1\mathbf T_1+S_2\mathbf T_2
+\mathbf r_J-\mathbf r_I.
\]

The code therefore uses the embedding-corrected derivative

\[
[D_\alpha X]_{IJ}(\mathbf k)
=
i\sum_{\mathbf S}
 d_{IJ,\alpha}(\mathbf S)
 X_{IJ}(\mathbf S)e^{2\pi i\mathbf k\cdot\mathbf S}.
\]

This is essential for the Ruby lattice: all intra-triangle hoppings have zero supercell shift, but nonzero `r_J-r_I`, so a derivative that kept only the cell translation would incorrectly remove their current contribution.

The code evaluates

\[
J_x=D_xH_0+D_x\Sigma,
\qquad
J_y=D_yH_0+D_y\Sigma,
\]

through

```python
supercell_h0_cartesian_derivatives(...)
spectral_cartesian_covariant_derivatives(...)
```

in `rubycgw/bulk_orbital_magnetization.py`.

---

## 5. Definition of `C_GW`

The symbol

\[
\boxed{\mathcal C_{\rm GW}}
\]

is a repository-defined shorthand for the linearization of the self-consistent GW self-energy functional.

Suppose the zero-field Green function is changed infinitesimally as

\[
G\rightarrow G+\epsilon X.
\]

Then

\[
\Sigma_{\rm GW}[G+\epsilon X]
=
\Sigma_{\rm GW}[G]
+
\epsilon\,\mathcal C_{\rm GW}[X]
+O(\epsilon^2).
\]

Equivalently,

\[
\boxed{
\mathcal C_{\rm GW}[X]
\equiv
\left.\frac{\delta\Sigma_{\rm GW}}{\delta G}\right|_{G=G^\ast}X.
}
\]

`C_GW` is therefore a **Jacobian-vector product**: input an arbitrary Green-function tangent `X=delta G`, output the associated first-order self-energy tangent `delta Sigma`.

The full Jacobian

\[
\frac{\delta\Sigma_{ab}(k,i\omega)}
{\delta G_{cd}(k',i\omega')}
\]

is never formed or stored explicitly. That object would be prohibitively large. The code only implements its action on a trial field `X`.

The corresponding routine is

```python
self_energy_tangent_from_G_tangent(G, W, Vq, X, grid, opts)
```

in `rubycgw/magnetic_self_energy.py`.

---

## 6. How `C_GW[X]` is evaluated in the split-GW implementation

The production self-energy is

\[
\Sigma=\Sigma_H+\Sigma_F+\Sigma_c,
\]

with

\[
\Sigma_c=-G*(W-V).
\]

For an arbitrary tangent

\[
X\equiv\delta G,
\]

we obtain

\[
\delta\Sigma
=
\delta\Sigma_H
+
\delta\Sigma_F
+
\delta\Sigma_{MT}
+
\delta\Sigma_{AL1}
+
\delta\Sigma_{AL2}.
\]

Thus

\[
\boxed{
\mathcal C_{\rm GW}[X]
=
\mathcal C_H[X]
+
\mathcal C_F[X]
+
\mathcal C_{MT}[X]
+
\mathcal C_{AL1}[X]
+
\mathcal C_{AL2}[X].
}
\]

### 6.1 Hartree part

The density variation is

\[
\delta n_a[X]
=
\frac{T}{N_k}
\sum_{\mathbf k,n}
X_{aa}(\mathbf k,i\omega_n).
\]

Therefore

\[
\delta\Sigma_H[X]
=V(0)\,\delta n[X].
\]

Code:

```python
_hartree_from_x(X, Vq[0, 0], grid)
```

### 6.2 Static Fock part

The equal-time density-matrix variation is

\[
\delta\rho(\mathbf k)
=T\sum_n X(\mathbf k,i\omega_n).
\]

Hence

\[
\delta\Sigma_F(\mathbf k)
=-\frac{1}{N_k}\sum_{\mathbf q}
\delta\rho(\mathbf k+\mathbf q)\circ V^T(\mathbf q).
\]

Code:

```python
_fock_from_x(X, Vq, grid, backend)
```

### 6.3 MT part

For

\[
\Sigma_c=-G*(W-V),
\]

the variation of the explicit `G` factor gives

\[
\boxed{
\delta\Sigma_{MT}[X]
=-X*(W-V).
}
\]

This is the Maki-Thompson-like contribution in the current code decomposition.

### 6.4 AL1 and AL2 parts

The screened interaction depends on `G` through the polarization,

\[
W^{-1}=V^{-1}-P.
\]

Therefore

\[
\delta W=W\,\delta P\,W.
\]

The polarization variation contains two terms,

\[
\delta P_{ab}(Q)
=
\frac{T}{N_k}\sum_k
\left[
X_{ab}(k+Q)G_{ba}(k)
+
G_{ab}(k+Q)X_{ba}(k)
\right].
\]

These two pieces generate the two Aslamazov-Larkin-like self-energy tangents,

\[
\delta\Sigma_{AL1},
\qquad
\delta\Sigma_{AL2},
\]

when inserted into

\[
-G*\delta W.
\]

The implementation uses exactly the same Hartree/Fock/MT/AL linear-response kernel as the existing covariant-GW code [3], but here it is applied to a general tangent `X` rather than only to `X=G Gamma G`.

---

## 7. Why `G_b` cannot simply be chosen

A common source of confusion is the phrase “set `delta G` and compute `delta Sigma`.”

That statement only defines how `C_GW` acts on an arbitrary tangent. In the physical uniform-field problem, `delta G` is **not chosen by hand**.

Eq. (A13) fixes it self-consistently:

\[
\boxed{
G_b=Y_b+G\Sigma_bG.
}
\]

At the same time, the GW self-energy responds to the actual Green-function derivative,

\[
\boxed{
\Sigma_b=\mathcal C_{\rm GW}[G_b].
}
\]

These two equations form a closed linear response problem.

Substituting the first into the second gives

\[
\Sigma_b
=
\mathcal C_{\rm GW}
\left[Y_b+G\Sigma_bG\right].
\]

Because `C_GW` is linear,

\[
\Sigma_b
=
\mathcal C_{\rm GW}[Y_b]
+
\mathcal C_{\rm GW}[G\Sigma_bG].
\]

Define

\[
A\equiv\mathcal C_{\rm GW}[Y_b]
\]

and

\[
L[S]\equiv\mathcal C_{\rm GW}[GSG].
\]

Then the unknown `S=Sigma_b` obeys

\[
\boxed{
(I-L)S=A.
}
\]

This is the equation solved numerically.

---

## 8. Fixed-point interpretation

Before discussing GMRES, the easiest way to understand the physics is the simple iteration

\[
\Sigma_b^{(0)}=0.
\]

Then

\[
G_b^{(0)}=Y_b,
\]

\[
\Sigma_b^{(1)}=\mathcal C_{\rm GW}[Y_b].
\]

Next,

\[
G_b^{(1)}
=Y_b+G\Sigma_b^{(1)}G,
\]

\[
\Sigma_b^{(2)}
=\mathcal C_{\rm GW}[G_b^{(1)}].
\]

The conceptual loop is therefore

\[
\boxed{
\Sigma_b^{(n)}
\longrightarrow
G_b^{(n)}=Y_b+G\Sigma_b^{(n)}G
\longrightarrow
\Sigma_b^{(n+1)}=\mathcal C_{\rm GW}[G_b^{(n)}].
}
\]

At convergence,

\[
\Sigma_b^{(n+1)}=\Sigma_b^{(n)},
\]

which is precisely

\[
\Sigma_b=\mathcal C_{\rm GW}[Y_b+G\Sigma_bG].
\]

This fixed-point picture is useful for understanding the physics, even though the production code uses GMRES rather than slow direct iteration.

---

## 9. GMRES implementation

Since the equation is linear,

\[
(I-L)\Sigma_b=A,
\]

we solve it with the same matrix-free restarted GMRES machinery used by the cGW vertex solver [3].

The right-hand side is computed once:

\[
A=\mathcal C_{\rm GW}[Y_b].
\]

For every trial self-energy tangent `S` supplied by GMRES, the code evaluates

\[
GSG,
\]

then

\[
\mathcal C_{\rm GW}[GSG],
\]

and finally the matrix-vector action

\[
\boxed{
\mathcal A[S]
=S-\mathcal C_{\rm GW}[GSG].
}
\]

Thus neither the full Jacobian `delta Sigma / delta G` nor the full matrix representation of `I-L` is ever stored.

The relevant code is schematically

```python
def apply_A(S):
    induced_G = G @ S @ G
    induced_sigma = C_GW(induced_G)
    return S - induced_sigma
```

with

```python
rhs = C_GW(Y_B)
```

and GMRES solves

```python
apply_A(Sigma_B) = rhs
```

for `Sigma_B`.

After convergence the code reconstructs

\[
\boxed{
G_b=Y_b+G\Sigma_bG
}
\]

and independently verifies the response equation

\[
\boxed{
\Sigma_b=\mathcal C_{\rm GW}[G_b].
}
\]

The printed `uniform-B` equation residual is the max norm of this final consistency check.

---

## 10. Relation to the ordinary cGW vertex equation

The logic is almost identical to the previously implemented covariant-GW source response.

For an ordinary external source `h`, one has

\[
G_h=G\Gamma_hG,
\]

and therefore

\[
\delta\Sigma_h
=\mathcal C_{\rm GW}[G\Gamma_hG].
\]

The dressed source vertex satisfies

\[
\Gamma_h=K_h+\mathcal C_{\rm GW}[G\Gamma_hG].
\]

This is the familiar cGW equation.

The uniform magnetic-field problem differs because Eq. (A13) contains an additional explicit geometric term,

\[
G_b=\boxed{Y_b}+G\Sigma_bG.
\]

Thus the mathematical kernel is the same linearized GW functional, but the source is not a simple bare vertex `K`; it is the geometric Green-function source `Y_b`.

---

## 11. Code-to-equation map

| Physics object | Equation | Code |
|---|---|---|
| physical `D_x H0`, `D_y H0` | embedding-corrected momentum derivative | `supercell_h0_cartesian_derivatives` |
| physical `D_x Sigma`, `D_y Sigma` | spectral covariant derivative | `spectral_cartesian_covariant_derivatives` |
| `J_x`, `J_y` | `J_alpha=-D_alpha G^{-1}` | assembled in `solve_checkpoint_uniform_B_self_energy_derivative` |
| `Y_b` | `-(i/2)(GJxGJyG-GJyGJxG)` | `geometric_uniform_B_green_source` |
| `C_GW[X]` | `(delta Sigma_GW / delta G) X` | `self_energy_tangent_from_G_tangent` |
| `A=C_GW[Y_b]` | magnetic source in self-energy space | `source_sigma` |
| `L[S]` | `C_GW[G S G]` | `apply_A` internal kernel |
| `Sigma_b` | `(I-L) Sigma_b=A` | `solve_uniform_B_self_energy_derivative` |
| `G_b` | `Y_b+G Sigma_b G` | `G_B_code` |
| `M_2` | second term of Ref. [1] Eq. (2) | `_nourafkan_field_self_energy_term` |

---

## 12. Validation strategy

The present implementation includes several levels of checks:

1. **A13 algebra check**: verifies that the `Y_b` three-G form is algebraically equivalent to the derivative form of Eq. (A13).
2. **Noninteracting limit**: for `V=0`, the self-energy derivative must vanish exactly,
   \[
   \Sigma_b=0,
   \]
   while `G_b=Y_b` remains finite in general.
3. **Linear equation residual**: after solving, verify
   \[
   \|\Sigma_b-\mathcal C_{\rm GW}[G_b]\|\ll1.
   \]
4. **TR-symmetric interacting sanity check**: a time-reversal-symmetric self-consistent GW background should not acquire a spurious total orbital magnetization from the second-term machinery.
5. **Mesh and Matsubara convergence**: `nk` and `nw` convergence remain necessary because `D_k Sigma` and all Matsubara sums are represented on finite grids.

A further desirable validation, especially before precision production use, is an independent gauge-consistent uniform-field benchmark of the gauge-invariant self-energy derivative. This is separate from the already implemented plaquette-flux finite-difference test.

---

## 13. Important scope/caveat for the tilde self-energy

The present GW implementation makes the following working identification at the GW level:

\[
\boxed{
\Sigma_b=\mathcal C_{\rm GW}[G_b].
}
\]

This is the natural diagrammatic linearization of the zero-field self-consistent GW functional after the gauge-dependent Peierls phase has been factored from the Green function, consistent with the statement in Ref. [1] that the linear magnetic-field dependence of the gauge-invariant self-energy arises through the linear magnetic-field dependence of the gauge-invariant Green function.

However, it is important not to confuse this with a statement explicitly written in Ref. [1] using the repository notation `C_GW`. The notation and the split-GW Hartree/Fock/MT/AL implementation are ours. The gauge-invariant identification should therefore continue to be checked against independent uniform-field or Ward-identity benchmarks when high-precision bulk magnetization is required.

---

## 14. Practical workflow

For a converged zero-source checkpoint, the full calculation is

\[
(G,\Sigma)
\rightarrow
(D_xG^{-1},D_yG^{-1})
\rightarrow
Y_b
\rightarrow
\Sigma_b
\rightarrow
G_b
\rightarrow
M_2
\rightarrow
M=M_1+M_2.
\]

The default command is

```bash
python analyze_bulk_magnetization.py CHECKPOINT \
  --field-tol 1e-8 \
  --field-max-iter 150 \
  --field-gmres-restart 12 \
  --npz bulk_M.npz \
  --json bulk_M.json
```

Useful quantities to inspect are

- `M1_code`, `M2_code`, `M_total_code`;
- `Sigma_B_code`;
- `G_B_code`;
- `geometric_G_source_code`;
- `Sigma_H_B_code`, `Sigma_F_B_code`, `Sigma_MT_B_code`, `Sigma_AL1_B_code`, `Sigma_AL2_B_code`;
- the uniform-B GMRES convergence flag and equation residual.

The decomposition lets one determine whether the second orbital-magnetization term is dominated by the static Fock response, MT response, or the screening-feedback AL contributions.

---

## References

[1] R. Nourafkan, G. Kotliar, and A.-M. S. Tremblay, **Orbital magnetization of correlated electrons with arbitrary band topology**, *Phys. Rev. B* **90**, 125132 (2014), DOI: 10.1103/PhysRevB.90.125132, arXiv:1404.3673. See especially Eq. (2), Appendix A, Eq. (A13), and the discussion of the gauge-invariant Green function and self-energy.

[2] L. Hedin, **New Method for Calculating the One-Particle Green's Function with Application to the Electron-Gas Problem**, *Phys. Rev.* **139**, A796 (1965), DOI: 10.1103/PhysRev.139.A796. Original formulation of the GW approximation and Hedin equations.

[3] H. Li, Z. Sun, Y. Su, H. Lin, H. Huang, and D. Li, **Linear Response Functions Respecting Ward-Takahashi Identity and Fluctuation-Dissipation Theorem within GW Approximation**, *Phys. Rev. B* **107**, 085106 (2023), DOI: 10.1103/PhysRevB.107.085106, arXiv:2208.10401. Relevant for the covariant linearization of a self-consistent GW functional and the Hartree/MT/AL response structure used in this repository.
