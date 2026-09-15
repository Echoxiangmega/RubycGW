# Ruby model and conventions

## Primitive cell

RubycGW uses a six-site primitive cell with orbitals `0,...,5`.  Sites `0,1,2` form triangle A and sites `3,4,5` form triangle B.  Reduced reciprocal coordinates are used throughout:

\[
k=(k_1,k_2),\qquad e^{2\pi i k\cdot R}.
\]

The public immutable model description is

```python
from rubycgw.api import RubyModel

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.1,
    Vcross=-0.05,
)
```

Setting `Vprime=Vcross=0` reproduces the baseline model.

## Hopping convention

`ti` is the hopping inside each elementary triangle.  `t1` and `t2` are the two inter-triangle hopping families.  The complete real-space hopping list is still defined by the validated historical `rubycgw.model` kernel; the new `RubyModel` is a public wrapper around the same convention rather than a new Hamiltonian implementation.

## Density interactions

The extended public model contains three short-range density interactions.

### Intra-triangle interaction `V`

`V` acts on the six bonds

```text
A: (0,1), (0,2), (1,2)
B: (3,4), (3,5), (4,5)
```

These bonds are intracell.

### Straight inter-triangle interaction `Vprime`

`Vprime` acts on the same six inter-triangle links used by the `t1/t2` hopping graph.  The real-space offsets are retained in the lattice interaction `V(q)`, so the interaction is momentum dependent when `Vprime != 0`.

### Crossed inter-triangle interaction `Vcross`

`Vcross` acts on the two crossed density links inside each of the same three neighboring A/B triangle pairs.  It does **not** add a long same-cell `A0-B0` bond.  In the six-orbital impurity projection, primitive-cell offsets are dropped and repeated projected pairs are summed; the six physical crossed bonds therefore project onto three orbital pairs with coupling `2*Vcross` each.

## Six-site cluster interaction

The lattice keeps the full real-space offsets.  The finite six-site impurity cannot retain those offsets independently, so its density interaction is the `q=0` six-orbital projection of the extended lattice interaction.  The cluster-GW subtraction and impurity ED use the same projected interaction so the local double-counting construction is internally consistent.

This distinction matters when interpreting large nonlocal `Vprime` or `Vcross`: the lattice and impurity are not identical spatial representations of the interaction.

## Current and pseudospin channels

For the original eta convention,

```text
triangle A loop: 0 -> 1 -> 2 -> 0
triangle B loop: 3 -> 4 -> 5 -> 3
```

the two drawn loops have opposite geometric handedness.  As a result, the old algebraic eta labels and the physical circulation labels are not the same:

```text
eta_plus  = physical opposite circulation
eta_minus = physical same circulation
```

The newer response layer uses `Ax Ay Az Bx By Bz`.  For user-facing A/B parity projections:

```text
z_even = same physical circulation on A and B
z_odd  = opposite physical circulation on A and B
```

This is the naming that should be used when discussing loop-current competition.

## Strong-coupling pseudospin couplings

For `t1=t2=t`, the leading projected model used by the effective ED workflow has

\[
H_{\rm eff}=\sum_{\langle ij\rangle_\gamma}
\left[
J_n\tau_i^{n_\gamma}\tau_j^{n_\gamma}
+J_m\tau_i^{m_\gamma}\tau_j^{m_\gamma}
+J_z\tau_i^z\tau_j^z
\right].
\]

With

\[
s=t^2/V,
\]

the current implementation uses

\[
J_n=\frac{5s}{9}+\frac{V'+V_\times}{18},
\]

\[
J_m=-\frac{s}{3}+\frac{-V'+V_\times}{6},
\]

\[
J_z=-\frac{s}{3}.
\]

The public method

```python
Jn, Jm, Jz = model.effective_couplings()
```

returns these leading-order coefficients.  They are a strong-coupling diagnostic, not an exact replacement for the microscopic fermion model.

## Momentum meshes

`Lx` and `Ly` in the lattice GW/cluster-ED+GW workflows specify the periodic momentum mesh of primitive cells.  They do not enlarge the impurity cluster: the impurity remains six correlated orbitals plus a finite bath.

For the effective pseudospin ED workflow, by contrast, `Lx x Ly` is the actual finite triangle-center quantum cluster with two pseudospins per primitive cell.
