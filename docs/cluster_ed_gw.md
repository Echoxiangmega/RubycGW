# Cluster ED+GW background

## Method

The production self-energy embedding uses a six-correlated-orbital Ruby impurity with a finite bath together with the lattice GW solution.  The embedded lattice self-energy is

\[
\Sigma_{\rm emb}(k,i\omega)=
\Sigma_{GW}^{lat}(k,i\omega)
-\Sigma_{GW}^{cluster}(i\omega)
+\Sigma_{ED}^{cluster}(i\omega).
\]

`Sigma_GW_cluster` is evaluated from the cluster-projected Green function using the same projected density interaction as the impurity.  `Sigma_ED_cluster` comes from exact diagonalization of the finite-bath impurity problem.

The polarization is still the lattice GW bubble.  The present method is therefore a self-energy embedding diagnostic, not a full GW+EDMFT implementation with an impurity polarization/vertex replacing the lattice bubble.

## Public workflow

```python
from rubycgw.api import RubyModel, BackgroundConfig, GridConfig, run_cluster_background

model = RubyModel(V=1.8, Vprime=-0.1, Vcross=-0.05)
config = BackgroundConfig(
    filling=2.0,
    grid=GridConfig(Lx=3, Ly=3, T=0.08),
)
run = run_cluster_background(model, config)
```

The workflow constructs `h0(k)` and the full extended `V(q)`.  The lattice Hartree, screening, and GW self-energy therefore use `V`, `Vprime`, and `Vcross`.  The finite-bath ED impurity treats only the strong six intra-triangle `V` bonds, and the cluster-GW double-counting subtraction is built from exactly that same `V`-only interaction subset.

## Fixed filling and chemical potential

The solver adjusts `mu` to satisfy the requested filling.  Raw `mu` is not a monotonic interaction diagnostic because Hartree and dynamic self-energy common shifts can move in opposite directions.  When comparing interaction points, useful quantities include

\[
\mu-\frac{1}{6}{\rm Tr}\,\Sigma_H
\]

and, when appropriate, a low-frequency common dynamic shift such as

\[
\mu-\frac{1}{6}{\rm Tr}\Sigma_H-
\frac{1}{6}{\rm Tr}\,{m Re}\,\bar\Sigma_{emb}(i\omega_0).
\]

Interpret these as diagnostics rather than uniquely defined observables in a gapped regime.

## Finite bath

The default public configuration uses six bath orbitals, giving a 12-orbital impurity including the six correlated sites.  The Weiss-field hybridization is fitted on the lowest Matsubara frequencies.  Important diagnostics are:

- `bath_fit_error`;
- `impurity_mismatch` between impurity and projected cluster Green functions;
- the raw embedding fixed-point residual;
- stability of `mu`, density, and low-frequency self-energy over the final iterations.

A small outer residual with a very poor bath fit can still signal a quantitatively limited impurity representation.

## Mixing

The accelerated solver mixes the lattice embedded self-energy and impurity self-energy as one coupled dynamic state.  The production defaults use scale-invariant Pulay/DIIS with history 8 and outer mixing 0.8.  `impurity_mixing=1` leaves damping to the coupled outer mixer.

For difficult points, reducing the mixing may help, but a systematically non-closing map should not be interpreted as a physical instability solely because aggressive damping cannot converge it.

## Restart versus continuation

`restart_mode="restart"` resumes the same parameter point with a fresh Pulay history.  `restart_mode="continuation"` reuses a converged embedded solution at a new interaction point and skips the need to approach the new point from a bare standalone GW initializer.

Structural quantities such as lattice mesh, temperature, frequency grids, hopping convention, filling, and bath size must remain compatible.  Interaction strengths may change under continuation.

## Extended interactions

The production partition is

\[
\mathrm{ED}(V)+\mathrm{GW}(V,V',V_x).
\]

The lattice keeps the true real-space offsets of `Vprime` and `Vcross`, so their momentum structure remains in the lattice Hartree/screening/GW map.  They are intentionally excluded from the six-site ED impurity because collapsing different neighboring triangles into one primitive-cell impurity converts distinct nonlocal bonds into repeated orbital pairs and can spuriously enhance an intra-triangle density/orbital mode.

The embedded correction is therefore

\[
\Sigma_{\rm emb}
=
\Sigma_{GW}^{\rm lattice}[V,V',V_x]
+
\left(\Sigma_{ED}^{\rm cluster}[V]
-\Sigma_{GW}^{\rm cluster}[V]\right).
\]

The two cluster terms always use the same interaction subset; this is required for a consistent double-counting subtraction.

## Broken symmetry

The nonlinear embedding can converge to symmetry-broken fixed points.  This does not automatically prove the microscopic zero-temperature ground state.  In particular, the six-site primitive-cell impurity selects a cluster orientation among three C3-related choices.  Use the [orientation-ensemble diagnostic](orientation_ensemble.md) before interpreting small C3-breaking density differences as intrinsic order.


### Physical-pair orientation workflow

The orientation-resolved extended-interaction research driver uses a more
specific partition than the generic public V-only workflow.  Each oriented
six-site impurity treats the intra-triangle V plus exactly one real A-B
neighbour pair (two Vprime and two Vcross bonds).  This avoids collapsing
different intercell neighbours onto the same impurity orbital pair.  A static
high-frequency Weiss contribution is separated from the finite-bath dynamic
hybridization so the bath is not asked to fit a nondecaying constant term.


### Limited-memory Broyden outer mixer

The accelerated cluster embedding supports `mixing_method="broyden"` in
addition to linear and Pulay mixing.  Broyden is applied only to the cluster
outer fixed point; the standalone SC-GW background solver is unchanged.

The real-linear state uses the nonredundant decomposition

```text
Sigma_weak = Sigma_emb - Sigma_imp(local)
X = (Sigma_H, Sigma_weak, Sigma_imp)
```

with block balancing before packing.  The chemical potential is not part of the
Broyden vector: for fixed filling it is solved from the density constraint after
each accepted self-energy update.  The mixer stores only a limited number of
secant pairs and applies a regularized multisecant inverse-Jacobian action.
Unsafe oversized steps fall back to the ordinary damped fixed-point step.


Broyden uses a trust-region safeguard rather than discarding every oversized
quasi-Newton proposal.  If a finite proposal exceeds `broyden_step_cap` times
the raw fixed-point step, its direction and multisecant history are retained
while only the accepted displacement is rescaled to the trust radius.  Only
non-finite proposals cause a true linear fallback/history restart.  This is
important when a slow fixed-point eigenmode legitimately requires a Newton step
many times larger than the raw residual.


## Source-audited V-only algorithm (11 November 2026)

This section applies to `RubyParameters(V=..., Vprime=Vcross=0)`, with only the
six intra-triangle density interactions. It does **not** describe the separate
physical-pair orientation driver that can put one selected inter-triangle
Vprime/Vcross pair into the impurity.

### What enters the impurity one-body matrix?

The actual solver uses

```python
h_cluster = np.mean(h0, axis=(0, 1))
g0_inv = np.linalg.inv(Gc) + sigma_imp
delta_raw = (1j*omega + mu)*I - h_cluster - g0_inv
static_shift, delta_dynamic = split_static_hybridization(delta_raw, omega)
h_impurity = h_cluster + static_shift
```

Thus `h_impurity` is **not simply the strict R=0 hopping matrix**.
`h_cluster` equals that matrix when the discrete lattice mesh does not
alias a nonzero primitive translation onto R=0. For `nk1=1` or `nk2=1`,
some intercell hoppings wrap around the finite torus and enter its local
block. The extra `static_shift` is extracted from the *target Weiss*
hybridization and absorbs its high-frequency constant. It is not a new
microscopic hopping and must not be added to `h0(k)`.

The practical ED solver defines its grand-canonical weights as
`exp[-beta(E_N-mu*N)]`; the stored one-body matrix does **not** include
an extra `-mu*I`. This is also why the bath resolvent contains
`i*omega+mu-eps_bath`.

### Explicit cluster double counting: three contributions

For the cluster interaction matrix `Vc` formed from the same six V-bonds
used in ED, the source `cluster_gw_self_energy` evaluates

```text
P_C(iOmega) = bubble[G_C, G_C]
W_C(iOmega) = [I - Vc P_C(iOmega)]^{-1} Vc

Sigma_H^C(a,b) = delta(a,b) sum_c Vc(a,c) n_c
Sigma_F^C(a,b) = - Vc(b,a) rho_C(a,b)
Sigma_c^C(iw)  = - T sum_{Omega} G_C(iw+iOmega) * [W_C(iOmega)-Vc]
                  (using the exact density-index convention in supercell_gw)
Sigma_DC^C = Sigma_H^C + Sigma_F^C + Sigma_c^C.
```

The last product includes the matrix-index ordering used in the GW code:
`G_ab W_ba`, rather than an ordinary matrix product. The new optional
`audit_cluster_gw_decomposition` diagnostic independently recomputes
these terms and reports `fock_error` and `reconstruction_error`.
No additional Hartree or Fock term is hidden in `h_impurity`; its static
Weiss shift is a separate effective one-body bookkeeping step.

The full lattice stores Hartree separately:
`Sigma_H_lattice` and `Sigma_GW_lattice = Sigma_F_lattice + Sigma_c_lattice`.
The embedding is

```text
Sigma_total(k,iw) = Sigma_H_lattice
                  + Sigma_GW_lattice(k,iw)
                  - Sigma_DC^C(iw)
                  + Sigma_ED^C(iw).
```

When **only V acts within the primitive cell**, the converged lattice and
cluster use the same `rho_C = average_k rho(k)` and `Vc`, so the
Hartree contributions are identical and the Fock contributions are identical:

```text
Sigma_H_lattice = Sigma_H^C
Sigma_F_lattice(k) = Sigma_F^C
Sigma_total(k,iw) = Sigma_ED^C(iw)
                  + Sigma_c_lattice(k,iw) - Sigma_c^C(iw).
```

This simplification applies only to consistent density/static conventions
at convergence, and not to added Vprime/Vcross intercell interactions.
Unlike `Sigma_H_lattice`, the full `Sigma_c_lattice` generally retains
momentum dependence even for intra-triangle V.

### Why the algorithm returns to momentum space

The correction `Sigma_ED^C - Sigma_DC^C` is independent of primitive
Bloch `k`, but `Sigma_GW_lattice(k)`, `G(k)`, `P(q)`, and `W(q)`
are not. Each outer iteration evaluates the full lattice GW map, then
broadcasts the same `(nf,6,6)` cluster correction over the two k axes:
`correction[:,None,None,:,:]`. Dyson, the common chemical potential, and
the next GW screening calculation all use the full `k` grid. The lack
of a separate `k` label in the ED stage is not a momentum-independent
lattice calculation.

### Real-space cross-cell hopping and cluster size

A strict one-primitive-cell 6-orbital impurity contains only those hopping
matrix elements whose **two endpoints both belong to that selected cell**.
A hopping to a different primitive cell remains in `h0(k)` and changes
`G_C` and the bath, rather than being inserted as an independent impurity
one-body matrix element. Wrapping under a finite torus is a discrete
boundary alias, not an extra physical intracluster bond.

One can instead (i) move the B triangle to a neighboring real cell while
keeping a physical six-site A-B cluster, at the cost of an orientation-
dependent cluster cut, or (ii) genuinely enlarge the correlated cluster.
For V-only and comparisons between the two triangle current channels,
the literal six-site primitive cell avoids unnecessary orientation bias
and contains both complete interacting triangles.

### Source audit C4--C10 (V-only)

| Check | Verified implementation and remaining limitation |
| --- | --- |
| C4: Weiss | `G0_C^{-1}=G_C^{-1}+Sigma_imp`; the target hybridization's static limit is separated, then its dynamic part fitted. The effective hybridization can contain GW environment contributions; causality and fit error need independent tests. |
| C5: ED solver | Dense spinless fixed-N sectors, grand-canonical weights `exp[-beta(E-mu*N)]`, complex off-diagonal Green functions and full Lehmann transitions. Dense solver currently limits total orbitals to 16. |
| C6: Bath | `delta` and `g0` fitting metrics, weighted low Matsubara frequencies; reported fit errors are diagnostic and are not enforced as a universal physical accuracy threshold. |
| C7: GW feedback | In the accelerated outer loop the lattice `P,W,Sigma_GW` are rebuilt from the current G each iteration; they are not permanently fixed at the initial GW solution. |
| C8: V-only DC | The six intra-triangle V bonds define both impurity interactions and cluster GW. `Sigma_DC^C = H+F+GW(W-V)`; the new audit regression checks the separate Fock and total terms. |
| C9: Symmetry | The default unperturbed bath fit has real couplings; time-reversal-odd source calculations require the complex-bath path. A single uniform impurity cannot spontaneously represent an intercell finite-Q CDW/LC. |
| C10: finite size | `h_cluster=mean_k h0(k)` intentionally matches the finite torus, including any `nk1=1` or `nk2=1` wrapping. Check mesh and bath convergence separately. |

### How many impurities?

Do not confuse the number of **independent impurity problems** `N_imp`
with the number of auxiliary bath orbitals `N_b` per impurity.

- Uniform primitive-cell phase: one six-site impurity (`N_imp=1`).
- Unconstrained period-three charge/current modulation: three independent
  six-site impurities (`N_imp=3`), with a shared lattice GW map and chemical
  potential; reducing this count requires a symmetry actually preserved by
  the putative ordered state.
- Two three-site triangle impurities do **not** reproduce a single six-site
  correlated A-B impurity: cross-triangle strong self-energy terms are lost
  even though their bare density V is zero.

For spinless full Fock space, the cost of one ED problem grows as
`2**(6+N_b)` before fixed-N sector decomposition. Multiple independent
impurities scale approximately linearly in total ED work if their sizes are
unchanged, and can be run in parallel. Increasing `N_b` from six to seven
increases the full Fock dimension from 4096 to 8192; this does not imply
a simple factor of two in dense-sector diagonalization time. The GW lattice
cost is shared and can dominate when the translation symmetry is relaxed.
