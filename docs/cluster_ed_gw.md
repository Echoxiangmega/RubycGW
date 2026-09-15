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

The workflow constructs `h0(k)` and the full extended `V(q)`, installs the model-specific q=0 cluster interaction only for the duration of the solve, and then restores the historical solver hooks.  This keeps the high-level interface free of the global patching that older launchers performed manually.

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

The lattice keeps the true real-space offsets of `Vprime` and `Vcross`.  The six-site impurity uses their q=0 projection.  Repeated crossed-bond orbital pairs are summed, so the impurity interaction is exactly the matrix used by the cluster-GW subtraction at q=0, even though the lattice retains additional spatial information.

## Broken symmetry

The nonlinear embedding can converge to symmetry-broken fixed points.  This does not automatically prove the microscopic zero-temperature ground state.  In particular, the six-site primitive-cell impurity selects a cluster orientation among three C3-related choices.  Use the [orientation-ensemble diagnostic](orientation_ensemble.md) before interpreting small C3-breaking density differences as intrinsic order.
