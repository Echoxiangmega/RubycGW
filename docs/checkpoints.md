# Checkpoints, restart, and continuation

## Standard background file

The maintained workflow saves cluster backgrounds with

```python
from rubycgw.api import save_background
save_background("results/background.npz", run)
```

The NPZ contains model parameters, grid metadata, convergence diagnostics, the lattice Green function and screened interaction, Hartree and embedded self-energies, the lattice/cluster GW pieces, impurity ED self-energy, cluster and impurity Green functions, finite-bath parameters, and iteration histories needed for diagnosis and restart.

Use `background_metadata(path)` when only scalar metadata are needed and avoid loading large arrays unnecessarily.

## Restart

A restart continues the same physical point.  It restores the saved embedded state and bath as an initializer but rebuilds the Pulay history:

```python
restart = load_restart_for_run(path, model, config)
run = run_cluster_background(
    model,
    config,
    restart=restart,
    restart_mode="restart",
)
```

This is appropriate when a previous run stopped before the requested tolerance or when a converged point is being recomputed with a larger iteration budget while keeping the physical and structural parameters unchanged.

## Parameter continuation

Continuation is intended for scans in interaction parameters:

```python
old = RubyModel(V=1.5, Vprime=-0.1, Vcross=-0.05)
new = RubyModel(V=1.6, Vprime=-0.1, Vcross=-0.05)
restart = load_restart_for_run("results/V1.5.npz", new, config)
run = run_cluster_background(new, config, restart=restart, restart_mode="continuation")
```

The continuation path starts the new embedded solve from the old converged embedded `G`, self-energy, chemical potential, impurity self-energy, and bath rather than requiring a separate new standalone SC-GW fixed point first.

## Compatibility checks

Continuation intentionally permits changes in `V`, `Vprime`, and `Vcross`.  It is not intended to silently reinterpret a checkpoint with incompatible structure.  Keep the following fixed unless a dedicated interpolation/conversion workflow is used:

```text
Lx, Ly
T
fermionic and bosonic frequency grids
filling
hopping convention / ti,t1,t2
number of bath orbitals
orbital ordering
```

If one of these changes, generate a new background or use a specifically validated interpolation utility from the research archive.

## Result metadata

A saved background records whether the point converged and stores at least the final embedding residual, impurity mismatch, bath fit error, chemical potential, density, mixing method, and per-iteration histories.  Downstream response calculations should reject or prominently warn about backgrounds whose residual exceeds the response calculation's acceptance threshold.

## Reproducibility practice

For production scans, keep the exact input NPZ used by a downstream JF run.  The response depends on the full nonlinear background, not just the scalar model parameters.  If a background was produced by continuation, retain the continuation chain or at least its source filename/metadata so branch following can be reconstructed.

Archived checkpoint conversion and interpolation tools remain under `research/`; they are not part of the stable public API.
