# RubycGW documentation

This directory describes the maintained, modular interface of RubycGW.  The repository was reorganized so that end users no longer need to navigate the large collection of development-time drivers that historically lived in the project root.

## Start here

1. [Getting started](getting_started.md) — installation, first background run, first effective-model run, and CLI entry points.
2. [Model and conventions](model_and_conventions.md) — Ruby lattice, hopping/interactions, `Vprime`/`Vcross`, momentum convention, and current-channel naming.
3. [Public API reference](api_reference.md) — the stable imports intended for notebooks, scripts, and a future GUI.
4. [Cluster ED+GW](cluster_ed_gw.md) — embedding equation, finite bath, self-consistency, continuation, and method limitations.
5. [Jacobian-free response](jf_response.md) — finite-q soft-mode calculations, bath tangent, solver options, and interpretation.
6. [Effective pseudospin ED](effective_pseudospin.md) — leading strong-coupling model and finite-size quantum ED.
7. [Cluster-orientation ensemble](orientation_ensemble.md) — why a single six-site cluster has an orientation bias and how the three-gauge diagnostic is used.
8. [Checkpoints and continuation](checkpoints.md) — saved backgrounds, restart versus continuation, and reproducibility.
9. [Numerics and validation](numerics_and_validation.md) — convergence requirements and validation philosophy.
10. [Repository layout](repository_layout.md) and [developer guide](developer.md) — where new functionality belongs and how to keep the public API stable.

The original [GW theory](gw_theory.md) and [cGW theory](cgw_theory.md) notes remain part of the maintained documentation because they describe the numerical kernels still used by the modular workflows.

## Public versus research interfaces

The supported user-facing layers are:

```text
rubycgw.api
rubycgw.models
rubycgw.solvers
rubycgw.workflows
rubycgw.analysis
rubycgw.io
scripts/
```

The `research/` directory contains historical scans, benchmarks, diagnostics, plotting scripts, and paper-specific drivers.  They are retained for reproducibility but are not treated as stable API.  Specialized historical documentation is stored under [research_notes/](research_notes/).

## Generated guide

GitHub Actions builds a PDF user guide from the maintained files in this directory.  The PDF is intended to reflect the current package layout rather than the historical development tree.
