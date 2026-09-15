# RubycGW documentation

This directory contains the maintained user documentation for RubycGW. It is used in two forms:

- a Sphinx/HTML website intended for Read the Docs;
- the PDF user guide built by GitHub Actions.

The HTML site uses the same PyData Sphinx theme family and general documentation structure as modern scientific-Python projects such as PythTB: a landing page, task-oriented tutorials, theory/method pages, and generated API documentation.

## Start here

1. [Website home](index.md) — landing page and quick example.
2. [Getting started](getting_started.md) — installation, first background run, first effective-model run, and CLI entry points.
3. [Tutorials](tutorials.md) — worked examples for the main maintained workflows.
4. [Model and conventions](model_and_conventions.md) — Ruby lattice, hopping/interactions, `Vprime`/`Vcross`, momentum convention, and current-channel naming.
5. [Public API reference](api_reference.md) — the stable imports intended for notebooks and scripts.
6. [Generated API documentation](generated_api.md) — Sphinx autodoc output from the maintained façade modules.
7. [Cluster ED+GW](cluster_ed_gw.md) — embedding equation, finite bath, self-consistency, continuation, and method limitations.
8. [Jacobian-free response](jf_response.md) — finite-q soft-mode calculations, bath tangent, solver options, and interpretation.
9. [Effective pseudospin ED](effective_pseudospin.md) — leading strong-coupling model and finite-size quantum ED.
10. [Cluster-orientation ensemble](orientation_ensemble.md), [checkpoints](checkpoints.md), and [numerics](numerics_and_validation.md) — diagnostics and reproducibility.

The original [GW theory](gw_theory.md) and [cGW theory](cgw_theory.md) notes remain part of the maintained documentation because they describe the numerical kernels still used by the modular workflows.

## Build the website locally

Install the package and documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

Build HTML:

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

## Read the Docs

The repository includes `.readthedocs.yaml`. After the GitHub repository is imported once into Read the Docs, builds are automatic and use `docs/conf.py` as the Sphinx configuration.

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

The `research/` directory contains historical scans, benchmarks, diagnostics, plotting scripts, and paper-specific drivers. They are retained for reproducibility but are not treated as stable API. Specialized historical documentation is stored under `research_notes/` and is intentionally excluded from the public Sphinx navigation.
