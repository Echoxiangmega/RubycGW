# Repository layout

The repository is organized so the root reflects the supported software rather than the chronology of the research project.

```text
RubycGW/
├─ rubycgw/          installable package
├─ scripts/          maintained command-line entry points
├─ docs/             current documentation
├─ tests/            regression and numerical tests
├─ research/         archived research drivers and diagnostics
├─ vprime_study/     compatibility package for older extended-model code
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## `rubycgw/`

The package contains two layers.  The historical top-level modules such as `gw.py`, `cluster_ed_gw_fast.py`, and `cluster_ed_gw_jf.py` remain the validated numerical kernels.  New user-facing modules sit above them:

```text
rubycgw/models/       model definitions and extended interactions
rubycgw/solvers/      stable solver imports
rubycgw/workflows/    end-to-end calculation assembly
rubycgw/symmetry/     C3 and cluster-orientation helpers
rubycgw/analysis/     reusable post-processing functions
rubycgw/io/           checkpoint and result I/O
rubycgw/api.py        compact public import surface
```

This layered organization lets internal kernels evolve without forcing notebooks and future GUI code to follow every file-level refactor.

## `scripts/`

Maintained CLI entry points are intentionally few:

```text
run_background.py      cluster ED+GW background
run_effective_ed.py    strong-coupling pseudospin ED
run_jf.py              baseline finite-q JF response
run_primitive_cgw.py   primitive GW/cGW reference driver
```

A new user feature should normally become a callable workflow first and a thin CLI second.  Avoid adding another large standalone root script.

## `research/`

This directory contains historical scans, finite-source validators, Fierz studies, SOX/SOSEX benchmarks, supercell branch searches, plotting scripts, specialized cluster drivers, and other experiment-specific code.  Files were moved rather than deleted so existing calculations remain reproducible.

The archive is deliberately outside the installed package list in `pyproject.toml`.  It may import private internals and does not promise API stability.

## `vprime_study/`

This package is retained as a compatibility layer because older extended-model scripts and the effective pseudospin implementation still import it.  Canonical public definitions of `Vprime` and `Vcross` now live in `rubycgw.models.ruby`.  New code should not add new model logic to `vprime_study/`.

## `docs/`

Top-level Markdown files describe the current public package.  Historical derivations and specialized validation records are under `docs/research_notes/`.  Archived notes are useful scientific records but may contain old command paths.

## Root policy

The repository root should contain project metadata and directories, not one-off scientific drivers.  When adding code:

- reusable physics/numerics -> `rubycgw/`;
- supported end-to-end workflow -> `rubycgw/workflows/`;
- supported CLI -> `scripts/`;
- temporary or paper-specific exploration -> `research/`;
- regression coverage -> `tests/`;
- maintained explanation -> `docs/`.

This policy is meant to keep a future interactive app simple: it should call `rubycgw.api` or workflow functions rather than discover and execute arbitrary research scripts.
