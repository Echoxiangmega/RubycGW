# Research archive

This directory contains the historical calculation drivers, benchmarks, scans,
diagnostics, plotting utilities, and validation scripts that accumulated while
the Ruby-lattice project was being developed.

They are kept here so published/intermediate calculations remain reproducible,
but they are **not** the supported user interface of RubycGW.  New code should
prefer:

- `rubycgw.api` for Python and notebook use;
- `rubycgw.models`, `rubycgw.solvers`, and `rubycgw.workflows` for modular use;
- `scripts/` for maintained command-line entry points.

Most archived scripts are unchanged apart from their path.  When an archived
script imports another historical driver by its old bare module name, run it as

```bash
python research/<script>.py ...
```

from the repository root so the `research/` directory is on Python's import
path.  Editable installation (`python -m pip install -e '.[dev]'`) is
recommended so imports from `rubycgw` are always available.

The documentation under `docs/research_notes/` is archival in the same sense:
it records specialized implementation and validation work, while the top-level
`docs/` files describe the maintained public workflows.
