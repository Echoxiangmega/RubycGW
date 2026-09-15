# Developer guide

## Design rule

Keep validated numerical kernels separate from the stable user interface.  A new feature should normally be implemented in this order:

1. numerical kernel or reusable helper under `rubycgw/`;
2. regression tests;
3. a high-level workflow/config dataclass if the feature is an end-to-end calculation;
4. export through `rubycgw.api` only after the interface is stable;
5. a thin CLI under `scripts/` if interactive command-line use is useful;
6. documentation in `docs/`.

Paper-specific parameter scans and diagnostics belong in `research/` unless they become generally reusable.

## Public API stability

`rubycgw.api` is intentionally small.  Treat names exported there as the compatibility surface.  Internal modules remain importable, but file-level private APIs may change during method development.

Prefer immutable dataclasses for user configuration.  The current pattern is:

```text
RubyModel
GridConfig
GWConfig
ClusterConfig
BackgroundConfig
EffectiveEDConfig
```

A future GUI can map widgets directly onto these objects without duplicating solver logic.

## Do not duplicate physics in CLIs

A maintained script should parse arguments, construct public configuration objects, call a workflow, and save/print results.  Avoid reimplementing interaction bonds, self-consistency loops, or checkpoint validation inside a CLI.

Historical drivers that still contain substantial calculation logic are kept in `research/` until the corresponding workflow is modularized.

## Extended interactions

Canonical `V`, `Vprime`, and `Vcross` definitions live in `rubycgw.models.ruby`.  `vprime_study/` is compatibility code.  New implementations should use `RubyModel`/`ExtendedRubyParameters` and `extended_cluster_interactions` rather than adding another patch-specific model copy.

## Testing

CI performs an editable installation, compiles the maintained and archived Python trees, and runs the full pytest suite:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q rubycgw scripts research vprime_study
python -m pytest -q
```

Tests that cover reusable mathematics should import package functions, not research drivers.  A small number of tests may continue to exercise archived scripts when the historical workflow itself is being preserved.

## Numerical regressions

Refactoring should preserve established numerical reference results.  When moving logic from a research script into the package, first extract the exact implementation, test the old and new paths on the same small problem, then switch the test to the package API.

For nonlinear solvers, do not use convergence alone as the only regression.  Record relevant combinations of residual, density, chemical potential, bath error, impurity mismatch, and selected response eigenvalues.

## Documentation policy

Top-level files in `docs/` describe supported workflows.  Detailed derivation notebooks, obsolete command paths, and experiment-specific validation reports go to `docs/research_notes/`.  Do not silently rewrite historical notes to look current; preserve them as records and write a new maintained page instead.

## Adding an interactive app

The future app should depend only on public models/configurations/workflows.  It should not shell out to `research/` scripts.  Long calculations should expose progress through workflow callbacks or a task layer rather than scraping stdout.  Keeping this boundary is the main reason for the current repository modularization.
