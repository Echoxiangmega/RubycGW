# RubycGW

RubycGW is a research codebase for interacting spinless fermions on the Ruby lattice. The maintained interface focuses on four workflows: model construction, self-consistent cluster ED+GW backgrounds, Jacobian-free response calculations, and the strong-coupling effective pseudospin ED model.

## Install

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Public Python API

```python
from rubycgw.api import RubyModel, GridConfig, BackgroundConfig, run_cluster_background

model = RubyModel(
    ti=0.4, t1=0.2, t2=0.2,
    V=1.8, Vprime=-0.1, Vcross=-0.05,
)
config = BackgroundConfig(
    filling=2.0,
    grid=GridConfig(Lx=3, Ly=3, T=0.08),
)
run = run_cluster_background(model, config)
print(run.result.mu, run.result.density)
```

For the leading strong-coupling pseudospin model:

```python
from rubycgw.api import EffectiveEDConfig, run_effective_ed

ed = run_effective_ed(model, EffectiveEDConfig(Lx=3, Ly=3))
print(ed.ground_energy, ed.gap)
```

## Maintained command-line entry points

```bash
python scripts/run_background.py --Lx 3 --Ly 3 --V 1.8 --Vp -0.1 --Vx -0.05 --out results/background.npz
python scripts/run_effective_ed.py --Lx 3 --Ly 3 --V 1.8 --Vp -0.1 --Vx -0.05
python scripts/run_jf.py results/background.npz --all-q --bath-rank 0 --bath-fd-step 2e-4 --stage full --no-rhs-recycle
python scripts/run_primitive_cgw.py --help
```

`run_jf.py` is the maintained baseline JF driver. Historical V-prime/V-cross wrappers and specialized validation drivers are preserved under `research/` while the high-level response workflow is being consolidated.

## Repository layout

```text
rubycgw/        installable package and numerical kernels
scripts/        maintained user-facing command-line entry points
docs/           current user and developer documentation
tests/          regression and numerical tests
research/       historical scans, diagnostics, benchmarks, and paper workflows
vprime_study/   compatibility layer for older V-prime/V-cross calculations
```

The development-time `run_*`, `scan_*`, `benchmark_*`, `diagnose_*`, `plot_*`, and `validate_*` scripts are intentionally kept out of the repository root. They remain available in `research/` for reproducibility but are not part of the stable public interface.

## Documentation

Start with [`docs/README.md`](docs/README.md). The maintained documents cover installation, model conventions, the public API, cluster ED+GW, JF response, effective pseudospin ED, cluster-orientation diagnostics, checkpoints, numerics, and repository development.

Specialized derivations and historical validation notes are retained under [`docs/research_notes/`](docs/research_notes/).

## Scope and caveats

The cluster embedding uses

```text
Sigma_emb(k,iw) = Sigma_GW^lat(k,iw)
                - Sigma_GW^cluster(iw)
                + Sigma_ED^cluster(iw)
```

with a finite-bath six-site impurity. The polarization remains the lattice GW bubble, so this is a self-energy embedding diagnostic rather than a full GW+EDMFT implementation. JF eigenmodes diagnose continuous soft modes of the chosen self-consistent background; they are not by themselves a free-energy comparison of competing ordered states.

For the extended model, `Vprime` acts on the straight inter-triangle links and `Vcross` on the crossed links of the same neighboring triangle pairs. Setting both to zero reproduces the baseline Ruby model.
