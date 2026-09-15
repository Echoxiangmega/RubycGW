# Getting started

## Installation

RubycGW requires Python 3.10 or newer.  For development or reproducible use from a cloned repository, install it in editable mode:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The core dependencies are NumPy, SciPy, and Matplotlib. Editable installation is recommended even when running archived scripts because it makes the `rubycgw` package available independently of the current working directory.

## First cluster ED+GW background

The preferred Python entry point is `rubycgw.api`:

```python
from rubycgw.api import (
    RubyModel,
    GridConfig,
    BackgroundConfig,
    run_cluster_background,
    save_background,
)

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.1,
    Vcross=-0.05,
)

config = BackgroundConfig(
    filling=2.0,
    grid=GridConfig(
        Lx=3,
        Ly=3,
        nw=55,
        nOmega=12,
        T=0.08,
    ),
)

run = run_cluster_background(model, config)
print("converged =", run.result.converged)
print("mu        =", run.result.mu)
print("density   =", run.result.density)

save_background("results/background.npz", run)
```

The equivalent maintained CLI is:

```bash
python scripts/run_background.py \
  --Lx 3 --Ly 3 \
  --V 1.8 --Vp -0.1 --Vx -0.05 \
  --filling 2 --T 0.08 \
  --out results/background.npz
```

On Windows `cmd.exe`, replace the trailing backslashes with `^`.

## Restart and continuation

A saved cluster background can be loaded through the public workflow:

```python
from rubycgw.api import load_restart_for_run, run_cluster_background

restart = load_restart_for_run("results/old.npz", model, config)

# Same physical parameter point:
run = run_cluster_background(model, config, restart=restart, restart_mode="restart")

# New V, Vprime, or Vcross with the same structural setup:
run = run_cluster_background(model, config, restart=restart, restart_mode="continuation")
```

`restart` is for continuing the same point. `continuation` intentionally allows interaction parameters to change while requiring compatible lattice size, temperature, frequency grids, filling, hoppings, and bath size.

## First effective pseudospin calculation

```python
from rubycgw.api import EffectiveEDConfig, run_effective_ed

ed = run_effective_ed(
    model,
    EffectiveEDConfig(Lx=3, Ly=3, nev=4),
)

print("Jn,Jm,Jz =", ed.Jn, ed.Jm, ed.Jz)
print("E0       =", ed.ground_energy)
print("gap      =", ed.gap)
print("z_same   =", ed.z_same)
```

CLI:

```bash
python scripts/run_effective_ed.py --Lx 3 --Ly 3 --V 1.8 --Vp -0.1 --Vx -0.05
```

The present direct ED implementation is intended for small pseudospin tori. A 3x3 triangle-center torus contains 18 pseudospins and is the practical production scale of the current parity-resolved sparse solver.

## Jacobian-free response

For the extended background created above, use:

```bash
python scripts/run_jf_extended.py results/background.npz \
  --all-q \
  --bath-rank 0 \
  --bath-fd-step 2e-4 \
  --stage full \
  --no-rhs-recycle
```

`run_jf_extended.py` reads `Vprime` and `Vcross` from the checkpoint and reuses the same production JF kernel with the canonical extended interaction. For a baseline checkpoint with `Vprime=Vcross=0`, `scripts/run_jf.py` is the direct driver.

The lower-level response objects are available from `rubycgw.solvers.response`; a fully configuration-driven high-level JF Python workflow is still being consolidated.

## Primitive GW/cGW driver

The historical primitive-cell driver is retained as a maintained CLI entry point:

```bash
python scripts/run_primitive_cgw.py --help
```

New notebooks should prefer the package imports under `rubycgw.solvers` rather than importing command-line drivers.

## Where old scripts went

Development-time scripts that used to fill the repository root were moved to `research/`. They were archived rather than deleted. If an old note says

```bash
python scan_supercell_cgw_vs_V.py ...
```

its archived equivalent is generally

```bash
python research/scan_supercell_cgw_vs_V.py ...
```

Current documentation does not rely on those scripts unless the workflow is explicitly described as research/legacy.
