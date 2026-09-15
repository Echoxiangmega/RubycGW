# Tutorials

This page collects short workflows built from the maintained RubycGW interface. The examples are intentionally small; production calculations should use convergence checks appropriate to the physics being studied.

## 1. Build the extended Ruby model

```python
import numpy as np
from rubycgw.api import RubyModel

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.10,
    Vcross=-0.05,
)

kpts = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [1.0 / 3.0, 1.0 / 3.0],
])

h0 = model.build_h0(kpts)
Vq = model.build_interaction(kpts)

print(h0.shape)   # (3, 6, 6)
print(Vq.shape)   # (3, 6, 6)
print(model.parameters())
```

The six orbitals are the six sites of the primitive Ruby-lattice cell. Reduced momenta are expressed in reciprocal-lattice coordinates.

For the lattice and sign conventions, see [Model and conventions](model_and_conventions.md).

## 2. Inspect the leading strong-coupling couplings

When `t1=t2=t`, RubycGW can evaluate the leading projected pseudospin couplings:

```python
from rubycgw.api import RubyModel

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.10,
    Vcross=-0.05,
)

Jn, Jm, Jz = model.effective_couplings()
print(Jn, Jm, Jz)
```

These are leading-order strong-coupling parameters. They should not be interpreted as an exact replacement for the microscopic fermion model.

## 3. Solve the finite-size effective pseudospin model

```python
from rubycgw.api import (
    RubyModel,
    EffectiveEDConfig,
    run_effective_ed,
)

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.10,
    Vcross=-0.05,
)

result = run_effective_ed(
    model,
    EffectiveEDConfig(
        Lx=3,
        Ly=3,
        nev=4,
        degeneracy_tol=1e-7,
    ),
)

print("E0 =", result.ground_energy)
print("gap =", result.gap)
print("ground-state degeneracy =", result.ground_degeneracy)

for q, current, xy in zip(
    result.q_centered,
    result.z_same,
    result.xy_max,
):
    print(q, current, xy)
```

A `3 x 3` effective torus has 18 pseudospins. It resolves Gamma and K/K' momenta. A `2 x 2` torus resolves Gamma and M-type momenta instead, so comparing the two cluster shapes is useful when the competing wave vector is not known in advance.

See [Effective pseudospin ED](effective_pseudospin.md) for normalization and interpretation.

## 4. Run a microscopic cluster ED+GW background

```python
from rubycgw.api import (
    RubyModel,
    GridConfig,
    GWConfig,
    ClusterConfig,
    BackgroundConfig,
    run_cluster_background,
)

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.5,
    Vprime=-0.10,
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
    gw=GWConfig(
        max_iter=160,
        tol=1e-8,
        mixing=0.25,
        mixing_method="pulay",
    ),
    cluster=ClusterConfig(
        max_iter=200,
        tol=2e-5,
        mixing=0.80,
        mixing_method="pulay",
        nbath=6,
    ),
)

run = run_cluster_background(model, config)

print("converged =", run.result.converged)
print("mu =", run.result.mu)
print("density =", run.result.density)
print("final error =", run.result.final_error)
```

A production result should be judged from the convergence history, bath-fit quality, impurity mismatch, and physical observables rather than from a single `converged` flag alone.

See [Cluster ED+GW](cluster_ed_gw.md) and [Numerics and validation](numerics_and_validation.md).

## 5. Restart or continue from a checkpoint

A same-parameter restart and an interaction continuation are different operations. Load the saved state against the new requested model/configuration, then choose the mode explicitly:

```python
from rubycgw.api import (
    RubyModel,
    BackgroundConfig,
    load_restart_for_run,
    run_cluster_background,
)

old_checkpoint = "results/background_V1.5.npz"

new_model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.6,
    Vprime=-0.10,
    Vcross=-0.05,
)
new_config = BackgroundConfig(filling=2.0)

restart = load_restart_for_run(
    old_checkpoint,
    new_model,
    new_config,
)

run = run_cluster_background(
    new_model,
    new_config,
    restart=restart,
    restart_mode="continuation",
)
```

Use `restart_mode="restart"` only for the same physical parameter point. Use `continuation` when the supported interaction parameters are changed while keeping the structural grid/bath configuration compatible.

See [Checkpoints and continuation](checkpoints.md).

## 6. Compute Jacobian-free response on a saved background

For a converged background with nonzero `Vprime` or `Vcross`, use the extended maintained wrapper:

```bash
python scripts/run_jf_extended.py results/background.npz \
  --all-q \
  --bath-rank 0 \
  --bath-fd-step 2e-4 \
  --stage full \
  --no-rhs-recycle
```

For a baseline `Vprime=Vcross=0` background, `scripts/run_jf.py` is sufficient.

The leading JF eigenmode is the softest *continuous fluctuation of the supplied background*. If that background already breaks a symmetry, the response describes fluctuations around that broken branch rather than the primary instability of the symmetric phase.

See [Jacobian-free response](jf_response.md).

## 7. Work from the stable façade

For reusable notebooks and scripts, prefer

```python
from rubycgw.api import ...
```

and the maintained façade modules

```text
rubycgw.models
rubycgw.solvers
rubycgw.workflows
rubycgw.analysis
rubycgw.io
```

Historical parameter scans and paper-specific drivers remain under `research/` for reproducibility but are not treated as a stable user API.
