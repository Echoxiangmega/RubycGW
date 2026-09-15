# Public API reference

RubycGW now has a deliberately small public surface.  New notebooks, maintained scripts, and future graphical interfaces should start from `rubycgw.api` rather than importing internal research modules directly.

## `rubycgw.api`

The public module exports:

```python
RubyParameters
ExtendedRubyParameters
RubyModel

GridConfig
GWConfig
ClusterConfig
BackgroundConfig
BackgroundRun
load_restart_for_run
run_cluster_background
save_background
background_metadata

EffectiveEDConfig
EffectiveEDResult
run_effective_ed
```

### `RubyModel`

```python
RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=0.2,
    Vprime=0.0,
    Vcross=0.0,
)
```

Important methods:

```python
model.parameters()                 # ExtendedRubyParameters
model.build_h0(kpts)               # (...,6,6)
model.build_interaction(qpts)      # (...,6,6)
model.cluster_interactions()       # q=0 impurity projection
model.effective_couplings()        # (Jn,Jm,Jz), requires t1=t2
```

## Background workflow

### `GridConfig`

```python
GridConfig(
    Lx=3,
    Ly=3,
    nw=55,
    nOmega=12,
    T=0.08,
)
```

`Lx` and `Ly` are the primitive-cell momentum-grid dimensions.  `nw` is the number of positive fermionic Matsubara indices stored on each side of zero; `nOmega` is the corresponding bosonic cutoff parameter used by `MatsubaraGrid`.

### `GWConfig`

Controls the standalone lattice GW initializer:

```python
GWConfig(
    max_iter=160,
    tol=1e-8,
    mixing=0.25,
    mixing_method="pulay",
    pulay_history=6,
    pulay_start=3,
    pulay_regularization=1e-10,
    momentum_backend="fft",
    verbose=True,
)
```

### `ClusterConfig`

Controls the coupled cluster-ED+GW embedding:

```python
ClusterConfig(
    max_iter=200,
    tol=2e-5,
    mixing=0.80,
    mixing_method="pulay",
    pulay_history=8,
    pulay_start=3,
    pulay_regularization=1e-7,
    pulay_step_cap=3.0,
    impurity_mixing=1.0,
    nbath=6,
    bath_fit_nfreq=12,
    bath_fit_max_nfev=300,
    bath_energy_window=4.0,
    bath_coupling_bound=4.0,
    bath_fit_xtol=1e-9,
    discard_weight_tol=1e-11,
    verbose=True,
)
```

### `BackgroundConfig`

Combines filling and the three numerical configuration groups:

```python
BackgroundConfig(
    filling=2.0,
    grid=GridConfig(...),
    gw=GWConfig(...),
    cluster=ClusterConfig(...),
)
```

### `run_cluster_background`

```python
run = run_cluster_background(
    model,
    config,
    restart=None,
    restart_mode="continuation",
)
```

`restart_mode` accepts:

- `fresh`: ignore restart data and solve a new background;
- `restart`: continue the same physical parameter point;
- `continuation`: reuse a converged embedded state while changing allowed interaction parameters.

The returned `BackgroundRun` contains the model/configuration, grid, `h0`, `Vq`, numerical result, and restart metadata.

The main solver result includes at least

```text
G, W, P
Sigma_H
Sigma_emb
Sigma_GW_lattice
Sigma_GW_cluster
Sigma_ED_cluster
G_cluster
G_impurity
mu
density
bath
converged
iterations
final_error
impurity_mismatch
bath_fit_error
residual histories
```

## Effective pseudospin ED

### `EffectiveEDConfig`

```python
EffectiveEDConfig(
    Lx=3,
    Ly=3,
    nev=4,
    tol=1e-10,
    maxiter=None,
    degeneracy_tol=1e-7,
)
```

### `run_effective_ed`

```python
result = run_effective_ed(model, EffectiveEDConfig(Lx=3, Ly=3))
```

The result contains the projected couplings `Jn,Jm,Jz`, ground-state energy, gap, low levels in both parity sectors, allowed q points, the full 6x6 equal-time pseudospin structure factor, leading eigenmodes, and the projected `z_same`, `z_opposite`, and `xy_max` diagnostics.

## Response solver façade

`rubycgw.solvers.response` provides stable imports for the lower-level JF machinery:

```python
from rubycgw.solvers.response import (
    BathTangentOptions,
    ClusterJFOptions,
    build_embedded_jacobian,
    response_matrix,
)
```

The end-to-end q-scan CLI is currently `scripts/run_jf.py`.  A fully consolidated high-level `JFConfig/run_jf` workflow is intentionally not advertised yet; extended-model wrappers remain in `research/` until that API is stabilized.

## Analysis helpers

Reusable self-energy analysis is available through `rubycgw.analysis`:

```python
from rubycgw.analysis import (
    realspace_to_k,
    k_to_realspace,
    dyson_kernel,
    decompose_kernel,
    green_from_kernel,
)
```

The gauge-insensitive Dyson kernel used by these helpers is

\[
K(k,i\omega)=\Sigma(k,i\omega)-\mu I
=i\omega I-h_0(k)-G^{-1}(k,i\omega).
\]

## Internal modules

Modules under the top level of `rubycgw/` still contain the validated numerical kernels and remain importable.  They are not guaranteed to keep their file-level API stable.  Code meant for reuse should prefer the façade modules under `models/`, `solvers/`, `workflows/`, `analysis/`, `io/`, and `symmetry/`.
