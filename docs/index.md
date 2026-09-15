# RubycGW

**Interacting-electron calculations on the Ruby lattice with GW, covariant response, cluster ED+GW, Jacobian-free response, and strong-coupling pseudospin models.**

RubycGW provides a compact Python interface for building the spinless extended-Hubbard Ruby-lattice model, solving embedded backgrounds, analyzing response channels, and comparing the microscopic fermion problem with its leading strong-coupling pseudospin description.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Getting started
:link: getting_started
:link-type: doc
Installation, your first model, a background calculation, and the maintained command-line entry points.
:::

:::{grid-item-card} 📘 Tutorials
:link: tutorials
:link-type: doc
Worked examples for model construction, cluster ED+GW, continuation, response calculations, and effective-model ED.
:::

:::{grid-item-card} 🧩 API reference
:link: api_reference
:link-type: doc
The small stable interface intended for reusable scripts and notebooks.
:::

:::{grid-item-card} 🧠 Theory and conventions
:link: model_and_conventions
:link-type: doc
Ruby-lattice conventions, interactions, current channels, and links to the GW/cGW formalism.
:::
::::

## Quick example

The public interface is exposed through `rubycgw.api`:

```python
from rubycgw.api import RubyModel, EffectiveEDConfig, run_effective_ed

model = RubyModel(
    ti=0.4,
    t1=0.2,
    t2=0.2,
    V=1.8,
    Vprime=-0.10,
    Vcross=-0.05,
)

print(model.effective_couplings())

result = run_effective_ed(
    model,
    EffectiveEDConfig(Lx=2, Ly=2, nev=4),
)

print(result.ground_energy)
print(result.gap)
```

For the full microscopic workflow, construct a `BackgroundConfig` and call `run_cluster_background`. See [Getting started](getting_started.md) and [Cluster ED+GW](cluster_ed_gw.md).

## Main capabilities

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Ruby model
Build the six-site Bloch Hamiltonian and momentum-dependent interaction matrix, including the extended `Vprime` and `Vcross` interactions.
:::

:::{grid-item-card} GW / cGW
Self-consistent lattice GW kernels and covariant-response machinery for interacting-electron calculations.
:::

:::{grid-item-card} Cluster ED+GW
Embed an exactly diagonalized six-site interacting cluster with a fitted finite bath into the lattice GW background.
:::

:::{grid-item-card} JF response
Compute finite-q linear response and soft modes using the Jacobian-free embedded response solver.
:::

:::{grid-item-card} Effective pseudospin ED
Solve the leading strong-coupling pseudospin Hamiltonian on finite tori and inspect current/charge structure factors.
:::

:::{grid-item-card} Analysis tools
Inspect self-energies, Dyson kernels, orientation dependence, checkpoints, and numerical convergence diagnostics.
:::
::::

```{admonition} Scope
:class: note
RubycGW is research software. The maintained public API is intentionally smaller than the collection of historical research scripts retained under `research/`.
```

```{toctree}
:hidden:
:maxdepth: 2

getting_started
tutorials
model_and_conventions
cluster_ed_gw
jf_response
effective_pseudospin
orientation_ensemble
checkpoints
numerics_and_validation
api_reference
generated_api
gw_theory
cgw_theory
repository_layout
developer
```
