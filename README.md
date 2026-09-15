# RubycGW

Reference implementation of self-consistent `GW` and covariant `GW` (cGW)
for the spinless six-sublattice Ruby-lattice density-interaction model.

The repository contains both a maintained **public package layer** and a larger
set of research/validation scripts accumulated during method development.  New
code should use the public package API under `rubycgw.api`; historical root-level
scripts remain available for reproducibility and will be migrated gradually.

## Install

For development or normal local use:

```bash
python -m pip install -e .
python -m pytest -q
```

The old `python -m pip install -r requirements.txt` workflow is still supported.

## Public Python API

The intended entry point for notebooks, reusable scripts, and future graphical
frontends is:

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
    Vprime=-0.10,
    Vcross=-0.05,
)

config = BackgroundConfig(
    filling=2.0,
    grid=GridConfig(Lx=3, Ly=3, T=0.08),
)

run = run_cluster_background(model, config)
save_background("results/background.npz", run)
```

The strong-coupling pseudospin ED has the same model object:

```python
from rubycgw.api import EffectiveEDConfig, run_effective_ed

result = run_effective_ed(
    model,
    EffectiveEDConfig(Lx=3, Ly=3),
)
```

Clean command-line frontends using this API live in `scripts/`, for example:

```bash
python scripts/run_background.py \
  --Lx 3 --Ly 3 --V 1.8 --Vp -0.1 --Vx -0.05 \
  --out results/background.npz

python scripts/run_effective_ed.py \
  --Lx 3 --Ly 3 --V 1.8 --Vp -0.1 --Vx -0.05
```

## Maintained module layout

The new public layout separates physical models, numerical solvers, workflows,
symmetry, IO, and post-processing:

```text
rubycgw/
  api.py                 stable small public surface
  models/                Ruby model and interaction definitions
  solvers/               GW, cluster ED+GW, JF, effective ED facades
  workflows/             app/notebook-ready high-level calculations
  symmetry/              C3 and cluster-orientation utilities
  io/                    checkpoint serialization
  analysis/              reusable result summaries
```

The original modules such as `rubycgw/gw.py`, `rubycgw/cluster_ed_gw_fast.py`,
and the root-level research scripts are deliberately retained so numerical
results and old commands remain reproducible.  The public layer calls those
validated kernels instead of duplicating their physics.

## Documentation

The maintained documentation lives in [`docs/`](docs/README.md). In particular:

- [`docs/getting_started.md`](docs/getting_started.md): installation and first run;
- [`docs/model_and_conventions.md`](docs/model_and_conventions.md): Ruby lattice and eta conventions;
- [`docs/gw_theory.md`](docs/gw_theory.md): self-consistent GW equations;
- [`docs/cgw_theory.md`](docs/cgw_theory.md): Hartree, MT, AL1, AL2 and the cGW vertex equation;
- [`docs/api_reference.md`](docs/api_reference.md): modules, classes, functions, inputs/outputs and array shapes;
- [`docs/numerics_and_validation.md`](docs/numerics_and_validation.md): convergence and validation checks;
- [`docs/convergence_scan.md`](docs/convergence_scan.md): automated `nw`, `nOmega`, and `nk` scans, continuation and fast MT mode;
- [`docs/performance_and_reuse.md`](docs/performance_and_reuse.md): performance bottlenecks and what can be reused between scan points;
- [`docs/orbital_moment.md`](docs/orbital_moment.md): checkpoint-to-bond-current and local plaquette orbital-moment post-processing;
- [`docs/electromagnetic_response.md`](docs/electromagnetic_response.md): Peierls-flux covariant response and finite-difference validation;
- [`docs/bulk_orbital_magnetization.md`](docs/bulk_orbital_magnetization.md): Nourafkan bulk-orbital-magnetization formula and complete `M1+M2` workflow;
- [`docs/uniform_B_self_energy_derivation.md`](docs/uniform_B_self_energy_derivation.md): detailed derivation of the uniform-`B` Green-function/self-energy response, the repository `C_GW` Jacobian-vector notation, Hartree/Fock/MT/AL decomposition, self-consistent `Sigma_B` equation, GMRES implementation, caveats, and references;
- [`docs/tutorial.md`](docs/tutorial.md): complete theory tutorial and main PDF source.

The GitHub Actions workflow `build tutorial PDF` automatically regenerates
`RubycGW_Tutorial.pdf` from the maintained Markdown files whenever relevant code
or documentation changes.

## Conventions

This repository preserves the earlier Ruby calculation conventions:

- sites are `0,1,2,3,4,5`;
- reduced reciprocal coordinates use `exp(2 pi i k.R)`;
- the hopping list is the previous `ti/t1/t2` 12-bond list;
- density interaction `V` acts on the six intra-triangle bonds only;
- `eta_A` uses `0 -> 1 -> 2 -> 0`;
- `eta_B` uses `3 -> 4 -> 5 -> 3`;
- `eta_plus = (eta_A + eta_B)/sqrt(2)` = **physical opposite circulation**;
- `eta_minus = (eta_A - eta_B)/sqrt(2)` = **physical same circulation**.

The public `RubyModel` additionally supports the current short-range extensions
`Vprime` (straight neighbouring-triangle links) and `Vcross` (crossed links in
the same neighbouring-triangle pair).  Setting both to zero recovers the
baseline interaction.

## Equations implemented

```text
G^{-1} = G0^{-1} - Sigma_H - Sigma_GW
P_ab(Q) = (T/Nk) sum_k G_ab(k+Q) G_ba(k)
W(Q) = V(Q) + V(Q) P(Q) W(Q)
Sigma_GW = Sigma_F + Sigma_c
Sigma_c,ab(k) = -(T/Nk) sum_Q G_ab(k+Q) [W(Q)-V(Q)]_ba
```

The supercell cGW layer uses the decomposition

```text
Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2
```

and the electromagnetic module applies the same functional derivative to a
periodic Peierls-flux source. At fixed filling it also includes `dmu/dphi` by
solving the additional chemical-potential vertex `K_mu=-I`.

## Research implementation map

Important lower-level modules retained for validation and backwards compatibility:

- `rubycgw/model.py`: baseline Ruby hopping, interaction matrix, eta vertices.
- `rubycgw/grids.py`: momentum/Matsubara grids and allocation-light `k+Q` shifts.
- `rubycgw/gw.py`: noninteracting reference plus self-consistent Hartree + GW solver.
- `rubycgw/cluster_ed_gw_fast.py`: production accelerated six-site cluster ED+GW embedding.
- `rubycgw/cluster_ed_gw_jf.py`: matrix-free Jacobian-free cluster response.
- `rubycgw/cgw.py`: primitive-cell q=0 cGW response.
- `rubycgw/supercell_cgw.py`: 18-site Hartree/Fock/MT/AL covariant response.
- `rubycgw/orbital_moment.py`: checkpoint Green function, bond currents, local plaquette moments.
- `rubycgw/electromagnetic.py`: Peierls source and covariant electromagnetic response.
- `rubycgw/bulk_orbital_magnetization.py`: physical momentum derivatives and Nourafkan `M1+M2` evaluation.
- `rubycgw/magnetic_self_energy.py`: gauge-invariant uniform-`B` self-energy derivative.
- `vprime_study/`: compatibility namespace for the earlier `Vprime/Vcross` study; new model code should import `rubycgw.models` instead.
- `tests/`: numerical conventions, regression tests, and public-API tests.

## Historical command-line workflows

Existing root-level research scripts remain usable.  For example:

```bash
python run_ruby_cgw.py
```

Local orbital moments from a converged zero-source 18-site GW checkpoint:

```bash
python analyze_orbital_moment.py checkpoints/example.npz \
  --csv orbital_moment.csv \
  --json orbital_moment.json
```

Electromagnetic covariant response from the same kind of checkpoint:

```bash
python analyze_em_response.py checkpoints/example.npz \
  --channel same \
  --npz em_same.npz
```

Validate it against two fully self-consistent GW calculations at `+/-delta_phi`:

```bash
python analyze_em_response.py checkpoints/example.npz \
  --channel same \
  --finite-difference 1e-4 \
  --json em_same_validation.json
```

Full staged convergence scan:

```bash
python convergence_scan.py --scan nomega --vertex-stage both
```

Fast exploratory scan without AL1/AL2:

```bash
python convergence_scan.py --scan nk \
  --vertex-stage mt \
  --base-nw 64 --base-nomega 16 \
  --nk-values 4 6 8
```

Compatible scans use continuation by default. Add `--no-continuation` to force every point to restart from the bare initial guess.

## Numerical note

The stored fermion Matsubara box is finite. Values of `G(i omega+i Omega)` outside the stored box are zero. Production results require explicit `nw`, `nOmega`, and `nk` convergence tests. For electromagnetic validation, decrease `delta_phi` until the finite-difference error stops improving; if it plateaus, increase `nw` before interpreting the mismatch as a vertex error.
