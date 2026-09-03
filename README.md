# RubycGW

Reference implementation of self-consistent `GW` and covariant `GW` (cGW)
for the spinless six-sublattice Ruby-lattice density-interaction model.

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
- [`docs/tutorial.md`](docs/tutorial.md): complete theory tutorial and main PDF source.

The GitHub Actions workflow `build tutorial PDF` automatically regenerates
`RubycGW_Tutorial.pdf` from the maintained Markdown files whenever relevant code
or documentation changes.

## Conventions

This repository preserves the earlier Ruby calculation conventions:

- sites are `0,1,2,3,4,5`;
- reduced reciprocal coordinates use `exp(2 pi i k.R)`;
- the hopping list is the previous `ti/t1/t2` 12-bond list;
- the same NN bonds carry density interaction `V`;
- `eta_A` uses `0 -> 1 -> 2 -> 0`;
- `eta_B` uses `3 -> 4 -> 5 -> 3`;
- `eta_plus = (eta_A + eta_B)/sqrt(2)` = **physical opposite circulation**;
- `eta_minus = (eta_A - eta_B)/sqrt(2)` = **physical same circulation**.

## Equations implemented

```text
G^{-1} = G0^{-1} - Sigma_H - Sigma_GW
P_ab(Q) = (T/Nk) sum_k G_ab(k+Q) G_ba(k)
W(Q) = V(Q) + V(Q) P(Q) W(Q)
Sigma_GW,ab(k) = -(T/Nk) sum_Q G_ab(k+Q) W_ba(Q)
```

The q=(0,0) cGW layer solves

```text
Gamma_eta = K_eta + Gamma_H + Gamma_MT + Gamma_AL1 + Gamma_AL2
```

and the response is

```text
chi_eta = -(T/Nk) sum_k Tr[K_eta G(k) Gamma_eta(k) G(k)] .
```

## Code layout

- `rubycgw/model.py`: Ruby hopping, NN interaction matrix, eta vertices.
- `rubycgw/grids.py`: momentum/Matsubara grids and allocation-light `k+Q` shifts.
- `rubycgw/gw.py`: noninteracting reference plus self-consistent Hartree + GW solver, with optional continuation warm start.
- `rubycgw/susceptibility.py`: `G0G0`, `GG`, and full-vertex eta response.
- `rubycgw/cgw.py`: q=0 Hartree, MT, AL1, AL2 vertex corrections; fused correction loop and vertex warm start.
- `rubycgw/orbital_moment.py`: reconstruct an 18-site checkpoint Green function and evaluate triangle bond currents/local plaquette moments.
- `run_ruby_cgw.py`: staged `G0G0 -> GG -> GW+MT -> full cGW` reference run.
- `convergence_scan.py`: automated cutoff convergence scans with CSV/PNG output.
- `analyze_orbital_moment.py`: checkpoint orbital-moment command-line post-processor.
- `tests/`: convention, filling, Hermiticity and V=0 regression checks.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_ruby_cgw.py
```

Local orbital moments from a converged zero-source 18-site GW checkpoint:

```bash
python analyze_orbital_moment.py checkpoints/example.npz \
  --csv orbital_moment.csv \
  --json orbital_moment.json
```

Add `--energy-unit-ev E0 --lattice-constant-angstrom a` to also report
charge current in amperes and plaquette moments in Bohr magnetons.  The direct
checkpoint tool measures an already-present loop current; the electromagnetic
covariant response `dG/dA` is a separate calculation and is not implied by this
post-processing step.

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

The stored fermion Matsubara box is finite. Values of `G(i omega+i Omega)` outside the stored box are zero. Performance-critical loops now operate only on the valid Matsubara slice, but production results still require explicit `nw`, `nOmega`, and `nk` convergence tests.
