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
- [`docs/convergence_scan.md`](docs/convergence_scan.md): automated `nw`, `nOmega`, and `nk` scans;
- [`docs/tutorial.md`](docs/tutorial.md): complete theory tutorial and main PDF source.

The GitHub Actions workflow `build tutorial PDF` automatically regenerates
`RubycGW_Tutorial.pdf` from the maintained tutorial Markdown files whenever
relevant code or documentation changes, and uploads it as the
`RubycGW-Tutorial-PDF` artifact.

## Conventions

This repository intentionally preserves the conventions used in the earlier
`ruby_selection_rule_check_physical_labels.py` calculation:

- sites are `0,1,2,3,4,5`;
- reduced reciprocal coordinates use `exp(2 pi i k.R)`;
- the hopping list is exactly the previous `ti/t1/t2` 12-bond list;
- the density interaction puts the same NN coupling `V` on those 12 bonds;
- `eta_A` uses `0 -> 1 -> 2 -> 0`;
- `eta_B` uses `3 -> 4 -> 5 -> 3`;
- `eta_plus = (eta_A + eta_B)/sqrt(2)` = **physical opposite circulation**;
- `eta_minus = (eta_A - eta_B)/sqrt(2)` = **physical same circulation**.

## Equations implemented

The GW layer uses

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

by differentiating the converged GW equations with respect to an external eta
source. The final response is

```text
chi_eta = -(T/Nk) sum_k Tr[K_eta G(k) Gamma_eta(k) G(k)] .
```

The current code is a **transparent reference implementation**. It uses
explicit momentum/frequency shifts and is intended first for small-grid
validation. The next optimization step is to replace the expensive
convolutions with FFT/Krylov implementations after all symmetry and limiting
checks pass.

## Code layout

- `rubycgw/model.py`: Ruby hopping, NN interaction matrix, eta vertices.
- `rubycgw/grids.py`: reduced momentum and Matsubara grids, `k -> k+q` shifts.
- `rubycgw/gw.py`: noninteracting reference plus self-consistent Hartree + GW solver.
- `rubycgw/susceptibility.py`: `G0G0`, `GG`, and full-vertex eta response.
- `rubycgw/cgw.py`: q=0 Hartree, MT, AL1, AL2 vertex corrections.
- `run_ruby_cgw.py`: staged `G0G0 -> GG -> GW+MT -> full cGW` reference run.
- `convergence_scan.py`: automated cutoff convergence scans with CSV/PNG output.
- `tests/`: convention, filling, Hermiticity and V=0 regression checks.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_ruby_cgw.py
```

For an automated convergence scan:

```bash
python convergence_scan.py --scan all
```

or scan one cutoff at a time:

```bash
python convergence_scan.py --scan nw
python convergence_scan.py --scan nomega
python convergence_scan.py --scan nk
```

Start with the tiny defaults. Then perform convergence tests in this order:

1. increase `nw` at fixed small momentum grid;
2. increase `nOmega`;
3. increase `nk1=nk2`;
4. reduce GW and vertex mixing dependence;
5. compare `G0G0 -> GG -> GW+MT -> full cGW` for both eta channels.

## Important numerical note

The stored fermion Matsubara box is finite. When a convolution requests
`G(i omega_n + i Omega_m)` outside that box, the reference code sets it to zero.
This makes the implementation simple and auditable, but production results must
be checked carefully versus `nw` (or later upgraded to an analytic
high-frequency tail treatment).
