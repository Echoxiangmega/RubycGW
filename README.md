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
- [`docs/electromagnetic_response.md`](docs/electromagnetic_response.md): Peierls-flux covariant response and finite-difference validation;
- [`docs/bulk_orbital_magnetization.md`](docs/bulk_orbital_magnetization.md): Nourafkan bulk-orbital-magnetization formula, physical k derivatives, and the GW field-self-energy caveat;
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

The bulk-magnetization module evaluates the first term of Nourafkan et al.
Phys. Rev. B 90, 125132 (2014), Eq. (2), using the full checkpoint `G`, the
full `Sigma=Sigma_H+Sigma_GW`, and physical Cartesian derivatives
`-dG^{-1}/dk = D H0 + D Sigma`. The derivatives include the Ruby orbital
embedding `r_J-r_I`; they are not ordinary finite differences of the cell-gauge
Bloch matrices. The full interacting Eq. (2) also contains
`d Sigma_tilde^(B)/dB`; that second term is not silently set to zero for
nonlocal GW.

## Code layout

- `rubycgw/model.py`: Ruby hopping, interaction matrix, eta vertices.
- `rubycgw/grids.py`: momentum/Matsubara grids and allocation-light `k+Q` shifts.
- `rubycgw/gw.py`: noninteracting reference plus self-consistent Hartree + GW solver.
- `rubycgw/susceptibility.py`: `G0G0`, `GG`, and full-vertex eta response.
- `rubycgw/cgw.py`: primitive-cell q=0 cGW response.
- `rubycgw/supercell_cgw.py`: 18-site Hartree/Fock/MT/AL covariant response.
- `rubycgw/orbital_moment.py`: checkpoint Green function, bond currents, local plaquette moments.
- `rubycgw/electromagnetic.py`: Peierls source, `Gamma_phi`, `G_phi`, `P_phi`, `W_phi`, self-energy derivatives, and finite-difference validation.
- `rubycgw/bulk_orbital_magnetization.py`: physical k derivatives and Nourafkan Eq. (2) bulk-M evaluation.
- `analyze_orbital_moment.py`: checkpoint local-orbital-moment post-processor.
- `analyze_em_response.py`: checkpoint electromagnetic covariant response and optional `+/-delta_phi` validation.
- `analyze_bulk_magnetization.py`: checkpoint bulk-orbital-magnetization post-processor.
- `tests/`: convention, GW/cGW, orbital-moment, electromagnetic-response, and bulk-M tests.

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

Bulk orbital magnetization, first Nourafkan term from the same kind of checkpoint:

```bash
python analyze_bulk_magnetization.py checkpoints/example.npz \
  --npz bulk_M.npz \
  --json bulk_M.json
```

For `V=0` this is the complete Eq. (2). For an interacting nonlocal GW
checkpoint the CLI reports `complete=False` unless an independently computed
`d Sigma_tilde/db`, `b=(e a^2/hbar)B_z`, is supplied through `--sigma-b-npz`.

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

The present Peierls-flux response is periodic in the supercell and is intended
for local/current-channel response and validation. It is not itself the
uniform-B self-energy derivative required by the second term of the exact
interacting bulk-M formula.

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

The stored fermion Matsubara box is finite. Values of `G(i omega+i Omega)` outside the stored box are zero. Production results require explicit `nw`, `nOmega`, and `nk` convergence tests. For electromagnetic validation, decrease `delta_phi` until the finite-difference error stops improving; if it plateaus, increase `nw` before interpreting the mismatch as a vertex error. Bulk orbital magnetization also requires explicit `nk` and `nw` convergence; a `3x3` self-energy mesh provides only a very short-ranged spectral representation of `D Sigma`.