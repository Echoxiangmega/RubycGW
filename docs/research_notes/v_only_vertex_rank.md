# V-only Ruby: ED–cGW static rank experiment

The original spinless model is implemented in
[`rubycgw/model.py`](../rubycgw/model.py).
Its six-site cell contains A=(0,1,2) and B=(3,4,5). The interaction is
**only** on (0,1), (0,2), (1,2), (3,4), (3,5), and (4,5); intertriangle
`t1/t2` bonds are hopping-only. The driver imports `RubyParameters` and
uses `ExactSmallRubyThermal` rather than constructing a different graph.

The driver is
[`research/diagnose_v_only_vertex_rank.py`](../research/diagnose_v_only_vertex_rank.py).

## Run

From the repository root, with the project dependencies installed:

```bash
# Small inexpensive system for numerical checks (six-site periodic torus)
python research/diagnose_v_only_vertex_rank.py \
    --L1 1 --L2 1 --V 0.0 0.2 0.6 \
    --T 0.08 --filling 3 --out results/v_only_vertex_rank

# Check finite-box / vertex-convergence sensitivity
python research/diagnose_v_only_vertex_rank.py \
    --L1 1 --L2 1 --V 0.6 --nw 85 --nomega 18

# Larger independent check (12-site ED: substantially more expensive)
python research/diagnose_v_only_vertex_rank.py \
    --L1 2 --L2 1 --V 0.6 --nw 55 --nomega 12
```

Defaults are `ti=0.4, t1=t2=0.2, filling=3` particles per
primitive cell, `T=0.08`. The cGW vertices include Hartree, Fock,
MT, AL1, and AL2 via the already validated `solve_vertex_q0_tail`.
The reference GW uses the existing static Fock/tail-consistent solver.
Both GW and ED are calculated on the *same finite periodic torus*.
No `Vprime` or `Vcross` is added.

## What is computed

The 36 normalized, Hermitian, one-cell form factors consist of six
densities, 15 real bonds and 15 imaginary bonds. The ED gives the exact
grand-canonical **connected** static susceptibility matrix
`chi_ed[a,b]` for these 36 sources. The full covariant-GW equation is
solved independently for each source to produce `chi_cgw[a,b]`.
Both the GW and exact-G bubbles are computed with the existing
analytic-tail-completed routine.

The default SVD target is the *common-reference response correction*

```text
D = sqrt(chi0_GW) @ (inverse(chi_cGW)-inverse(chi_ED)) @ sqrt(chi0_GW)
```

where all inverses are computed in the **same resolved positive
subspace** of `chi0_GW`, using truncated pseudoinverses. Sort the
eigenvalues of `D` by absolute magnitude and plot

```text
R(r) = sum_{j <= r} |d_j|^2 / sum_j |d_j|^2 .
```

The corresponding rank-r approximation to the inverse-response
correction is used to reconstruct a susceptibility matrix and its
projections onto `LC_same`, `LC_opposite`, and `CO_AminusB`.

The file also saves

```text
Delta K_eff =
  (inverse(chi0_ED) - inverse(chi_ED))
  - (inverse(chi0_GW) - inverse(chi_cGW)).
```

This is an **effective static kernel**; it includes effects due to the
change in the propagator. It should not be identified with the full
frequency-dependent four-leg irreducible particle-hole vertex.

Physical circulation follows the *original model* convention:
`eta_minus=LC_same`, `eta_plus=LC_opposite`.
The third target `CO_AminusB` is only one charge form factor,
not an exhaustive set of possible charge orders.

## Outputs and interpretation

Each `V_<value>/` folder contains `rank_diagnostic.npz`,
`spectrum.png`, `rank_capture.png`, `mode_weights.png`, and
`target_reconstruction.png`.

A fast rise in `R(r)` establishes low rank **in this static,
intracell, q=0, metric-weighted space only**. It is not enough to
establish that a rank-r approximation predicts ordered-state free
energies or other momenta/frequencies. In particular:

1. Increase `--nw` and `--nomega` to check numerical convergence.
   Compare the raw bubbles and ED–cGW response before interpreting
   the spectrum. Inspect `vertex_residuals` in the NPZ.
2. Compare the 1x1 and 2x1 tori; small finite clusters can artificially
   select modes.
3. A negligible Frobenius tail can still determine which nearly
   degenerate LC/CO mode wins. The plot
   `target_reconstruction.png` is therefore as important as
   `rank_capture.png`.
4. If a rank-truncated inverse response has a resolved negative
   eigenvalue, its reconstructed response is stored as NaN rather than
   treating it as physical.
5. To study the actual irreducible vertex, extend to
   `chi_{ab,cd}(nu,nu';Omega)` and invert its full Bethe–Salpeter
   equation with frequency regularization. This has **not** yet been
   implemented by the static driver.

The baseline `V=0` case is particularly useful for checking
operator normalization, ED/GW consistency, and Matsubara tail accuracy.


## Follow-up: stable response-difference diagnostic (2026-10-16)

The original eigenvalue plot is unstable in tiny-response directions
because of explicit inverse susceptibilities. For the original
`1x1` data at `V=0.6`, varying the relative cutoff from `0.001`
to `0.01` changes whether the first four reconstructed inverse
modes correct the loop currents at all. Do not interpret its original
rank-1 result as proof that the irreducible vertex is rank one.

The additional script
[`research/analyze_v_only_response_rank.py`](../../research/analyze_v_only_response_rank.py)
instead diagonalizes
`DeltaChi = chi_ED - chi_cGW` **without matrix inversion**.
This captures response-error modes, not a fully irreducible
or particle-hole irreducible four-point vertex.

Original 1x1 `V=0.2`/`0.6`:

* first four DeltaChi eigenvectors capture about 82.7% / 94.7%
  of the squared Frobenius error;
* 95% squared-error capture needs 7 / 5 modes, respectively;
* the LC response errors are largely captured by the first two
  directions in both cases.
* Reusing the **subspace** learned at V=0.2 for V=0.6 and refitting
  its coefficients with ED at V=0.6 nearly reproduces all three selected
  target susceptibilities after seven directions. This is a
  transferability *oracle*, not a prediction without ED.

### Independent 12-site tests

The following were run on GitHub Actions with `L1=2,L2=1`,
`V=0.6`, `T=0.08`, and filling=3 per primitive cell,
with the preliminary `nw=28,nOmega=7` Matsubara window:

| Primitive momentum | Quantity | ED | covariant GW |
|---|---|---:|---:|
| (0,0) | LC_same | 1.93325 | 2.838101 |
| (0,0) | LC_opposite | 1.882308 | 2.640437 |
| (0,0) | CO_AminusB | 0.5777883 | 0.429482 |
| (1/2,0) | LC_same | 1.882308 | 2.640437 |
| (1/2,0) | LC_opposite | 1.93325 | 2.838101 |
| (1/2,0) | CO_AminusB | 0.5597473 | 0.445984 |

Direct-DeltaChi squared-Frobenius weights:
* Q=(0,0): R(4)=0.886613, rank95=5;
* Q=(1/2,0): R(4)=0.846301, rank95=6.

These are finite-torus static responses; only the *specific* A/B
charge-difference form factor is tested as CO here. They cannot establish
the leading CO instability over all q, nor an ordered-state free-energy
comparison. No fitted correction is inserted into the self-energy.

Workflow logs and downloadable NPZ/PNG artifacts:
* [Q=(0,0) pilot](https://github.com/Echoxiangmega/RubycGW/actions/runs/38911132976)
* [Q=(1/2,0) pilot](https://github.com/Echoxiangmega/RubycGW/actions/runs/38911786935)

### Real finite-Q sources

The driver also accepts `--Q1 0.5 --Q2 0` for the 2x1 torus.
This implements **real** alternating cell phases and embeds them as
Hermitian source matrices into the twelve-site supercell, allowing
reuse of the ordinary supercell q=0 cGW source-derivative solver.
It does not yet handle generic complex-momentum sources;
those require separate cosine/sine Hermitian components.

The preferred next benchmark is convergence in Matsubara windows
and the comparison of additional finite momenta and charge
form factors before making a claim about the ordering competition.


### Refined 12-site q=0 convergence and cross-V modes

A second [12-site q=0 job](https://github.com/Echoxiangmega/RubycGW/actions/runs/38911508679)
used `nw=55,nOmega=12` at the same temperature and filling.

| V | channel | ED | cGW |
|---|---|---:|---:|
| 0.2 | LC_same | 1.624014 | 1.691236 |
| 0.2 | LC_opposite | 1.542006 | 1.595567 |
| 0.2 | CO_AminusB | 0.6074731 | 0.5881776 |
| 0.6 | LC_same | 1.933250 | 2.853599 |
| 0.6 | LC_opposite | 1.882308 | 2.654854 |
| 0.6 | CO_AminusB | 0.5777883 | 0.4284481 |

The GW/ED bubble matrix relative errors are about `0.00347` and
`0.02685` at V=0.2 and V=0.6. The cGW vertex residuals at V=0.6
are below `4.6e-9`; the GW residual is about `1.75e-8`.
Direct `DeltaChi` spectra give `R4=0.787943` and `0.886689`,
with 95% capture requiring 7 and 5 modes respectively.

The **first two** response-difference eigenvectors at V=0.2 and V=0.6
have unsigned inner products approximately `0.99964` and `0.99905`,
respectively. The seven-dimensional leading subspaces also show
strong overlap (their seven principal-angle singular values are
all above ~0.996). At V=0.6 the leading eigenvectors are dominated
by imaginary-bond (current) operators, while the seventh important
eigenvector predominantly contributes to the A-minus-B charge
response.

Projecting the V=0.6 DeltaChi onto the *fixed seven-dimensional
subspace* derived at V=0.2 while obtaining **new amplitudes from ED
at V=0.6** yields about -1.05%, -1.48%, and +0.42% errors for the
same, opposite and CO observables. These values diagnose structural
transferability but do not validate an ED-independent prediction.

Comparing V=0.6 q=0 cGW using `nw=28,nOmega=7` versus
`nw=55,nOmega=12` changes the shown current responses by ~0.5%;
this is encouraging but is **not** a rigorous cutoff extrapolation.


### Refined 12-site finite-Q comparison

The [refined Q=(1/2,0) run](https://github.com/Echoxiangmega/RubycGW/actions/runs/38912441224)
uses `nw=55,nOmega=12` for both `V=0.2` and `V=0.6`.
This modulation is relative to the primitive cell and is represented
by opposite real phases on the two cells of the finite torus.

| V | Q | target | ED | cGW |
|---|---|---|---:|---:|
| 0.2 | (1/2,0) | LC_same | 1.542006 | 1.595567 |
| 0.2 | (1/2,0) | LC_opposite | 1.624014 | 1.691236 |
| 0.2 | (1/2,0) | CO_AminusB | 0.7030093 | 0.6888156 |
| 0.6 | (1/2,0) | LC_same | 1.882308 | 2.654854 |
| 0.6 | (1/2,0) | LC_opposite | 1.933250 | 2.853599 |
| 0.6 | (1/2,0) | CO_AminusB | 0.5597473 | 0.4447949 |

Direct-DeltaChi R(4)=0.794573/0.845350 and rank95=7/6
at V=0.2/0.6. The first seven eigenspaces at these two V values
have principal-angle singular values in the interval
`[0.99671555,0.99992052]`. Transferring the seven-dimensional
basis from V=0.2 to 0.6, **while refitting its amplitudes with test-V
ED**, yields approximately -1.48%, -1.05%, and -0.004% errors
in LC_same, LC_opposite and CO_AminusB.

The sign and ordering of physical currents must be interpreted
with the geometric eta convention in `rubycgw/model.py`; in the
2x1 finite torus the two listed current responses exchange
values between Q=0 and Q=(1/2,0). The exchange is a finite-torus
fact, not an established thermodynamic symmetry.

The leading two direct-response eigenmodes at V=0.6, Q=0
are entirely in the time-reversal-odd imaginary-bond sector,
with about 77% and 83% of their squared form-factor weight
on the six triangle-internal imaginary-bond operators. The
remaining 23%/17% belongs to intertriangle bilinears. Thus
a pure two-triangle loop-current ansatz is a good starting point
but need not be an exact basis for the correction.

Finally, `DeltaChi` is **not** `Delta I_ph`:
`DeltaChi` may be low-rank because critical susceptibility modes
amplify selected directions of a full-rank irreducible correction.
The inverse-response effective kernel remains strongly sensitive
to its weak-eigenvalue truncation. A genuine dynamic `I_ph`
test requires the generalized frequency-resolved susceptibility.
