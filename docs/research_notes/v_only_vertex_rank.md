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
