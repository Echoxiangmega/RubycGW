# 12-site exact-G four-layer benchmark

`diagnose_ed12_four_layer.py` is a small-cluster diagnostic designed to separate
one-particle GW errors from two-particle cGW-kernel errors.

The default finite torus is the ordinary rectangular 2x1 primitive-cell Ruby
cluster (12 sites).  All particle-number sectors are fully diagonalized; the
largest block at n=3 is `C(12,6)=924`.  The exact benchmark is grand canonical,
with the exact chemical potential chosen so that `<N>=6`, while SC-GW separately
adjusts its chemical potential to the same average total particle number.

For each requested channel the script compares

\[
C_{ED}(\tau),\qquad
C_{bubble[G_{ED}]}(\tau),\qquad
C_{bubble[G_{GW}]}(\tau),\qquad
C_{cGW[G_{GW}]}(\tau).
\]

The exact one-particle Green function uses the full Lehmann representation.  To
avoid doing expensive basis/eigenbasis transforms for thermally irrelevant
initial states, the code retains states until the cumulative exact
Grand-canonical Boltzmann weight reaches `1 --thermal-discard`.  Final states in
the Lehmann sums are not truncated.  The default discarded initial-state weight
is `1e-12` and is printed/saved.

The most useful diagnostic decomposition is

\[
\Delta C_{background}=C_{bubble[G_{ED}]}-C_{bubble[G_{GW}]},
\]

\[
\Delta C_{vertex}^{exact}=C_{ED}-C_{bubble[G_{ED}]},
\]

\[
\Delta C_{vertex}^{cGW}=C_{cGW[G_{GW}]}-C_{bubble[G_{GW}]}.
\]

If the two bubbles are close but the two vertex corrections differ strongly,
the dominant error is in the irreducible two-particle response kernel rather
than in the one-particle GW background.  Conversely, a large
`bubble[G_ED]-bubble[G_GW]` difference already diagnoses an inaccurate GW
one-particle background.

The driver also prints the full and low-Matsubara relative error between
`G_GW` and `G_ED`, exact particle-number sector weights, midpoint values at
`tau=beta/2`, and static `iOmega=0` values.

Default run:

```bat
python diagnose_ed12_four_layer.py ^
  --V 1 ^
  --filling 3 ^
  --T 0.08 ^
  --channels x_even z_same z_opposite ^
  --out ed12_four_layer.npz
```

The same 2x1 torus can probe the folded primitive `M1=(1/2,0)` harmonic with
`--q m1`.  Gamma is the default.

The full cGW layer uses the production tail-consistent finite-frequency solver,
including H/F/MT/AL.  The bubble layers and cGW inverse Matsubara transform use
the same represented `nw/nOmega` window.  The direct exact ED `C(tau)` is not
bosonic-frequency truncated; therefore the midpoint is the cleanest part of the
curve for quantitative comparison, while endpoint differences should still be
checked against `nOmega` convergence.
