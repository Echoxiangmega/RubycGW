# Bare SOX check for the 12-site O(V^2) current ledger

`diagnose_ed12_sox_order2.py` tests the simplest beyond-GW candidate suggested by
the weak-coupling ledger: the **bare second-order exchange (SOX)** self-energy.
This is deliberately a strict second-order diagnostic, not yet a self-consistent
GW+SOX production solver.

## Why this is the next test

The existing 12-site ledger found that the exact and cGW current responses agree at
linear order, while at quadratic order the repeated Fock term is strongly positive
and the existing GW direct (MT+AL) terms cancel only a small part of it. The exact
ED response therefore requires a sizeable additional negative O(V^2) contribution.

GW contains the second-order **direct/ring** self-energy through the expansion

\[
W=V+VPV+\cdots,
\qquad
\Sigma_{GW,c}=-G(W-V),
\]

but it does not contain the crossed second-order-exchange skeleton. SOX is therefore
the first missing self-energy topology to check.

## Spinless density-density SOX convention

For the RubycGW spinless density-density interaction matrix `v`, the bare SOX
self-energy in the repository Matsubara convention is

\[
\Sigma^{(2x)}_{ij}(\tau)
=
\sum_{kl}
 v_{il}v_{kj}
 G_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau).
\]

The sign is the exchange sign in the spin-orbital second-Born/GF2 expression. It is
opposite to the direct contraction after the latter is written in the same
second-Born bookkeeping.

For a static source `K`, cGW uses

\[
X=GKG.
\]

Differentiating the SOX skeleton therefore gives three terms, according to which of
the three internal Green functions is replaced by `X`:

\[
\Gamma_{SOX}^{(2)}
=
D\Sigma_{SOX}^{(2)}[X].
\]

The code evaluates free `G(tau)` and its Frechet derivative exactly from the finite
one-body spectrum. A Gauss-Legendre transform then produces the Matsubara vertex.
The scalar interaction strength is removed from `v_unit`, so the returned response
is directly the coefficient multiplying `V^2`.

## Internal normalization check

The same driver independently evaluates the strict O(V^2) direct GW derivative:

\[
W_c^{(2)}=V P_0 V.
\]

Differentiating `-G W_c^(2)` produces strict MT and AL coefficients. These should
match the `V -> 0` extrapolation of `MT(1)+AL(1)` from
`scan_ed12_order2_ledger.py`. Agreement is an important sign/normalization check
before interpreting SOX.

## Run

After the O(V^2) ledger has already been generated:

```bat
python diagnose_ed12_sox_order2.py ^
  --ledger results\ed12_order2_ledger\ledger_fits.json ^
  --out results\ed12_sox_order2
```

This does **not** rerun ED or self-consistent GW. It uses only the V=0 one-body
problem and normally finishes quickly.

If the sibling `ledger_scan.npz` exists, the driver also refits the saved scan with
`Vmax=0.02, 0.03, 0.05` without recalculating any physics. This checks whether the
comparison between the missing quadratic coefficient and SOX is stable against
higher-order leakage in the weak-coupling fit.

Outputs:

- `sox_order2.json`
- `sox_order2.png` (when the saved ledger is available)

The key comparison is

\[
b_{SOX}
\quad\hbox{vs}\quad
b_{exact}-b_{cGW}.
\]

If they have the same sign and similar magnitude, SOX identifies the dominant
missing second-order self-energy/kernel topology. Only then is it justified to
move on to a consistent self-energy extension

\[
\Sigma_{GW+SOX}=\Sigma_{GW}+\Sigma_{SOX}
\]

and its covariant derivative. A successful diagnostic does not by itself prove that
bare SOX is the best finite-coupling approximation; SOSEX or more general vertex
corrections can differ at O(V^3) and higher.
