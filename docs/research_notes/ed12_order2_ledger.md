# 12-site weak-coupling O(V^2) cGW response ledger

`scan_ed12_order2_ledger.py` is a static, Gamma-point diagnostic for the 12-site
`2x1` Ruby torus. It asks: once exact ED and Fock current feedback agree at
linear order in `V` but separate at quadratic order, which production-cGW
pieces account for the `O(V^2)` coefficient?

The diagnostic is intentionally restricted to `z_same` and `z_opposite`. In
the unbroken TR-symmetric normal state the static Hartree response to either
current source vanishes, although the production tail-consistent H/F source
treatment is kept in the calculation.

## Additive response ledger

For a source `K`, write

\[
\Gamma=K+H[\Gamma]+F[\Gamma]+MT[\Gamma]+AL[\Gamma].
\]

At weak coupling, `F` starts at `O(V)`, while correlation MT and AL start at
`O(V^2)`. Therefore the **response correction** through second order can be
organized as

\[
\Delta\chi_{\rm cGW}
=H^{(1)}+F^{(1)}+F^{(2)}+MT^{(1)}+AL^{(1)}+O(V^3).
\]

`F(2)` is the second application of the affine production Fock map, i.e. the
repeated-Fock feedback term. It is not a new irreducible diagram. Because the
production tail map is affine, `F(1)` includes both the source-tail constant
and the linear Fock action. `F(2)` is evaluated by the exact interacting-bond
reduction in `rubycgw.fock_diagnostic`.

`MT(1)` and `AL(1)` are one applications of those cGW kernels to the bare
source. Feedback involving MT/AL enters only at `O(V^3)` and higher, so it is
left in the closure remainder rather than the second-order subtotal.

The script also solves full cGW and reports

\[
\delta_{\rm close}(V)=\Delta\chi_{\rm full}
-[H^{(1)}+F^{(1)}+F^{(2)}+MT^{(1)}+AL^{(1)}].
\]

A successful weak-coupling audit has no `O(V^2)` term in this closure, up to
fit and numerical error.

## Exact reference and tails

The exact vertex correction is

\[
\Delta\chi_{\rm exact}=\chi_{\rm ED}-B_{\rm ED}^{\rm tail},
\]

where the ED bubble is completed with the analytic reference-tail remainder.
This removes the spurious `V=0` offset caused by a finite fermionic Matsubara
box.

The cGW vertex correction is

\[
\Delta\chi_{\rm cGW}=\chi_{\rm full,box}-B_{\rm GW,box}.
\]

An additive final observable bubble-tail completion cancels in this
difference, so it does not change the additive vertex ledger.

## Fits

Quantities beginning at linear order are fit as

\[
y(V)=aV+bV^2+cV^3,
\]

while `F(2)`, `MT(1)`, `AL(1)` and the closure are fit as

\[
y(V)=bV^2+cV^3.
\]

The default weak window is

```text
V = 0, 0.005, 0.01, 0.02, 0.03, 0.05
```

with `V <= 0.05` used in the fit.

The important output is the `b` ledger:

- exact ED vertex coefficient;
- full cGW vertex coefficient;
- additive cGW pieces from H(1), F(1) dressing, F(2) feedback, MT(1), AL(1);
- `full-subtotal`, which should be near zero at second order;
- `exact-full cGW`, the second-order response still missing from cGW.

A nonzero exact-minus-cGW coefficient does **not** uniquely identify SOX,
SOSEX, second-Born, or another beyond-GW correction. It only localizes the
first missing response physics and supplies a quantitative target for a later
consistent self-energy/kernel extension.

## Run

```bat
python scan_ed12_order2_ledger.py ^
  --values 0 .005 .01 .02 .03 .05 ^
  --fit-vmax .05 ^
  --out results\ed12_order2_ledger
```

Outputs:

- `ledger_scan.npz`
- `ledger_scan.csv`
- `ledger_fits.json`
- `order2_ledger.png`

The scan redoes exact 12-site ED and same-torus SC-GW at each `V`. Points are
traversed in increasing `V`, reusing the previous GW solution as the next
initial state.
