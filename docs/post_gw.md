# Covariant post-GW screening in RubycGW

## Motivation from the ED benchmark

The 12-site same-torus benchmark separates a robust weak-coupling result from a
stronger-coupling background problem.

For the latest benchmark (`T=0.08`, `ti=0.4`, `t1=t2=0.2`), the weak-coupling
current response is essentially repaired by bare SOX.  For example at `V=0.1`,

- `z_same`: ED `2.964522`, cGW `2.992316`, cGW+SOX `2.964405`;
- `z_opposite`: ED `2.815379`, cGW `2.837528`, cGW+SOX `2.815379`.

At `V=1`, however, ordinary cGW strongly over-enhances the current channels,
while bare SOX over-corrects them:

- `z_same`: ED `4.032216`, cGW `11.419454`, cGW+SOX `2.309702`;
- `z_opposite`: ED `4.033763`, cGW `10.679889`, cGW+SOX `2.560204`.

This is the regime in which it is useful to ask whether improving the screened
interaction and the resulting one-particle background can move the response
back toward ED.

## Which chi belongs in post-GW?

The interaction in this repository is a two-index orbital density interaction,
so the response needed to reconstruct the physical screened interaction is the
**full reducible orbital density-density susceptibility**

\[
\chi^{nn}_{ab}(Q)
=\frac{\delta\langle n_a(Q)\rangle}{\delta\phi_b(Q)},
\qquad Q=(q,i\Omega_m).
\]

For a right density source

\[
K_b=|b\rangle\langle b|,
\]

and the RubycGW positive tangent convention

\[
X_b(k;Q)=G(k+Q)\Gamma_b(k;Q)G(k)=-\frac{dG}{dh_b},
\]

we use

\[
\boxed{
\chi^{nn}_{ab}(Q)
=-\frac{T}{N_k}\sum_{k,n}[X_b(k;Q)]_{aa}
}.
\]

This is a `norb x norb` matrix for every bosonic transfer.  It is **not**
`chi_z_same`, `chi_z_opposite`, `chi_x_even`, and it is not an irreducible
polarization to be fed into another RPA resummation.

## Sign convention and W_post

The GW implementation uses

\[
P=+GG,\qquad
W=V+VPW=(I-VP)^{-1}V.
\]

The physical covariant density response carries the opposite bubble sign.
Consequently the post-GW screened interaction in the present convention is

\[
\boxed{
W_{\rm post}(Q)=V(q)-V(q)\chi^{nn}_{\rm cov}(Q)V(q)
}.
\]

Keeping only the Hartree/RPA vertex reproduces the ordinary GW screening
identity, which is a useful sign check.

## Full dynamic transfer kernel

Post-GW needs the entire `chi_nn(q,iOmega)`, not only the static `q=0` block.
`rubycgw.transfer_cgw` therefore extends the production cGW tangent to a general
bosonic transfer.

The represented response is

\[
X(k;Q)=G(k+Q)\Gamma(k;Q)G(k).
\]

Hartree and bare-Fock derivatives use the same analytic tail machinery as the
production map.  MT differentiates the internal electron line and uses `W-V`.
The AL terms differentiate `W` and require an off-diagonal bosonic object
schematically

\[
dW(Q_i-Q_e,Q_i)
=W(Q_i-Q_e)\,[L_1+L_2]\,W(Q_i).
\]

When `Q_i-Q_e` lies outside the stored bosonic box, the required diagonal
background `W` is evaluated from the same `G` and `V` on demand rather than
zero-padded.

The transfer implementation is regression-tested to reduce to the established
production kernel at `q=0,m_ext=0`.

## Including or excluding SOX

The post engine is common to both one-particle backgrounds.

### GW background

Use

\[
\chi^{nn}_{\rm cov}[G_{GW},W_{GW};\,H/F/MT/AL].
\]

### GW+SOX background

Use

\[
\chi^{nn}_{\rm cov}[G_{GW+SOX},W_{GW+SOX};\,H/F/MT/AL/SOX].
\]

The latter includes the functional derivative of the SOX skeleton, implemented
for a general external momentum/frequency in `rubycgw.sox_transfer`.  At
`Q=0` it reduces to the existing static SOX vertex.

## One-shot post update

The post construction proceeds in the following order:

1. compute the covariant density response on the chosen background;
2. construct
   \[
   W_{post}=V-V\chi^{nn}_{cov}V;
   \]
3. evaluate the post self-energy with the background Green function;
4. solve Dyson once to obtain `G_post`, including a new fixed-filling chemical
   potential when requested.

In the split RubycGW notation,

\[
\Sigma_{\rm post}
=\Sigma_H^{bg}
+\Sigma_F[G_{bg},V]
+\Sigma_c[G_{bg},W_{\rm post}-V]
+\Sigma_{extra}[G_{bg}],
\]

where `Sigma_extra=0` for GW and `Sigma_extra=Sigma_SOX` for GW+SOX.

## Benchmark policy

The original production comparison remains

- ED,
- GG,
- cGW,
- cGW+SOX.

For a selected post method, however, the benchmark does **not** analyze a
`GG[post]` curve and does **not** keep `G` fixed while replacing only `W`.
The post state is treated as the updated pair

\[
\boxed{(G_{post},W_{post})}.
\]

The three physical pseudospin channels are then recomputed on that updated
state with the same q=0 covariant response machinery:

### ordinary post-GW

\[
\chi_{post-GW}^{\mu\nu}
\equiv
\chi_{cGW}^{\mu\nu}[G_{post},W_{post}],
\]

for `x_even`, `z_same`, and `z_opposite`.

The relevant comparison is therefore

\[
\boxed{ED\quad vs\quad cGW\quad vs\quad post-GW\;\chi.}
\]

### post-(GW+SOX)

When the SOX-post path is selected, the final response uses the updated
`(G_post,W_post)` together with the SOX vertex contribution:

\[
\chi_{post-(GW+SOX)}^{\mu\nu}
\equiv
\chi_{cGW+SOX}^{\mu\nu}[G_{post},W_{post}].
\]

The corresponding comparison is

\[
\boxed{ED\quad vs\quad cGW+SOX\quad vs\quad post-(GW+SOX)\;\chi.}
\]

Green-function information is still useful and is saved separately as

\[
\frac{\|G-G_{ED}\|_F}{\|G_{ED}\|_F}
\]

for the full represented Matsubara box and for the lowest-frequency subset.
The Green plots are therefore complementary to, not substitutes for, the
susceptibility plots.

Strictly, the q=0 response computed on the updated post background is not a
second functional derivative of the entire one-shot post construction, because
that would additionally differentiate the covariant density response entering
`W_post`.  The benchmark quantity is the covariant response of the updated
post state, which is the intended comparison for deciding whether the post
update improves the current-channel susceptibility.

## Independent post switches

Run only ordinary post-GW on the GW background:

```bash
python benchmark_ed_gw_cgw_sox.py --post-gw
```

Run only post-(GW+SOX):

```bash
python benchmark_ed_gw_cgw_sox.py --post-gw-sox
```

Run both:

```bash
python benchmark_ed_gw_cgw_sox.py --post-gw --post-gw-sox
```

A cheap static-frequency debugging run can be applied to whichever post path is
selected, for example

```bash
python benchmark_ed_gw_cgw_sox.py --post-gw --post-mmax 0
```

but this is explicitly a **windowed diagnostic**, not the full post-GW method;
outside the selected frequency window it retains the background `W`.

The ordinary `--post-gw` option does **not** invoke the much more expensive
finite-transfer SOX response.  This is the recommended first run when the goal
is to isolate plain post-GW before combining it with SOX.

## Plotting saved results

Every benchmark run writes figures automatically.  An existing `benchmark.npz`
can also be plotted without rerunning the many-body calculation:

```bash
python plot_ed_gw_cgw_sox_benchmark.py path/to/benchmark.npz
```

The main outputs are

- `benchmark_response_summary.png`: original ED/GG/cGW/cGW+SOX benchmark;
- `benchmark_post_gw_chi_*.png`: ED vs cGW vs post-GW susceptibility using
  `(G_post,W_post)`;
- `benchmark_post_gw_chi_relative_error.png`: cGW and post-GW susceptibility
  errors relative to ED;
- `benchmark_post_gw_sox_chi_*.png`: independently selected SOX-post response;
- `benchmark_green_relative_error.png`: full Matsubara Green-function error;
- `benchmark_green_lowfreq_error.png`: low-frequency Green-function error.
