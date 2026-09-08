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

This is exactly the regime in which it is useful to separate a missing exchange
topology from an inaccurate screened-interaction/background feedback.

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
polarization to be fed back into another RPA resummation.

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
background `W` is evaluated from the same `G` and `V` on demand.  It is not set
to zero.

The transfer implementation is regression-tested to reduce to the established
production kernel at `q=0,m_ext=0`.

## Including or excluding SOX

The post engine itself does not know which one-particle approximation generated
the background.  It accepts a matching covariant response provider.

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

The latter must include the functional derivative of the SOX skeleton.  It is
not consistent to use the ordinary cGW density vertex on a GW+SOX background
and call the result post-(GW+SOX).

`rubycgw.sox_transfer` implements the SOX derivative for a general external
momentum/frequency, including the endpoint phases of all three differentiated
Green lines and an analytic high-frequency transfer tail.  At `Q=0` it reduces
to the existing static SOX vertex.

## One-shot post Dyson step

The canonical post construction evaluates the new self-energy with the
**background** Green function and then solves Dyson once.  In the split RubycGW
notation,

\[
\Sigma_{\rm post}
=\Sigma_H^{bg}
+\Sigma_F[G_{bg},V]
+\Sigma_c[G_{bg},W_{\rm post}-V]
+\Sigma_{extra}[G_{bg}],
\]

where `Sigma_extra=0` for GW and `Sigma_extra=Sigma_SOX` for GW+SOX.

The Hartree potential is kept at its background value and the chemical
potential is re-solved for the requested filling.

An important interpretation point is that canonical post-GW does **not**
directly recompute a smaller Fock self-energy from `G_post`: the Fock piece is
still evaluated on `G_bg`.  Therefore post-GW can compensate an overly strong
Fock-driven tendency through improved dynamic screening/correlation and the
resulting `G_post`, but it is not an empirical rescaling of Fock.

## Benchmark policy: separate G, bubble chi, and W/vertex effects

The earlier ED decomposition already showed that replacing the GW Green
function by the exact ED Green function changes the bubble susceptibility much
less than the full cGW-versus-ED discrepancy.  Therefore a post benchmark that
only reports `||G_post-G_ED||` is not sufficient.

`benchmark_ed_gw_cgw_sox.py` now records three logically separate diagnostics.

### 1. Production response

The original comparison is kept uncluttered:

- ED,
- `GG[GW]`,
- cGW,
- cGW+SOX.

The script also stores `GG[GW+SOX]` for background decomposition.

### 2. Bubble on the post Green function

For every selected post path the script computes

\[
\chi^{GG}[G_{post}]
=-\frac{T}{N_k}\sum_{kn}
\mathrm{Tr}[K G_{post} K G_{post}],
\]

with the same analytic observable tail completion as the other benchmark
bubbles.  This directly answers whether the new one-particle background changes
`x_even`, `z_same`, and `z_opposite` substantially.

The bubble plot contains, as available,

- `GG[GW]`,
- `GG[GW+SOX]`,
- `GG[post-GW]`,
- `GG[post-(GW+SOX)]`.

ED full susceptibility is shown only as a reference; it is not labelled as an
ED bubble.

### 3. Response diagnostics using W_post

Because the dominant response error need not come from `G`, the benchmark also
asks directly what the new screened interaction does to the q=0 covariant
response.

For ordinary post-GW it saves

\[
\chi_{\rm diag}^{(W)}
=\mathrm{cGW}[G_{GW},W_{post}],
\]

which isolates the screening change, and

\[
\chi_{\rm diag}^{(G,W)}
=\mathrm{cGW}[G_{post},W_{post}],
\]

which includes both the changed Green function and the changed screened
interaction.

The corresponding SOX-selected path analogously uses the cGW+SOX vertex
functional.

These curves are deliberately named **diagnostics**, not strict post-GW
susceptibilities.  A strict derivative of the complete one-shot post map would
also differentiate the covariant density response inside

\[
W_{post}=V-V\chi^{nn}_{cov}V,
\]

and therefore requires an additional higher-order response construction.  The
present diagnostics are nevertheless the useful quantities for answering the
ED question: is the current-channel error mainly changed by the improved `W`,
by the new `G`, or by neither?

## Independent post switches

The two expensive post paths are independently selectable.

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
finite-transfer SOX response.  This is the recommended first benchmark when the
goal is to isolate the effect of post-GW screening before studying its
combination with SOX.

## Plotting saved results

Every benchmark run writes figures automatically.  An existing `benchmark.npz`
can also be plotted without rerunning the many-body calculation:

```bash
python plot_ed_gw_cgw_sox_benchmark.py path/to/benchmark.npz
```

The plotting layer keeps the different questions separate:

- `benchmark_response_summary.png`: production ED/GG/cGW/cGW+SOX response;
- `benchmark_bubble_summary.png`: one-particle-background effect on chi;
- `benchmark_post_gw_response_*.png`: ordinary post-GW W-only and G+W response diagnostics;
- `benchmark_post_gw_sox_response_*.png`: independently selected SOX-post diagnostics;
- `benchmark_green_relative_error.png`: full Matsubara Green-function error;
- `benchmark_green_lowfreq_error.png`: low-frequency Green-function error.
