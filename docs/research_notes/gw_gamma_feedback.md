# Covariant GW-Gamma_P feedback

This note documents the screening-vertex feedback implemented in
`rubycgw/hedin_gamma.py` and the accelerated solver in
`rubycgw/hedin_gamma_fast.py`.

The current approximation is **GW-Gamma_P**, not a full Hedin `GWGamma`
closure: the covariant density vertex corrects the polarization/screening
sector, while the self-energy retains the GW form and does not yet contain an
explicit three-point Gamma factor.

## 1. Hedin and response conventions

RubycGW uses

\[
W=V+VPW,
\]

with irreducible polarization `P`, while the physical reducible density
susceptibility is defined with the convention

\[
\chi_{nn}=-(1-PV)^{-1}P.
\]

Therefore

\[
W=(1-VP)^{-1}V
\]

is exactly equivalent to

\[
W=V-V\chi_{nn}V.
\]

A covariant reducible susceptibility can be converted back to the corresponding
Hedin irreducible polarization without inverting `V`:

\[
P_{\Gamma}=-\chi_{nn}^{\rm cov}(1-V\chi_{nn}^{\rm cov})^{-1}.
\]

The implementation checks numerically that

\[
(1-VP_{\Gamma})^{-1}V
=
V-V\chi_{nn}^{\rm cov}V.
\]

## 2. Inner covariant density-vertex solve

For every orbital-density source

\[
K_b=|b\rangle\langle b|,
\]

the dressed vertex at fixed background `G,W` obeys

\[
\Gamma_b
=
K_b+\mathcal L_{GW}[\Gamma_b],
\]

or equivalently

\[
(1-\mathcal L_{GW})\Gamma_b=K_b.
\]

The GW covariant kernel contains the functional derivatives of the Hartree,
static Fock, and retarded GW self-energy pieces.  In the current decomposition
this gives

\[
\Gamma_b=K_b+
\Gamma_H+
\Gamma_F+
\Gamma_{MT}+
\Gamma_{AL1}+
\Gamma_{AL2}.
\]

The linear equation is solved by matrix-free GMRES.  With

\[
X_b(k;Q)=G(k+Q)\Gamma_b(k;Q)G(k),
\]

the full orbital density response is

\[
\chi_{ab}^{\rm cov}(Q)
=-\frac{T}{N_k}\sum_{kn}[X_b(k;Q)]_{aa}.
\]

## 3. Outer GW-Gamma_P self-consistency

Start from a converged ordinary self-consistent GW solution

\[
G_0,\quad W_0,\quad \Sigma_{H,0},\quad\Sigma_{GW,0},\quad\mu_0.
\]

At outer iteration `n`, the state is

\[
G_n,\quad W_n,\quad\Sigma_{H,n},\quad\Sigma_{GW,n},\quad\mu_n.
\]

The fixed-point map is:

1. Compute density and the Hartree target

\[
\Sigma_{H,n}^{*}=\Sigma_H[G_n].
\]

2. Compute the static bare-`V` Fock term

\[
\Sigma_{F,n}=\Sigma_F[G_n,V].
\]

3. At fixed `G_n,W_n`, solve all covariant density vertices

\[
(1-\mathcal L_n)\Gamma_{b,n}=K_b.
\]

4. Form the reducible density susceptibility

\[
\chi_n^{\rm cov}=-G_n\Gamma_nG_n.
\]

5. Convert to the Hedin irreducible polarization

\[
P_{\Gamma,n}
=-\chi_n^{\rm cov}
(1-V\chi_n^{\rm cov})^{-1}.
\]

6. Build the vertex-corrected screened interaction

\[
W_n^{*}
=(1-VP_{\Gamma,n})^{-1}V
=V-V\chi_n^{\rm cov}V.
\]

7. Build the GW-form self-energy target

\[
\Sigma_{GW,n}^{*}
=\Sigma_{F,n}-G_n(W_n^{*}-V).
\]

8. Monitor the raw outer residual

\[
r_n=\max\left(
\|\Sigma_{H,n}^{*}-\Sigma_{H,n}\|_\infty,
\|\Sigma_{GW,n}^{*}-\Sigma_{GW,n}\|_\infty,
\|W_n^{*}-W_n\|_\infty
\right).
\]

9. Mix or DIIS-extrapolate `Sigma_H`, `Sigma_GW`, and `W`, then solve Dyson

\[
G_{n+1}^{-1}=G_0^{-1}-\Sigma_{H,n+1}-\Sigma_{GW,n+1},
\]

while readjusting `mu` at fixed filling.

The loop is therefore

\[
\boxed{
G_n,W_n
\rightarrow
\Gamma_{\rho,n}
\rightarrow
\chi_n^{\rm cov}
\rightarrow
P_{\Gamma,n}
\rightarrow
W_n^{*}
\rightarrow
\Sigma_n^{*}
\rightarrow
G_{n+1}
}.
\]

For `m_max=0`, only the static bosonic sector is vertex-corrected; unsolved
nonzero Matsubara transfers retain the current background `W_n`.

## 4. Pulay / DIIS acceleration

The reference feedback solver uses linear mixing.  The fast solver treats the
three blocks

\[
X=(\Sigma_H,\Sigma_{GW},W)
\]

jointly.  Its residual is

\[
R=(\Sigma_H^*-\Sigma_H,
   \Sigma_{GW}^*-\Sigma_{GW},
   W^*-W).
\]

The DIIS metric normalizes the inner product of each block by its number of
elements before summing, so the much larger dynamic self-energy and screened
interaction arrays do not overwhelm the Hartree block only because of array
size.  Early iterations use ordinary linear mixing; after the Pulay history is
populated, the next iterate is extrapolated from previous fixed-point targets.
A step limiter and damping are applied to suppress large nonphysical DIIS
extrapolations when the vertex-corrected screening changes strongly.

## 5. Performance optimizations in the fast solver

`rubycgw/hedin_gamma_fast.py` and related fast modules add only numerical
optimizations; the fixed-point equations are unchanged.

- **Cross-outer-iteration Gamma warm starts:** every explicitly solved
  `(m,q,density-source)` vertex is cached and reused at the next outer
  iteration.  Since mixed `G,W` change smoothly, later GMRES solves usually
  start close to their new solution.
- **Prepared transfer FFT kernel:** for a fixed external transfer, all six
  density sources share the same MT/AL linear operator.  FFTs of background
  `G`, `W-V`, shifted `W`, and other Gamma-independent factors are computed once
  and reused for all six sources and all Krylov matrix-vector products.
- **No duplicate static Fock evaluation:** the already-computed Fock term is
  reused when forming the new GW self-energy target.
- **No redundant final BSE solve:** if an evaluated outer iterate satisfies the
  convergence tolerance, its existing covariant response is returned directly.
  If `max_iter` is reached, the last explicitly evaluated state is returned
  rather than making an unevaluated mixing step and then recomputing another
  full response.
- **Timing diagnostics:** every outer iteration reports wall-clock time split
  into vertex, screening algebra, self-energy, and fixed-filling Dyson pieces,
  together with total GMRES iterations and exact-transfer cache hits.

## 6. What is still missing from full GWGamma

The present closure modifies the polarization/screening side only:

\[
P\rightarrow P_\Gamma,
\qquad
W\rightarrow W_\Gamma,
\]

while the self-energy remains

\[
\Sigma=\Sigma_H+\Sigma_F-G(W_\Gamma-V).
\]

A full Hedin vertex-corrected closure would also require a consistent
three-point vertex in the self-energy,

\[
\Sigma\sim-GW\Gamma,
\]

with careful frequency/momentum routing and double-counting control.  Thus the
current solver should be interpreted as a controlled diagnostic of whether
vertex-corrected density screening is responsible for the ordinary-GW current
branch, not as the final conserving `GWGamma` approximation.
