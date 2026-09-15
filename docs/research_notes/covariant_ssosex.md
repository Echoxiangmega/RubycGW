# Static covariant screened SOSEX on the Ruby same-torus benchmark

This note defines the covariant response used by
`benchmark_ed_ssosex_covariant_post.py`.

## 1. Background closure

The no-Gamma screened-SOSEX background is the self-consistent approximation

\[
P[G]=GG,
\qquad
W[G]=(1-VP[G])^{-1}V,
\]

\[
\Sigma[G]
=
\Sigma_H[G]
+
\Sigma_F[G]
-
G\,(W[G]-V)
+
\Sigma_{\rm sSOSEX}[G,W[G]].
\]

For the default `oneW-sym` static screened-SOSEX functional,

\[
\Sigma_{\rm sSOSEX}
=
\frac12\{\mathcal S[V,W_0]+\mathcal S[W_0,V]\},
\qquad
W_0(q)=W(q,i\Omega=0),
\]

where

\[
\mathcal S[U,X]_{ij}(\tau)
=
\sum_{kl}
U_{il}X_{kj}
G_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau).
\]

`twoW` replaces this by \(\mathcal S[W_0,W_0]\).

## 2. Covariant derivative

Use a static source

\[
h_0(\phi)=h_0-\phi K.
\]

With the RubycGW convention define

\[
\Gamma_K=\frac{\delta G^{-1}}{\delta\phi},
\qquad
X_K=G\Gamma_KG=-\frac{\delta G}{\delta\phi}.
\]

Differentiating Dyson gives the static Bethe-Salpeter/covariant equation

\[
\Gamma_K
=K
+D\Sigma_H[X_K]
+D\Sigma_F[X_K]
+D\Sigma_c[X_K]
+D\Sigma_{\rm sSOSEX}[X_K].
\]

The GW correlation derivative contains the usual MT and AL paths.  For sSOSEX
there are two qualitatively different contributions.

### 2.1 Explicit Green-function derivative

At fixed interactions,

\[
D_G\mathcal S[U,X;Y]_{ij}
=
\sum_{kl}U_{il}X_{kj}\Big[
Y_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau)
+G_{ik}(\tau)Y_{kl}(-\tau)G_{lj}(\tau)
+G_{ik}(\tau)G_{kl}(-\tau)Y_{lj}(\tau)
\Big].
\]

Thus `oneW-sym` contains

\[
\frac12\left[
D_G\mathcal S[V,W_0;X_K]
+D_G\mathcal S[W_0,V;X_K]
\right].
\]

These are the screened analogues of the three bare-SOX derivative terms.

### 2.2 Screened-line derivative

Because \(W_0\) is not an external constant but a functional of \(G\), the
complete derivative also contains

\[
\frac12\left[
\mathcal S[V,DW_0[X_K]]
+
\mathcal S[DW_0[X_K],V]
\right].
\]

For ordinary GW screening,

\[
DW=W\,(DP)\,W,
\]

and, schematically,

\[
DP[X_K]=X_KG+GX_K,
\]

with the appropriate transfer routing.  Therefore the screened-line term is a
genuine extra vertex correction; it is absent in bare SOX and also absent if a
screened interaction is simply frozen while differentiating the explicit
three-G skeleton.

For `twoW`, the screened-line contribution is instead

\[
\mathcal S[DW_0,W_0]+\mathcal S[W_0,DW_0].
\]

## 3. Why the benchmark uses a finite-difference tangent

For a general finite external transfer, coding the full routing of
\(DW/\delta\phi\) inside screened SOSEX is substantially more complicated than
for bare SOX.  On the 2x1 same-torus benchmark all static primitive momenta are
represented as orbital structure inside one 12-orbital cell.  We can therefore
compute the *exact static tangent of the approximate closure* by central finite
differences:

\[
\chi_{AB}^{\rm sSOSEX}
=
\frac{\langle A\rangle_{+\epsilon B}
      -\langle A\rangle_{-\epsilon B}}
     {2\epsilon}
+O(\epsilon^2),
\]

where each \(+\epsilon\) and \(-\epsilon\) point is a fully converged
self-consistent GW+sSOSEX solution and the chemical potential is kept fixed at
the unperturbed value.

This numerical derivative automatically includes every derivative path listed
above:

\[
D\Sigma_H,
\quad D\Sigma_F,
\quad D\Sigma_{GW}\;(\mathrm{MT+AL}),
\quad D_G\Sigma_{\rm sSOSEX},
\quad
\frac{\partial\Sigma_{\rm sSOSEX}}{\partial W}\frac{dW}{dG}.
\]

It is therefore a useful correctness reference before implementing a faster
matrix-free analytic screened-SOSEX vertex.

The response is nonperturbative in the restricted sense that the complete
nonlinear fixed point is differentiated.  Equivalently, the associated BSE
resums the chosen kernel to all orders.  It does **not** mean that all exact
many-body diagrams at every order in bare \(V\) are present.

## 4. Current susceptibilities

For the normalized same/opposite loop-current operators,

\[
\chi_{JJ}
=
\left.\frac{d\langle K_J\rangle}{dh_J}\right|_{h_J=0},
\qquad
J\in\{z_{\rm same},z_{\rm opposite}\}.
\]

The benchmark compares these directly with the exact grand-canonical finite-T
ED susceptibility on the same 2x1 torus.

## 5. Static density response and post screened interaction

For each site/orbital projector \(P_b=|b\rangle\langle b|\),

\[
\chi^{nn}_{ab}(0)
=
\frac{n_a(+\epsilon P_b)-n_a(-\epsilon P_b)}{2\epsilon}.
\]

This is evaluated at fixed chemical potential.  The raw matrix is kept for a
reciprocity check and its Hermitian part is used in the post identity

\[
W_{\rm post}(i\Omega=0)
=
V-V\chi^{nn}_{\rm cov}(0)V.
\]

The first benchmark is deliberately static: all \(i\Omega\ne0\) sectors retain
the self-consistent background \(W\).

## 6. One-shot post Green function

Using the background Green function, recompute both pieces that depend on the
screened interaction:

\[
\Sigma_{GW,\rm post}
=
\Sigma_F[G_{bg},V]
-
G_{bg}(W_{\rm post}-V),
\]

\[
\Sigma_{\rm sSOSEX,post}
=
\Sigma_{\rm sSOSEX}[G_{bg},W_{\rm post}].
\]

The Hartree field is kept at the background value for this one-shot diagnostic.
Then

\[
G_{\rm post}
=
\left[
G_0^{-1}
-
\Sigma_H^{bg}
-
\Sigma_{GW,\rm post}
-
\Sigma_{\rm sSOSEX,post}
\right]^{-1},
\]

with the chemical potential readjusted only in this final Dyson step to recover
the target filling.  `Gerr` and the low-frequency `Gerr` are then compared with
exact ED on the same torus.

## 7. Interpretation

If the screened-SOSEX covariant current susceptibility moves toward ED while
`G_post` also lowers the full Green-function error, the missing screened-line
vertex feedback is quantitatively important.  If the current susceptibility
improves but `G_post` does not, that indicates observable-level cancellation or
remaining self-energy topology errors.  If neither improves, a more complete
channel resummation (dynamic G3W2, ladder/parquet, or a strong-coupling cluster
reference) is required rather than further static post dressing.
