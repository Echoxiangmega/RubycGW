# Screened SOSEX in the Ruby-lattice GWΓP workflow

## 1. Why go beyond bare SOX?

The bare crossed-exchange skeleton used in the project is

\[
\Sigma^{\rm SOX}_{ij}(\tau)
=\sum_{kl}V_{il}V_{kj}
G_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau).
\]

It is the lowest non-trivial exchange/self-energy vertex topology and starts at
\(O(V^2)\).  At \(V\sim1\), however, a single bare \(V^2\) diagram need not
capture the higher-order exchange diagrams generated when an interaction line
is screened.

Define the bilinear crossed skeleton

\[
\mathcal S[U,X]_{ij}(\tau)
=\sum_{kl}U_{il}X_{kj}
G_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau).
\]

Then bare SOX is simply

\[
\Sigma^{\rm SOX}=\mathcal S[V,V].
\]

## 2. Screened interaction used here

The Gamma_P loop supplies the full reducible density response and the screened
interaction

\[
W_\Gamma(Q)=V(Q)-V(Q)\chi_{nn}^{\rm cov}(Q)V(Q).
\]

For the first affordable screened-exchange diagnostic we take its static slice

\[
W_0(q)\equiv W_\Gamma(q,\Omega=0).
\]

This is a static-screening approximation, not the fully dynamic G3W2
self-energy.

## 3. Default: symmetrized one-screened-line SOSEX

The production diagnostic uses

\[
\boxed{
\Sigma^{\rm sSOSEX}_{1W}
=\frac12\left[\mathcal S[V,W_0]+\mathcal S[W_0,V]\right].
}
\]

The symmetrization avoids privileging either of the two crossed interaction
lines.  Most importantly,

\[
W_0\to V
\quad\Longrightarrow\quad
\Sigma^{\rm sSOSEX}_{1W}\to\Sigma^{\rm SOX}
\]

exactly.

Using the formal static screening expansion

\[
W_0=V+VP_0V+VP_0VP_0V+\cdots,
\]

we obtain

\[
\Sigma^{\rm sSOSEX}_{1W}
=\Sigma^{(2)}_{\rm SOX}
+\frac12\{\mathcal S[V,VP_0V]+\mathcal S[VP_0V,V]\}
+O(V^4).
\]

Thus the bare \(O(V^2)\) crossed topology is retained while bubble-screened
higher-order crossed-exchange diagrams are resummed on either interaction line.

## 4. Optional: two-screened-line static G3W2/SOSEX

The code also supports

\[
\boxed{
\Sigma^{\rm sSOSEX}_{2W}=\mathcal S[W_0,W_0].
}
\]

This is the static two-screened-line version.  It also reduces to bare SOX when
\(W_0=V\), but is more aggressive and more expensive because both interaction
neighbor tables become dense in the finite-torus real-space representation.

The CLI option is

```text
--ssosex-mode twoW
```

while the recommended first test is

```text
--ssosex-mode oneW-sym
```

## 5. Relation to fully dynamic screened exchange

A complete second-order term in the screened interaction has the schematic
three-G/two-W structure

\[
\Sigma^{G3W2}(1,2)
=-\int d3\,d4\,
G(1,3)W(1,4)G(3,4)G(4,2)W(3,2),
\]

up to the project's sign convention.  Because both \(W\) lines are dynamic,
there are two independent internal times/frequencies.  This cannot be reduced
to the single-\(\tau\) product used by the existing bare-SOX implementation.
The present static screened SOSEX is therefore an intermediate diagnostic,
chosen because it adds a controlled class of higher-order screened-exchange
terms without introducing the double-frequency cost of full dynamic G3W2.

## 6. Self-consistent closure used in the code

At outer iteration \(n\), the screened-SOSEX solver performs

\[
G_n,W_n
\rightarrow \Gamma_{\rho,n}
\rightarrow \chi_n^{\rm cov}
\rightarrow W_n^*,
\]

with

\[
W_n^*=V-V\chi_n^{\rm cov}V.
\]

The GW target is

\[
\Sigma_{GW,n}^*
=\Sigma_F[G_n]-G_n(W_n^*-V),
\]

and the screened crossed-exchange target is

\[
\Sigma_{{\rm sSOSEX},n}^*
=\frac12\left[
\mathcal S[V,W_n^*(0);G_n]
+\mathcal S[W_n^*(0),V;G_n]
\right]
\]

for the default mode.  The total non-Hartree target is

\[
\boxed{
\Sigma_{{\rm corr},n}^*
=\Sigma_{GW,n}^*+\Sigma_{{\rm sSOSEX},n}^*.
}
\]

The joint \((\Sigma_H,\Sigma_{\rm corr},W)\) state is mixed with the existing
outer Pulay/DIIS machinery and Dyson is solved again at fixed filling.

## 7. What is not yet included

The density-vertex kernel remains the GW H/F/MT/AL kernel.  We do **not** yet
include

\[
\frac{\delta\Sigma_{\rm sSOSEX}}{\delta G},
\]

because this derivative contains both derivatives of the three internal Green
functions and the derivative of the screened line \(W_0[G]\).  Adding only the
former would not be a consistent derivative of the screened-SOSEX functional.

Thus the present implementation should be interpreted as a self-energy-side
screened-exchange diagnostic on top of the Gamma_P screening feedback, not a
full conserving Hedin closure.

## 8. Files

- `rubycgw/ssosex_static.py`: static screened-SOSEX kernel.
- `rubycgw/hedin_gamma_ssosex_fast.py`: self-consistent Gamma_P+sSOSEX solver.
- `scan_primitive_finite_source_gw_gamma_ssosex_fast.py`: ED/GW/Gamma/bare-SOX/sSOSEX comparison.
- `tests/test_ssosex_static.py`: bare-limit and bilinear-scaling regression tests.
