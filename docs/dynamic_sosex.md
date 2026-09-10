# Frequency-dependent SOSEX / G3W2 decomposition

The static `oneW-sym` approximation used previously replaces the screened line
by `W(q,Omega=0)`.  The dynamic implementation keeps the complete bosonic
Matsubara dependence.

For the second-order screened-exchange topology, in the RubycGW orbital
convention,

\[
\Sigma_{ij}^{G3W2}(K)=
\left(\frac{T}{N_k}\right)^2
\sum_{Q,Q'}\sum_{kl}
W_{il}(Q)W_{kj}(Q')
G_{ik}(K+Q)G_{kl}(K+Q+Q')G_{lj}(K+Q').
\]

This is the Matsubara counterpart of the standard real-time `GWGWG` / `G3W2`
second-order self-energy.  The overall sign here follows the existing RubycGW
SOX convention, so the bare limit reproduces the tested bare-SOX code.

Write

\[
W(Q)=V(q)+W_p(Q),\qquad W_p(Q)=W(Q)-V(q).
\]

Then

\[
\Sigma^{G3W2}
=\Sigma^{VV}+\Sigma^{W_pV}+\Sigma^{VW_p}+\Sigma^{W_pW_p}.
\]

The first term is bare SOX.  Standard dynamic SOSEX contains one dynamically
screened line and one bare line.  Since the two mixed line orderings are equal
in the exact infinite-frequency/translationally invariant expression, the
production implementation uses their symmetric average,

\[
\boxed{
\Sigma_{\rm dSOSEX}
=\Sigma_{\rm SOX}
+\frac12\left(\Sigma^{W_pV}+\Sigma^{VW_p}\right).
}
\]

This is the direct dynamic counterpart of the project's static `oneW-sym`
choice.  `mode=2sosex` retains both mixed contributions,

\[
\Sigma_{2\rm SOSEX}
=\Sigma_{\rm SOX}+\Sigma^{W_pV}+\Sigma^{VW_p}.
\]

The genuinely double-dynamic term \(\Sigma^{W_pW_p}\) is **not** part of this
first implementation; adding it gives full dynamic G3W2 and requires a double
bosonic-frequency sum.

## Single-frequency form used in code

The bare line is instantaneous and should not be truncated by the represented
bosonic box.  Therefore the code analytically collapses the frequency carried
by the bare line.  After unfolding momentum to the finite real-space torus,

\[
\Sigma^{W_pV}_{ij}(i\omega_n)
=T\sum_m\sum_{kl}
W_{p,il}(i\Omega_m)
G_{ik}(i\omega_n+i\Omega_m)
B_{klj}(i\Omega_m)V_{kj},
\]

with

\[
B_{klj}(i\Omega_m)
=T\sum_r
G_{kl}(i\omega_r+i\Omega_m)G_{lj}(i\omega_r),
\]

and

\[
\Sigma^{VW_p}_{ij}(i\omega_n)
=T\sum_m\sum_{kl}
V_{il}C_{ikl}(i\Omega_m)
W_{p,kj}(i\Omega_m)
G_{lj}(i\omega_n+i\Omega_m),
\]

\[
C_{ikl}(i\Omega_m)
=T\sum_r
G_{ik}(i\omega_r)G_{kl}(i\omega_r+i\Omega_m).
\]

Thus only \(W_p=W-V\), which decays at large \(|\Omega|\), sees the explicit
`nOmega` cutoff.  The `(V,V)` term is evaluated by the existing tail-completed
SOX quadrature.

## Tail completion of B and C

Let the static H+F reference have eigenvalues \(\xi_a=\epsilon_a-\mu\) and
Fermi occupations \(f_a\).  The exact reference pair sum is

\[
T\sum_n
\frac{1}{i\omega_n+i\Omega_m-\xi_a}
\frac{1}{i\omega_n-\xi_b}
=
\frac{f_a-f_b}{\xi_a-\xi_b-i\Omega_m},
\]

with the removable `m=0`, degenerate limit \(f'(\xi_a)\).  The code adds the
finite-box difference between the interacting and reference pair products to
this exact reference result.  Shifted Green functions outside the represented
fermionic box are continued by the same static reference.

## Self-consistent background

`rubycgw/gw_dynamic_sosex.py` solves

\[
P=GG,\qquad W=(1-VP)^{-1}V,
\]

\[
\Sigma_{\rm corr}
=\Sigma_F-G(W-V)+\Sigma_{\rm dSOSEX}[G,W],
\]

plus the usual Hartree term, to a full nonlinear fixed point.  Therefore the
frequency dependence of `W` changes together with `G`; it is not a one-shot
correction.

The first intended benchmark is `benchmark_ed_dynamic_sosex.py`, which compares
ordinary GW, static oneW-sym, and dynamic SOSEX against exact ED on the same
2x1 Ruby torus.  Only after checking that the dynamic background actually
reduces the Green-function error should the substantially more complicated
covariant derivative of the dynamic SOSEX functional be implemented.

Reference for the G3W2 decomposition and fully dynamic frequency structure:
F. Bruneval and A. Forster, *J. Chem. Theory Comput.* **20**, 3218-3230 (2024),
DOI 10.1021/acs.jctc.4c00090.
