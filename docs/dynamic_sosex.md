# Frequency-dependent SOSEX / full G3W2

The static `oneW-sym` approximation used previously replaces the screened line
by `W(q,Omega=0)`.  The dynamic implementation keeps the bosonic Matsubara
frequency dependence and now supports the full two-screened-line `W,W` G3W2
skeleton.

For the second-order screened-exchange topology, in the RubycGW orbital
convention,

\[
\Sigma_{ij}^{G3W2}(K)=
\left(\frac{T}{N_k}\right)^2
\sum_{Q,Q'}\sum_{kl}
W_{il}(Q)W_{kj}(Q')
G_{ik}(K+Q)G_{kl}(K+Q+Q')G_{lj}(K+Q').
\]

Write

\[
W(Q)=V(q)+W_p(Q),\qquad W_p(Q)=W(Q)-V(q).
\]

Then

\[
\boxed{
\Sigma^{G3W2}
=\Sigma^{VV}+\Sigma^{W_pV}+\Sigma^{VW_p}+\Sigma^{W_pW_p}.
}
\]

The first term is the tested bare SOX skeleton.  The project modes are

\[
\Sigma_{\rm sosex}
=\Sigma_{\rm SOX}
+\frac12\left(\Sigma^{W_pV}+\Sigma^{VW_p}\right),
\]

\[
\Sigma_{2\rm sosex}
=\Sigma_{\rm SOX}+\Sigma^{W_pV}+\Sigma^{VW_p},
\]

and

\[
\boxed{
\Sigma_{\rm g3w2}
=\Sigma_{\rm SOX}+\Sigma^{W_pV}+\Sigma^{VW_p}+\Sigma^{W_pW_p}.
}
\]

Thus `--dynamic-mode g3w2` is the full dynamic `W,W` calculation requested for
the Ruby benchmark.

## Mixed terms

The bare line is instantaneous and should not be truncated by the represented
bosonic box.  The mixed terms are therefore reduced to a single explicit
bosonic sum.  On the unfolded finite torus,

\[
\Sigma^{W_pV}_{ij}(i\omega_n)
=T\sum_m\sum_{kl}
W_{p,il}(i\Omega_m)
G_{ik}(i\omega_n+i\Omega_m)
B_{klj}(i\Omega_m)V_{kj},
\]

where

\[
B_{klj}(i\Omega_m)
=T\sum_r G_{kl}(i\omega_r+i\Omega_m)G_{lj}(i\omega_r),
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
=T\sum_r G_{ik}(i\omega_r)G_{kl}(i\omega_r+i\Omega_m).
\]

The pair sums are completed analytically with the static H+F reference.

## Double-screened WW term

The extra piece in full G3W2 is

\[
\boxed{
\Sigma^{W_pW_p}_{ij}(i\omega_n)
=T^2\sum_{m,m'}\sum_{kl}
W_{p,il}(i\Omega_m)W_{p,kj}(i\Omega_{m'})
G_{ik}(i\omega_n+i\Omega_m)
G_{kl}(i\omega_n+i\Omega_m+i\Omega_{m'})
G_{lj}(i\omega_n+i\Omega_{m'}).
}
\]

Both explicit bosonic sums are truncated at the represented `nOmega`, but only
`Wp=W-V` appears in them.  Since `Wp -> 0` at large bosonic frequency, this is
well behaved; the non-decaying bare `V,V` contribution remains in the separate
tail-completed SOX term.

The three shifted Green functions are continued outside the represented
fermionic box by the same static H+F reference used elsewhere in the SOX code.
For the 12-site 2x1 benchmark, the inner orbital contraction is evaluated by
batched matrix multiplication rather than a generic five-factor einsum.

## Self-consistent background

`rubycgw/gw_dynamic_sosex.py` solves

\[
P=GG,\qquad W=(1-VP)^{-1}V,
\]

\[
\Sigma_{\rm corr}
=\Sigma_F-G(W-V)+\Sigma_{\rm exchange}[G,W],
\]

plus Hartree, to a full nonlinear fixed point.  With
`--dynamic-mode g3w2`, `Sigma_exchange` is the full expression above, so both
screened interaction lines and the Green function evolve self-consistently.

The main benchmark command is

```text
python benchmark_ed_dynamic_sosex.py --V 1.0 --filling 2 --L1 2 --L2 1 --T 0.08 --nw 55 --nomega 12 --nquad 128 --dynamic-mode g3w2 --initial static --max-iter 1200 --mixing 0.08 --tol 2e-8
```

Because the WW term contains a double bosonic sum, it is substantially more
expensive than one-screened-line SOSEX.  `nOmega=12` is the first production
benchmark; repeat with 16 or 20 to check bosonic-cutoff convergence before
interpreting small differences in Gerr.

Reference for the G3W2 decomposition and fully dynamic frequency structure:
F. Bruneval and A. Forster, *J. Chem. Theory Comput.* **20**, 3218-3230 (2024),
DOI 10.1021/acs.jctc.4c00090.
