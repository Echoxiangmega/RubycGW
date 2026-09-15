# Tail-consistent production cGW

## Why this exists

Production SC-GW treats the instantaneous bare interaction separately from the retarded screened part,

\[
\Sigma_{GW}=\Sigma_F+\Sigma_c,\qquad \Sigma_c=-G*(W-V),
\]

and evaluates the equal-time density matrix entering Hartree/static Fock with an analytic reference-Green-function Matsubara-tail subtraction,

\[
\rho=\rho_{ref}+T\sum_{n\in box}(G-G_{ref}),
\qquad
G_{ref}^{-1}=i\omega_n+\mu-h_0-\Sigma_H.
\]

The historical cGW H/F kernel instead differentiated only the represented finite fermion box through

\[
X=G\Gamma G.
\]

A direct fixed-\(\mu\) finite-source test on the n=3, V=1, T=0.08 same-torus benchmark showed an approximately 1.15% mismatch between production SC-GW and the historical static cGW.  Replacing production SC-GW by a diagnostic finite-box map reduced that mismatch to 1.2e-4 for `z_same` and 7.7e-7 for `z_opposite`, establishing that H/F/MT/AL diagram routing is structurally correct and that the remaining production mismatch is a tail-discretization inconsistency.

## Production-tail tangent

For external transfer \(Q=(q,i\Omega_m)\), define the positive cGW tangent

\[
X(k;Q)=G(k+Q)\Gamma(k;Q)G(k)=-\partial_h G.
\]

For a static one-body source \(S\), the missing reference tail is

\[
C_Q[S]
=
T\sum_{n=-\infty}^{\infty}G_{ref}(k+Q)S G_{ref}(k)
-
T\sum_{n\in box}G_{ref}(k+Q)S G_{ref}(k).
\]

The infinite reference bubble is evaluated analytically in the eigenbasis of \(h_0+\Sigma_H\), using

\[
\frac{f(\epsilon_{kb})-f(\epsilon_{k+q,a})}
{i\Omega_m+\epsilon_{kb}-\epsilon_{k+q,a}},
\]

with the usual derivative limit for a degenerate \(\Omega_m=0\) denominator.

Because `G_ref` contains `h0 + Sigma_H`, its tangent is driven by both the explicit source and the Hartree response.  Therefore

\[
R=T\sum_{box}X+C_Q[K+\Gamma_H].
\]

The Hartree part obeys a small orbital-space linear equation and is solved exactly.  The static Fock tangent is then obtained by applying the ordinary bare-V Fock convolution to `R`.

The resulting vertex still satisfies

\[
\Gamma=K+\Gamma_H+\Gamma_F+\Gamma_{MT,c}+\Gamma_{AL1}+\Gamma_{AL2}.
\]

Internally the source-dependent constant tail pieces are moved to the right-hand side so the large matrix-free GMRES problem remains linear.

## What is intentionally unchanged

MT and AL differentiate the explicitly represented retarded `W-V` part of the production self-energy.  They therefore keep the existing finite Matsubara convolution/routing.  No ad-hoc tail is added to MT/AL.

The final susceptibility contraction used in the finite-source FDT check also remains the represented finite-box contraction.  This is deliberate: the direct source observable used by `validate_cgw_finite_source.py` is the same finite Matsubara sum for traceless pseudospin/current operators.  Improving the observable's own asymptotic tail is a separate convergence task and must not be mixed with the functional-derivative consistency fix.

## New production-tail APIs

- `rubycgw.response_tail.build_tail_reference`
- `rubycgw.production_cgw.solve_vertex_q0_tail`
- `rubycgw.production_dynamic_cgw.solve_vertex_iomega_tail`
- `rubycgw.production_finite_q_cgw.solve_vertex_finite_q_tail`

The old finite-box solvers remain available as regression/diagnostic implementations.

## Drivers

Use the `_tail` drivers for production-tail response calculations:

```bat
validate_cgw_tail_fdt.py
run_supercell_cgw_tail.py
scan_primitive_cgw_q_tail.py
benchmark_ed18_cgw_tail.py
augment_ed18_cgw_full_tau_tail.py
decompose_ed18_cgw_stages_tail.py
diagnose_ed18_cgw_fixed_modes_tail.py
```

For an existing expensive ED benchmark, no ED or SC-GW checkpoint needs to be regenerated.  Reuse the old benchmark NPZ and checkpoint with the tail stage/dynamic wrappers.

## Validation hierarchy

1. `tests/test_response_tail.py` checks a direct production fixed-\(\mu\) finite-source derivative against static tail-cGW.
2. The dynamic tail solver at `m_ext=0` must reproduce the static tail solver.
3. The finite-q tail solver at `q=0` must reproduce the static tail solver.
4. `validate_cgw_tail_fdt.py` is the end-to-end benchmark-scale check.  It should reduce the earlier ~1% production mismatch to the finite-difference/solver scale.

The finite-box diagnostic from PR #20 remains useful for isolating discretization effects, but it is not a production replacement.
