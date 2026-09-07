# Production primitive-cell GW and q=0 cGW

The six-site primitive-cell path is now intentionally aligned with the mature
18-site supercell implementation.  This document records the production
conventions so that future finite-q work starts from the same background and
vertex equations rather than from the older prototype formulas.

## 1. Primitive SC-GW background

The production entry point is

```python
from rubycgw import solve_gw
```

which routes through `rubycgw.primitive_gw.solve_gw`.  The primitive Bloch
Hamiltonian and interaction are still the ordinary 6x6 `build_h0` and
`build_interaction`; only the numerical/self-energy kernel is shared with the
arbitrary-matrix supercell solver.

For the instantaneous density interaction we use

\[
\Sigma_{GW}=\Sigma_F+\Sigma_c,
\]

with

\[
\Sigma_F(k)=-\frac{1}{N_k}\sum_q \rho(k+q)\circ V(q)^T,
\]

and

\[
\Sigma_c(k,i\omega_n)
=-\frac{T}{N_k}\sum_{q,m}
G(k+q,i\omega_n+i\Omega_m)\circ[W(q,i\Omega_m)-V(q)]^T.
\]

The equal-time density matrix entering `Sigma_F` is evaluated with analytic
high-frequency tail subtraction.  Only the decaying `W-V` part is truncated by
the finite bosonic Matsubara box.  This removes the artificial `nOmega`
dependence of the non-decaying bare-V contribution present in the historical
primitive implementation.

Fixed-filling calculations inherit the cached static tail, warm-started
safeguarded Newton solve for `mu`, adaptive inner filling tolerance and strict
final fixed-point verification from the production matrix solver.

`rebuild_primitive_fixed_point(result, params, grid)` performs one independent
full split-GW map evaluation and reports separate Hartree/GW residuals.  A cGW
response should not be interpreted unless this residual is small.

## 2. Primitive q=0 cGW

The production response entry point remains

```python
from rubycgw import VertexOptions, solve_vertex_q0
```

but `rubycgw.cgw` is now a compatibility facade over the same arbitrary-matrix
kernel used by the supercell response.  It differentiates the split SC-GW map:

\[
\Gamma
=K+\Gamma_H+\Gamma_F+\Gamma_{MT,c}+\Gamma_{AL1}+\Gamma_{AL2}.
\]

The important decomposition is:

- `Gamma_H`: Hartree derivative;
- `Gamma_F`: derivative of the static bare-V Fock self-energy;
- `Gamma_MT`: MT derivative of the retarded `W-V` self-energy;
- `Gamma_AL1`, `Gamma_AL2`: derivatives of screened W.  These contain the full
  W because `delta(W-V)=delta W=W(delta P)W`.

The equation is linear in the full vertex.  The default solver therefore treats

\[
(I-L)\Gamma=K
\]

with restarted matrix-free GMRES.  `solver="linear"` retains the older damped
fixed-point iteration only for diagnostics/regression.

The modern call should pass the full `Vq` array.  For old scripts,
`solve_vertex_q0` still accepts a 6x6 `Vq0`; it is broadcast over q.  This is
exact for the current Ruby model because the implemented V bonds are all
intra-triangle/intra-cell and hence V(q) is q independent.

## 3. q=0 pseudospin/orbital channels

`rubycgw.pseudospin` supplies Pauli-normalized primitive vertices:

- `Ax, Ay, Az`, `Bx, By, Bz`;
- `x_even, x_odd`, `y_even, y_odd`;
- `z_same`, `z_opposite` (aliases of physical-frame z even/odd).

The x/y components are TR-even intra-triangle E-type charge/orbital order;
z is the TR-odd loop chirality.  z is Pauli-normalized, so a diagonal new
`chi_zz` is one third of the historical eta-current susceptibility for the same
physical current channel.

The command-line driver is now unified:

```bash
python run_ruby_cgw.py --V 1.0 --filling 2 --chi x_even,x_even
python run_ruby_cgw.py --V 1.0 --filling 2 --chi y_even,y_even
python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_same,z_same
python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_opposite,z_opposite
```

or solve a full q=0 response matrix:

```bash
python run_ruby_cgw.py --V 1.0 --filling 2 \
  --channels x_even y_even z_same z_opposite
```

Response stages are:

- `--stage gg`: dressed bubble only;
- `--stage split-mt`: H + F + MT(W-V);
- `--stage full`: add AL1 + AL2.

The driver first verifies that the SC-GW background is a fixed point of the
same split map used in the derivative.

## 4. What is intentionally deferred

This modernization is still external q=0.  Arbitrary finite-q cGW is a separate
extension because the response propagator becomes

\[
G_h(k;q)=G(k+q)\Gamma(k;q)G(k),
\]

and requires corresponding momentum bookkeeping throughout H/F/MT/AL.  The
finite-q extension should be built on the production primitive background and
vertex decomposition documented here, not on the historical primitive
prototype.
