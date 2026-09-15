# Production primitive-cell GW and cGW

The six-site primitive-cell path is aligned with the mature 18-site supercell implementation.  The same split-GW background and H/F/MT/AL covariant derivative are now available both at q=0 and at finite external primitive momentum on the numerical k mesh.

## 1. Primitive SC-GW background

The production entry point is

```python
from rubycgw import solve_gw
```

which routes through `rubycgw.primitive_gw.solve_gw`.  The primitive Bloch Hamiltonian and interaction are the ordinary 6x6 `build_h0` and `build_interaction`; the numerical/self-energy kernel is shared with the arbitrary-matrix supercell solver.

For the instantaneous density interaction,

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

The equal-time density matrix entering `Sigma_F` is evaluated with analytic high-frequency tail subtraction.  Only the decaying `W-V` part is truncated by the finite bosonic Matsubara box.  This removes the artificial `nOmega` dependence of the non-decaying bare-V contribution present in the historical primitive implementation.

Fixed-filling calculations inherit the cached static tail, warm-started safeguarded Newton solve for `mu`, adaptive inner filling tolerance and strict final fixed-point verification from the production matrix solver.

`rebuild_primitive_fixed_point(result, params, grid)` performs one independent full split-GW map evaluation and reports separate Hartree/GW residuals.  A cGW response should not be interpreted unless this residual is small.

## 2. Primitive q=0 cGW

The q=0 production entry point is

```python
from rubycgw import VertexOptions, solve_vertex_q0
```

and differentiates the split SC-GW map,

\[
\Gamma
=K+\Gamma_H+\Gamma_F+\Gamma_{MT,c}+\Gamma_{AL1}+\Gamma_{AL2}.
\]

The decomposition is:

- `Gamma_H`: Hartree derivative;
- `Gamma_F`: derivative of the static bare-V Fock self-energy;
- `Gamma_MT`: MT derivative of the retarded `W-V` self-energy;
- `Gamma_AL1`, `Gamma_AL2`: derivatives of screened W, using full W because `delta(W-V)=delta W=W(delta P)W`.

The equation is linear in the full vertex,

\[
(I-L)\Gamma=K,
\]

and is solved by restarted matrix-free GMRES by default.  `solver="linear"` retains the older damped fixed-point iteration only for diagnostics/regression.

The modern call should pass the full `Vq` array.  For old scripts, `solve_vertex_q0` still accepts a 6x6 `Vq0`; it is broadcast over q.  This is exact for the current Ruby model because all implemented V bonds are intra-triangle/intra-cell and hence V(q) is q independent.

## 3. q=0 pseudospin/orbital channels

`rubycgw.pseudospin` supplies Pauli-normalized primitive vertices:

- `Ax, Ay, Az`, `Bx, By, Bz`;
- `x_even, x_odd`, `y_even, y_odd`;
- `z_same`, `z_opposite`.

The x/y components are TR-even intra-triangle E-type charge/orbital order; z is the TR-odd loop chirality.  z is Pauli-normalized, so a diagonal new `chi_zz` is one third of the historical eta-current susceptibility for the same physical current channel.

The q=0 command-line driver is

```bash
python run_ruby_cgw.py --V 1.0 --filling 2 --chi x_even,x_even
python run_ruby_cgw.py --V 1.0 --filling 2 --chi y_even,y_even
python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_same,z_same
python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_opposite,z_opposite
```

or

```bash
python run_ruby_cgw.py --V 1.0 --filling 2 \
  --channels x_even y_even z_same z_opposite
```

Response stages are `gg`, `split-mt`, and `full`.

## 4. Finite external q

The finite-q extension is implemented in

```python
from rubycgw import (
    FiniteQVertexOptions,
    solve_vertex_finite_q,
    susceptibility_matrix_finite_q,
)
```

with response propagator

\[
G_h(k;q)=G(k+q)\Gamma(k;q)G(k).
\]

The same H/F/MT/AL decomposition is retained.  The external q changes the polarization and screened-interaction tangent routing, while the SC-GW background itself remains the same translationally invariant six-site solution.

The implementation currently accepts any q on the same discrete reciprocal mesh as the background calculation,

\[
q=(i_1/n_{k1},i_2/n_{k2}).
\]

This is the controlled finite-q resolution of the numerical k mesh; no interpolation of the frequency-dependent interacting self-energy is used.

The dedicated scan driver is

```bash
python scan_primitive_cgw_q.py --V 1.0 --filling 2 \
  --nk1 6 --nk2 6 \
  --chi x_even,x_even --all-q
```

or, for Q=(1/3,1/3),

```bash
python scan_primitive_cgw_q.py --V 1.0 --filling 2 \
  --nk1 6 --nk2 6 \
  --chi x_even,x_even --q 0.3333333333333333 0.3333333333333333
```

The SC-GW background is solved only once and reused for every external q.  See `finite_q_cgw.md` for the complete finite-q AL derivation, momentum conventions, q/-q checks and recommended scanning workflow.
