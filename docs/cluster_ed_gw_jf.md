# Jacobian-free finite-q response for cluster ED+GW

This note documents `rubycgw/cluster_ed_gw_jf.py` and
`scan_cluster_ed_gw_jf_pseudospin.py`.

## Scope

The zero-field embedding is

\[
\Sigma_{\rm tot}
=
\Sigma_H^{\rm lat}
+
\Sigma_{GW}^{\rm lat}
+
\Sigma_{\rm imp}
-
\Sigma_{GW}^{C}.
\]

The response code differentiates this *actual embedded approximation*.  The
lattice-GW and cluster-GW pieces use the existing finite-q cGW kernels.  The
finite-bath ED impurity self-energy is differentiated numerically through a
Jacobian-vector product.  No explicit analytic ED vertex is required.

At one external momentum \(q\),

\[
\delta G(k;q)
=
G(k+q)\,\Gamma(k;q)\,G(k),
\]

and the unknowns are the total lattice response vertex \(\Gamma\), the local
impurity self-energy tangent \(S_{\rm imp}\), and at \(q=0\) the fixed-filling
chemical-potential tangent \(d\mu\).  The matrix-free equations are

\[
\Gamma
-L_{GW,q}[\Gamma]
-S_{\rm imp}
+L_{GW,C}[\delta G_C]
+d\mu I
=K,
\]

\[
S_{\rm imp}
-D\Sigma_{\rm imp}[\mathcal G_{0,C}^{-1}]
[\delta \mathcal G_{0,C}^{-1}]
=0,
\]

with

\[
\delta G_C=\langle\delta G(k;q)\rangle_k,
\qquad
\delta\mathcal G_{0,C}^{-1}
=-G_C^{-1}\delta G_C G_C^{-1}+S_{\rm imp}.
\]

For \(q=0\), \(d\mu\) is fixed by \(\delta N=0\).  For \(q\ne0\), the
chemical-potential tangent is constrained to zero.

## Channel basis

The recommended discovery basis is

```text
Ax Ay Az Bx By Bz
```

rather than preselecting even/odd combinations.  These are the Pauli-normalized
pseudospin operators of the two Ruby triangles.  Thus the leading eigenvector
of the full 6x6 static susceptibility matrix simultaneously determines

- whether the soft direction is mainly \(\tau_x\), \(\tau_y\), or \(\tau_z\);
- the relative A/B amplitude and phase;
- whether the mode reduces to an even/odd pattern at a symmetry momentum.

The driver also prints the leading vector in the `x/y/z_even/odd` basis when
the default six local channels are used.

## q dependence

The external momentum is restricted to the same discrete reciprocal mesh as
the converged embedding checkpoint.  The driver scans all mesh points by
default.  Raw responses satisfy, in the exact calculation,

\[
\chi_{ab}(q)=\chi_{ba}(-q)^*.
\]

After all q points are available the driver therefore uses

\[
\chi_H(q)
=
\frac12\left[\chi(q)+\chi(-q)^\dagger\right]
\]

before diagonalization.

## Krylov acceleration

The large response equation is solved with `scipy.sparse.linalg.gcrotmk`, not
plain restarted GMRES.  This is useful here because six right-hand sides are
solved for the same Jacobian at fixed q.  The recycle space is retained across
`Ax...Bz` at the same q.  It is reset when q changes because the Jacobian itself
changes.  The converged solution of a channel at the previous q is nevertheless
used as the initial guess at the next q.

The q points are traversed in serpentine order so neighboring solves remain
close.

A cheap Neumann preconditioner can be enabled with

```text
--preconditioner-order 1
```

or `2`.  It approximately inverts the lattice-GW block,

\[
(I-L_{GW})^{-1}
\simeq
I+L_{GW}+L_{GW}^2+\cdots,
\]

while leaving the local ED correction to the outer GCROT solve.  Because the
ED Jacobian-vector product is normally much more expensive than one FFT-based
GW tangent, spending one or two additional GW applications per Krylov step can
still reduce total wall time.

## Bath derivative modes

### `linearized` (recommended for scans)

The fitted finite bath is

\[
\Delta_{ab}(i\omega)
=
\sum_p
\frac{V_{ap}V^*_{bp}}
{i\omega+\mu-\epsilon_p}.
\]

The code builds the real least-squares Jacobian with respect to

\[
\theta=(\epsilon_p,\Re V_{ap},\Im V_{ap})
\]

once at the converged zero-field bath.  A pseudoinverse gives the minimum-norm
bath tangent for each \(\delta\Delta\).  Small singular values, including bath
phase-gauge directions, are removed by `--bath-svd-rcond`.

This avoids a nonlinear least-squares optimization inside every Krylov
matvec.  Only the perturbed impurity ED solve remains expensive.

The approximation is a Gauss-Newton derivative of the bath fit.  It is most
reliable when the saved bath-fit residual is already small and no bath
parameter is sitting against a bound.

### `refit` (validation mode)

`--bath-derivative refit` reruns the nonlinear complex-bath fit at every
finite-difference perturbation.  It is considerably slower and can be noisier
because nearby optimizations can move between nearly equivalent bath gauges.
Use it on the final candidate q/channel to validate the faster linearized-bath
result, not for the first full mesh scan.

## Forward versus central impurity difference

`--difference-scheme forward` needs one perturbed ED solve per Jacobian-vector
product and reuses a cached zero-field impurity self-energy.  It is therefore
the recommended exploratory setting.

`--difference-scheme central` needs two ED solves per Jacobian-vector product,
but removes the leading finite-step error.  Use it to validate the winning mode
and to check `--fd-rel-step` convergence.

The Krylov tolerance should not be pushed far below the numerical noise of the
impurity directional derivative.  A useful sequence is

```text
2e-4 -> 5e-5 -> 2e-5 -> 1e-5
```

for `--fd-rel-step`, together with a response tolerance around `1e-5` initially.

## How `nbath` changes speed and error

For six correlated Ruby orbitals the impurity contains

\[
N_{\rm orb}=6+N_{\rm bath}
\]

spinless orbitals.  The full Fock dimension is \(2^{N_{\rm orb}}\), while the
largest fixed-particle sector is roughly

\[
\binom{N_{\rm orb}}{N_{\rm orb}/2}.
\]

With the current dense sector diagonalization this makes `nbath` the dominant
cost knob.  Representative largest-sector dimensions are

```text
nbath=4:  N_orb=10, C(10,5)=252
nbath=6:  N_orb=12, C(12,6)=924
nbath=8:  N_orb=14, C(14,7)=3432
nbath=10: N_orb=16, C(16,8)=12870
```

Dense diagonalization grows approximately cubically with sector dimension, so
moving from 6 to 8 bath orbitals can be tens of times more expensive in the
worst sector.  `nbath=10` is not realistic for repeated JF matvecs with the
current dense solver.

Increasing `nbath` generally lowers the bath-discretization error, especially
at low Matsubara frequency, but a smaller fit residual does **not** guarantee a
more accurate instability.  Convergence should be judged on

1. the leading susceptibility eigenvalue;
2. the overlap of the leading eigenvector between bath sizes;
3. the winning q;
4. the zero-field embedding Green function / impurity mismatch.

A practical production sequence is therefore

```text
nbath=4   coarse debugging only
nbath=6   main full-q scan
nbath=8   validate only the few candidate q points if affordable
```

Changing `nbath` requires rerunning the zero-field embedding first.  The JF
scanner intentionally does not replace the bath size of an existing checkpoint,
because doing so would linearize around a state that is no longer a fixed point.

## Other bath-fit controls

`bath_fit_nfreq`
: Increasing it grows the least-squares residual and Jacobian linearly in the
  number of fitted Matsubara frequencies.  This cost is small compared with ED.
  Values around 8-16 are sensible.  Too few points can overfit the first one or
  two frequencies; too many can spend bath freedom on high-frequency structure
  that is less relevant to the instability.

`bath_fit_xtol` and `bath_fit_max_nfev`
: These matter primarily for `refit`.  A very tight tolerance reduces Jv noise
  but increases optimization time.  For `linearized`, the nonlinear fit is not
  repeated inside Krylov.

`bath_svd_rcond`
: Controls which bath-fit tangent singular directions are kept.  Too small a
  value amplifies gauge/ill-conditioned directions and makes Krylov noisy.  Too
  large a value removes physical bath flexibility.  Start at `1e-9`; compare
  `1e-8`, `1e-9`, `1e-10` for the final mode.

`bath_tikhonov`
: Optional regularization of the bath tangent pseudoinverse.  Leave it at zero
  unless the reported bath Jacobian condition number is very large and the JF
  residual is erratic.

`discard_weight_tol`
: Loosening it can reduce finite-temperature Green-function work, especially at
  low T where only a small set of impurity eigenstates carries appreciable
  weight.  It must be converged on the leading eigenvalue/eigenvector.

## Typical run

```bash
python scan_cluster_ed_gw_jf_pseudospin.py \
  results/cluster_ed_gw/cluster_ed_gw_L4x4_V1.5_fill2.npz \
  --channels Ax Ay Az Bx By Bz \
  --bath-derivative linearized \
  --difference-scheme forward \
  --gcrot-m 20 --gcrot-k 8 \
  --preconditioner-order 1 \
  --jf-rtol 2e-5
```

For a final candidate q, rerun only that point with

```bash
python scan_cluster_ed_gw_jf_pseudospin.py CHECKPOINT.npz \
  --q-index IQ1 IQ2 \
  --bath-derivative refit \
  --difference-scheme central \
  --jf-rtol 5e-6
```

## Interpretation

The largest eigenvalue of the Hermitian static susceptibility gives the softest
linear mode of the *selected symmetric embedding branch*.  This is the correct
quantity for deciding which channel first becomes unstable in a continuous
transition.

It is not a proof of the globally stable ordered phase when the transition is
first order.  If two nonlinear branches cross before the symmetric-branch
Jacobian becomes singular, the true winner still requires a free-energy
comparison of the ordered solutions.
