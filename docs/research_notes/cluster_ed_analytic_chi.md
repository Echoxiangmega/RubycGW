# Analytic static cluster-ED+GW susceptibility on Hex18 PBC

This calculation is DIFFERENT from a post-processed GG bubble on an
ED-corrected Green function. It differentiates the **whole** finite-bath
embedding self-consistent map at the same zero-field fixed point used
by the finite-source reference.

## Reproducible three-way comparison

From the repository root, PowerShell:

```powershell
python research/compare_ed_cluster_chi_analytic.py `
  --V 0.2 --n 2 --T 0.125 `
  --Q 0.333333333333 0.333333333333 `
  --channels x_even y_even z_same z_opposite `
  --nbath 2 --nw 19 --nomega 5 `
  --ed-samples 6 --ed-tau-points 17 `
  --cluster-seed free `
  --cluster-max-iter 60 --baseline-max-iter 80 `
  --baseline-tol 3e-6 --source-tol 1e-6 `
  --source-step 0.001 --source-workers 2 `
  --analytic-tol 1e-5 --analytic-maxiter 12 `
  --out results/hex18_three_chi
```

The default bath has six orbitals. A 1-2 bath run is **only a fast
method check**; it cannot be used to infer quantitative strong-coupling
physical susceptibilities. Increase nbath and nw after the analytic
and finite-source methods agree at small sizes. For strong V you may
use `--cluster-seed density-gw`; it still verifies the convergence
of the independent embedding fixed point.

The script writes:
- `chi_comparison.csv`: ED, cluster finite-source, cluster analytic
  for every requested channel, ED relative errors, finite-source/
  analytic disagreement and time.
- `chi_comparison.png`: channel-wise three-method plot.
- `diagnostics.json`: convergence status and failures for every
  stage, including zero-field complex-bath and each source.
Partial results are clearly marked failed/unavailable: a failed
linear solver or a nonstationary bath fit is not given a numeric chi.

For noninteger total N, ED uses the existing selected-sector
grand-canonical finite-T reference; for integer N, canonical momentum
ED. Both cluster variants always fix the **mean total lattice N**.

## Why all three compare the same lattice and zero-field state

All use `build_torus_geometry("hex18")`: the same skew 18-site PBC
hopping and intra-triangle density V. Start from a one-impurity
translation-symmetric cluster-ED+GW solution. Then **relax once at
zero source using the complex-bath three-impurity solver**.

The final converged zero-field complex-bath solution is replicated
without re-solving for every ±h source and is also the expansion
point for the analytic tangent solver.

This is necessary because the historical uniform cluster embedding
fits a REAL bath while the existing finite-source source solver fits
a COMPLEX bath. Those are different finite-dimensional optimization
manifolds; comparing a derivative of one against a finite difference
of the other is not legitimate until they share a zero-field
reference fixed point.

## Analytic matrix-free fixed-point derivative

Let h be the full 18x18 one-body Hamiltonian, with source
h_source = h - eta K, and let S_H and S_emb denote the separately
stored lattice Hartree and remaining embedding self-energies.

    G = [iw + mu - h_source - S_H - S_emb]^-1.
    S_emb = S_GW_lattice + blockdiag(S_ED_C - S_GW_C).

For a real Hermitian static source, delta G is exactly

    dG = G (dh_source + dS_H + dS_emb - dmu I) G.

The chemical potential response dmu is chosen so that Tr(d rho)=0.
The one-body density response uses the same analytic static
reference/tail subtraction as the production code. For degenerate
static reference eigenenergies, use the derivative of the matrix
Fermi function through divided differences.

The GW variations are exact derivatives of the *actual discretized
implementation*, including:

    dP = P[dG,G] + P[G,dG].
    dW = W dP W.
    dSigma_GW = dSigma_F[rho] +
                Sigma_c[dG,W-V] + Sigma_c[G,dW].
    dSigma_H = diag(V d density).

For each six-site impurity c, the Weiss target satisfies

    Delta_star = (iw+mu)I - h_loc - Gc^-1 - Sigma_imp.
    dDelta_star = dmu I - dh_loc +
                  Gc^-1 (dGc) Gc^-1 - dSigma_imp.

The frequency-even high-frequency static Weiss extraction
`split_static_hybridization()` is linear, so the same operation
gives its exact static tangent dS and dynamic tangent dDelta.

## Implicit finite-bath fit and impurity ED

The finite bath is represented by real relative pole energies
xi_p=epsilon_p-mu and complex couplings v_ap. The fit minimizes
the actual low-frequency weighted residual

    F(theta) = 1/2 ||Delta_fit(theta)-Delta_target||_w^2,

where theta=(xi,Re v,Im v). In the interior of the parameter
bounds, differentiate the stationary condition J^T r = 0:

    H_theta dtheta = J^T dDelta_target_weighted.
    H_theta = J^T J + sum_i r_i Hessian(r_i).

The **second term is retained**; Gauss–Newton-only differentiation
does not generally reproduce the actual finite bath fit when
the fitting residual is appreciable. Bath gauge/phase zero-modes
are handled by a reported SVD pseudoinverse. An active bound, a
nonstationary base fit, or an inconsistent linearized equation
is NOT silently accepted as a valid analytic chi.

The finite-bath impurity Hamiltonian is ED-diagonalized only at
zero source. Its analytic Green derivative includes:
1. grand-canonical Boltzmann weight variations;
2. variations of the thermal source states through off-diagonal
   eigenstate perturbation theory;
3. two-resolvent insertions of the Hamiltonian perturbation in
   the opposite-particle-number sector.

Degenerate eigenspaces are rotated by the tangent one-body operator
before the directional derivative. In particular, complex
loop-current sources require a TRANSPOSED resolvent insertion for
the upper-source Lehmann contribution; this is unit-tested against
independent full impurity ED finite differences.

Then

    dSigma_ED = dG0_fit^-1 +
                Gimp^-1 (dGimp) Gimp^-1.

The full map's derivative is assembled in the GW+ED double-counting
formula, and the response is the solution of

    (I - dF/dSigma) dSigma = (dF/deta)|_Sigma,

with real matrix-free GMRES. The complete linearized physical map
has no Pulay mixing: mixing only locates the zero-field fixed point.

For Q=(1/3,1/3), the Hermitian cos/sin physical sources are solved
and added with EXACTLY the same normalization as the finite-source
reference. The source is static Omega=0; dynamic Omega != 0 would
require a full time-dependent Weiss/impurity four-point vertex, not
only static bath-parameter responses.

## Validation and scope limitations

Run on V=0 first, where the analytic result must reproduce the
noninteracting finite-source PBC response and the GW/ED derivatives
must vanish. Then use weak nonzero V and small bath to test the
full nontrivial linear map against the finite-source calculation.
Finally increase V, temperature resolution, nw and nbath.

A converged linear solver is not proof of physical bath convergence.
Monitor the absolute finite bath fit error, Gimp/Gc mismatch,
stationarity, the normal-state residual, linear residual and
stability under increasing bath orbitals.

The analytic tangent reuses one ED eigenproblem per inequivalent
cluster, but a matvec still requires expensive many-body resolvent
insertions. The algorithm is not guaranteed to be faster for
every small problem. On a large bath or close to a soft mode, a
linearized fixed point can be ill conditioned and the matrix-free
GMRES may need stronger preconditioning. Never replace a failed
analytic chi with a GG bubble and call it cluster-covariant.
