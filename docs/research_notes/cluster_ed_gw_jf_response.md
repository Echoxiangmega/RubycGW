# Jacobian-free pseudospin response of cluster ED+GW

## Purpose

The self-consistent cluster embedding uses

\[
\Sigma_{\rm emb}(k,i\omega)
=\Sigma_{\rm GW}^{\rm lat}(k,i\omega)
+\Sigma_{\rm imp}^{C}(i\omega)
-\Sigma_{\rm GW}^{C}(i\omega).
\]

A GW-only covariant vertex is therefore not the derivative of the approximation
that generated the embedded Green function.  The Jacobian-free (JF) response
implemented in `rubycgw/cluster_ed_gw_jf.py` differentiates the full embedded
self-energy, including the response of the finite-bath ED impurity correction.

The intended application is a strong-coupling competition among the local
pseudospin components of the triangle E doublet.  Rather than assuming a
current pattern in advance, the default response basis is

```text
Ax Ay Az Bx By Bz
```

with all components Pauli-normalized.  At every external momentum q the full
susceptibility matrix is diagonalized.  Its leading eigenvector therefore
selects tau_x/tau_y/tau_z character and the relative A/B phase simultaneously.
For tau_z, A/B-even is physical same circulation and A/B-odd is physical
opposite circulation.

## Linear equation

For a static source with external primitive momentum p,

\[
X(k;p)=G(k+p)\Gamma(k;p)G(k),
\qquad
\delta G_C(p)=\frac{1}{N_k}\sum_k X(k;p).
\]

The embedded vertex obeys

\[
\left[I-\mathcal L_{\rm GW}^{\rm lat}(p)
      -\mathcal L_{\rm imp}^{C}
      +\mathcal L_{\rm GW}^{C}\right]\Gamma_p=K_p.
\]

The lattice GW contribution uses the existing finite-q H/F/MT/AL kernel.  The
cluster-GW double-counting derivative is evaluated with the same GW functional
on the local six-site Green function.  The remaining difficult term is the ED
impurity derivative.

## Low-rank finite-bath impurity tangent

Writing the real bath parameter vector as

\[
\theta=(\epsilon_p,\operatorname{Re}V_{ap},\operatorname{Im}V_{ap}),
\]

the code constructs the analytic Jacobian of the fitted hybridization (or the
Weiss Green function) with respect to theta.  An SVD

\[
J_{\rm bath}=U S V^T
\]

identifies the bath directions that actually change the fitted object.  Only
the retained singular directions are differentiated through ED.  For each
retained mode r, a centered finite difference gives

\[
D_r(i\omega)=\frac{\partial\Sigma_{\rm imp}(i\omega)}{\partial a_r}.
\]

This is the expensive step, but it is done once.  Every subsequent Krylov
matrix-vector product uses only the precomputed D_r, FFT GW convolutions and a
small dense solve.

The impurity map is implicit because

\[
\mathcal G_{0,C}^{-1}=G_C^{-1}+\Sigma_{\rm imp}.
\]

Hence

\[
\delta\mathcal G_{0,C}^{-1}
=-G_C^{-1}\delta G_CG_C^{-1}+\delta\Sigma_{\rm imp}.
\]

Projecting this equation into the retained bath tangent space produces a small
matrix `(I-M)`.  It is formed and factorized once; its condition number is
saved as a diagnostic.

## Why GCROT(m,k)

There are six right-hand sides at the same q but the matrix-free operator is
identical.  `gcrotmk` recycles an approximate invariant subspace between these
right-hand sides.  This is generally preferable to restarting six unrelated
GMRES solves near a soft response mode.  The q mesh is also traversed in a
serpentine path and the previous-q vertex is used as the initial guess for the
next q.

The two acceleration mechanisms address different costs:

1. low-rank bath tangent removes ED diagonalization from every Krylov matvec;
2. Krylov recycling and q warm starts reduce the number of matvecs.

`gmres` remains available as a regression/fallback solver.

## Bath parameters: speed versus physical error

`nbath` is the dominant physical and computational bath parameter.  With six
correlated sites the impurity has `6 + nbath` orbitals, so the full Fock-space
size is

\[
2^{6+n_{\rm bath}}.
\]

For dense finite-temperature ED, increasing `nbath` therefore becomes expensive
very quickly.  A smaller bath can converge the outer embedding residual to a
very small number while still having a substantial discretization error.  The
outer residual and the bath discretization error must not be confused.

`bath_fit_nfreq` controls how much of the Matsubara Weiss field is represented
by the finite bath.  Too few points can miss frequency structure; too many can
make a small bath spend parameters on high-frequency details at the expense of
the low-energy response.  At strong coupling the repository also provides a
`g0` fitting metric, which measures the quantity propagated into the impurity,
rather than Delta itself.

`bath_energy_window` and `bath_coupling_bound` are optimization bounds.  A
result that hits either bound is not converged with respect to the bath
parameterization.  Making these bounds much larger than necessary can worsen
the conditioning of the fit and slow convergence.

`bath_fit_max_nfev` and the least-squares tolerances control optimizer effort.
Tightening them below the finite-bath representation error only increases
runtime.  `discard_weight_tol` controls the thermal-state truncation used when
forming the impurity Green function; it affects Green-function cost and error,
but does not remove the dominant dense diagonalization cost.

For the JF response there are three additional bath-tangent parameters:

- `bath_rank`: retained SVD rank.  This is the main speed/accuracy knob for the
  response.  A coarse q scan can use a smaller rank; the leading q points must
  then be repeated at larger rank.
- `bath_svd_rcond`: discards nearly null bath-fit directions.  Retaining very
  small singular directions can amplify bath-fit noise and make `(I-M)` ill
  conditioned.
- `bath_fd_step`: finite difference used only in the one-time ED tangent build.
  It must show a plateau when varied.  Too large gives nonlinear error; too
  small amplifies diagonalization and finite-bath numerical noise.

## Recommended convergence order

For a phase-selection calculation, do not tighten every parameter at once.
Use the following hierarchy.

1. Obtain a well-converged zero-field cluster-ED+GW background and record its
   outer residual, `Gimp/Gc` mismatch and bath-fit error.
2. At a few representative q points, converge `bath_rank` and `bath_fd_step`.
3. Tighten the JF Krylov tolerance until the leading eigenvalue/eigenvector are
   stable.  A residual much smaller than the bath-tangent error is unnecessary.
4. Scan the q mesh.  Use a coarse mesh first if the Brillouin-zone maximum is
   not known, then refine around the leading q.
5. Finally repeat the physically leading result versus `nbath` and
   `bath_fit_nfreq`.  This last step tests the finite-bath approximation itself.

A practical speed-first starting point is `nbath=6`, `bath_rank=12-24`,
centered `bath_fd_step=2e-4`, and GCROT recycling.  These are starting values,
not convergence claims.

## Running

First generate the converged embedding in the usual way.  Then run, for
example,

```bash
python scan_cluster_ed_gw_jf_q.py cluster_ed_gw_L6x6_V2_fill2.npz \
    --all-q \
    --channels Ax Ay Az Bx By Bz \
    --solver gcrotmk \
    --bath-rank 24 \
    --out results/jf_V2.npz
```

For a rank check at the winning momentum,

```bash
python scan_cluster_ed_gw_jf_q.py INPUT.npz --q-index IQ1 IQ2 --bath-rank 12
python scan_cluster_ed_gw_jf_q.py INPUT.npz --q-index IQ1 IQ2 --bath-rank 24
python scan_cluster_ed_gw_jf_q.py INPUT.npz --q-index IQ1 IQ2 --bath-rank 0
```

The output contains the raw and q/-q-Hermitianized response matrices, all
eigenvalues/eigenvectors, the leading-mode even/odd projections, per-channel
Krylov iterations/residuals, retained bath singular values and the condition
number of the local implicit bath response.

## Meaning of the “winning” channel

The largest eigenvalue of the Hermitian static susceptibility identifies the
softest *linear* mode of the symmetric embedded solution.  This is the correct
quantity for determining which continuous instability occurs first.  It does
not by itself prove the global thermodynamic winner if a first-order transition
preempts the linear instability.  In that case the JF result should be used to
seed the small set of candidate ordered branches, followed by a free-energy
comparison.
