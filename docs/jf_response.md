# Jacobian-free cluster response

## What the calculation measures

The Jacobian-free (JF) response differentiates the converged nonlinear cluster-ED+GW embedding without explicitly constructing the full many-variable Jacobian.  For a source coupled to one of the six local pseudospin channels

```text
Ax Ay Az Bx By Bz
```

the solver obtains the linear response at a chosen external momentum q and assembles the corresponding susceptibility matrix.

The leading eigenvalue/eigenvector identifies the softest **continuous fluctuation of the chosen background**.  It does not by itself compare nonlinear free energies of distinct ordered branches and cannot diagnose a first-order transition solely from the largest susceptibility eigenvalue.

## Current and charge-like channels

The x/y components are density/bond-order-like pseudospin channels.  The z component is loop current.  For A/B parity projections the user-facing convention is

```text
z_even = same physical circulation on A and B
z_odd  = opposite physical circulation on A and B
```

At finite q, the full six-channel matrix should be inspected rather than assuming a channel label in advance.

## Maintained CLI

For a converged baseline background:

```bash
python scripts/run_jf.py results/background.npz \
  --all-q \
  --bath-rank 0 \
  --bath-fd-step 2e-4 \
  --stage full \
  --no-rhs-recycle
```

Useful options include:

```text
--q-index i j            one commensurate q point
--q q1 q2                one reduced momentum
--all-q                  full Lx x Ly mesh
--bath-rank 0            keep all SVD-retained bath-tangent modes
--bath-fd-step 2e-4      production finite-difference scale used in current studies
--solver gcrotmk|gmres
--jf-tol 1e-8
--jf-maxiter 80
--krylov-m 24
--stage split-mt|full
--no-rhs-recycle         disable RHS recycling when soft modes poison recycled spaces
```

The driver automatically retries failed q points with a fresh larger Krylov space and can fall back from GCROT to GMRES.  Partial q-scan checkpoints are written so an interruption does not discard completed momenta.

## Bath tangent

The finite-bath fit is part of the self-consistent impurity map, so a consistent derivative must include how the fitted bath changes under the external source.  The bath tangent is built once and reused across q points.  Its SVD rank is a controlled approximation unless `--bath-rank 0` is used.

Important diagnostics are the retained rank, tangent condition number, finite-difference step dependence, and the residual of every linear solve.  A converged Krylov solve at one rank is not sufficient evidence that the bath-tangent approximation itself is converged.

## Momentum meshes

The allowed q points are those commensurate with the background `Lx x Ly` mesh.  For example:

- 2x2 contains Gamma and M-type points but not K;
- 3x3 contains Gamma and K/K' but not M;
- a larger mesh is required when both M and K competition must be represented in the same microscopic response calculation.

The q mesh belongs to the lattice embedding; it does not enlarge the six-site impurity.

## Background symmetry matters

A JF calculation differentiates the saved background actually supplied to it.  If that background already breaks C3 or another symmetry, the resulting susceptibility answers the question "what is soft around this broken branch?" rather than "which instability first leaves the symmetric phase?"

This distinction is particularly important for the six-site Ruby cluster because the primitive-cell cluster cut has an orientation bias.  See [Cluster-orientation ensemble](orientation_ensemble.md).

## Extended-model wrappers

The historical `Vprime`/`Vcross` JF wrappers remain under `research/` and patch the same production JF kernel with the extended interaction.  They are kept for exact reproducibility of existing calculations.  The current public façade `rubycgw.solvers.response` exposes the underlying solver objects, but a fully consolidated high-level extended-model JF workflow has not yet been declared stable.

## Validation

For important production points, validate JF against finite-source calculations when feasible and scan at least the bath finite-difference step and retained tangent rank.  The archived validation utilities are in `research/` and their detailed development notes are in `docs/research_notes/`.
