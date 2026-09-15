# Numerics and validation

RubycGW contains nonlinear fixed-point solvers, finite Matsubara boxes, a finite-bath impurity representation, and matrix-free response calculations.  A small final residual at one numerical setting is not enough to establish a physical result.

## Background convergence

For cluster ED+GW record at least:

```text
final embedding residual
impurity residual / impurity mismatch
bath fit error
mu and density
number of outer iterations
mixing method and Pulay fallbacks
```

Inspect the histories when a point is close to a soft mode or when continuation changes branch.  A residual that decreases while the bath fit deteriorates can still indicate a limited impurity representation.

## Momentum and frequency convergence

The lattice calculation uses finite `Lx x Ly`, fermionic `nw`, and bosonic `nOmega` grids.  Production conclusions should be checked against increases of all three where computationally feasible.

The stored fermionic box is finite.  Frequency-shifted Green functions outside the stored box are not magically exact; tail handling and cutoff convergence are part of the numerical approximation.  Historical seam/tail diagnostics are preserved in `research/` and `docs/research_notes/`.

## Fixed filling

Check the total density independently of the apparent stability of `mu`.  At strong interaction, `mu` can move non-monotonically because Hartree and dynamic self-energy common shifts compensate each other.  Compare self-energy-corrected chemical-potential diagnostics when trying to identify a genuine low-energy reconstruction.

## Finite bath

For the default six-bath-orbital impurity, scan the bath representation when a conclusion is sensitive to low-frequency structure.  Useful variations include number of bath orbitals when computationally possible, fit-frequency window, bath fit bounds, and bath optimizer budget.

A finite bath can reproduce the fitted Weiss field well while still limiting derivatives.  JF response therefore needs an additional bath-tangent convergence analysis.

## JF response validation

For important soft-mode results:

1. require a converged background below the chosen acceptance threshold;
2. inspect the Krylov residual for every q/channel solve;
3. compare bath-tangent ranks, ideally including `--bath-rank 0`;
4. vary `--bath-fd-step` around the production value;
5. disable RHS recycling if a soft mode produces false stagnation;
6. compare with a direct finite-source derivative for representative channels when feasible.

The JF result is the derivative of the actual nonlinear approximation.  It is not made more physical by symmetrizing labels after the fact if the background itself is already symmetry broken.

## Effective ED

For the strong-coupling pseudospin model, finite-size effects are momentum-selective.  Compare at least 2x2 and 3x3 when M versus K competition matters.  A near-degenerate pair of finite-size levels should be treated as a ground manifold using a physically justified `degeneracy_tol` before computing structure factors.

## Symmetry diagnostics

Small C3-breaking densities from one six-site cluster orientation should be checked against the three-orientation gauge ensemble before being interpreted as intrinsic symmetry breaking.  The failed convergence of the old directly projected C3 solver is not itself a phase diagnostic.

## Reproducibility checklist

For a result intended for a figure or paper, archive the exact model/configuration, source checkpoint, software commit, convergence histories, and output NPZ.  For continuation scans, keep enough metadata to reconstruct the branch-following path.  For JF scans, retain the partial/final response file together with the exact background it differentiates.
