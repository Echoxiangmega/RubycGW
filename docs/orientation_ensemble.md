# Three-orientation cluster diagnostic

## Why the diagnostic exists

The physical Ruby lattice is C3 symmetric when the hopping and interaction parameters respect the corresponding three bond directions.  A six-site primitive-cell impurity, however, must choose which B triangle is assigned to the same primitive cell as a given A triangle.  That cluster cut internalizes one of three equivalent A/B neighboring directions.

Consequently, the lattice Hamiltonian can be exactly C3 symmetric while the **approximation defined by one oriented six-site impurity** has a preferred cluster direction.  At strong coupling a small density anisotropy can therefore reflect the cluster partition rather than an intrinsic microscopic C3-broken state.

## Why direct C3 projection is not the production method

A trial solver projected the lattice embedded self-energy back to a C3-symmetric subspace after every outer iteration while retaining one oriented impurity map.  In practice that map failed to close even at interaction strengths where the ordinary embedding converged.  The impurity Weiss problem and the projected lattice feedback were no longer the same fixed-point map.

That constrained solver is retained only as historical research code.  It should not be used as evidence for or against physical symmetry breaking simply because it fails to converge.

## Three gauge-related cluster orientations

The robust diagnostic instead runs the ordinary convergent embedding in three unit-cell gauges.  A is kept fixed while the B-triangle cell assignment is shifted by

```text
orientation 0: ( 0, 0)
orientation 1: ( 0, 1)
orientation 2: (-1, 0)
```

At the lattice level these are related by the same-k unitary transformation

\[
X_s(k)=D_s(k)X(k)D_s^\dagger(k),
\]

with a common phase applied to the three B orbitals.  The one-particle spectrum is unchanged at every k, and the q=0 six-orbital interaction projection is unchanged.  What changes is which inter-triangle bonds lie inside the finite six-site cluster.

The three intracell A/B bond pairs are:

```text
orientation 0: A1-B1, A2-B2
orientation 1: A0-B2, A1-B0
orientation 2: A0-B1, A2-B0
```

## Archived driver

The current implementation is preserved under

```text
research/run_cluster_ed_gw_vprime_vcross_orientation.py
research/analyze_cluster_orientation_ensemble.py
```

because the public high-level workflow has not yet absorbed the orientation ensemble.  These drivers remain part of reproducibility support, not the stable API.

## Interpretation

Suppose one orientation produces a pattern such as one site on each triangle having a slightly larger density.  If the special density direction rotates together with the cluster orientation and the three gauge-transformed results become equivalent, that is strong evidence that the anisotropy is dominated by the six-site cluster cut.

If, after transforming all solutions into a common physical frame, the same order direction remains preferred independently of cluster orientation, the case for genuine spontaneous C3 breaking is stronger.  Even then, a free-energy or more exact microscopic calculation is needed before identifying the true thermodynamic ground state.

## Response calculations

Do not simply average three nonlinear backgrounds and treat the average as a new impurity fixed point.  For response, the consistent procedure is conceptually

\[
\chi_{C3}(q)=\frac13\sum_{r=0}^{2}
\mathcal R_r^{-1}\chi^{(r)}(q)\mathcal R_r,
\]

where each `chi^(r)` is computed around its own converged orientation and then transformed to a common physical frame before averaging.

This orientation-resolved JF averaging is the preferred future public implementation because it removes cluster-cut bias without imposing a non-closing projected impurity map.
