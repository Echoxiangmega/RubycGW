# Audit: triangle3 vs cell6 hex18 CO fixed points

This note records architecture checks of the two SEET-like ED+GW partitions,
not a determination of the thermodynamic CO phase boundary.

## Important: compare the same filling first

A triangle3 log at `n=1, V=0..5` and a cell6 log at `n=1.3, V=.5`
**cannot** establish a partition disagreement: in the three-cell torus their
global average particle numbers are 3 and 3.9, respectively.

At fixed `n=1.3`, the two solvers must also use identical
`T,t,t1,t2,K,nw,nomega,modes`, and identical seed families.
Different bath counts imply different fitting errors and should be scanned
separately.

## Exact architectural checks

The regression `tests/test_triangle3_partition_architecture_audit.py`
verifies:

1. `triangle3` consists of six **independent** ED impurities with site
   blocks `[0:3],[3:6],...,[15:18]`, i.e. A0/B0/A1/B1/A2/B2.
   Each has exactly the three intra-triangle physical interaction links.
   Summing the six cluster interaction matrices reproduces the full folded
   physical interaction matrix.
2. The full 18x18 `h0` retains all intertriangle `t1,t2` hoppings.
   At V=0, the triangle Weiss field constructed from the projected
   lattice Green matches the exact Schur complement of the remaining
   15 physical orbitals at *every Matsubara frequency*.
3. For a deliberately non-C3 and nontranslation-symmetric density matrix,
   the sum over six local Hartree and Fock contributions is identical
   to the full-lattice Hartree/Fock from the same density interaction.
   There is no first-order loss/double count of local V bonds.
4. The `co_qK_A0` self-energy seed at `V=.5, n=1.3` acts on
   independent three-site blocks and induces a nonzero finite-Q CO
   order parameter in the full lattice after the initial seeded Dyson
   calculation. No symmetry constraint projects out CO at initialization.

This does **not** prove that the fully converged triangle3 ED+GW fixed
point has CO: an initial finite-Q source is immediately removed, and the
nonlinear zero-source solver may return to a symmetric stationary point.

## Where triangle3 and cell6 genuinely differ

- Triangle3: A/B intertriangle propagation enters the complete lattice
  Green and the triangle Weiss baths. Correlation corrections crossing the
  triangle boundaries remain in low-level full-lattice GW.
- Cell6: a six-orbital impurity includes two coupled triangles, so ED
  can nonperturbatively reproduce some additional intertriangle
  self-energy/vertex diagrams. Its local subtraction is correspondingly
  the GW calculation on the full six-orbital cluster.
- The full lattice Hamiltonian and global chemical potential constraint
  are identical; each candidate is free to develop CO, LC and finite Q.

The default finite bath sizes also differ:
`triangle3+3` fits one three-orbital Weiss function with only
three bath poles, while `cell6+6` fits a six-orbital Weiss with
six bath poles. The environments and fitting problems are different.
A triangle3 bath relative error around 0.3 is an actual quantitative
limitation; do not interpret it as proof of symmetry preservation.

There is also a **numerical symmetry question** on the cell6 side.
The branch named `symmetric` has *no enforced symmetry constraint*;
it is merely initialized with zero order seed. A density-GW iteration
with independent nonlinear bath fits can develop CO either because
the symmetric fixed point is unstable or because bath-fit differences
act as small symmetry-breaking perturbations. Check its CO amplitude,
Gimp/Gloc mismatch, bath-fit quality, independent initializations,
and free-energy reliability before asserting a stable CO phase.
The `phase_family=CO` label only checks an order threshold.

## Small, controlled, same-point experiment

First compare only `n=1.3`, `V=0,.5`; select the minimal
`symmetric,co_q0_A0,co_qK_A0` branches.

```powershell
git pull

python research/scan_hex18_triangle_ed_gw_vn.py `
  --V-scan 0 0.5 --fillings 1.3 `
  --T 0.125 --t 1 --t1 -0.5 --t2 0.5 `
  --K 0.333333333333 0.333333333333 `
  --modes density `
  --candidates symmetric co_q0_A0 co_qK_A0 `
  --seed-amplitudes 0.08 0.3 `
  --nw 19 --nomega 10 --nbath 3 `
  --max-iter 160 --mixing-method pulay `
  --bootstrap free `
  --out results/triangle3_n13_same_point

python research/scan_hex18_cluster_order_vn.py `
  --cluster-partition cell6 `
  --V-scan 0 0.5 --fillings 1.3 `
  --T 0.125 --t 1 --t1 -0.5 --t2 0.5 `
  --K 0.333333333333 0.333333333333 `
  --modes density `
  --candidates symmetric co_q0_A0 co_qK_A0 `
  --seed-amplitudes 0.08 0.3 `
  --nw 19 --nomega 10 --nbath 6 `
  --max-iter 160 --mixing-method pulay `
  --bootstrap free `
  --out results/cell6_n13_same_point
```

For each branch compare `co_amplitude`, `dominant_channel`,
`density_range`, `G_K_KplusQ_norm`, `g_error`,
`bath_fit_max`, `gimp_gc_max` and `F_per_pc`, together with the
`free_energy_reliable` flag. Next repeat triangle3 with
`--nbath 6` in a fresh directory to separate finite-bath accuracy
from cluster-size error. Since the branch-specific seed is applied
only once, include larger seed amplitude or a source-ramp analysis
before concluding that a nontrivial fixed point does not exist.

A finite-temperature 18-site ED Green reference does not by itself
assign an ordered phase. The ED reference must be converged in
thermal eigenstate count and particle-number sector coverage, and
charge/current structure factors should be checked independently.
