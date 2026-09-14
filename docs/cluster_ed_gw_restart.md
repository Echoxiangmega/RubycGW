# Restarting a cluster-ED+GW embedding

Both `run_cluster_ed_gw.py` and `run_cluster_ed_gw_accel.py` support restarting
from a previously saved cluster-ED+GW `.npz` result through

```bash
--restart-from PATH_TO_OLD_RESULT.npz
```

A restart restores the expensive dynamic embedding state

- lattice Green function `G`;
- Hartree self-energy `Sigma_H`;
- embedded dynamic self-energy `Sigma_emb`;
- impurity/ED cluster self-energy `Sigma_ED_cluster`;
- chemical potential `mu`;
- finite-bath energies and couplings.

The Pulay/DIIS history is intentionally **not** restored.  A restart therefore
continues from the old physical fixed-point iterate but builds a fresh Pulay
history using the current solver implementation.  This is important when the
mixing algorithm or regularization has changed between runs.

The restart loader requires the same physical problem and representation:
`Lx`, `Ly`, `V`, `ti`, `t1`, `t2`, filling, temperature, Matsubara grids and
`nbath` must match.  Convergence-control parameters such as `--embed-tol`,
`--embed-max`, `--embed-mixing`, Pulay settings and bath optimizer effort may be
changed.

For example, to continue an older calculation with the scale-invariant Pulay
launcher and a tighter target tolerance,

```bash
python run_cluster_ed_gw_accel.py \
    --Lx 6 --Ly 6 --V 2.0 --filling 2 --T 0.08 \
    --nbath 6 \
    --restart-from results/cluster_ed_gw/cluster_ed_gw_L6x6_V2_fill2.npz \
    --embed-max 100 \
    --embed-tol 2e-5
```

The saved output records the restart source path.  The new run starts its
iteration counter at one; this counter refers only to the continuation segment,
not the cumulative number of outer iterations across all runs.
