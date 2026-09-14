# Attractive V' Ruby study

This directory keeps the V' extension isolated from the baseline `rubycgw`
model.  The launchers stay at repository root so the workflow is the same as the
existing cluster ED+GW/JF scripts.

## Model

The original interaction remains `V` on the six bonds forming the two
triangles.  We add

\[
H_{V'} = V' \sum_{\langle ij\rangle_{\rm inter}} n_i n_j,
\]

on **all six inter-triangle bonds that carry `t1`/`t2` hopping**.  `V' < 0` is
attractive.  Their primitive-cell offsets are kept in the lattice interaction,
so `V(q)` is momentum dependent.

The six-site impurity cannot retain an inter-cell bond displacement.  Its exact
ED interaction is therefore the primitive-cell / `q=0` projection of the same
interaction: all six inter-triangle bond types are represented between their
corresponding correlated orbitals.  The cluster-GW double counting uses the
same `V(q=0)` matrix.  The lattice GW and JF response still use the full
momentum-dependent interaction.  This is the deliberate cluster approximation
for the V' study.

## Minimal L2x2 comparison

First make a converged V' background, for example the previously interesting
small attraction `V'=-0.05`:

```bat
python run_cluster_ed_gw_vprime.py ^
    --Lx 2 --Ly 2 ^
    --V 1.2 --Vp -0.05 --filling 2 --T 0.08 ^
    --embed-mixing-method pulay --embed-pulay-history 8
```

Then compute the same full six-channel all-q JF response:

```bat
python scan_cluster_ed_gw_jf_q_vprime.py ^
    results\vprime_cluster_bg\cluster_ed_gw_vprime_L2x2_V1.2_Vp-0.05_fill2.npz ^
    --all-q --bath-rank 0 --bath-fd-step 2e-4 --stage full
```

For a small V' scan, repeat the background + response for e.g.
`V'=0,-0.01,-0.05,-0.10`, then compare the response files with

```bat
python analyze_vprime_jf.py results\vprime_response\*.npz ^
    --csv results\vprime_response\vprime_summary.csv
```

The summary follows the quantities that were useful in the V'=0 calculation:

- global leading susceptibility and its q/mode;
- uniform `z_same` and `z_opposite` at Gamma;
- maximum `z_same` and `z_opposite` over q;
- largest eigenvalue in the full `(Ax,Ay,Bx,By)` x/y subspace over q.

The final L2x2 step is only a qualitative comparison because it samples four q
points.  Once the trend with V' is clear, repeat the same workflow on an L6x6
background to resolve the ordering wavevector.
