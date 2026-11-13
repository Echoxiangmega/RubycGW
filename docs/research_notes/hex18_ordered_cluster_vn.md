# Periodic hex18 ordered-state cluster ED+GW: V/filling scan

`research/scan_hex18_cluster_order_vn.py` searches zero-source ordered
solutions for the **18-site periodic skew Ruby torus** (3 primitive cells).
This is not a Hartree–Fock approximation and not an 18-orbital impurity ED.
It uses three six-correlated-orbital ED impurities, their finite baths, and
the full 18x18 lattice GW Green matrix. Every candidate runs the same
multi-impurity Fierz-consistent embedding map at the requested V and filling.

**No susceptibilities are computed.** All ordering information is extracted
from the converged lattice normal Green function. A single 18-site
finite-temperature ED Green reference is computed **once per (V,n) point**
and compared to every converged cluster branch, using the full 6x6
matrix at the selected primitive momentum K. ED employs the existing
thermal Lanczos truncation (not a full thermal eigenbasis), so its
convergence controls remain important.

## Alternative triangle3 impurity partition (3 correlated + 3 bath)

Use the new dedicated script `research/scan_hex18_triangle_ed_gw_vn.py`
for the two-triangle-per-primitive-cell partition. It invokes the same
ordered-state scan, Green-function benchmark and real-space plotting logic
as `scan_hex18_cluster_order_vn.py`, but uses
`--cluster-partition triangle3`, **default `--nbath 3`**.

On periodic hex18, this means **six independently self-consistent Anderson
impurities**, each containing three interacting triangle sites and
three auxiliary bath orbitals. The existing `cell6` branch remains
available through the original script; it uses three six-site impurities
and six bath orbitals by default. Both use the **exact same 18-site
lattice hopping Hamiltonian**, including every `t1` and `t2` between
different triangles. The triangular impurity's bare Hamiltonian contains
only the three intra-triangle `t` hoppings; the intertriangle hopping
is encoded by the full lattice Dyson equation and each triangle's
self-consistent Weiss hybridization.

In triangle3, the local ED and the subtracted local GW treat only
the three V bonds on the **same** A or B triangle. The lattice GW
remains an 18-site calculation and is never downfolded to isolated
triangles. For `pseudospin`/`bj` options, the local Fierz vertices
are projected from the correctly matched six-site A/B triangle
definition before local GW subtraction; the physical ED impurity
interaction is independent of Fierz convention.

Example (PowerShell):

```powershell
git pull
python research/scan_hex18_triangle_ed_gw_vn.py `
  --V-start 0 --V-stop 5 --V-step 0.5 `
  --fillings 2.0 2.1 2.2 2.3 `
  --T 0.125 --t 1 --t1 -0.5 --t2 0.5 `
  --K 0.333333333333 0.333333333333 `
  --modes density `
  --nw 19 --nomega 10 `
  --max-iter 160 --mixing 0.25 --mixing-method pulay `
  --bath-energy-window 8 --bath-fit-restarts 2 `
  --plot-branches `
  --out results/hex18_triangle3_vn
```

Or use `python research/scan_hex18_cluster_order_vn.py
--cluster-partition triangle3` with the same remaining arguments.
To test bath convergence, explicitly pass `--nbath 4` or
`--nbath 6`. A 3+3 impurity has only `2^6=64` Fock states,
compared with `2^12=4096` for the old 6+6 impurity; the 18-site GW
and the ED Green **reference** cost do not decrease.

The **only filling constraint** is the global folded-lattice
`N_target=3*n`, enforced by a common chemical potential mu and
full matrix Dyson equation. Neither the A and B occupancies nor
the six individual impurity fillings are fixed separately. Each
converged branch's JSON/CSV records `N_lattice`,
`filling_error`, and the three A/B triangle occupancies;
its `*_orders.npz` contains `triangle_occupations`
of shape `(3,2)`. At `n=2.2`, the lattice target is 6.6,
not six separately fixed occupancies.

The same V/n branch-specific warm start, finite-Q zero-source
self-consistency, ED Green comparison at the requested K, and optional
density/bond/current triptych plotting are reused unchanged.
The first triangle3 point starts from a free-lattice seed;
the `--bootstrap uniform` setting (for legacy six-site warm-up)
is intentionally not used for a triangle3 bootstrap, because it
would require a different impurity partition. In both partitions,
only **certified converged states of the same partition** are used
for cross-V or cross-filling warm starts; restart checkpoint shapes
and physical signatures distinguish the two.

The triangle3 density-mode Luttinger-Ward free energy now sums
six three-site ED-minus-local-GW Phi corrections instead of three
six-site terms, using the same full-lattice GW and global Helmholtz
Legendre transform. It is an approximate finite-T ranking only
when bath and Gimp/Gloc consistency thresholds are satisfied.
It is **not** automatically more accurate than cell6; the intertriangle
two-particle diagrams are treated by lattice GW rather than ED.
The two partitions must be benchmarked against the same ED Green
reference and checked for phase-boundary sensitivity.

## Suggested run

From the repository root, in PowerShell:

```powershell
git pull

python research/scan_hex18_cluster_order_vn.py `
  --V-start 0 --V-stop 5 --V-step 0.5 `
  --fillings 2.0 2.1 2.2 2.3 `
  --T 0.125 --t 1 --t1 -0.5 --t2 0.5 `
  --K 0.333333333333 0.333333333333 `
  --modes density `
  --nbath 6 --nw 19 --nomega 10 `
  --max-iter 160 --tol 3e-5 `
  --mixing-method pulay --mixing 0.25 `
  --bath-energy-window 8 --bath-fit-restarts 2 `
  --seed-amplitudes 0.08 `
  --out results/hex18_ordered_vn
```

The default candidate list includes a symmetric branch; primitive-cell CO;
Q=(1/3,1/3) CO on both triangle sublattices with even/odd relative sign;
same/opposite triangle loop-current branches at Q=0 and Q=K; two
star-of-David/inverse-star-of-David-*like* trial real-bond motifs; and a
mixed CO/LC seed. These are a **set of trial self-energy patterns**, not
pre-labelled exact phases. For more exhaustive testing include
`--candidates` followed by names from
`rubycgw.hex18_order.FAMILIES`, e.g.
`co_qK_A1 co_qK_A2 co_qK_sin_A0 co_qK_odd_sin_A0
lc_qK_same_sin lc_qK_opposite_sin bond_qK_c3 bond_q0_c3
random_complex`, and multiple `--seed-amplitudes 0.04 0.12`.
The independent finite-Q sine and cosine seeds allow different spatial
phases within the three-cell torus, not merely different orbital directions.

The hex18 ring3 bond geometry can be organized into C3/C6 bond classes
as in the preceding hexagon HF study. The SD/ISD-like seeds here probe
real-bond modulation on the **actual Ruby hopping graph**. They are not
automatically equivalent to kagome SD/ISD states. The script reports
actual real-space bond observables so the distinction can be checked.

## Real-space density, bond and current visualization

The plotter `research/plot_hex18_cluster_order.py` consumes the
`*_orders.npz` files saved by this scan **without any new ED or GW
calculations**. It produces a single, matched three-panel figure:
site densities `n_i - mean(n)`, real-bond order, and oriented physical
bond currents on the true six-triangle hexagon.

Its site permutation is the *exact* equivalence verified in
`research/verify_hex18_equivalence.py`, so it uses the same Hamiltonian
as the native hex18 solver while showing the original hexagonal drawing.
All 36 physical hoppings are represented; the six wrapped bonds are
shown as dashed periodic-image links. Arrows follow the signs of the
stored bond currents, including the hopping amplitude convention.
The default real-bond panel subtracts the mean within each hopping
type `t`, `t1`, and `t2`, exposing symmetry-breaking modulation
rather than a uniform kinetic background.

For one converged candidate, from the repository root:

```powershell
python research/plot_hex18_cluster_order.py `
  --orders results/hex18_ordered_vn/n_2p2/V_1/density__symmetric_orders.npz `
  --formats png svg --annotate
```

For all converged branches in an existing scan, or just the reliably
selected minimum-F branch at each point:

```powershell
python research/plot_hex18_cluster_order.py `
  --scan-root results/hex18_ordered_vn `
  --selection converged --modes density --V 1 --fillings 2.2
```

Use `--selection winners` only for points with reliable phase
ranking in `phase_summary.csv`. Unresolved points have no selected
winner. Optional `--bond-reference zero` displays the raw real
bond expectation without subtracting each hopping type's mean.

To produce three-panel images automatically after each *converged*
candidate in a new scan, add to the scan command:

```powershell
--plot-branches --plot-formats png svg
```

Plots are saved beside each candidate's NPZ result as
`<mode>__<candidate>_orders_realspace.png` (and `.svg` when requested).
Failed/unconverged branches do not produce misleading state plots.
The plotter and the automatic scan hook share the same implementation.

## What --K and --modes mean

`--K K1 K2` selects the **primitive momentum at which the already
converged 18-site Green function is projected to a 6x6 matrix** for
comparison with the thermal Lanczos ED reference. The default is
`K=(1/3,1/3)`; `K=(0,0)` is also allowed. It is *not* a
symmetry-breaking wave vector and does not constrain the ordered state.
The output additionally contains the off-diagonal momentum mixing
`G(K,K+Q;iomega)` with the fixed commensurate
`Q=(1/3,1/3)`. To find finite-Q order, the **candidate seed** and
the 18-site multi-impurity solver are responsible, not the `--K` flag.

`--modes` chooses the lattice GW Fierz decoupling matched to the
six-site impurity GW subtraction: `density`, `pseudospin` (A1+E)
or `bj` (bond/current). All use the **same physical interaction
Hamiltonian** and the same exact impurity solver. By default the
script uses `--modes density`; requesting all three compares
their approximate Green functions and order patterns, at extra cost.
Only the density-mode Luttinger-Ward implementation currently
supports the available free-energy ranking. Other modes' phase
summary remains `unresolved` even when a particular order is
well established.

## Why finite Q is included

Q=(1/3,1/3) is commensurate with the three-cell hex18 PBC. The seed
perturbs the initial **full 18x18 self-energy** with a cos(Q.R) or sin(Q.R)
profile (and the requested six-site orbital form factor). The physical
hopping matrix contains **no persistent source**. The subsequent
multi-impurity solver independently updates all three impurity baths
and their self-energies. A nonuniform Q profile may survive after the
seed is removed, yielding a real translation-breaking self-consistent
fixed point. A uniform single-impurity calculation alone could not do
this. The Q=K seed is only one harmonic; mixed order is possible.

The scanning order is outer n and inner V. For each candidate + Fierz
mode, use its last converged solution at the preceding V of the same n,
or at the same V of the preceding n; if neither is available, initialize
from the same-point symmetric solution, then a previously converged
symmetric point, then a uniform/free bootstrap. Across V/n the warm
state carries lattice G, Sigma_H, Sigma_emb, the **three different**
impurity self-energies, mu, and all complex bath poles/couplings. The
new target's lattice/cluster GW, mu and self-consistency map are
recomputed. The perturbation is applied only to the new starting
Sigma_emb. **Failed or unconverged states never become warm starts.**

## Output files

At the scan root:
- `branches.csv`: one row per (V,n,Fierz mode,candidate,seed amplitude),
  containing convergence, inferred order family, dominant form factor,
  CO/LC/bond amplitudes, full-matrix iomega Green errors vs ED,
  bath error, Gimp/Gc mismatch, mu, the finite-Q momentum-mixing norm
  ||G(K,K+Q;iomega)||/||G(K,K;iomega)||, and any free-energy diagnostic;
- `phase_summary.csv`: a cautious per-(V,n,mode) phase choice, reliable
  minimum Helmholtz F, competing F and near-degeneracy label. If the
  supplied finite-bath approximation does not permit reliable ranking,
  the result is `unresolved`;
- `phase_map.png`: a density-mode phase summary for successful rankings;
- `scan_manifest.json`: physical inputs and point-level statuses.

In each `n_<...>/V_<...>/` directory:
- `ed_Gk.npz`: once-per-point ED matrix G(K,iw) and thermal weight data;
- `<mode>__<candidate>.npz`: **only after convergence**, the full
  zero-source 18x18 matrix G(iw), self-energy and bath restart state,
  rho, 6x6 G(K,K;iw), the off-diagonal 6x6 G(K,K+Q;iw)
  and its fixed-point residual history;
- `<mode>__<candidate>_orders.npz`: rho, 18 site densities, all
  36 bond currents, real bond expectation values and G(K,K+Q;iw);
- `<mode>__<candidate>_bonds.csv`: physically oriented bond observables;
- `<mode>__<candidate>.json`: per-branch status and all projected
  order parameters for q=0 and q=K, including failed branches.

Use `--resume` with the **same physical scan arguments** to skip
previously converged branches and reload their saved states; failing
branches are retried. Solver mixing/iteration limits may be changed,
but model/grid/ED and seed definitions must remain unchanged.

## Thermodynamics / limitations

`--modes density` is the recommended default because the existing
`evaluate_multicell_free_energy` implements the **physical density
GW** Luttinger–Ward functional plus the cluster ED correction.
Only density-mode candidate endpoints with converged self-consistency,
`max Gimp/Gc <= --free-energy-mismatch-max` (default .05), and
`max bath fit error <= --free-energy-bath-max` (default .15) are
considered for a lowest-F branch choice. Other Fierz modes may be
used to compare Green functions and orders, but their free energies
are **not** intercompared using a density-only functional.
Use `--no-free-energy` for a faster exploratory survey; then all
reported **ground-state choices remain unresolved**, even if a branch
has a clear nonzero CO/LC order parameter.

Converged self-consistent branches can be metastable and a seed list
is not exhaustive. The finite-bath and Matsubara-grid approximation
must be checked; bath fitting errors like .44 at V=5 are too large
for trustworthy close free-energy rankings. Finally, T=0.125 is
a finite-temperature phase survey, not an exact zero-T ground state.
The exact finite 18-site ED thermal state itself cannot spontaneously
choose a broken-symmetry expectation without a source or selected
symmetry-breaking state; its Green function remains a reference for
the unbroken ensemble, not an automatically phase-matched ordered
branch. Thus a large G error against ED can partly reflect different
symmetry sectors, even for a physically meaningful ordered solution.
