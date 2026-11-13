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
`co_qK_A1 co_qK_A2 lc_qK_same_sin lc_qK_opposite_sin bond_qK_c3
bond_q0_c3 random_complex`, and multiple `--seed-amplitudes 0.04 0.12`.

The hex18 ring3 bond geometry can be organized into C3/C6 bond classes
as in the preceding hexagon HF study. The SD/ISD-like seeds here probe
real-bond modulation on the **actual Ruby hopping graph**. They are not
automatically equivalent to kagome SD/ISD states. The script reports
actual real-space bond observables so the distinction can be checked.

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
