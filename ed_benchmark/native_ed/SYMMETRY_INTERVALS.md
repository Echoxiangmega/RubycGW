# Fixed-filling ED symmetry-interval analysis

Run after any `scan_susceptibility.py` scan with fixed `n`/`Nf`, `t1`, `t2` and varying `V`. Uses the already saved `V_*/sectors.json`, `responses.json`, and `scan_metadata.json`. **Does not need new ED unless you opt into the `--refine` mode.**

## Windows PowerShell / CMD

```powershell
cd E:\26\博后\code\RubycGW\ed_benchmark\native_ed
python analyze_symmetry_intervals.py --input results\chi_E_matrix_gaps_n2
```

Or choose your own scan:

```powershell
python analyze_symmetry_intervals.py --input results\chi_t-06_t06_n3
```

It writes under `SCAN/symmetry_intervals/`:

- `intervals.md`, `intervals.csv`: adjacent *sampled* V points with identical complete ground-multiplet symmetry labels (including momentum, rotation and compatible mirror); no claim of thermodynamic phases.
- `crossing_brackets.csv`: `[V_left,V_right]` when the ground multiplet changes. An optional **linear interpolation hint** compares the competing sector energies, but it is not a calculated transition point.
- `symmetry_levels.png`: actual ED energies per symmetry sector, measured relative to `E0`. Different sectors can cross exactly; a single plotted line does not give a thermodynamic phase.
- `symmetry_regions.png`: a categorical one-dimensional diagram of observed ground-state symmetry regimes. Grey intervals have unresolved boundaries.
- `excitation_gaps.png`: lowest nonzero energy above each ground multiplet.
- `symmetry_points.csv`: full parsed ground labels at every scanned V, along with existing LC current projections and CO channel fields if available.
- `sector_levels.csv`: all saved low-lying multi-particle eigenvalues and their quantum numbers for independent inspection.

For a more precise level crossing bracket, opt into *energy-only* additional ED solutions, which reuse a cached symmetry basis (no Green/chi calculation):

```powershell
python analyze_symmetry_intervals.py --input results\chi_E_matrix_gaps_n2 --refine --refine-tol 0.02 --refine-steps 8
```

Refinement uses the same `RubyPBC` geometry, fixed particle number, hopping signs, `VScanSolver` and all symmetry sectors. These midpoint eigenvalues are not added to the original susceptibility scan (and do not have full response data). More than one crossing inside an old scan interval may require a new, finer ordinary `scan_susceptibility.py` scan. The `--refine` option may take substantial CPU time for 18-site half filling, and 24-site large bases are not optimized.

## What the interval labels mean

The sector labels are **exact quantum numbers of the finite PBC Hamiltonian**. They do not by themselves identify LC/CO/SC phases. A finite-size state can be unique and have zero static current/density modulation even when a corresponding susceptibility/structure factor is large.

- `K` is the integer character label of the independent finite-translation generator, not always a literal reciprocal vector. For `ring3`: `[0] = Gamma`, `[1] = (1/3,1/3)`, `[2] = (2/3,2/3)`.
- `rotation=['C6',3]` means eigenvalue `exp(-2 pi i 3/6) = -1`; `mirror=0` means even if the code included a compatible mirror block. The other mirrors may not commute with that rotation; one cannot necessarily label both simultaneously.
- Signed rotation (when `t1=-t2`) includes a particle-number-dependent gauge phase; the `m` label by itself is not the ordinary bare spatial C6 eigenvalue.
- `ground_dim` is counted from distinct eigenvalues across the symmetry-block spectrum with the configured energy tolerance (`degeneracy_tol`, often `1e-7`), not just the number of sectors. Near-degenerate levels inside this tolerance are treated as one ground multiplet.
- Opposite-LC `Jmax` in the saved CSV means the maximized current inside that *exact/near*-degenerate manifold; it is evidence of an accessible finite-size current-carrying ground-state combination, not sufficient proof of a thermodynamic LC phase.
- The adaptive charge-leading momentum can be compared with the ground total momentum, but they represent **different objects**: `q` is the wavevector of charge fluctuations, while `K` labels the **many-body ground state's** total momentum.

## Grounded example, n=2 ring3, t1=t2=0.5

From five **actual, previously saved** ED points (`V=1,2,2.5,3,3.5`):

- V sampled at 1 and 2: a *twofold* `K/K'` ground multiplet, each `C3:m0`, mirror-even.
- V sampled at 2.5, 3, 3.5: a *unique* `Gamma` ground state with `C6:m3`, mirror-even (`C6 = -1`).
- Therefore an observed finite-cluster symmetry change occurs **somewhere in `(2,2.5)`**. A linear interpolation of the two competing symmetry-sector energies gives `~2.158` as a **heuristic hint**, not a refined ED calculation.
- The strong `S(K)` and charge susceptibility at higher `V` can coexist with a unique `Gamma` ground state and uniform mean site density. Symmetry alone cannot assign the thermodynamic region to CO vs LC.

To turn these finite-torus symmetry regimes into *candidate physical phase* intervals, combine with identical-Hamiltonian CO and LC operators, low-energy matrix elements, finite-size scaling, and small source fields. Avoid comparing raw CO and LC susceptibilities of differently normalized operators as if they were a phase-classification score.
## E0(V) and energy-slope figures

The report now writes `ground_energy.png` (total E0 and E0 per primitive
cell), `ground_energy_slope.png` (only secant slopes within the same
ground-state symmetry segment), and `energy_slopes.csv`. No extra ED
computation is needed when analyzing an existing scan. A finite-size
different-irrep level crossing changes the lowest-energy branch, but an
observable discontinuity in its *first derivative* additionally requires
the slopes of the two branches to differ. The E0 curve can look smooth
at the scale of its extensive background, so compare the sector-resolved
levels and the intra-interval slopes. A finite-torus level crossing does
not by itself certify a thermodynamic LC/CO phase transition.
