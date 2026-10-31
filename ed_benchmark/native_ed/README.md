# Native Ruby-lattice ED (no QuSpin)

A zero-temperature **spinless** Ruby-lattice exact-diagonalization implementation using **NumPy and SciPy only**. This folder is separate from `../quspin_zeroT/` and reuses the same verified six-orbital geometry, PBC connectivity and positive-hopping convention.

**Status:** The finite-cluster Hamiltonian, symmetry projectors, zero-temperature Lanczos, selected correlation functions and Matsubara Green functions are running and have independent regression tests. The 18-site problems are the primary validated targets. The 24-site half-filled calculation (`Nf=12`, dimension 2,704,156) **has not been performance-validated** and exceeds the default symmetry-projector memory cap. Do not treat a finite-size correlation maximum as definitive evidence of a thermodynamic ordered phase.

## Installation (Windows PowerShell or CMD)

In the cloned `RubycGW` repository:

```powershell
cd ed_benchmark\native_ed
python -m pip install -r requirements.txt
python test_native.py
```

No QuSpin, Numba, C++ extensions or Visual C++ DLLs are required by the **native solver**. An existing QuSpin package does not need to be uninstalled. To refresh an already cloned repository run `git pull origin main` from the repository root.

## Zero-temperature ED: first runs

```powershell
python run_zeroT.py --cluster ring3 --nf 6 --V 0 --symmetry none --no-observables --out results\free_plain
python run_zeroT.py --cluster ring3 --nf 6 --V 0 --symmetry point --no-observables --out results\free_sym
```

Both runs should find `E0 = -9.162277660168...`, independent of the symmetry block strategy. The next interacting benchmark is:

```powershell
python run_zeroT.py --cluster ring3 --nf 6 --V 1 --symmetry point --pair --chi-lanczos 60 --max-current --save-psi --out results\ring3_N6_V1
python run_zeroT.py --cluster 3x1 --nf 6 --V 1 --symmetry point --save-psi --out results\3x1_N6_V1
```

Expected reference energies are `E0(ring3,6,1)=-7.993849322158...` and `E0(3x1,6,1)=-8.272976846672...`.

To inspect general `t1,t2` and the gauge-corrected rotation with `t1=-t2`:

```powershell
python run_zeroT.py --cluster ring3 --nf 6 --V 1 --t1 0.6 --t2 0.2 --no-observables --out results\generic
python run_zeroT.py --cluster ring3 --nf 6 --V 1 --t1 0.5 --t2 -0.5 --no-observables --out results\signed_C6
```

`--symmetry none` uses only the fixed particle number. `--symmetry translation` adds momentum blocks. `--symmetry point` (default) uses allowed rotations and compatible mirrors in each momentum little group. Each calculation searches **all** generated symmetry sectors; it does not assume the ground state is at Gamma.

## Symmetry details

- For general `t1,t2`, the `2x2` and `ring3` tori retain ordinary `C3` and valid mirrors; the `3x1` torus retains a mirror, **not** `C3`.
- `t1=t2` allows ordinary `C6` and more mirrors in the symmetric tori.
- `t1=-t2` allows gauge-signed `C6` for `ring3`/`2x2`, and gauge-signed `C2` for `3x1`. The native implementation applies these operations directly **in the original hopping gauge**, including the product of site phases and fermionic permutation sign. For odd `Nf`, `(G C6)^6=(-1)^Nf`, so its eigenvalues are shifted by a half-unit.
- Mirrors not commuting with rotation (or not preserving the given momentum) are not simultaneously diagonalized. For signed rotations, the additional gauge-dressed mirrors are **not yet simultaneously implemented**. This does not omit energy eigenstates, since all retained rotation sectors are included.
- The decomposition dimension sum and eigenvector residuals are checked. The group projectors are built explicitly, so for large `Nf` this step can be memory- and CPU-intensive.

A previous QuSpin prototype used a gauge that made signed `C6` a plain permutation, but **ordinary translations are generally not plain permutations after that gauge change**. The native solver avoids this issue by representing the signed rotation directly. The signed-symmetry implementation was checked against the original Hamiltonian residual, not just a symmetry-label count.

## Scan V and filling

```powershell
python scan_phase.py --cluster ring3 --nf 6 7 8 9 --V 0 0.5 1 2 3 --chi-lanczos 60 --pair --output results\ring3_scan
```

The `summary.csv` is updated after each successful point and the scan skips any point that already has `summary.json`. Default `t=1`, `t1=t2=0.5`. The `--t1` and `--t2` options are also available in the scan script. Filling per primitive cell is `n=Nf/3` for both 18-site clusters, `n=Nf/4` for the 24-site cluster.

## Static observables, flux and susceptibility

An ED parameter-point folder contains:

- `geometry.json`: site/orbital numbering, PBC hopping, model parameters, and conventions.
- `sectors.json`: dimension, labels and lowest eigenvalues of every spatial symmetry block.
- `summary.json`: global ground energy, the sector of the selected ground eigenvector, degeneracy count and available correlations.
- `equal_time.npz`: densities, connected charge covariance, `rho[i,j]=<c_i^dag c_j>`, plaquette vertex lists, real-space plaquette current expectations and current-current correlation matrix; eigenvalues of charge/current correlation matrices; nearest-neighbor spinless bond-pair Gram matrix if `--pair`; zero-temperature static selected LC/CO susceptibilities and spectral poles/weights if `--chi-lanczos > 0`.
- `ground_state.npz` if `--save-psi`: ground eigenvector in full fixed-`Nf` ascending-integer basis, for later Green-function calculations.

`loop_rho_flux` is the **gauge-invariant correlation loop phase** `arg(prod_<ij> <c_i^dag c_j>)`, not a magnetic Peierls flux nor a uniquely defined interacting self-consistent hopping flux. It is saved as `NaN` when the underlying product vanishes. A phase `pi` is not automatically time-reversal breaking.

For current ordering, `--max-current` diagonalizes `P J_loop P` in the specified near-degenerate eigenstate manifold. This is a **maximization over linear combinations**, not evidence of spontaneous current in a unique finite-size ground state. Change `--degeneracy-tol` explicitly if needed, and examine energy differences. The code does not automatically declare a phase boundary. Static LC response uses physically CCW-oriented plaquette currents; the original Γ± vertices had opposite geometric loop orientations between the A and B triangles.

The pairing Gram matrix at fixed `Nf` tests spinless **intersite** pairing correlations. It is **not** the dynamical pairing susceptibility (`Nf±2` sectors), which is future work. The current implementation does not automatically assign an `s/p/f` representation to pairing eigenvectors.

For `--chi-lanczos`, current and charge **Kubo** susceptibility data are approximated from Krylov poles and weights at zero temperature. The charge field is the strongest connected density-covariance eigenmode; for a specified CO pattern a dedicated source operator should be selected. Near-zero poles are flagged. This is not the same as evaluating `beta*variance`.

## One-particle Green functions

```powershell
python response_iw.py --input results\ring3_N6_V1 --orbitals 0 1 2 3 4 5 --nfreq 30 --krylov 80
```

This uses fixed `Nf+1` and `Nf-1` resolvents and saves `green_iw.npz`. The Green function definition and the finite-size addition/removal chemical-potential interval are recorded in that file. `beta_reference` only sets the Matsubara-frequency sampling grid, **not a finite temperature**. The default orbital indices are the six sites in primitive cell 0; this is not yet a translationally averaged or momentum-resolved 6×6 Green matrix. For comparison against lattice GW, specify the required site indexing and perform the appropriate Fourier transform or cluster averaging.

The noninteracting `3x1, Nf=6` Green function was checked against `(iω+μ-h0)^-1` to roughly `1e-13` numerical accuracy for a selected block.

## Resource and numerical limits

- The default solver explicitly assembles CSR Hamiltonians, and full fixed-N basis projectors are built for each symmetry block. The optional `--method matrixfree` avoids storing the complete hopping CSR matrix but **still constructs the explicit projectors**. Its large-size performance has not been tuned.
- The default projector cap is `--max-basis 900000`, and equal-time observable cap is `--max-observable-basis 70000`. 18-site `Nf=6,9` fits both; 24-site half filling does not. Raising caps may cause very high memory use, especially on Windows.
- Equal-time observables are currently evaluated in a full fixed-`Nf` basis; they are slower than the symmetry-reduced eigensolver. Use `--no-observables` for the first convergence check.
- The finite-cluster ground state may be exactly degenerate; an individual eigenvector in that manifold is not unique. To interpret current correlations and response in such cases, compare the entire manifold or include a controlled source field. The current prototype records near-degenerate states and an optional maximum-current diagnostic, not a full manifold-averaged `G`.
- Pair-pair correlation currently restricted to 18-site clusters in `observables.py`.
- Only selected diagonal susceptibilities have been implemented; full site-resolved four-point response is not implemented yet.

## Regression tests

```powershell
python test_native.py
```

This tests PBC geometry, ordinary/signed point-group projectors, dimensions, Hermiticity, fermionic signs, noninteracting and interacting energies, signed-rotation eigenstate residuals, equal-time data and the noninteracting Green-function identity. It imports **no QuSpin**.

## Interpreting V–n scans (new)

See [PHASE_DIAGNOSTICS.md](PHASE_DIAGNOSTICS.md). The solver computes order-channel
correlations, but **does not automatically assign a thermodynamic CO/LC/SC
phase**. Use `channel_diagnostics.py` to extract momentum-resolved charge
correlations and physically CCW-normalized same/opposite loop-current modes.
For example (Windows CMD or PowerShell, single-line commands):

```powershell
python scan_phase.py --cluster ring3 --t1 0.6 --t2 0.2 --nf 6 7 8 9 --V -2 -1 -0.5 0 0.5 1 2 3 --pair --output results\ring3_t06_t02
python channel_diagnostics.py --input results\ring3_t06_t02
```

The updated scanner records `n=Nf/Nc`, `t,t1,t2` in the summary CSV and
incorporates those hopping values into parameter-directory names, preventing
silent reuse of the wrong parameter point. Existing scan roots from earlier
versions should be kept separate from new ones.

Optional channel heatmaps (requires `python -m pip install matplotlib`):

```powershell
python plot_channels.py --input results\ring3_t06_t02
```

Each figure is a distinct **channel-strength map**, not an automatic
categorical phase diagram.

## Fixed-filling susceptibility vs V (new)

Use [`scan_susceptibility.py`](scan_susceptibility.py) for a direct **zero-temperature ED scan at fixed `n` or `Nf`**, with CSV + line plots for `CO_AB`, `CO_leading`, physical `LC_same`, `LC_opposite`, square/hexagon currents, and optional `Nf±2` bond pairing susceptibility. The analysis checks exact degeneracies and records undefined singular responses as `NaN` with separate zero-mode weights. For full source definitions, run options and Windows commands, see [`SUSCEPTIBILITY_SCAN.md`](SUSCEPTIBILITY_SCAN.md).

```powershell
python test_susceptibility.py
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --Vmin -1 --Vmax 2 --Vstep 0.5 --depth 60 --no-extra-loops --output results\chi_ring3_n2
python plot_susceptibility.py --input results\chi_ring3_n2
```

This scan is a separate workflow from [`scan_phase.py`](scan_phase.py), which saves primarily equal-time correlation maxima and optional selected legacy χ; the new scan uses explicit fixed source definitions and saves full Lanczos spectra for each channel.

## Full E-charge 4x4 response and gap diagnostics (new)

The fixed-filling `scan_susceptibility.py` now includes the complete real symmetric static susceptibility matrix in the four HF pseudospin charge sources `(even_x, even_y, odd_x, odd_y)` and saves its full `4x4` entries, eigenvalues, **leading eigenspace projector**, and zero-mode matrix. The same scan stores low-energy level gaps, source-active spectral gaps, ground-manifold degeneracy and leading charge-mode momentum/overlap. Solid vs dashed segments flag unique vs degenerate finite-size grounds. For full definitions and Windows commands, see [FULL_E_MATRIX_AND_GAPS.md](FULL_E_MATRIX_AND_GAPS.md).

```powershell
python test_susceptibility.py
python scan_susceptibility.py --cluster ring3 --n 2 --t1 0.5 --t2 0.5 --Vmin 1 --Vmax 5 --Vstep 0.25 --depth 80 --no-extra-loops --output results\chi_E_matrix_gaps_n2
python plot_e_matrix.py --input results\chi_E_matrix_gaps_n2 --V 3
```

Old scan directories from the previous channel schema must not be reused; the scanner explicitly prevents accidental mixing.
## Degenerate loop-current ground multiplets (fixed Nf, new)

For (t_1=-0.6, t_2=+0.6, n=3), the 18-site ring cluster can have a
four-dimensional ground space for every V in a scan. If a current operator
has matrix elements inside that space, the zero-temperature total static
susceptibility is intentionally reported as NaN with `status=zero_mode`.
This is **not** failed ED convergence nor a zero current response.

See [LC_DEGENERACY_DIAGNOSTICS.md](LC_DEGENERACY_DIAGNOSTICS.md).
From the native_ed directory, existing scans can be analyzed immediately:

```powershell
python postprocess_lc_degeneracy.py --input results\\chi_t-06_t06_n3
```

This creates `lc_regular_vs_V.png`, `lc_zero_mode_vs_V.png` and
`lc_degenerate_diagnostics.csv` without rerunning ED. If the original scan
used `--save-state`, it also reconstructs `lc_Jmax_vs_V.png` from saved
wavefunctions. Without those wavefunctions, **Jmax cannot be derived from
`zero_weight` alone**.

New scans additionally save `lc_ground_projected.npz`,
`chi_lc_regular.png`, `lc_ground_Jmax.png` and per-point
`LC_same_ground_Jmax_abs` and `LC_opposite_ground_Jmax_abs` in the CSV.
These are currents optimized inside the exactly degenerate ground-state
subspace, not by mixing arbitrary excited states.

## Real-space bond-current maps (new)

[Current-map guide](CURRENT_PATTERN.md) and [plot_current_pattern.py](plot_current_pattern.py)
produce **physical directed currents on every hopping bond**, plus the six
triangle CCW circulations, PBC image arrows, charge-continuity residual, and
plaquette correlation-loop phases. The plotted state is the exact ED
ground-multiplet superposition maximizing the specified same/opposite
triangle-current operator (not an arbitrary excited-state mixture).

Windows PowerShell or CMD, from this directory:

```powershell
python test_current_pattern.py
python plot_current_pattern.py --input results\chi_ring3_n2 --V 1 2 3 --channel opposite
```

Replace the input with your actual scan folder. If the scan stored
`ground_states.npz` via `--save-state`, plotting needs **no new ED**;
otherwise only the V points requested for the plot are solved again.
Outputs are in `SCAN/current_patterns/V_+.../bond_current.png`,
`triangle_current.png`, bond/plaquette CSVs and reproducible NPZ data.

## Site charge density, local variance, and CO-polarized ground states

The new [plot_charge_pattern.py](plot_charge_pattern.py) complements the
current diagrams. It saves real-space site densities for both the **unbiased
equal ground-state mixture** and the **maximally charge-disproportionated pure
ground-state superposition**, along with on-site quantum variances,
connected site-density covariances, physical triangle pseudospins and
variance-vs-V plots. See [CHARGE_PATTERN.md](CHARGE_PATTERN.md) for the
difference between `Var(n_i)=<n_i>(1-<n_i>)` and the **spatial variance of
the site-mean densities**, which is more relevant to CO.

```powershell
python test_charge_pattern.py
python plot_charge_pattern.py --input results\chi_ring3_n2 --V 1 2 3 --kind triangle
python plot_charge_pattern.py --input results\chi_ring3_n2 --V 1 2 3 --source CO_E_odd_x --polarity positive --output results\chi_ring3_n2\CO_E_odd_x_patterns
```

Existing scan folders can be used: saved `ground_states.npz` avoids ED
reruns, otherwise only the selected V values are re-solved. The script
never interprets nonzero on-site variance alone as evidence of charge order.

## All-q six-orbital charge structure factor (new)

The new [charge_structure.py](charge_structure.py) and
[plot_charge_structure.py](plot_charge_structure.py) evaluate the **complete
6x6 connected charge structure-factor matrix at every allowed PBC momentum**
(`Gamma,K,K'` for `ring3`; `Gamma` plus M-type momenta for `2x2`).
They export all eigenvalues/eigenvectors, real-space leading *fluctuation*
templates, and the fraction of the leading eigenmode in triangle-internal
HF E-type pseudospin charge vs triangle-total charge channels.

The ground-state degeneracy is treated with the basis-independent equal
ground-manifold ensemble. If a ground multiplet is available, a pure
ground-state superposition can additionally be selected using a projected
finite-q density source; the real density map is always labeled separately
from the fluctuation template. Strong S(K) alone does **not** imply nonzero
one-point CO in a finite ED ground state.

See [CHARGE_STRUCTURE_FACTOR.md](CHARGE_STRUCTURE_FACTOR.md) for Fourier
conventions, formulas, data files and options. CMD/PowerShell:

```powershell
python test_charge_structure.py
python plot_charge_structure.py --input results\chi_ring3_n2 --V 1 2 3
python plot_charge_structure.py --input results\chi_ring3_n2 --V 1 2 3 --cache-only
```

If `plot_charge_pattern.py` was already used, its saved
`charge_patterns/V_.../charge_pattern.npz` can be read **without rerunning ED**.
New `plot_charge_pattern.py` runs automatically export these q-resolved
results into each `charge_patterns/V_.../momentum_charge` subfolder.

## Fixed-n symmetry interval map and crossing brackets

Use [analyze_symmetry_intervals.py](analyze_symmetry_intervals.py) to convert
already computed `scan_susceptibility.py` sector energies into a one-dimensional
map of *finite-torus ground-state symmetry regions*, with multi-particle
momentum, C3/C6 representation, compatible mirror parity, near-degenerate
ground multiplet, energy gaps, and observed change brackets. The interval
table includes the CO leading wavevector and LC ground-multiplet current
diagnostics whenever they are saved. It deliberately **does not auto-assign
thermodynamic CO/LC phase labels based only on a symmetry sector**.

```powershell
python test_symmetry_intervals.py
python analyze_symmetry_intervals.py --input results\chi_E_matrix_gaps_n2
python analyze_symmetry_intervals.py --input results\chi_E_matrix_gaps_n2 --refine --refine-tol 0.02
```

The first command reads existing scan files only; the `--refine` variant
adds ED diagonalizations **only at bracket midpoints**, reusing the symmetry
basis and not recomputing Green functions or susceptibilities. See
[SYMMETRY_INTERVALS.md](SYMMETRY_INTERVALS.md) for output paths and limits.

## Compact fixed-filling V-scan plots and ground-state energy

By default `scan_susceptibility.py` now makes a **compact set of figures**:
`ground_energy.png` (total and per-cell E0), main CO response
(`CO_leading` and `CO_E_matrix`), physical LC_same/opposite, low gaps,
ground-state degeneracy and zero-mode diagnostics. The full four-dimensional
E-type CO matrix, each individual component, eigenvalues, eigenvectors,
spectra and all metadata are still **computed and saved**. To draw the
additional comparison graphs, pass `--detailed-plots` when scanning or
when plotting existing results:

```powershell
python plot_susceptibility.py --input results\chi_ring3_n2
python plot_susceptibility.py --input results\chi_ring3_n2 --detailed-plots
python analyze_symmetry_intervals.py --input results\chi_ring3_n2
```

The symmetry-interval report also saves `ground_energy.png`,
`ground_energy_slope.png` and `energy_slopes.csv`. The slope is a
within-symmetry *finite-difference secant*. Crossings are excluded from
these slopes. By Hellmann-Feynman, the exact derivative of E0 is the sum
of triangle-bond density correlations in a differentiable, nondegenerate
ground-state branch; the plotted secants only approximate it.

## Full loop-current S(q,V), off-site correlations and V=0 control

The new [plot_lc_structure.py](plot_lc_structure.py) analyzes the
**complete triangle-triangle connected current correlation matrix** in the
fixed-N exact ground-state multiplet and builds all allowed PBC momentum
**2x2 A/B** LC structure-factor matrices. It also separates local
p=q self-correlations from nonlocal p!=q correlations, resolves exact
ground-multiplet zero modes versus finite-energy fluctuations, and
compares all these quantities with the **same model at V=0**.
For each V it archives the full matrices, pair-distance correlations
and maximal same/opposite J in the ground manifold.

```powershell
python test_lc_structure.py
python plot_lc_structure.py --input results\chi_ring3_n2 --V 0 1 2 3 4
python plot_lc_structure.py --input results\chi_ring3_n2 --cache-only
```

Old scan results with `ground_states.npz` can be reused; otherwise missing
full ground multiplets (including V=0 if absent) are re-solved at the
requested V values only. A scalar `susceptibility.csv` is not enough to
reconstruct the complete inter-triangle correlation matrix.

The plots `lc_structure/lc_comparison.png`,
`lc_structure/lc_Sq_vs_V.png`, `lc_structure/lc_nonlocal_vs_V.png`
compare all-q structure factors, V=0-subtracted correlations, ground zero
modes, and Jmax **without treating nonzero Jmax as an LC phase label**.
The off-site-only Fourier diagnostic can be negative and is not a
positive-definite structure factor. See
[LC_STRUCTURE_BENCHMARK.md](LC_STRUCTURE_BENCHMARK.md) for normalization,
file layouts and caveats on finite-size order.

## ED loop-phase maps: triangles, squares, and hexagons together

`plot_loop_flux.py` reads existing native ED scans and draws **all elementary
triangle, square and hexagon loops on the same real-space map**. It exports
signed and absolute phase maps, per-loop CSV data and a V-scan overview.

From `ed_benchmark/native_ed` (PowerShell or CMD):

```powershell
python plot_loop_flux.py --input results\YOUR_2x2_SCAN --V 1 1.4 1.5 2 --channel same
python plot_loop_flux.py --input results\YOUR_2x2_SCAN --channel opposite
python plot_loop_flux.py --input results\YOUR_2x2_SCAN --V 1 2 --channel same --flux-mode correlation
python test_loop_flux.py
```

The **default** `--flux-mode fock` builds an effective complex hopping
from the one-body density matrix of the selected exact ED state, using

```text
  H_kin = sum_ij t_ij c_i^dag c_j + h.c.
  H_V   = V sum_(intra-triangle ij) n_i n_j
  t_eff(j <- i) = t_ij - V_ij <c_i^dag c_j>
  Phi_Fock(p) = arg product_(CCW i -> j in p) [t_eff(j <- i) / t_ij]
```

The division by the bare real hopping removes the reference sign/phase.
**This is a Fock effective Peierls phase**, not actual electromagnetic
magnetic flux: the exact ED Hamiltonian used in this scan has real hopping
and contains no externally imposed Peierls flux. Only the t-type
(intra-triangle) bonds carry V in this model, so the Fock phase of an
elementary hexagon containing no t-type edge is identically zero.
One cannot expect the +6phi hexagon, -2phi square, zero triangle pattern
of Guan et al., Phys. Rev. B 108, 085121 (2023), from this Hamiltonian:
that article *puts Peierls phases into the bare hexagon hopping bonds*.

For the original density-matrix loop phase instead, choose
`--flux-mode correlation`, which evaluates
`Phi_corr(p) = arg product_(CCW i->j in p) <c_i^dag c_j>`.
This can differ from the Fock phase and is not an inserted Peierls flux.
Both modes are gauge invariant for the stated conventions.
A phase of pi can respect time-reversal symmetry; the CSV also contains
`sin_phi` and signed current for interpretation.

Without `--V`, all saved `V_*/responses.json` points are processed.
Saved `ground_states.npz` is reused; otherwise the requested V is re-solved.
Use `--V` to avoid costly full-scan recomputation.

The default output is `SCAN/loop_flux_fock_same/` (or the corresponding
flux mode and current channel), containing for each V:

- `loop_flux_abs.png` and `loop_flux_signed.png`, with all three shapes together;
- `loop_flux.csv`, with both Fock and correlation phases, loop current,
  bare hopping phase and validity flags;
- `loop_flux_summary.json`, with per-shape statistics and selection warning.

At the scan level, `overview.csv`, `overview.json`, and
`loop_flux_overview.png` give the maximum/mean |Phi| of each shape versus V.
Phases with products smaller than `--min-loop-product` (default 1e-12)
are marked `n/a`, since their angle is not reliable.

If the maximal projected `LC_same` or `LC_opposite` eigenvalue is zero or
degenerate, the selected state is not unique; this is reported explicitly.
