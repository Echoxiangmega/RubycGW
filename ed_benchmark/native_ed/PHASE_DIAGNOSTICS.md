# Zero-temperature ED: candidate orders and how to interpret the V–n scan

This document defines **candidate channels**, not a fully automatic thermodynamic phase classifier. A finite PBC ED eigenstate normally preserves unbroken symmetries (unless degenerate), so the safest first output is a *correlation/instability map*, not categorical CO/LC/SC phases.

## Model and filling

- Spinless fermions on six orbital primitive cells, hopping `+t=+1`, fixed input `t1,t2`, density repulsion/attraction `V` **only inside each triangle**.
- Canonical `Nf` is an integer: `n=Nf/Nc` particles per primitive cell, `Nc=3` (`3x1`, `ring3`) or `Nc=4` (`2x2`). This is an intrinsically **discrete** filling grid.
- `ring3`: crystal momenta Γ,K,K′; `2x2`: Γ and M-type points; `3x1`: Γ and two uniaxial points. A particular finite-Q charge mode cannot necessarily be tested in all three clusters.

## Proposed candidate channels

| Candidate | Field/operator | Existing ED signal | Status |
|---|---|---|---|
| Charge order (CO), including intra-cell and finite-q | `O_CO(q)=sum_R,alpha f_alpha(q) exp(i q·R) (n_Ralpha - <n_Ralpha>)` | Connected charge covariance; strongest 6x6 orbital charge-correlation eigenmode at each allowed q; leading-mode static susceptibility | Implemented equal-time; response is along leading **full** covariance eigenvector, not a user-fixed q pattern |
| Loop current, physical same direction on A/B triangles | normalized sum of CCW plaquette currents with equal signs | `LC_same_S`, `LC_same_chi` | Implemented |
| Loop current, physical opposite direction on A/B triangles | normalized sum of CCW plaquette currents with opposite signs on the two triangles | `LC_opposite_S`, `LC_opposite_chi` | Implemented |
| Other current patterns | arbitrary eigenvectors of current-current matrix over triangles/squares/hexagons, possibly at finite q | `current_corr`, loops, current eigenvalues; need symmetry labeling of eigenvectors | Correlation matrix exists; systematic pattern classification is still TODO |
| Spinless pairing tendency | `Delta_ij=c_i c_j`, i!=j on NN bonds | pair Gram matrix and eigenvalues | Implemented equal-time for 18-site only; **dynamical pair susceptibility needs Nf ± 2** and symmetry classification remains TODO |
| Phase separation (PS) / clustering | non-convex energy vs particle density; spatial charge correlations | `E0(Nf,V)` and second differences; density structure factor | Calculate from scan after covering adjacent Nf; negative second difference by itself does not distinguish PS from pairing or finite-size effects |
| Bond/plaquette order | modulation of kinetic bonds and plaquette energies | not yet separately measured | TODO |
| Unresolved / normal-like | no robust growing channel | spectra plus all channels | descriptive only; do not infer true thermodynamic absence of order from 18-site data |

## Equal-time metrics emitted by channel_diagnostics.py

For every elementary triangle p, let `I_p` be the **counterclockwise** particle-current sum divided by the triangle length 3, as in `observables.current_terms`. Define `C_pq=<I_p I_q>-<I_p><I_q>` for triangles (a Hermitian positive-semidefinite Gram matrix). For Ntri=2 Nc:

- `O_same=(sum_p I_p)/Ntri`;
- `O_opposite=(sum_p s_p I_p)/Ntri`, `s_p=+1` for A triangles (orbitals 0,1,2), `-1` for B (orbitals 3,4,5).
- `LC_same_S=<O_same^2>_c`; `LC_opposite_S=<O_opposite^2>_c`.
- Ratio `LC_same_S/LC_opposite_S` is meaningful **within these equally normalized LC channels**, but not sufficient to declare an ordered phase.
- The file `equal_time.npz` uses old code's *operator* conventions but all triangle cycles are stored CCW in real space. These are **physical** same/opposite signs; the old `Gamma_+`/`Gamma_-` labels use opposite relative geometrical triangle orientations.
- Both static `chi` values (`LC_same_chi_static`, `LC_opposite_chi_static`) use the same normalization as the corresponding `O` operators.

For charge, project the site charge covariance into momentum q:
`C_ab(q) = (1/Nc) sum_{R,R'} exp(i q·(R'-R)) C_(R,a),(R',b)`.
`CO_qmax` is the maximum eigenvalue among all q and six-orbital charge eigenvectors; `CO_q1,CO_q2` identify its momentum in primitive reciprocal coordinates. The exact uniform mode is removed automatically by working at fixed Nf with connected C. **A large eigenvalue at V=0 is not itself CO**. The full q-dependent eigenvectors are saved as `channel_diagnostics.q_modes.json`.

`pair_lambda_max` is an equal-time NN bond-pair matrix eigenvalue. It is **not** directly comparable in magnitude to charge/current maxima; no SC phase label is assigned. Its growth relative to the noninteracting reference, long-distance correlations, spectral pair gap and N±2 susceptibility should be checked.

`loop_rho_flux=arg(prod_loop <c_i^dag c_j>)` is gauge invariant where the product is nonzero. `0` or `pi` can occur in time-reversal-symmetric states; it is neither a Peierls magnetic flux nor a sufficient LC phase diagnostic.

## Protocol for a defensible candidate phase map

1. Choose a fixed `(t,t1,t2)`. Scan integer `Nf` and `V`, **including V=0** for a normal reference and both negative and positive V if relevant.
2. At each point, compute the global lowest eigenstate **across all symmetry sectors**, low-energy gaps, site charge covariance, current matrix, and pair correlation. Use `--chi-lanczos` for a few competitive regions (expensive for a full scan).
3. Plot separate heatmaps of charge `CO_qmax` and its q/eigenvector, `LC_same_S`, `LC_opposite_S`, their two static susceptibilities, and `pair_lambda_max` (when enabled). Compare *within a normalized channel and across V* rather than comparing raw charge/current/pair numbers.
4. Identify **candidates**: a given channel increases with interaction, has an interpretable symmetry pattern, and is supported by low-lying states / larger susceptibility / a small source-field response. Save ties and coexisting tendencies rather than forcing an exclusive label.
5. Check any candidate with all available supercells, bearing in mind their **different sets of allowed q**, and ideally with larger N/canonical energy-size scaling. Examine near-degenerate ground states carefully: an arbitrary eigenvector inside a degenerate multiplet can obscure TR-breaking LC or CO.
6. For attraction, check `E0(Nf-1)+E0(Nf+1)-2 E0(Nf)` and its convex hull to assess PS. Negative local finite difference may also reflect bound pairs and parity effects, so **it is not by itself a PS label**.
7. Before making categorical thermodynamic CO/LC/SC phase claims, add pattern-fixed source-field susceptibilities, pair N±2 susceptibility, and finite-size or carefully qualified extrapolation.

## Commands (single line, Windows PowerShell or CMD)

```shell
python scan_phase.py --cluster ring3 --t1 0.6 --t2 0.2 --nf 6 7 8 9 --V -2 -1 -0.5 0 0.5 1 2 3 --pair --output results/ring3_t06_t02
python channel_diagnostics.py --input results/ring3_t06_t02
```

The scan writes `summary.csv` with `n=Nf/Nc` and a unique parameterized folder for each point. The postprocessor writes `channel_diagnostics.csv` and `channel_diagnostics.q_modes.json`. These are **numerical indicators**, not assigned phase names.

For an initial cheap pass use `--no-observables` and later rerun to calculate the full observables in a separate result tree or remove incomplete point directories. Do not use `--pair` for `2x2`: that feature is currently disabled for 24 sites. QuSpin is not required.

For separate zero-temperature heatmaps after the scan and diagnostics:

```shell
python -m pip install matplotlib
python plot_channels.py --input results\ring3_t06_t02
```

This plots the charge and current metrics, the physical same/opposite ratio,
optional Kubo susceptibilities and optional pairing eigenvalue. Missing
observables show up as gray cells rather than fabricated values.