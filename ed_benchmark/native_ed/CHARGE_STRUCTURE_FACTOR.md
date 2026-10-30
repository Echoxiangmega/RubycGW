# Momentum-resolved CO from exact ED (Ruby lattice)

This adds **all allowed wavevectors** and the complete **6×6 connected charge structure-factor matrix** to the existing site-density/variance workflow. The implementation is pure NumPy/SciPy/Matplotlib; there is no QuSpin dependency.

## Definition and normalization

For the six sites in primitive cell `R`, `n_(R,a)`, `a=0..5`, and a finite torus with `Nc` primitive cells, define the connected covariance for the **equal ensemble over the complete degenerate ground manifold**:

`C_(Ra,R'b) = <n_(R,a) n_(R',b)>_0 - <n_(R,a)>_0 <n_(R',b)>_0`.

Here `<...>_0 = Tr(P0 ...)/d0`, with `P0` the projector onto the full ground manifold, dimension `d0`. This prescription is invariant under rotations within that degenerate manifold. The Fourier basis used by the earlier susceptibility scan is unchanged:

`F_(Ra,b)(q) = delta_ab * exp(+2*pi*i*(q1*R1+q2*R2))/sqrt(Nc)`;

`S_ab(q) = [F(q)^dag C F(q)]_ab`.

The six real nonnegative eigenvalues of each Hermitian 6×6 matrix quantify **equal-time charge fluctuations** at q. They are not a static susceptibility; no energy denominators are included. The largest eigenvector `u_a(q)` is a *charge fluctuation mode*. Its corresponding site-space template is `F(q)u(q)` (arbitrary global complex phase). It is **not** automatically the density expectation of any exact ED eigenstate.

The code includes Γ and every allowed finite momentum: `ring3` has Γ and K/K′ (fractional `(1/3,1/3)` and `(2/3,2/3)` in our primitive reciprocal convention), `2x2` has the three M-type points, `3x1` has its three strip momenta. `K` and `K′` are conjugate for real density correlations.

## Separation of internal E pseudospin from triangle total charge

For each leading orbital eigenvector, the six orthonormal reference directions are `total_A=(1,1,1,0,0,0)/sqrt(3)`, `total_B=(0,0,0,1,1,1)/sqrt(3)`, `E_Ax=(2,-1,-1,0,0,0)/sqrt(6)`, `E_Ay=(0,-1,1,0,0,0)/sqrt(2)`, `E_Bx=(0,0,0,2,-1,-1)/sqrt(6)`, and `E_By=(0,0,0,0,1,-1)/sqrt(2)`. The B `y` sign agrees with the previous physical pseudospin convention. `E_internal_total` sums the four E weights. A finite-K peak with E weight near one represents *intercell-modulated triangle-internal charge disproportionation*, directly related to the HF `2+1` density mode.

The response is still not an ordered phase diagnosis by itself. For a unique translationally symmetric ED ground state, `n_selected=n_ensemble` even if `S(K)` is huge. For a degenerate ground manifold, a separate finite-q Hermitian density source is projected to `P0 O_q P0`, diagonalized, and a pure *exact ground state* is chosen for its largest absolute deviation from the ensemble-average source. That is shown as `selected_density`; it may still be uniform if the ground manifold does not carry an appropriate finite-q order. Both are saved with explicit labels.

## Windows CMD / PowerShell quick start

Update code from the repository's root, then work from `ed_benchmark\native_ed`:

```powershell
git pull origin main
cd ed_benchmark\native_ed
python test_charge_structure.py
python plot_charge_structure.py --input results\chi_ring3_n2 --V 1 2 3
```

To specifically investigate a finite-q peak around larger V:

```powershell
python plot_charge_structure.py --input results\chi_ring3_n2 --V 2.5 3 3.5 --output results\chi_ring3_n2\K_charge_modes
```

If you previously ran `plot_charge_pattern.py`, the tool can reuse its `charge_patterns\V_...\charge_pattern.npz` **without recomputing ED**. Use `--cache-only` to prohibit solving:

```powershell
python plot_charge_structure.py --input results\chi_ring3_n2 --V 1 2 3 --cache-only
```

Otherwise, when original `--save-state` was enabled it loads the exact `ground_states.npz`; if neither dataset exists, it only solves the requested V points. For 18-site half filling the full multiplet can be expensive; it is never silently converted into a unique-ground approximation.

`plot_charge_pattern.py --input ... --V ...` also automatically saves momentum-resolved S(q) under each `charge_patterns\V_...\momentum_charge\` output from this update onward.

## Files

- `charge_structure/charge_Sq_vs_V.png` and `.csv`: leading eigenvalue of S(q) for **each** allowed momentum, against V.
- `charge_structure/V_.../charge_structure.npz`: the whole Hermitian `S_q` array `(Nq,6,6)`, all eigenvalues `(Nq,6)` and eigenvectors `(Nq,6,6)`, complex site-space mode and real template, and ensemble vs selected ground-state densities.
- `charge_structure/V_.../charge_structure_modes.csv`: six eigenvalues at each q.
- `charge_structure/V_.../charge_orbital_character.csv`: E-type triangle-internal vs triangle-total charge weights for each q's strongest mode.
- `charge_structure/V_.../charge_structure_summary.json`: leading q, S value, E fraction, whether a polarized pure ground state was actually constructed.
- `charge_structure/V_.../q_00/`, `q_01/` etc.: real/imag 6×6 matrices, all eigenvalues and **real-space leading fluctuation templates**.
- `charge_structure/V_.../ensemble_density.png`, `selected_density.png`: compare actual one-point density in symmetric mixture and projected extremal exact ground state; unlike a mode template, these are true expectation values.

## Verification

`python test_charge_structure.py` verifies full 6×6 matrices for three supercells; Hermiticity, nonnegative eigenvalues, complete momentum projector sum, total number conservation, Fourier spectral sum rule `sum_q Tr S(q)=Tr C`, and equality with the already computed `CO_leading` at the same ED point. It also validates ground-subspace basis invariance, the E-mode decomposition, cache-only/no-recompute mode, CLI entrypoint and export. `python test_charge_pattern.py` continues to test the site-density visualizations.