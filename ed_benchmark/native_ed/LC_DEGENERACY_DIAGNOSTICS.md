# Ground-multiplet loop-current diagnostics for the native Ruby ED

In an exact ground multiplet of dimension `d`, a static `LC_same` or
`LC_opposite` response can be `NaN` because the current operator has finite
matrix elements among exactly degenerate ground eigenvectors. This is **not**
an ED convergence failure, a zero response, or proof of an LC phase.

For the existing normalized, physically CCW-oriented triangle-current
operators `O_same` and `O_opposite`, this update computes

- `P0 O P0` in the orthonormal ED ground-multiplet basis;
- `Jmax_abs = max(abs(eigvalsh(P0 O P0)))` and `Jmin_signed`, `Jmax_signed`;
- `Jmax_per_triangle = Jmax_abs / sqrt(Ntriangles)` under the current
  normalization `O=(sum_p s_p I_p)/sqrt(Ntriangles)`;
- `W0 = Tr[(P0 O P0 - mean * identity)^2]/d`,
  `mean = Tr(P0 O P0)/d`, cross-checked against the previously saved
  `LC_*_zero_weight`. This `W0` is **not** the value of `Jmax**2`.

For a truly degenerate multiplet, any normalized superposition stays a ground
state; a tiny source field `H-h O` selects an extremal current eigenvector.
At finite size this does not prove thermodynamic spontaneous LC order, because
symmetry-protected degeneracy can exist without an LC phase transition.

## Existing scan: plots without recomputing ED

```powershell
cd ed_benchmark\native_ed
python postprocess_lc_degeneracy.py --input results\chi_t-06_t06_n3
```

This **never modifies** the existing `susceptibility.csv`. It writes
`lc_degenerate_diagnostics.csv`, `lc_regular_vs_V.png`, and
`lc_zero_mode_vs_V.png`. If `--save-state` was used in the original scan,
then it also opens each `V_.../ground_states.npz` and calculates
`lc_Jmax_vs_V.png` without rerunning ED.

If the original run did **not** save the full ground multiplet, a wavefunction
reconstruction is necessary to obtain `Jmax`, since the CSV's `W0` is only a
Frobenius-norm invariant. The utility leaves Jmax `NaN` and states clearly
that it cannot recover it from scalar CSV data.

## New scans (Jmax calculated automatically)

The updated `scan_susceptibility.py` computes `P0 O P0` for both triangle
current channels at each V point and writes:

- Additional CSV fields `LC_same_ground_Jmax_abs`,
  `LC_opposite_ground_Jmax_abs`, signed extrema, mean, RMS and
  `Jmax_per_triangle`.
- Per V: `lc_ground_projected.npz` with full matrices and eigenvectors.
- Whole scan: `chi_lc_regular.png` and `lc_ground_Jmax.png`, in addition
  to the existing `chi_zero_mode_weight.png`.

Run a small test scan first:

```powershell
python test_susceptibility.py
python scan_susceptibility.py --cluster ring3 --n 3 --t1 -0.6 --t2 0.6 --V 1 3 5 --depth 60 --no-extra-loops --save-state --output results\chi_n3_LC_projection
```

**Use a new output directory.** The updated channel schema is
`full_E_matrix_gaps_lcJ_v3`, deliberately preventing silent reuse of an old
point that lacks projected-current matrices. All old energy/response
conventions are otherwise unchanged. The additional matrices must not be
confused with a finite static `chi(0)` when `status=zero_mode`.