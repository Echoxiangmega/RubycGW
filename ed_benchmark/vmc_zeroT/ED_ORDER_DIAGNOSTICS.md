# ED order diagnostics (Ruby supercell)

All historical outputs are unchanged. New fields distinguish triangle-internal
charge polarization (CO_E) from total triangle charge (CO_N).
For CO, CO_E, CO_N:
- S_*_over_Ntriangle = leading full raw structure-factor eigenvalue / (2 Ncell).
- S_*_local and S_*_intertriangle partition that SAME mode. The latter is signed.
- S_*_intertriangle_qmax instead maximizes the matrix with all intratriangle
  blocks removed. It need not select the same q or mode.
- CO_leading_E_fraction and CO_leading_N_fraction are orbital weights of the
  selected full charge mode. If its top eigenvalue is degenerate, mode-specific
  quantities depend on the chosen eigenvector (multiplicity is reported).

S_LC_local = sum_i <I_i^2>/Ntriangle^2. S_LC_same_intertriangle and
S_LC_opposite_intertriangle subtract this common diagonal background.
Positive intertriangle values indicate constructive correlations in that
channel, not proof of long-range order.

For complete equally weighted ED ground manifolds only, each channel also
reports <O^dag P0 O>, the complementary excited-space contribution, their
fraction (null when the full strength vanishes), and max |<O>| within P0.
The CO mode O is complex at finite q; the maximum is its projected numerical
radius. LC uses the historical mean signed triangle current. The CO amplitude
normalization is 1/sqrt(Ntriangle) times a unit site-space Fourier mode.
CO and LC still have different microscopic units; these are not Pauli-projected
pseudospin observables.

Independent postprocessing (does not diagonalize H or optimize VMC):

    python ed_benchmark/vmc_zeroT/analyze_ed_order.py --input results/supercell_1x1_Nf4_V8p6

Uses saved ed/final_state.npz and settings.json. If settings are missing,
supply --t --t1 --t2 and --L1 --L2 if not 1x1. Writes ed/order_diagnostics.json
without modifying original summary.json. A complete ground space is required.
New ED runs include the fields directly in summary.json and benchmark CSVs.
No phase assignment is made from finite-size correlations alone.
