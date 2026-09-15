# Discrete cGW/FDT validation

The production SC-GW solver evaluates equal-time Hartree/Fock quantities with analytic Matsubara-tail subtraction, whereas the static cGW kernel differentiates the explicitly represented finite fermionic box through `X = G Gamma G`.

Therefore the existing production finite-source check mixes two questions:

1. Is the H/F/MT/AL implementation the correct derivative of the discretized GW equations?
2. How large is the residual finite-Matsubara-box error relative to the production analytic-tail equal-time sums?

`validate_cgw_discrete_fdt.py` isolates (1). It re-equilibrates the production checkpoint to a diagnostic finite-box SC-GW fixed point at the same chemical potential, using

- `n = 1/2 + T/Nk sum G_aa`,
- `rho(k) = 1/2 I + T sum_n G(k,iw_n)`,
- the same static bare-V Fock plus dynamic `W-V` split as production.

For this diagnostic map the finite-source derivative and the static cGW vertex use the same finite-frequency discretization. Agreement to solver and central-difference precision therefore validates the H/F/MT/AL derivative implementation independently of analytic-tail effects.

This diagnostic finite-box map is not intended to replace production tail-corrected SC-GW.
