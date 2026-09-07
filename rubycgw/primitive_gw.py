"""Modern 6-site primitive-cell self-consistent GW.

This module is the production primitive-cell entry point.  It deliberately
uses the same matrix-valued kernel as the mature supercell solver, but with the
ordinary six-orbital Ruby Hamiltonian and interaction.  In particular,

    Sigma_GW = Sigma_F + Sigma_c,
    Sigma_F  = - <G V>_{tau=0-},
    Sigma_c  = - G * (W - V),

so the non-decaying instantaneous bare interaction is not truncated by the
finite bosonic Matsubara box.  Fixed-filling calculations also inherit the
cached-tail, safeguarded-Newton chemical-potential solve and the strict final
fixed-point verification used by the supercell implementation.

Finite external-q response is intentionally *not* implemented here; this file
only modernizes the translationally invariant primitive background.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult
from .model import RubyParameters, build_h0, build_interaction
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import solve_matrix_gw_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_components,
    compute_sigma_gw_split_matrix,
    compute_static_fock_matrix,
    one_body_density_matrix_tail,
)


def primitive_background_arrays(
    params: RubyParameters,
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(h0, Vq)`` for the six-site primitive cell."""
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    return np.asarray(h0, dtype=complex), np.asarray(Vq, dtype=complex)


def solve_gw(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
) -> GWResult:
    """Solve production primitive-cell SC-GW with the static-Fock/W-V split.

    The numerical implementation is exactly the arbitrary-matrix fast solver
    used by the supercell code.  The only primitive-specific work performed
    here is constructing the 6x6 Bloch Hamiltonian and interaction.
    """
    h0, Vq = primitive_background_arrays(params, grid)
    return solve_matrix_gw_fast(h0, Vq, grid, opts=opts, initial=initial)


def rebuild_primitive_fixed_point(
    result: GWResult,
    params: RubyParameters,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> dict[str, np.ndarray | float]:
    """Rebuild one full split-GW map from a saved/in-memory primitive result.

    This is useful before a cGW response calculation: a response should only be
    trusted when the supplied background is actually a fixed point of the same
    self-energy map that is differentiated.
    """
    h0, Vq = primitive_background_arrays(params, grid)
    G = dyson_from_sigma_matrix(
        h0, grid, result.mu, result.Sigma_H, result.Sigma_GW
    )
    density = density_from_G_matrix(
        G, grid, h0=h0, mu=result.mu, sigma_h=result.Sigma_H
    )
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G,
        W,
        Vq,
        grid,
        h0,
        result.mu,
        result.Sigma_H,
        backend=backend,
    )
    r_h = float(np.max(np.abs(sigma_h_out - result.Sigma_H)))
    r_gw = float(np.max(np.abs(sigma_gw_out - result.Sigma_GW)))
    return {
        "h0": h0,
        "Vq": Vq,
        "G": G,
        "P": P,
        "W": W,
        "density": density,
        "Sigma_H_out": sigma_h_out,
        "Sigma_GW_out": sigma_gw_out,
        "residual_H": r_h,
        "residual_GW": r_gw,
        "residual": max(r_h, r_gw),
    }


__all__ = [
    "primitive_background_arrays",
    "solve_gw",
    "rebuild_primitive_fixed_point",
    "one_body_density_matrix_tail",
    "compute_static_fock_matrix",
    "compute_sigma_gw_split_components",
    "compute_sigma_gw_split_matrix",
]
