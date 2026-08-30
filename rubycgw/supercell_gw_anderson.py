"""Compatibility layer for the lightweight 18-site GW solver.

Historically the driver imported this module for the Anderson implementation.
The active solver now lives in :mod:`rubycgw.supercell_gw_block`: it keeps the
same public API but uses an analytic Hartree cold start, separate Hartree/GW
mixing, one expensive GW map per outer iteration, and late Anderson only.
"""

from .supercell_gw_block import (
    AndersonOptions,
    solve_matrix_gw_anderson,
    solve_supercell_gw_anderson,
)

__all__ = [
    "AndersonOptions",
    "solve_matrix_gw_anderson",
    "solve_supercell_gw_anderson",
]
