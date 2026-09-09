"""Fast wrapper around the production self-consistent GW+SOX solver.

The fixed-point equations are unchanged.  Only the periodic SOX self-energy
kernel is replaced by :func:`compute_sox_self_energy_periodic_fast` during the
solve.  The wrapper is intentionally small so the reference solver remains the
single implementation of the GW+SOX iteration logic.
"""
from __future__ import annotations

from . import gw_sox as _gw_sox
from .sox_fast import compute_sox_self_energy_periodic_fast


def solve_matrix_gw_sox_fast(*args, **kwargs):
    """Run ``solve_matrix_gw_sox`` with the optimized SOX self-energy kernel."""
    old = _gw_sox.compute_sox_self_energy_periodic
    _gw_sox.compute_sox_self_energy_periodic = compute_sox_self_energy_periodic_fast
    try:
        return _gw_sox.solve_matrix_gw_sox(*args, **kwargs)
    finally:
        _gw_sox.compute_sox_self_energy_periodic = old


def solve_primitive_gw_sox_fast(*args, **kwargs):
    """Primitive-cell entry point using the optimized SOX self-energy kernel."""
    old = _gw_sox.compute_sox_self_energy_periodic
    _gw_sox.compute_sox_self_energy_periodic = compute_sox_self_energy_periodic_fast
    try:
        return _gw_sox.solve_primitive_gw_sox(*args, **kwargs)
    finally:
        _gw_sox.compute_sox_self_energy_periodic = old


__all__ = ["solve_matrix_gw_sox_fast", "solve_primitive_gw_sox_fast"]
