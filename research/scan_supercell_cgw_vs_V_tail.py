#!/usr/bin/env python3
"""Tail-consistent wrapper for ``scan_supercell_cgw_vs_V.py``.

The normal SC-GW continuation/checkpoint logic is unchanged.  At every
requested V, the current-response H/F tangent differentiates the production
analytic-tail equal-time map; MT/AL remain the derivative of the represented
retarded W-V part.
"""

from __future__ import annotations

import scan_supercell_cgw_vs_V as base
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell import build_supercell_h0


_REFERENCE = None
_CURRENT_PARAMS = None
_ORIGINAL_BUILD_INTERACTION = base.build_supercell_interaction
_ORIGINAL_SOLVE_RESPONSE = base._solve_cgw_response


def _build_interaction(qmesh, params):
    global _CURRENT_PARAMS
    _CURRENT_PARAMS = params
    return _ORIGINAL_BUILD_INTERACTION(qmesh, params)


def _solve_vertex(G, W, Vq, K, grid, opts, initial_gamma=None):
    if _REFERENCE is None:
        raise RuntimeError("tail reference was not initialized for this V")
    return solve_vertex_q0_tail(
        G,
        W,
        Vq,
        K,
        grid,
        reference=_REFERENCE,
        opts=opts,
        initial_gamma=initial_gamma,
    )


def _solve_response(gw, Vq, grid, args):
    global _REFERENCE
    if _CURRENT_PARAMS is None:
        raise RuntimeError("current Ruby parameters were not captured")
    h0 = build_supercell_h0(grid.kmesh(), _CURRENT_PARAMS, source_strength=0.0)
    _REFERENCE = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    return _ORIGINAL_SOLVE_RESPONSE(gw, Vq, grid, args)


def main():
    base.build_supercell_interaction = _build_interaction
    base.solve_vertex_q0 = _solve_vertex
    base._solve_cgw_response = _solve_response
    print("NOTE: V-scan cGW uses tail-consistent production H/F tangents")
    base.main()


if __name__ == "__main__":
    main()
