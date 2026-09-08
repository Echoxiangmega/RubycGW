#!/usr/bin/env python3
"""Tail-consistent wrapper for ``scan_primitive_cgw_q.py``.

The primitive SC-GW background is unchanged.  H/F in every finite-q vertex
solve differentiate the same analytic-tail equal-time map as that background.
"""

from __future__ import annotations

import scan_primitive_cgw_q as base
from rubycgw.model import build_h0
from rubycgw.production_finite_q_cgw import solve_vertex_finite_q_tail
from rubycgw.response_tail import build_tail_reference


_REFERENCE = None
_ORIGINAL_REBUILD = base.rebuild_primitive_fixed_point


def _rebuild(gw, params, grid, backend="fft"):
    global _REFERENCE
    out = _ORIGINAL_REBUILD(gw, params, grid, backend=backend)
    h0 = build_h0(grid.kmesh(), params)
    _REFERENCE = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    return out


def _solve(G, W, Vq, K, q_index, grid, opts, initial_gamma=None):
    if _REFERENCE is None:
        raise RuntimeError("tail reference was not initialized")
    return solve_vertex_finite_q_tail(
        G,
        W,
        Vq,
        K,
        q_index,
        grid,
        reference=_REFERENCE,
        opts=opts,
        initial_gamma=initial_gamma,
    )


def main():
    base.rebuild_primitive_fixed_point = _rebuild
    base.solve_vertex_finite_q = _solve
    print("NOTE: primitive finite-q cGW uses tail-consistent production H/F tangents")
    base.main()


if __name__ == "__main__":
    main()
