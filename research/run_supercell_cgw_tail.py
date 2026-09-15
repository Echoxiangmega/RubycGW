#!/usr/bin/env python3
"""Tail-consistent wrapper for ``run_supercell_cgw.py``."""

from __future__ import annotations

import run_supercell_cgw as base
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference


_REFERENCE = None
_ORIGINAL_REBUILD = base._verify_and_rebuild


def _verify_and_rebuild(seed, params, grid, backend):
    global _REFERENCE
    out = _ORIGINAL_REBUILD(seed, params, grid, backend)
    h0 = out[0]
    _REFERENCE = build_tail_reference(h0, seed.mu, seed.Sigma_H, grid)
    return out


def _solve(G, W, Vq, K, grid, opts, initial_gamma=None):
    if _REFERENCE is None:
        raise RuntimeError("tail reference was not initialized")
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


def main():
    base._verify_and_rebuild = _verify_and_rebuild
    base.solve_vertex_q0 = _solve
    print("NOTE: supercell cGW uses tail-consistent production H/F tangents")
    base.main()


if __name__ == "__main__":
    main()
