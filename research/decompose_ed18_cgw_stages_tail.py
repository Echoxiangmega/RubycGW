#!/usr/bin/env python3
"""Tail-consistent wrapper for ``decompose_ed18_cgw_stages.py``.

Reuses the existing ED/GG benchmark and all plotting/output logic, but every
static cGW stage differentiates the production tail-subtracted Hartree/Fock map.
"""

from __future__ import annotations

import decompose_ed18_cgw_stages as base
from benchmark_ed18_cgw import _rebuild_scgw as _benchmark_rebuild
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference


_REFERENCE = None


def _rebuild_scgw(seed, params, grid, backend):
    global _REFERENCE
    out = _benchmark_rebuild(seed, params, grid, backend)
    h0 = out[0]
    _REFERENCE = build_tail_reference(h0, seed.mu, seed.Sigma_H, grid)
    return out


def _solve_vertex_q0(G, W, Vq, K, grid, opts, initial_gamma=None):
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
    base._rebuild_scgw = _rebuild_scgw
    base.solve_vertex_q0 = _solve_vertex_q0
    print("NOTE: stage decomposition uses tail-consistent production H/F tangents")
    base.main()


if __name__ == "__main__":
    main()
