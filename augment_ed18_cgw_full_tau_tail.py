#!/usr/bin/env python3
"""Add tail-consistent full cGW C(tau) curves to an existing ED/GG benchmark.

The ED trace is reused from ``benchmark_ed18_cgw.py``.  The SC-GW checkpoint is
unchanged.  Only the covariant H/F tangent is upgraded so it differentiates the
same analytic-tail equal-time map used by production SC-GW; MT/AL frequency
routing is unchanged.

The old benchmark's saved static cGW result used the finite-box H/F tangent, so
this wrapper deliberately disables that obsolete m=0 equality check.  The new
dynamic m=0 implementation is independently regression-tested against the new
static tail-consistent solver.

Example
-------
    python augment_ed18_cgw_full_tau_tail.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --nw 55 ^
      --vertex-workers 2 ^
      --out bench_n3_V1_T008_full_tau_tail.npz
"""

from __future__ import annotations

import numpy as np

import augment_ed18_cgw_full_tau as base
from rubycgw.production_dynamic_cgw import solve_vertex_iomega_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell import build_supercell_h0
from rubycgw.dynamic_cgw import susceptibility_matrix_iomega


_REFERENCE = None
_ORIGINAL_LOAD = base._load
_ORIGINAL_REBUILD = base._rebuild_scgw


def _load_without_legacy_static(path):
    data = _ORIGINAL_LOAD(path)
    # The saved static cGW diagonal belongs to the historical finite-box H/F
    # tangent.  Comparing the new m=0 block against it would intentionally fail.
    data.pop("cgw_static_chi", None)
    return data


def _rebuild_scgw(seed, params, grid, backend):
    global _REFERENCE
    out = _ORIGINAL_REBUILD(seed, params, grid, backend)
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    _REFERENCE = build_tail_reference(h0, seed.mu, seed.Sigma_H, grid)
    return out


def _solve_chain(
    qname,
    channel,
    Kleft,
    Kright,
    m_nonnegative,
    G,
    W,
    Vq,
    grid,
    opts,
):
    if _REFERENCE is None:
        raise RuntimeError("tail reference was not initialized")
    vals = np.zeros(len(m_nonnegative), dtype=complex)
    iterations = np.zeros(len(m_nonnegative), dtype=int)
    residuals = np.zeros(len(m_nonnegative), dtype=float)
    previous = None
    for j, m in enumerate(m_nonnegative):
        print(f"  dynamic tail-full {qname:5s} {channel:12s} m={int(m):+d}")
        res = solve_vertex_iomega_tail(
            G,
            W,
            Vq,
            Kright,
            int(m),
            grid,
            reference=_REFERENCE,
            opts=opts,
            initial_gamma=previous,
        )
        if not res.converged:
            raise RuntimeError(
                f"tail dynamic vertex failed for {qname}/{channel}/m={m}: "
                f"residual={res.final_error:.3e}"
            )
        vals[j] = susceptibility_matrix_iomega(
            G,
            Kleft[None, ...],
            [res.Gamma],
            int(m),
            grid,
        )[0, 0]
        iterations[j] = int(res.iterations)
        residuals[j] = float(res.final_error)
        previous = res.Gamma
    return vals, iterations, residuals


def main():
    base._load = _load_without_legacy_static
    base._rebuild_scgw = _rebuild_scgw
    base._solve_chain = _solve_chain
    print("NOTE: using tail-consistent production H/F tangent at every external Omega")
    base.main()


if __name__ == "__main__":
    main()
