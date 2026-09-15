#!/usr/bin/env python3
"""Validate tail-consistent production cGW against direct SC-GW finite sources.

This reuses the robust source-solving/extrapolation machinery of
``validate_cgw_finite_source.py`` but replaces the historical finite-box H/F
vertex tangent by the analytic derivative of the same tail-subtracted
Hartree/Fock map used by production SC-GW.

Example
-------
    python validate_cgw_tail_fdt.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --h-values 0.02 0.01 0.005 ^
      --gw-tol 1e-10 ^
      --out validate_n3_V1_tail_fdt.npz
"""

from __future__ import annotations

import numpy as np

import validate_cgw_finite_source as base
from rubycgw.ed_cgw_benchmark import static_response_from_gammas
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell_cgw import SupercellVertexOptions


_REFERENCE = None
_ORIGINAL_REBUILD = base._rebuild_zero_background


def _rebuild_zero_background(seed, params, grid, backend):
    global _REFERENCE
    out = _ORIGINAL_REBUILD(seed, params, grid, backend)
    h0 = out[0]
    _REFERENCE = build_tail_reference(h0, seed.mu, seed.Sigma_H, grid)
    return out


def _cgw_scalar(G, W, Vq, K, grid, args):
    if _REFERENCE is None:
        raise RuntimeError("tail reference was not initialized")
    opts = SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        mixing=float(args.vertex_mixing),
        solver=str(args.vertex_solver),
        gmres_restart=int(args.vertex_gmres_restart),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=bool(args.vertex_verbose),
        momentum_backend=str(args.momentum_backend),
    )
    result = solve_vertex_q0_tail(
        G, W, Vq, K, grid, reference=_REFERENCE, opts=opts
    )
    if not result.converged:
        raise RuntimeError(
            f"tail-consistent cGW vertex did not converge: residual={result.final_error:.3e}"
        )
    chi = static_response_from_gammas(
        G, np.asarray(K)[None, ...], [result.Gamma], grid
    )[0, 0]
    return complex(chi), result


def main():
    base._rebuild_zero_background = _rebuild_zero_background
    base._cgw_scalar = _cgw_scalar
    print("NOTE: using tail-consistent production H/F cGW tangent")
    base.main()


if __name__ == "__main__":
    main()
