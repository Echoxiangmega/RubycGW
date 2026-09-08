#!/usr/bin/env python3
"""Run the full same-torus ED/GW benchmark with production-tail cGW.

This is a drop-in wrapper around ``benchmark_ed18_cgw.py`` for future benchmark
runs.  It preserves the ED/GG implementation and replaces only the static cGW
H/F tangent by the derivative of the production tail-subtracted equal-time map.
For reusing an already expensive ED benchmark without re-running ED, use
``decompose_ed18_cgw_stages_tail.py`` for static stages and
``augment_ed18_cgw_full_tau_tail.py`` for the full C(tau) overlay.
"""

from __future__ import annotations

import numpy as np

import benchmark_ed18_cgw as base
from rubycgw.ed_cgw_benchmark import static_response_from_gammas
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell_cgw import SupercellVertexOptions, physical_symmetric_susceptibility


_REFERENCE = None
_ORIGINAL_REBUILD = base._rebuild_scgw


def _rebuild_scgw(seed, params, grid, backend):
    global _REFERENCE
    out = _ORIGINAL_REBUILD(seed, params, grid, backend)
    h0 = out[0]
    _REFERENCE = build_tail_reference(h0, seed.mu, seed.Sigma_H, grid)
    return out


def _solve_static_cgw(G, W, Vq, vertices, grid, args, label):
    vertices = np.asarray(vertices, dtype=complex)
    if args.stage == "gg":
        gammas = [np.broadcast_to(K, G.shape).copy() for K in vertices]
    else:
        if _REFERENCE is None:
            raise RuntimeError("tail reference was not initialized")
        opts = SupercellVertexOptions(
            max_iter=args.vertex_max_iter,
            tol=args.vertex_tol,
            mixing=args.vertex_mixing,
            solver=args.vertex_solver,
            gmres_restart=args.vertex_gmres_restart,
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=(args.stage == "full"),
            verbose=args.vertex_verbose,
            momentum_backend=args.momentum_backend,
        )
        gammas = []
        for i, K in enumerate(vertices, start=1):
            print(f"  {label}: tail vertex {i}/{len(vertices)} ({args.stage})")
            res = solve_vertex_q0_tail(
                G, W, Vq, K, grid, reference=_REFERENCE, opts=opts
            )
            if not res.converged:
                raise RuntimeError(
                    f"{label} tail vertex {i} failed: residual={res.final_error:.3e}"
                )
            gammas.append(res.Gamma)
    raw = static_response_from_gammas(G, vertices, gammas, grid)
    sym, imag_max = physical_symmetric_susceptibility(raw)
    return np.asarray(sym, dtype=float), float(imag_max)


def main():
    base._rebuild_scgw = _rebuild_scgw
    base._solve_static_cgw = _solve_static_cgw
    print("NOTE: static cGW uses tail-consistent production H/F tangents")
    base.main()


if __name__ == "__main__":
    main()
