"""Tail-consistent production static cGW response.

The legacy :mod:`rubycgw.supercell_cgw` kernel correctly differentiates the
explicit finite Matsubara box.  Production SC-GW, however, evaluates the
Hartree density and static Fock density matrix with analytic reference-tail
subtraction.  This module keeps the already validated MT/AL implementation but
replaces H/F by the exact derivative of that production equal-time map.

Because the reference Green function depends explicitly on the source through
h0, the H/F tail derivative contains a small source-dependent constant.  The
vertex equation is therefore written as

    [I - L_tail] Gamma = K_eff,

where K_eff contains those constant H/F tail pieces and L_tail remains a linear
matrix-free operator.  At the converged solution the reported decomposition
still satisfies

    Gamma = K + H + F + MT + AL1 + AL2.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .gw import _check_backend
from .response_tail import TailReference, build_tail_hf_context
from .supercell_cgw import (
    SupercellVertexOptions,
    SupercellVertexResult,
    _check_vertex_solver,
    _dynamic_corrections_direct,
    _dynamic_corrections_fft,
    _gmres_matrix_free,
    _initial_gamma_field,
    _maxabs,
    _x_field,
)


def _dynamic_parts(G, W, Vq, X, grid, opts):
    backend = _check_backend(opts.momentum_backend)
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, ...]
    dyn = _dynamic_corrections_fft if backend == "fft" else _dynamic_corrections_direct
    return dyn(
        G,
        W,
        Wc,
        X,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )


def solve_vertex_q0_tail(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    reference: TailReference,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> SupercellVertexResult:
    """Solve the production q_sc=0 cGW vertex with H/F tail consistency."""
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    norb = int(G.shape[-1])
    if K.shape != (norb, norb):
        raise ValueError(f"K shape {K.shape} != {(norb, norb)}")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    solver = _check_vertex_solver(opts.solver)
    backend = _check_backend(opts.momentum_backend)
    Kfield = np.broadcast_to(K, G.shape).copy()
    ctx = build_tail_hf_context(
        reference,
        K,
        Vq,
        grid,
        q_index=(0, 0),
        m_ext=0,
        backend=backend,
        include_hartree=opts.include_hartree,
        include_fock=opts.include_fock,
    )
    hconst, fconst = ctx.constant_vertex_parts(G)
    rhs = Kfield + hconst + fconst
    Gamma0 = _initial_gamma_field(initial_gamma, rhs)

    def linear_kernel(field):
        field = np.asarray(field, dtype=complex)
        X = _x_field(G, field)
        gh, gf = ctx.linear_vertex_parts(X, G)
        gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, grid, opts)
        return gh + gf + gmt + gal1 + gal2

    if solver == "gmres":
        def apply_A(field):
            field = np.asarray(field, dtype=complex)
            return field - linear_kernel(field)

        Gamma, converged, it, err = _gmres_matrix_free(
            apply_A,
            rhs,
            Gamma0,
            tol=float(opts.tol),
            max_iter=int(opts.max_iter),
            restart=int(opts.gmres_restart),
            verbose=bool(opts.verbose),
        )
    else:
        Gamma = np.array(Gamma0, copy=True)
        converged = False
        err = float("inf")
        it = 0
        for it in range(1, int(opts.max_iter) + 1):
            residual = rhs + linear_kernel(Gamma) - Gamma
            err = _maxabs(residual)
            if opts.verbose:
                print(
                    f"tail-consistent cGW iter {it:4d}: residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * residual

    X = _x_field(G, Gamma)
    gh, gf = ctx.total_vertex_parts(X, G)
    gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, grid, opts)
    total = gh + gf + gmt + gal1 + gal2
    err = _maxabs(Kfield + total - Gamma)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    return SupercellVertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_F=gf,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        solver=solver,
    )


__all__ = ["solve_vertex_q0_tail"]
