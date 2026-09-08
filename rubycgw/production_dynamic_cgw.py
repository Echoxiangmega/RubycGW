"""Tail-consistent finite-external-frequency cGW at q_sc=0.

This is the production-tail counterpart of :mod:`rubycgw.dynamic_cgw`.  MT/AL
frequency routing is unchanged; only the Hartree/static-Fock tangent is replaced
by the analytic derivative of the production equal-time tail-subtracted map.
"""

from __future__ import annotations

import numpy as np

from .dynamic_cgw import (
    DynamicVertexOptions,
    DynamicVertexResult,
    _bare_vertex_field_iomega,
    _dynamic_corrections_iomega_direct,
    _dynamic_corrections_iomega_fft,
    _mask_external_window,
    _x_field_iomega,
    normalize_external_m,
)
from .grids import MatsubaraGrid
from .gw import _check_backend
from .response_tail import TailReference, build_tail_hf_context
from .supercell_cgw import (
    _check_vertex_solver,
    _gmres_matrix_free,
    _initial_gamma_field,
    _maxabs,
)


def _dynamic_parts(G, W, Vq, X, m_ext, grid, opts):
    backend = _check_backend(opts.momentum_backend)
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, ...]
    dyn = (
        _dynamic_corrections_iomega_fft
        if backend == "fft"
        else _dynamic_corrections_iomega_direct
    )
    return dyn(
        G,
        W,
        Wc,
        X,
        int(m_ext),
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )


def solve_vertex_iomega_tail(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    reference: TailReference,
    opts: DynamicVertexOptions = DynamicVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> DynamicVertexResult:
    """Solve one production-tail cGW block at external bosonic index m_ext."""
    m_ext = normalize_external_m(m_ext, grid)
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    norb = int(G.shape[-1])
    if K.shape != (norb, norb):
        raise ValueError("tail-consistent dynamic solver requires a static orbital K")

    solver = _check_vertex_solver(opts.solver)
    backend = _check_backend(opts.momentum_backend)
    Kfield = _bare_vertex_field_iomega(K, G, m_ext, grid)
    ctx = build_tail_hf_context(
        reference,
        K,
        Vq,
        grid,
        q_index=(0, 0),
        m_ext=m_ext,
        backend=backend,
        include_hartree=opts.include_hartree,
        include_fock=opts.include_fock,
    )
    hconst, fconst = ctx.constant_vertex_parts(G)
    rhs = _mask_external_window(Kfield + hconst + fconst, grid, m_ext)
    Gamma0 = _mask_external_window(
        _initial_gamma_field(initial_gamma, rhs), grid, m_ext
    )

    def linear_kernel(field):
        field = _mask_external_window(np.asarray(field, dtype=complex), grid, m_ext)
        X = _x_field_iomega(G, field, m_ext, grid)
        gh, gf = ctx.linear_vertex_parts(X, G)
        gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, m_ext, grid, opts)
        return _mask_external_window(gh + gf + gmt + gal1 + gal2, grid, m_ext)

    if solver == "gmres":
        def apply_A(field):
            field = _mask_external_window(np.asarray(field, dtype=complex), grid, m_ext)
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
        Gamma = Gamma0.copy()
        converged = False
        err = float("inf")
        it = 0
        for it in range(1, int(opts.max_iter) + 1):
            residual = rhs + linear_kernel(Gamma) - Gamma
            err = _maxabs(residual)
            if opts.verbose:
                print(
                    f"tail dynamic cGW m={m_ext:+d} iter {it:4d}: "
                    f"residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * residual
            Gamma = _mask_external_window(Gamma, grid, m_ext)

    Gamma = _mask_external_window(Gamma, grid, m_ext)
    X = _x_field_iomega(G, Gamma, m_ext, grid)
    gh, gf = ctx.total_vertex_parts(X, G)
    gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, m_ext, grid, opts)
    gh, gf, gmt, gal1, gal2 = tuple(
        _mask_external_window(x, grid, m_ext)
        for x in (gh, gf, gmt, gal1, gal2)
    )
    err = _maxabs(Kfield + gh + gf + gmt + gal1 + gal2 - Gamma)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    return DynamicVertexResult(
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


__all__ = ["solve_vertex_iomega_tail"]
