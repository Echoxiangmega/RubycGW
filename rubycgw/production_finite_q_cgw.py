"""Tail-consistent static finite-q cGW on the primitive momentum mesh."""

from __future__ import annotations

import numpy as np

from .finite_q_cgw import (
    FiniteQVertexOptions,
    FiniteQVertexResult,
    _bare_vertex_field,
    _dynamic_corrections_finite_q_direct,
    _dynamic_corrections_finite_q_fft,
    _x_field_finite_q,
    normalize_q_index,
    q_reduced_from_index,
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


def _dynamic_parts(G, W, Vq, X, q_index, grid, opts):
    backend = _check_backend(opts.momentum_backend)
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, ...]
    dyn = (
        _dynamic_corrections_finite_q_fft
        if backend == "fft"
        else _dynamic_corrections_finite_q_direct
    )
    return dyn(
        G,
        W,
        Wc,
        X,
        q_index,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )


def solve_vertex_finite_q_tail(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    q_index,
    grid: MatsubaraGrid,
    reference: TailReference,
    opts: FiniteQVertexOptions = FiniteQVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> FiniteQVertexResult:
    """Solve one static finite-q production cGW block with tail-consistent H/F."""
    p = normalize_q_index(q_index, grid)
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    norb = int(G.shape[-1])
    if K.shape != (norb, norb):
        raise ValueError("tail-consistent finite-q solver requires a static orbital K")

    solver = _check_vertex_solver(opts.solver)
    backend = _check_backend(opts.momentum_backend)
    Kfield = _bare_vertex_field(K, G)
    ctx = build_tail_hf_context(
        reference,
        K,
        Vq,
        grid,
        q_index=p,
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
        X = _x_field_finite_q(G, field, p)
        gh, gf = ctx.linear_vertex_parts(X, G)
        gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, p, grid, opts)
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
                    f"tail finite-q cGW q={q_reduced_from_index(p, grid)} "
                    f"iter {it:4d}: residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * residual

    X = _x_field_finite_q(G, Gamma, p)
    gh, gf = ctx.total_vertex_parts(X, G)
    gmt, gal1, gal2 = _dynamic_parts(G, W, Vq, X, p, grid, opts)
    err = _maxabs(Kfield + gh + gf + gmt + gal1 + gal2 - Gamma)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    return FiniteQVertexResult(
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


__all__ = ["solve_vertex_finite_q_tail"]
