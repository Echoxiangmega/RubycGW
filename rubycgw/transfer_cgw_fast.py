"""Prepared-FFT transfer cGW kernel for repeated density-vertex solves.

For a fixed external transfer Q and fixed background G/W, all density sources
share the same linear dynamic MT/AL operator.  The reference implementation
recomputes several FFTs of G and W-V inside every GMRES matrix-vector product.
This module precomputes those background-only transforms once per Q and reuses
them for all orbital density sources and all Krylov iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices
from .gw import _check_backend
from .response_tail import TailReference, build_tail_hf_context
from .supercell_cgw import (
    SupercellVertexOptions,
    _check_vertex_solver,
    _gmres_matrix_free,
    _initial_gamma_field,
    _maxabs,
)
from .supercell_gw import _reverse_fft_spectrum
from .transfer_cgw import (
    TransferVertexResult,
    _bare_vertex_field,
    _build_w_lookup,
    _mask_transfer,
    normalize_q_index,
    solve_vertex_transfer_tail,
    transfer_x_field,
)


@dataclass
class _PreparedEntry:
    im: int
    src2: slice
    dst2: slice
    mleft: int
    src1: slice
    dst1: slice
    Wchat_minus: np.ndarray | None
    G2hat: np.ndarray | None
    G1hat_minus: np.ndarray | None
    Wright: np.ndarray | None
    Wleft: np.ndarray | None


@dataclass
class PreparedTransferFFT:
    q_index: tuple[int, int]
    m_ext: int
    entries: list[_PreparedEntry]
    grid: MatsubaraGrid
    opts: SupercellVertexOptions


def prepare_transfer_fft_context(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> PreparedTransferFFT | None:
    """Precompute background-only FFT factors for one external transfer.

    Returns ``None`` for the direct backend, for which the reference solver is
    retained unchanged.
    """
    backend = _check_backend(opts.momentum_backend)
    if backend != "fft":
        return None
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    p = normalize_q_index(q_index, grid)
    m_ext = int(m_ext)
    Wc = W - Vq[None, ...]
    Wlookup = _build_w_lookup(G, W, Vq, m_ext, grid, backend)
    ip1, ip2 = p
    entries: list[_PreparedEntry] = []

    for im, mi_raw in enumerate(grid.m_values):
        mi = int(mi_raw)
        src2, dst2 = frequency_shift_slices(grid.nf, mi)
        if src2.stop == src2.start:
            continue

        Wchat_minus = None
        if opts.include_mt:
            WcT = np.swapaxes(Wc[im], -1, -2)[None, :, :, :, :]
            Wchat_minus = _reverse_fft_spectrum(WcT, axes=(1, 2))

        mleft = mi - m_ext
        src1, dst1 = frequency_shift_slices(grid.nf, mleft)
        G2hat = None
        G1hat_minus = None
        Wright = None
        Wleft = None
        if opts.include_al:
            G2hat = np.fft.fftn(G[src2], axes=(1, 2))
            if src1.stop != src1.start:
                G1T = np.swapaxes(G[dst1], -1, -2)
                G1hat_minus = _reverse_fft_spectrum(G1T, axes=(1, 2))
            Wright = W[im]
            Wleft = np.roll(
                Wlookup[mleft], shift=(ip1, ip2), axis=(0, 1)
            )

        entries.append(
            _PreparedEntry(
                im=im,
                src2=src2,
                dst2=dst2,
                mleft=mleft,
                src1=src1,
                dst1=dst1,
                Wchat_minus=Wchat_minus,
                G2hat=G2hat,
                G1hat_minus=G1hat_minus,
                Wright=Wright,
                Wleft=Wleft,
            )
        )
    return PreparedTransferFFT(
        q_index=p,
        m_ext=m_ext,
        entries=entries,
        grid=grid,
        opts=opts,
    )


def _dynamic_parts_fft_prepared(
    X: np.ndarray,
    prepared: PreparedTransferFFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = prepared.grid
    opts = prepared.opts
    m_ext = int(prepared.m_ext)
    pref = float(grid.T) / float(grid.nk)
    gmt = np.zeros_like(X)
    gal1 = np.zeros_like(X)
    gal2 = np.zeros_like(X)

    for e in prepared.entries:
        Xsrc_hat = None
        if opts.include_mt:
            Xsrc_hat = np.fft.fftn(X[e.src2], axes=(1, 2))
            gmt[e.dst2] -= pref * np.fft.ifftn(
                Xsrc_hat * e.Wchat_minus, axes=(1, 2)
            )

        if not opts.include_al:
            continue

        if e.src1.stop == e.src1.start:
            L1 = np.zeros_like(e.Wright)
        else:
            if e.src1 == e.src2 and Xsrc_hat is not None:
                X1hat = Xsrc_hat
            else:
                X1hat = np.fft.fftn(X[e.src1], axes=(1, 2))
            C1 = pref * np.fft.ifftn(
                np.sum(X1hat * e.G1hat_minus, axis=0), axes=(0, 1)
            )
            ip1, ip2 = prepared.q_index
            L1 = np.roll(C1, shift=(ip1, ip2), axis=(0, 1))

        X2 = X[e.dst2]
        X2T = np.swapaxes(X2, -1, -2)
        X2hat_minus = _reverse_fft_spectrum(X2T, axes=(1, 2))
        L2 = pref * np.fft.ifftn(
            np.sum(e.G2hat * X2hat_minus, axis=0), axes=(0, 1)
        )

        M1 = np.matmul(np.matmul(e.Wleft, L1), e.Wright)
        M2 = np.matmul(np.matmul(e.Wleft, L2), e.Wright)
        M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
        M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
        M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
        M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))
        gal1[e.dst2] -= pref * np.fft.ifftn(
            e.G2hat * M1hat_minus, axes=(1, 2)
        )
        gal2[e.dst2] -= pref * np.fft.ifftn(
            e.G2hat * M2hat_minus, axes=(1, 2)
        )

    return (
        _mask_transfer(gmt, m_ext, grid),
        _mask_transfer(gal1, m_ext, grid),
        _mask_transfer(gal2, m_ext, grid),
    )


def solve_vertex_transfer_tail_fast(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
    reference: TailReference,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
    initial_gamma: np.ndarray | None = None,
    extra_kernel: Callable[[np.ndarray], np.ndarray] | None = None,
    prepared: PreparedTransferFFT | None = None,
) -> TransferVertexResult:
    """Transfer vertex solve using a shared prepared FFT background kernel."""
    backend = _check_backend(opts.momentum_backend)
    if backend != "fft":
        return solve_vertex_transfer_tail(
            G, W, Vq, K, q_index, m_ext, grid, reference,
            opts=opts, initial_gamma=initial_gamma, extra_kernel=extra_kernel,
        )

    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    p = normalize_q_index(q_index, grid)
    m_ext = int(m_ext)
    solver = _check_vertex_solver(opts.solver)
    if prepared is None:
        prepared = prepare_transfer_fft_context(
            G, W, Vq, p, m_ext, grid, opts
        )
    if prepared is None or prepared.q_index != p or prepared.m_ext != m_ext:
        raise ValueError("prepared transfer context does not match requested Q")

    Kfield = _bare_vertex_field(K, G, m_ext, grid)
    ctx = build_tail_hf_context(
        reference,
        K,
        Vq,
        grid,
        q_index=p,
        m_ext=m_ext,
        backend=backend,
        include_hartree=opts.include_hartree,
        include_fock=opts.include_fock,
    )
    hconst, fconst = ctx.constant_vertex_parts(G)
    rhs = Kfield + hconst + fconst
    Gamma0 = _initial_gamma_field(initial_gamma, rhs)
    Gamma0 = _mask_transfer(Gamma0, m_ext, grid)

    def parts(field):
        field = _mask_transfer(np.asarray(field, dtype=complex), m_ext, grid)
        X = transfer_x_field(G, field, p, m_ext, grid)
        gh, gf = ctx.linear_vertex_parts(X, G)
        gmt, gal1, gal2 = _dynamic_parts_fft_prepared(X, prepared)
        gextra = (
            _mask_transfer(extra_kernel(field), m_ext, grid)
            if extra_kernel is not None
            else np.zeros_like(field)
        )
        return gh, gf, gmt, gal1, gal2, gextra

    def linear_kernel(field):
        pp = parts(field)
        return sum(pp[1:], pp[0])

    if solver == "gmres":
        def apply_A(field):
            field = _mask_transfer(np.asarray(field, dtype=complex), m_ext, grid)
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
                    f"transfer cGW fast q={p}, m={m_ext:+d} iter {it:4d}: "
                    f"residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * residual
            Gamma = _mask_transfer(Gamma, m_ext, grid)

    Gamma = _mask_transfer(Gamma, m_ext, grid)
    X = transfer_x_field(G, Gamma, p, m_ext, grid)
    gh, gf = ctx.total_vertex_parts(X, G)
    gmt, gal1, gal2 = _dynamic_parts_fft_prepared(X, prepared)
    gextra = (
        _mask_transfer(extra_kernel(Gamma), m_ext, grid)
        if extra_kernel is not None
        else np.zeros_like(Gamma)
    )
    total = gh + gf + gmt + gal1 + gal2 + gextra
    err = _maxabs(Kfield + total - Gamma)
    converged = bool(np.isfinite(err) and err < float(opts.tol))
    return TransferVertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_F=gf,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        Gamma_extra=gextra,
        q_index=p,
        m_ext=m_ext,
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        solver=solver,
    )


__all__ = [
    "PreparedTransferFFT",
    "prepare_transfer_fft_context",
    "solve_vertex_transfer_tail_fast",
]
