"""Tail-consistent covariant-GW response at a general bosonic transfer.

This module extends the existing static finite-q production kernel to

    Q = (q, i Omega_m)

on the discrete momentum/Matsubara grid.  It is primarily used by post-GW,
which needs the full dynamic density-density response rather than only the
static pseudospin channels used in phase-diagram scans.

The positive tangent convention is the one used throughout RubycGW,

    X(k;Q) = G(k+Q) Gamma(k;Q) G(k) = - dG/dh_Q.

For a density source K_b=|b><b| the physical reducible density response is

    chi_ab(Q) = - (T/Nk) sum_k,n X_b(k;Q)_{aa}.

Hartree/Fock tails are differentiated with :mod:`rubycgw.response_tail`.
MT uses W-V and AL uses the full screened W.  For an AL left bosonic frequency
that lies outside the stored W box, W is evaluated on demand from the same
background G; this is preferable to either zero padding or silently replacing
it by a different approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import _check_backend
from .response_tail import (
    TailReference,
    build_tail_hf_context,
    build_tail_reference,
    reference_tail_remainder,
)
from .supercell_cgw import (
    SupercellVertexOptions,
    _check_vertex_solver,
    _gmres_matrix_free,
    _initial_gamma_field,
    _maxabs,
)
from .supercell_gw import _reverse_fft_spectrum


@dataclass
class TransferVertexResult:
    Gamma: np.ndarray
    Gamma_H: np.ndarray
    Gamma_F: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    Gamma_extra: np.ndarray
    q_index: tuple[int, int]
    m_ext: int
    converged: bool
    iterations: int
    final_error: float
    solver: str = "gmres"


def normalize_q_index(q_index, grid: MatsubaraGrid) -> tuple[int, int]:
    arr = np.asarray(q_index, dtype=int).reshape(2)
    return int(arr[0] % grid.nk1), int(arr[1] % grid.nk2)


def negative_transfer(
    q_index: tuple[int, int], m_ext: int, grid: MatsubaraGrid
) -> tuple[tuple[int, int], int]:
    p1, p2 = normalize_q_index(q_index, grid)
    return ((-p1) % grid.nk1, (-p2) % grid.nk2), -int(m_ext)


def _static_kfield(field: np.ndarray, grid: MatsubaraGrid, norb: int) -> np.ndarray:
    arr = np.asarray(field, dtype=complex)
    if arr.shape == (norb, norb):
        return np.broadcast_to(arr, (grid.nk1, grid.nk2, norb, norb)).copy()
    expected = (grid.nk1, grid.nk2, norb, norb)
    if arr.shape != expected:
        raise ValueError(f"static field shape {arr.shape} != {expected}")
    return np.array(arr, copy=True)


def _mask_transfer(field: np.ndarray, m_ext: int, grid: MatsubaraGrid) -> np.ndarray:
    arr = np.asarray(field, dtype=complex)
    out = np.zeros_like(arr)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst] = arr[dst]
    return out


def _bare_vertex_field(
    K: np.ndarray, G: np.ndarray, m_ext: int, grid: MatsubaraGrid
) -> np.ndarray:
    norb = int(G.shape[-1])
    kfield = _static_kfield(K, grid, norb)
    out = np.zeros_like(G, dtype=complex)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst] = np.broadcast_to(kfield, out[dst].shape)
    return out


def transfer_x_field(
    G: np.ndarray,
    Gamma: np.ndarray,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return X(k;Q)=G(k+Q) Gamma(k;Q) G(k) on base-frequency indices."""
    p1, p2 = normalize_q_index(q_index, grid)
    G = np.asarray(G, dtype=complex)
    Gamma = np.asarray(Gamma, dtype=complex)
    if G.shape != Gamma.shape:
        raise ValueError("G and Gamma must have identical transfer-field shapes")
    out = np.zeros_like(G)
    src, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if src.stop == src.start:
        return out
    Gq = roll_spatial(G[src], p1, p2)
    out[dst] = np.einsum(
        "nxyab,nxybc,nxycd->nxyad",
        Gq,
        Gamma[dst],
        G[dst],
        optimize=True,
    )
    return out


def estimate_transfer_static_vertex(
    Gamma: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    edge_points: int = 2,
) -> np.ndarray:
    """Linear high-frequency estimate of Gamma(k;Q) for a finite transfer."""
    arr = np.asarray(Gamma, dtype=complex)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop == dst.start:
        return np.zeros(arr.shape[1:], dtype=complex)
    valid = np.arange(dst.start, dst.stop, dtype=int)
    p = min(max(int(edge_points), 1), max(len(valid) // 2, 1))
    idx = np.concatenate((valid[:p], valid[-p:]))
    return np.mean(arr[idx], axis=0)


def _polarization_at_m_direct(
    G: np.ndarray, m: int, grid: MatsubaraGrid
) -> np.ndarray:
    norb = int(G.shape[-1])
    out = np.zeros((grid.nk1, grid.nk2, norb, norb), dtype=complex)
    src, dst = frequency_shift_slices(grid.nf, int(m))
    if src.stop == src.start:
        return out
    pref = float(grid.T) / float(grid.nk)
    for iq1 in range(grid.nk1):
        for iq2 in range(grid.nk2):
            Gq = roll_spatial(G[src], iq1, iq2)
            out[iq1, iq2] = pref * np.einsum(
                "nxyab,nxyba->ab", Gq, G[dst], optimize=True
            )
    return out


def _polarization_at_m_fft(
    G: np.ndarray, m: int, grid: MatsubaraGrid
) -> np.ndarray:
    norb = int(G.shape[-1])
    out = np.zeros((grid.nk1, grid.nk2, norb, norb), dtype=complex)
    src, dst = frequency_shift_slices(grid.nf, int(m))
    if src.stop == src.start:
        return out
    A = G[src]
    B = np.swapaxes(G[dst], -1, -2)
    Ahat = np.fft.fftn(A, axes=(1, 2))
    Bhat_minus = _reverse_fft_spectrum(B, axes=(1, 2))
    product = np.sum(Ahat * Bhat_minus, axis=0)
    out = (float(grid.T) / float(grid.nk)) * np.fft.ifftn(
        product, axes=(0, 1)
    )
    return out


def _screened_at_m(
    G: np.ndarray,
    Vq: np.ndarray,
    m: int,
    grid: MatsubaraGrid,
    backend: str,
) -> np.ndarray:
    P = (
        _polarization_at_m_fft(G, m, grid)
        if backend == "fft"
        else _polarization_at_m_direct(G, m, grid)
    )
    norb = int(G.shape[-1])
    eye = np.eye(norb, dtype=complex)
    lhs = eye[None, None, :, :] - np.matmul(Vq, P)
    return np.linalg.solve(lhs, Vq)


def _build_w_lookup(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    backend: str,
) -> dict[int, np.ndarray]:
    stored = {int(m): W[i] for i, m in enumerate(grid.m_values)}
    needed = {int(mi) - int(m_ext) for mi in grid.m_values}
    lookup: dict[int, np.ndarray] = dict(stored)
    for m in sorted(needed):
        if m not in lookup:
            lookup[m] = _screened_at_m(G, Vq, m, grid, backend)
    return lookup


def _dynamic_parts_direct(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    Wlookup: dict[int, np.ndarray],
    X: np.ndarray,
    q_index: tuple[int, int],
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmt = np.zeros_like(X)
    gal1 = np.zeros_like(X)
    gal2 = np.zeros_like(X)
    if not opts.include_mt and not opts.include_al:
        return gmt, gal1, gal2

    ip1, ip2 = q_index
    pref = float(grid.T) / float(grid.nk)
    for im, mi_raw in enumerate(grid.m_values):
        mi = int(mi_raw)
        src2, dst2 = frequency_shift_slices(grid.nf, mi)
        if src2.stop == src2.start:
            continue

        if opts.include_mt:
            for iq1 in range(grid.nk1):
                for iq2 in range(grid.nk2):
                    Xq = roll_spatial(X[src2], iq1, iq2)
                    gmt[dst2] -= (
                        pref
                        * Xq
                        * Wc[im, iq1, iq2].T[None, None, None, :, :]
                    )

        if not opts.include_al:
            continue

        mleft = mi - int(m_ext)
        src1, dst1 = frequency_shift_slices(grid.nf, mleft)
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                if src1.stop == src1.start:
                    L1 = np.zeros_like(W[im, iq1, iq2])
                else:
                    Xleft = roll_spatial(
                        X[src1], iq1 - ip1, iq2 - ip2
                    )
                    L1 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Xleft, G[dst1], optimize=True
                    )

                Gq = roll_spatial(G[src2], iq1, iq2)
                L2 = pref * np.einsum(
                    "nxyef,nxyfe->ef", Gq, X[dst2], optimize=True
                )
                Wleft = Wlookup[mleft][
                    (iq1 - ip1) % grid.nk1,
                    (iq2 - ip2) % grid.nk2,
                ]
                Wright = W[im, iq1, iq2]
                M1 = Wleft @ L1 @ Wright
                M2 = Wleft @ L2 @ Wright
                gal1[dst2] -= pref * Gq * M1.T[None, None, None, :, :]
                gal2[dst2] -= pref * Gq * M2.T[None, None, None, :, :]

    return (
        _mask_transfer(gmt, m_ext, grid),
        _mask_transfer(gal1, m_ext, grid),
        _mask_transfer(gal2, m_ext, grid),
    )


def _dynamic_parts_fft(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    Wlookup: dict[int, np.ndarray],
    X: np.ndarray,
    q_index: tuple[int, int],
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmt = np.zeros_like(X)
    gal1 = np.zeros_like(X)
    gal2 = np.zeros_like(X)
    if not opts.include_mt and not opts.include_al:
        return gmt, gal1, gal2

    ip1, ip2 = q_index
    pref = float(grid.T) / float(grid.nk)
    for im, mi_raw in enumerate(grid.m_values):
        mi = int(mi_raw)
        src2, dst2 = frequency_shift_slices(grid.nf, mi)
        if src2.stop == src2.start:
            continue

        if opts.include_mt:
            Xhat = np.fft.fftn(X[src2], axes=(1, 2))
            WcT = np.swapaxes(Wc[im], -1, -2)[None, :, :, :, :]
            Wchat_minus = _reverse_fft_spectrum(WcT, axes=(1, 2))
            gmt[dst2] -= pref * np.fft.ifftn(
                Xhat * Wchat_minus, axes=(1, 2)
            )

        if not opts.include_al:
            continue

        mleft = mi - int(m_ext)
        src1, dst1 = frequency_shift_slices(grid.nf, mleft)
        if src1.stop == src1.start:
            L1 = np.zeros_like(W[im])
        else:
            X1hat = np.fft.fftn(X[src1], axes=(1, 2))
            G1T = np.swapaxes(G[dst1], -1, -2)
            G1hat_minus = _reverse_fft_spectrum(G1T, axes=(1, 2))
            C1 = pref * np.fft.ifftn(
                np.sum(X1hat * G1hat_minus, axis=0), axes=(0, 1)
            )
            # L1(Q;Qe)=C1(Q-qe).
            L1 = np.roll(C1, shift=(ip1, ip2), axis=(0, 1))

        G2 = G[src2]
        X2 = X[dst2]
        G2hat = np.fft.fftn(G2, axes=(1, 2))
        X2T = np.swapaxes(X2, -1, -2)
        X2hat_minus = _reverse_fft_spectrum(X2T, axes=(1, 2))
        L2 = pref * np.fft.ifftn(
            np.sum(G2hat * X2hat_minus, axis=0), axes=(0, 1)
        )

        Wright = W[im]
        Wleft = np.roll(
            Wlookup[mleft], shift=(ip1, ip2), axis=(0, 1)
        )
        M1 = np.matmul(np.matmul(Wleft, L1), Wright)
        M2 = np.matmul(np.matmul(Wleft, L2), Wright)

        M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
        M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
        M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
        M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))
        gal1[dst2] -= pref * np.fft.ifftn(
            G2hat * M1hat_minus, axes=(1, 2)
        )
        gal2[dst2] -= pref * np.fft.ifftn(
            G2hat * M2hat_minus, axes=(1, 2)
        )

    return (
        _mask_transfer(gmt, m_ext, grid),
        _mask_transfer(gal1, m_ext, grid),
        _mask_transfer(gal2, m_ext, grid),
    )


def solve_vertex_transfer_tail(
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
) -> TransferVertexResult:
    """Solve the tail-consistent cGW vertex at one general transfer Q.

    ``extra_kernel`` is an optional *linear* functional of Gamma.  It is used by
    GW+SOX so the same transfer solver remains the common cGW backbone.
    """
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    p = normalize_q_index(q_index, grid)
    m_ext = int(m_ext)
    norb = int(G.shape[-1])
    expected_g = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if G.shape != expected_g:
        raise ValueError("unexpected G shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")

    backend = _check_backend(opts.momentum_backend)
    solver = _check_vertex_solver(opts.solver)
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
    Wc = W - Vq[None, ...]
    Wlookup = _build_w_lookup(G, W, Vq, m_ext, grid, backend)
    dyn = _dynamic_parts_fft if backend == "fft" else _dynamic_parts_direct

    def parts(field):
        field = _mask_transfer(np.asarray(field, dtype=complex), m_ext, grid)
        X = transfer_x_field(G, field, p, m_ext, grid)
        gh, gf = ctx.linear_vertex_parts(X, G)
        gmt, gal1, gal2 = dyn(
            G, W, Wc, Wlookup, X, p, m_ext, grid, opts
        )
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
                    f"transfer cGW q={p}, m={m_ext:+d} iter {it:4d}: "
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
    gmt, gal1, gal2 = dyn(G, W, Wc, Wlookup, X, p, m_ext, grid, opts)
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


def transfer_response_tail_completed(
    G: np.ndarray,
    left_vertices: np.ndarray,
    right_gammas: list[np.ndarray] | np.ndarray,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
    h_static_ref: np.ndarray,
    mu: float,
    *,
    edge_points: int = 2,
) -> dict[str, np.ndarray]:
    """Raw and analytic-tail-completed response at one general transfer."""
    G = np.asarray(G, dtype=complex)
    K = np.asarray(left_vertices, dtype=complex)
    if K.ndim == 2:
        K = K[None, ...]
    gammas = [np.asarray(x, dtype=complex) for x in right_gammas]
    p = normalize_q_index(q_index, grid)
    raw = np.zeros((len(K), len(gammas)), dtype=complex)
    pref = -float(grid.T) / float(grid.nk)
    for b, gamma in enumerate(gammas):
        X = transfer_x_field(G, gamma, p, int(m_ext), grid)
        raw[:, b] = pref * np.einsum(
            "aij,nxyji->a", K, X, optimize=True
        )

    h = np.asarray(h_static_ref, dtype=complex)
    norb = int(G.shape[-1])
    ref = build_tail_reference(
        h,
        float(mu),
        np.zeros((norb, norb), dtype=complex),
        grid,
    )
    correction = np.zeros_like(raw)
    for b, gamma in enumerate(gammas):
        S = estimate_transfer_static_vertex(
            gamma, int(m_ext), grid, edge_points=edge_points
        )
        C = reference_tail_remainder(
            ref, S, grid, q_index=p, m_ext=int(m_ext)
        )
        correction[:, b] = -(1.0 / float(grid.nk)) * np.einsum(
            "aij,xyji->a", K, C, optimize=True
        )
    return {
        "raw": raw,
        "tail_correction": correction,
        "completed": raw + correction,
    }


__all__ = [
    "TransferVertexResult",
    "normalize_q_index",
    "negative_transfer",
    "transfer_x_field",
    "estimate_transfer_static_vertex",
    "solve_vertex_transfer_tail",
    "transfer_response_tail_completed",
]
