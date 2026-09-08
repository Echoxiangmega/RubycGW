"""Finite-external-frequency covariant-GW response at supercell q_sc=0.

This module is the imaginary-frequency counterpart of ``finite_q_cgw``. It
keeps the external *spatial* supercell momentum at zero (primitive Gamma and
+/-Q remain available through the 18-orbital harmonic vertices), but allows a
nonzero external bosonic Matsubara index ``m_ext``.

For a fixed external transfer Q=(q_sc=0,iOmega_m), define

    X(p;Q) = G(p+Q) Gamma(p;Q) G(p).

Different external bosonic frequencies are independent linear-response blocks:

    [I - L(Q)] Gamma(Q) = K(Q).

Within one block the internal fermionic frequencies remain coupled by H/F/MT/AL
convolutions. The frequency routing follows the same off-diagonal self-energy
convention used by ``finite_q_cgw``:

    dW(K-Q,K) = W(K-Q) [L1(K;Q)+L2(K;Q)] W(K),

with

    L1(K;Q) = int_p X(p+K-Q;Q) G(p),
    L2(K;Q) = int_p G(p+K) X(p;Q).

At m_ext=0 the implementation reduces to the established static
``supercell_cgw`` kernel. The finite Matsubara box uses the same
zero-padding/truncation convention as the production GW code: shifted
fermionic or bosonic frequencies outside the represented box contribute zero.

The split-GW decomposition is retained:

    Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2,

where MT contains W-V while AL contains the full screened W.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import _check_backend
from .supercell_cgw import (
    SupercellVertexOptions,
    SupercellVertexResult,
    _check_vertex_solver,
    _fock_from_x,
    _gmres_matrix_free,
    _hartree_from_x,
    _initial_gamma_field,
    _maxabs,
)
from .supercell_gw import _reverse_fft_spectrum


DynamicVertexOptions = SupercellVertexOptions
DynamicVertexResult = SupercellVertexResult


def normalize_external_m(m_ext: int, grid: MatsubaraGrid) -> int:
    """Validate and return one represented external bosonic Matsubara index."""
    m = int(m_ext)
    if m < -int(grid.nOmega) or m > int(grid.nOmega):
        raise ValueError(
            f"external bosonic index m={m} is outside represented range "
            f"[-{grid.nOmega},+{grid.nOmega}]"
        )
    return m


def _boson_array_index(m: int, grid: MatsubaraGrid) -> int | None:
    m = int(m)
    if m < -int(grid.nOmega) or m > int(grid.nOmega):
        return None
    return m + int(grid.nOmega)


def _external_dst_slice(grid: MatsubaraGrid, m_ext: int) -> slice:
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    return dst


def _mask_external_window(
    field: np.ndarray,
    grid: MatsubaraGrid,
    m_ext: int,
) -> np.ndarray:
    """Zero entries for which the external shifted G(p+Q) is unavailable."""
    out = np.zeros_like(field)
    dst = _external_dst_slice(grid, m_ext)
    if dst.stop != dst.start:
        out[dst] = np.asarray(field)[dst]
    return out


def _x_field_iomega(
    G: np.ndarray,
    Gamma: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return X(p;Q)=G(p+Q) Gamma(p;Q) G(p) on the base fermion grid."""
    m_ext = normalize_external_m(m_ext, grid)
    out = np.zeros_like(Gamma, dtype=complex)
    src, dst = frequency_shift_slices(grid.nf, m_ext)
    if src.stop == src.start:
        return out
    out[dst] = np.einsum(
        "...ab,...bc,...cd->...ad",
        G[src],
        np.asarray(Gamma)[dst],
        G[dst],
        optimize=True,
    )
    return out


def _dynamic_corrections_iomega_direct(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    include_mt: bool,
    include_al: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros_like(X)
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()
    if not include_mt and not include_al:
        return gmt, gal1, gal2

    pref = float(grid.T) / float(grid.nk)
    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        src, dst = frequency_shift_slices(grid.nf, m)
        if src.stop == src.start:
            continue

        # MT: X(p+K;Q) W_c(K). All external-Q dependence is already in X.
        Xsrc = X[src]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = roll_spatial(Xsrc, iq1, iq2)
                if include_mt:
                    WcK = Wc[im, iq1, iq2]
                    gmt[dst] -= pref * Xq * WcK.T[None, None, None, :, :]

                if not include_al:
                    continue

                # Code convention: dW(K-Q,K), so the left screened
                # interaction carries internal index m-m_ext.
                mleft = m - int(m_ext)
                ileft = _boson_array_index(mleft, grid)
                if ileft is None:
                    continue

                # L1(K;Q)=int_p X(p+K-Q;Q) G(p).
                src1, dst1 = frequency_shift_slices(grid.nf, mleft)
                if src1.stop == src1.start:
                    continue
                X1q = roll_spatial(X[src1], iq1, iq2)
                G1base = G[dst1]
                L1 = pref * np.einsum(
                    "nxyef,nxyfe->ef", X1q, G1base, optimize=True
                )

                # L2(K;Q)=int_p G(p+K) X(p;Q).
                Gq = roll_spatial(G[src], iq1, iq2)
                Xbase = X[dst]
                L2 = pref * np.einsum(
                    "nxyef,nxyfe->ef", Gq, Xbase, optimize=True
                )

                Wleft = W[ileft, iq1, iq2]
                Wright = W[im, iq1, iq2]
                M1 = Wleft @ L1 @ Wright
                M2 = Wleft @ L2 @ Wright

                # Outer self-energy electron line G(p+K), matching the
                # established finite-q off-diagonal convention.
                gal1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                gal2[dst] -= pref * Gq * M2.T[None, None, None, :, :]

    return gmt, gal1, gal2


def _dynamic_corrections_iomega_fft(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    include_mt: bool,
    include_al: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same frequency routing, using FFT only for spatial convolutions."""
    zero = np.zeros_like(X)
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()
    if not include_mt and not include_al:
        return gmt, gal1, gal2

    pref = float(grid.T) / float(grid.nk)
    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        src, dst = frequency_shift_slices(grid.nf, m)
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xhat = np.fft.fftn(Xsrc, axes=(1, 2))
        if include_mt:
            WcT = np.swapaxes(Wc[im], -1, -2)[None, :, :, :, :]
            Wc_hat_minus = _reverse_fft_spectrum(WcT, axes=(1, 2))
            mt_conv = np.fft.ifftn(Xhat * Wc_hat_minus, axes=(1, 2))
            gmt[dst] -= pref * mt_conv

        if not include_al:
            continue

        mleft = m - int(m_ext)
        ileft = _boson_array_index(mleft, grid)
        if ileft is None:
            continue
        src1, dst1 = frequency_shift_slices(grid.nf, mleft)
        if src1.stop == src1.start:
            continue

        # L1(K;Q)=int_p X(p+K-Q;Q) G(p).
        X1hat = np.fft.fftn(X[src1], axes=(1, 2))
        G1base_T = np.swapaxes(G[dst1], -1, -2)
        G1base_hat_minus = _reverse_fft_spectrum(G1base_T, axes=(1, 2))
        L1_product = np.sum(X1hat * G1base_hat_minus, axis=0)
        L1 = pref * np.fft.ifftn(L1_product, axes=(0, 1))

        # L2(K;Q)=int_p G(p+K) X(p;Q).
        Gsrc = G[src]
        Xbase = X[dst]
        Ghat = np.fft.fftn(Gsrc, axes=(1, 2))
        Xbase_T = np.swapaxes(Xbase, -1, -2)
        Xbase_hat_minus = _reverse_fft_spectrum(Xbase_T, axes=(1, 2))
        L2_product = np.sum(Ghat * Xbase_hat_minus, axis=0)
        L2 = pref * np.fft.ifftn(L2_product, axes=(0, 1))

        Wleft = W[ileft]
        Wright = W[im]
        M1 = np.matmul(np.matmul(Wleft, L1), Wright)
        M2 = np.matmul(np.matmul(Wleft, L2), Wright)

        M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
        M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
        M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
        M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))
        gal1[dst] -= pref * np.fft.ifftn(Ghat * M1hat_minus, axes=(1, 2))
        gal2[dst] -= pref * np.fft.ifftn(Ghat * M2hat_minus, axes=(1, 2))

    return gmt, gal1, gal2


def vertex_corrections_iomega(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: DynamicVertexOptions = DynamicVertexOptions(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (H,F,MTc,AL1,AL2) for one external bosonic-frequency block."""
    backend = _check_backend(opts.momentum_backend)
    m_ext = normalize_external_m(m_ext, grid)
    X = _x_field_iomega(G, Gamma, m_ext, grid)
    zero = np.zeros_like(Gamma)

    gh = _hartree_from_x(X, Vq[0, 0], grid) if opts.include_hartree else zero.copy()
    gf = _fock_from_x(X, Vq, grid, backend) if opts.include_fock else zero.copy()

    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, :, :, :, :]
    dyn = _dynamic_corrections_iomega_fft if backend == "fft" else _dynamic_corrections_iomega_direct
    gmt, gal1, gal2 = dyn(
        G,
        W,
        Wc,
        X,
        m_ext,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )

    # Gamma(p;Q) is represented only where p and p+Q both lie in the finite
    # fermion box. Keep every tangent consistent with the bubble truncation.
    return tuple(
        _mask_external_window(part, grid, m_ext)
        for part in (gh, gf, gmt, gal1, gal2)
    )


def _kernel_sum_iomega(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: DynamicVertexOptions,
):
    parts = vertex_corrections_iomega(G, W, Vq, Gamma, m_ext, grid, opts)
    total = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
    return total, parts


def _bare_vertex_field_iomega(
    K: np.ndarray,
    G: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    arr = np.asarray(K, dtype=complex)
    if arr.shape == G.shape[-2:]:
        field = np.broadcast_to(arr, G.shape).copy()
    elif arr.shape == G.shape:
        field = np.array(arr, copy=True)
    else:
        raise ValueError(
            f"K shape {arr.shape} must be orbital matrix {G.shape[-2:]} "
            f"or full vertex field {G.shape}"
        )
    return _mask_external_window(field, grid, m_ext)


def solve_vertex_iomega(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: DynamicVertexOptions = DynamicVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> DynamicVertexResult:
    """Solve one cGW vertex at q_sc=0 and external bosonic index m_ext."""
    m_ext = normalize_external_m(m_ext, grid)
    norb = int(G.shape[-1])
    if np.asarray(K).shape not in ((norb, norb), G.shape):
        raise ValueError(f"unexpected K shape {np.asarray(K).shape}")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    solver = _check_vertex_solver(opts.solver)
    Kfield = _bare_vertex_field_iomega(K, G, m_ext, grid)
    Gamma0 = _initial_gamma_field(initial_gamma, Kfield)
    Gamma0 = _mask_external_window(Gamma0, grid, m_ext)

    if solver == "gmres":
        def apply_A(field):
            field = _mask_external_window(np.asarray(field, dtype=complex), grid, m_ext)
            kernel, _ = _kernel_sum_iomega(G, W, Vq, field, m_ext, grid, opts)
            return field - kernel

        Gamma, converged, it, err = _gmres_matrix_free(
            apply_A,
            Kfield,
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
            kernel, _ = _kernel_sum_iomega(G, W, Vq, Gamma, m_ext, grid, opts)
            residual = Kfield + kernel - Gamma
            err = _maxabs(residual)
            if opts.verbose:
                print(
                    f"dynamic cGW m={m_ext:+d} iter {it:4d}: residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * residual
            Gamma = _mask_external_window(Gamma, grid, m_ext)

    Gamma = _mask_external_window(Gamma, grid, m_ext)
    kernel, parts = _kernel_sum_iomega(G, W, Vq, Gamma, m_ext, grid, opts)
    gh, gf, gmt, gal1, gal2 = parts
    err = _maxabs(Kfield + kernel - Gamma)
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


def susceptibility_matrix_iomega(
    G: np.ndarray,
    left_vertices: np.ndarray,
    right_gammas: list[np.ndarray] | np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return chi_ab(Q)=-T/Nk sum_p Tr[K_a G(p+Q) Gamma_b G(p)]."""
    m_ext = normalize_external_m(m_ext, grid)
    K = np.asarray(left_vertices, dtype=complex)
    gammas = [np.asarray(x, dtype=complex) for x in right_gammas]
    if K.ndim != 3 or K.shape[-2:] != G.shape[-2:]:
        raise ValueError("left_vertices must have shape (N,norb,norb)")

    src, dst = frequency_shift_slices(grid.nf, m_ext)
    out = np.zeros((K.shape[0], len(gammas)), dtype=complex)
    if src.stop == src.start:
        return out
    Gp = G[src]
    G0 = G[dst]
    pref = -(float(grid.T) / float(grid.nk))
    for b, gamma in enumerate(gammas):
        if gamma.shape != G.shape:
            raise ValueError("every driven Gamma must have the same shape as G")
        out[:, b] = pref * np.einsum(
            "aij,nxyjk,nxykl,nxyli->a",
            K,
            Gp,
            gamma[dst],
            G0,
            optimize=True,
        )
    return out
