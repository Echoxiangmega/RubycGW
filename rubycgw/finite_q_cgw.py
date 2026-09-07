"""Static finite-external-q covariant-GW response in the six-site Ruby cell.

This module extends the production q=0 cGW kernel to a discrete external
primitive-cell momentum ``p`` on the same reciprocal mesh as the background
SC-GW calculation.  The response convention is

    X(k;p) = dG(k+p,k)/dh_p = G(k+p) Gamma(k;p) G(k),

with zero external bosonic frequency.  The self-energy tangent is decomposed in
exactly the same way as the production split GW map,

    Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2,

where MT contains ``W-V`` and the AL terms contain full W.

The finite-p polarization tangent is represented in the momentum convention of
the existing GW code.  For an internal bosonic momentum Q, the screened-
interaction tangent entering the self-energy is

    dW(Q-p,Q) = W(Q-p) [L1(Q;p)+L2(Q;p)] W(Q),

with

    L1_ab(Q;p) = int_k X_ab(k+Q-p;p) G_ba(k),
    L2_ab(Q;p) = int_k G_ab(k+Q) X_ba(k;p).

These formulas reduce identically to the established q=0 H/F/MT/AL kernel when
``p=(0,0)``.  The implementation supports both direct and FFT momentum
backends and is regression-tested against the q=0 production path.

Important scope: ``p`` must lie on the discrete k mesh, i.e.
``p=(ip1/nk1, ip2/nk2)`` modulo reciprocal lattice vectors.  This is the
natural finite-q resolution of a periodic finite k mesh and avoids any
interpolation of the interacting self-energy.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import _check_backend
from .supercell_cgw import (
    SupercellVertexOptions,
    SupercellVertexResult,
    _check_vertex_solver,
    _gmres_matrix_free,
    _initial_gamma_field,
    _maxabs,
)
from .supercell_gw import _reverse_fft_spectrum
from .supercell_gw_split import compute_static_fock_matrix


FiniteQVertexOptions = SupercellVertexOptions
FiniteQVertexResult = SupercellVertexResult


def normalize_q_index(
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[int, int]:
    """Return a periodic external momentum index on the background mesh."""
    arr = np.asarray(q_index, dtype=int).reshape(2)
    return int(arr[0] % grid.nk1), int(arr[1] % grid.nk2)


def q_index_from_reduced(
    q: tuple[float, float] | list[float] | np.ndarray,
    grid: MatsubaraGrid,
    atol: float = 1e-10,
) -> tuple[int, int]:
    """Convert reduced reciprocal coordinates to an exact mesh index.

    Raises when q is not commensurate with the current k mesh.  Reciprocal
    lattice shifts are accepted, so e.g. q=-1/nk is equivalent to nk-1.
    """
    q = np.asarray(q, dtype=float).reshape(2)
    raw = np.asarray([q[0] * grid.nk1, q[1] * grid.nk2], dtype=float)
    nearest = np.rint(raw).astype(int)
    if np.max(np.abs(raw - nearest)) > float(atol):
        raise ValueError(
            "external q must lie on the current k mesh: "
            f"q={tuple(q)}, nk=({grid.nk1},{grid.nk2})"
        )
    return normalize_q_index((int(nearest[0]), int(nearest[1])), grid)


def q_reduced_from_index(
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[float, float]:
    ip1, ip2 = normalize_q_index(q_index, grid)
    return float(ip1) / float(grid.nk1), float(ip2) / float(grid.nk2)


def negative_q_index(
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[int, int]:
    ip1, ip2 = normalize_q_index(q_index, grid)
    return (-ip1) % grid.nk1, (-ip2) % grid.nk2


def _x_field_finite_q(
    G: np.ndarray,
    Gamma: np.ndarray,
    q_index: tuple[int, int],
) -> np.ndarray:
    """Return X(k;p)=G(k+p) Gamma(k;p) G(k)."""
    ip1, ip2 = q_index
    Gp = roll_spatial(G, ip1, ip2)
    return np.einsum("...ab,...bc,...cd->...ad", Gp, Gamma, G, optimize=True)


def _hartree_from_x_finite_q(
    X: np.ndarray,
    Vq: np.ndarray,
    q_index: tuple[int, int],
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Finite-q Hartree tangent.

    In the bosonic Fourier convention used by the GW code an electron response
    carrying +p couples to the Hartree interaction at -p.  For the present Ruby
    model V(q) is q independent, but retaining the index makes the convention
    explicit and future-proof.
    """
    norb = int(X.shape[-1])
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    im1, im2 = negative_q_index(q_index, grid)
    diag = Vq[im1, im2] @ response_density
    mat = np.zeros((norb, norb), dtype=complex)
    mat[np.diag_indices(norb)] = diag
    return np.broadcast_to(mat, X.shape).copy()


def _fock_from_x_finite_q(
    X: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str,
) -> np.ndarray:
    """Static bare-V exchange tangent at external p.

    The equal-time off-diagonal density matrix is absolutely convergent because
    X decays as 1/omega^2.  The internal interaction momentum convolution is
    the same as at q=0; the external p dependence is already carried by X.
    """
    delta_rho = grid.T * np.sum(X, axis=0)
    gf_static = compute_static_fock_matrix(delta_rho, Vq, grid, backend=backend)
    return np.broadcast_to(gf_static[None, ...], X.shape).copy()


def _dynamic_corrections_finite_q_direct(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    q_index: tuple[int, int],
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

    ip1, ip2 = q_index
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xbase = X[dst]
        Gsrc = G[src] if include_al else None
        Gbase = G[dst] if include_al else None

        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                # MT differentiates the internal electron line and therefore
                # has the same Q convolution as q=0, with X carrying p.
                Xq = roll_spatial(Xsrc, iq1, iq2)
                if include_mt:
                    WcQ = Wc[im, iq1, iq2]
                    gmt[dst] -= pref * Xq * WcQ.T[None, None, None, :, :]

                if include_al:
                    # The code's bosonic convention uses dW(Q-p,Q).
                    Xq_minus_p = roll_spatial(
                        Xsrc, iq1 - ip1, iq2 - ip2
                    )
                    Gq = roll_spatial(Gsrc, iq1, iq2)
                    L1 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Xq_minus_p, Gbase, optimize=True
                    )
                    L2 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Gq, Xbase, optimize=True
                    )
                    Wleft = W[im, (iq1 - ip1) % grid.nk1, (iq2 - ip2) % grid.nk2]
                    Wright = W[im, iq1, iq2]
                    M1 = Wleft @ L1 @ Wright
                    M2 = Wleft @ L2 @ Wright
                    gal1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                    gal2[dst] -= pref * Gq * M2.T[None, None, None, :, :]

    return gmt, gal1, gal2


def _dynamic_corrections_finite_q_fft(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    q_index: tuple[int, int],
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

    ip1, ip2 = q_index
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xhat = np.fft.fftn(Xsrc, axes=(1, 2))

        if include_mt:
            WcT = np.swapaxes(Wc[im], -1, -2)[None, :, :, :, :]
            Wc_hat_minus = _reverse_fft_spectrum(WcT, axes=(1, 2))
            mt_conv = np.fft.ifftn(Xhat * Wc_hat_minus, axes=(1, 2))
            gmt[dst] -= pref * mt_conv

        if include_al:
            Xbase = X[dst]
            Gsrc = G[src]
            Gbase = G[dst]
            Ghat = np.fft.fftn(Gsrc, axes=(1, 2))

            Gbase_T = np.swapaxes(Gbase, -1, -2)
            Xbase_T = np.swapaxes(Xbase, -1, -2)
            Gbase_hat_minus = _reverse_fft_spectrum(Gbase_T, axes=(1, 2))
            Xbase_hat_minus = _reverse_fft_spectrum(Xbase_T, axes=(1, 2))

            # C1[r] = int_k X(k+r;p) G^T(k), then L1(Q;p)=C1[Q-p].
            C1_product = np.sum(Xhat * Gbase_hat_minus, axis=0)
            C1 = pref * np.fft.ifftn(C1_product, axes=(0, 1))
            L1 = np.roll(C1, shift=(ip1, ip2), axis=(0, 1))

            # L2(Q;p)=int_k G(k+Q) X^T(k;p).
            C2_product = np.sum(Ghat * Xbase_hat_minus, axis=0)
            L2 = pref * np.fft.ifftn(C2_product, axes=(0, 1))

            Wm = W[im]
            Wleft = np.roll(Wm, shift=(ip1, ip2), axis=(0, 1))
            M1 = np.matmul(np.matmul(Wleft, L1), Wm)
            M2 = np.matmul(np.matmul(Wleft, L2), Wm)

            M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
            M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
            M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
            M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))
            gal1[dst] -= pref * np.fft.ifftn(
                Ghat * M1hat_minus, axes=(1, 2)
            )
            gal2[dst] -= pref * np.fft.ifftn(
                Ghat * M2hat_minus, axes=(1, 2)
            )

    return gmt, gal1, gal2


def vertex_corrections_finite_q(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
    opts: FiniteQVertexOptions = FiniteQVertexOptions(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(H,F,MTc,AL1,AL2)`` for one finite-p trial vertex."""
    backend = _check_backend(opts.momentum_backend)
    p = normalize_q_index(q_index, grid)
    X = _x_field_finite_q(G, Gamma, p)
    zero = np.zeros_like(Gamma)
    gh = (
        _hartree_from_x_finite_q(X, Vq, p, grid)
        if opts.include_hartree
        else zero.copy()
    )
    gf = (
        _fock_from_x_finite_q(X, Vq, grid, backend)
        if opts.include_fock
        else zero.copy()
    )
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, :, :, :, :]
    dyn = (
        _dynamic_corrections_finite_q_fft
        if backend == "fft"
        else _dynamic_corrections_finite_q_direct
    )
    gmt, gal1, gal2 = dyn(
        G,
        W,
        Wc,
        X,
        p,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )
    return gh, gf, gmt, gal1, gal2


def _kernel_sum_finite_q(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    q_index: tuple[int, int],
    grid: MatsubaraGrid,
    opts: FiniteQVertexOptions,
):
    parts = vertex_corrections_finite_q(
        G, W, Vq, Gamma, q_index, grid, opts
    )
    total = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
    return total, parts


def _bare_vertex_field(K: np.ndarray, G: np.ndarray) -> np.ndarray:
    arr = np.asarray(K, dtype=complex)
    if arr.shape == G.shape[-2:]:
        return np.broadcast_to(arr, G.shape).copy()
    if arr.shape == G.shape:
        return np.array(arr, copy=True)
    raise ValueError(
        f"K shape {arr.shape} must be orbital matrix {G.shape[-2:]} "
        f"or full vertex field {G.shape}"
    )


def solve_vertex_finite_q(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
    opts: FiniteQVertexOptions = FiniteQVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> FiniteQVertexResult:
    """Solve ``(I-L_p) Gamma_p = K_p`` at one static external mesh momentum."""
    p = normalize_q_index(q_index, grid)
    norb = int(G.shape[-1])
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    solver = _check_vertex_solver(opts.solver)
    Kfield = _bare_vertex_field(K, G)
    Gamma0 = _initial_gamma_field(initial_gamma, Kfield)

    if solver == "gmres":
        def apply_A(field):
            kernel, _ = _kernel_sum_finite_q(
                G, W, Vq, field, p, grid, opts
            )
            return np.asarray(field, dtype=complex) - kernel

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
        Gamma = np.asarray(Gamma0, dtype=complex).copy()
        converged = False
        err = float("inf")
        it = 0
        for it in range(1, int(opts.max_iter) + 1):
            kernel, _ = _kernel_sum_finite_q(
                G, W, Vq, Gamma, p, grid, opts
            )
            equation_residual = Kfield + kernel - Gamma
            err = _maxabs(equation_residual)
            if opts.verbose:
                qred = q_reduced_from_index(p, grid)
                print(
                    f"finite-q cGW linear iter {it:4d}: q={qred}, "
                    f"residual_max={err:.3e}"
                )
            if err < float(opts.tol):
                converged = True
                break
            Gamma += float(opts.mixing) * equation_residual

    kernel, parts = _kernel_sum_finite_q(G, W, Vq, Gamma, p, grid, opts)
    gh, gf, gmt, gal1, gal2 = parts
    err = _maxabs(Kfield + kernel - Gamma)
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


def susceptibility_matrix_finite_q(
    G: np.ndarray,
    left_vertices: np.ndarray,
    gammas: list[np.ndarray] | np.ndarray,
    q_index: tuple[int, int] | list[int] | np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return chi_ab(p)=-int_k Tr[K_a G(k+p) Gamma_b(k;p) G(k)].

    ``left_vertices`` are constant orbital matrices.  This is the appropriate
    cell-periodic form-factor convention for the local x/y/z pseudospin
    operators used in this repository.
    """
    p = normalize_q_index(q_index, grid)
    K = np.asarray(left_vertices, dtype=complex)
    if K.ndim == 2:
        K = K[None, :, :]
    nleft = int(K.shape[0])
    nright = len(gammas)
    chi = np.zeros((nleft, nright), dtype=complex)
    Gp = roll_spatial(G, p[0], p[1])
    pref = -(grid.T / grid.nk)
    for b in range(nright):
        Gamma = np.asarray(gammas[b], dtype=complex)
        chi[:, b] = pref * np.einsum(
            "iab,nxybc,nxycd,nxyda->i",
            K,
            Gp,
            Gamma,
            G,
            optimize=True,
        )
    return chi


def hermitianize_q_pair(
    chi_q: np.ndarray,
    chi_minus_q: np.ndarray,
) -> np.ndarray:
    """Return the static Hermitian response from the q/-q pair.

    Exact equilibrium response obeys chi_ab(q)=chi_ba(-q)^*.  Averaging the two
    numerical estimates removes finite-box / iterative noise without assuming
    inversion symmetry.
    """
    a = np.asarray(chi_q, dtype=complex)
    b = np.asarray(chi_minus_q, dtype=complex)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("q-pair Hermitianization requires square matrices of equal shape")
    return 0.5 * (a + b.conj().T)


__all__ = [
    "FiniteQVertexOptions",
    "FiniteQVertexResult",
    "normalize_q_index",
    "q_index_from_reduced",
    "q_reduced_from_index",
    "negative_q_index",
    "vertex_corrections_finite_q",
    "solve_vertex_finite_q",
    "susceptibility_matrix_finite_q",
    "hermitianize_q_pair",
]
