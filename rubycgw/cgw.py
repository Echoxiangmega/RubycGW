"""q=(0,0) covariant-GW vertex solver for eta_+/-.

The default ``momentum_backend='fft'`` evaluates the periodic two-dimensional
momentum convolutions by FFT while retaining the Matsubara sums explicitly.
The ``direct`` backend preserves the transparent explicit-q implementation for
validation.  Hartree, MT, AL1, and AL2 still follow exactly the same equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .model import NSUB


@dataclass
class VertexOptions:
    max_iter: int = 100
    tol: float = 1e-8
    mixing: float = 0.25
    include_hartree: bool = True
    include_mt: bool = True
    include_al: bool = True
    verbose: bool = True
    momentum_backend: str = "fft"  # "fft" or "direct"


@dataclass
class VertexResult:
    Gamma: np.ndarray
    Gamma_H: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    converged: bool
    iterations: int


def _check_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"fft", "direct"}:
        raise ValueError("momentum_backend must be 'fft' or 'direct'")
    return backend


def _reverse_fft_spectrum(field: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Return F_hat(-r), without complex-conjugating the physical field."""
    return np.conj(np.fft.fftn(np.conj(field), axes=axes))


def _x_field(G: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return np.einsum("...ab,...bc,...cd->...ad", G, Gamma, G, optimize=True)


def _hartree_from_x(X: np.ndarray, Vq0: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    diag = Vq0 @ response_density
    mat = np.zeros((NSUB, NSUB), dtype=complex)
    mat[np.diag_indices(NSUB)] = diag
    return np.broadcast_to(mat, X.shape).copy()


def _vertex_corrections_q0_direct(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
    include_hartree: bool,
    include_mt: bool,
    include_al: bool,
):
    """Transparent explicit-q reference implementation."""
    X = _x_field(G, Gamma)
    zero = np.zeros_like(Gamma)
    gh = _hartree_from_x(X, Vq0, grid) if include_hartree else zero.copy()
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()

    if not include_mt and not include_al:
        return gh, gmt, gal1, gal2

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
                Xq = roll_spatial(Xsrc, iq1, iq2)
                WQ = W[im, iq1, iq2]

                if include_mt:
                    gmt[dst] -= (
                        pref * Xq
                        * WQ.T[None, None, None, :, :]
                    )

                if include_al:
                    Gq = roll_spatial(Gsrc, iq1, iq2)
                    L1 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Xq, Gbase, optimize=True
                    )
                    L2 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Gq, Xbase, optimize=True
                    )
                    M1 = WQ @ L1 @ WQ
                    M2 = WQ @ L2 @ WQ
                    gal1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                    gal2[dst] -= pref * Gq * M2.T[None, None, None, :, :]

    return gh, gmt, gal1, gal2


def _vertex_corrections_q0_fft(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
    include_hartree: bool,
    include_mt: bool,
    include_al: bool,
):
    """FFT implementation of the q=0 Hartree/MT/AL corrections.

    For every bosonic Matsubara index m, all q points are evaluated at once.
    The frequency window is still treated by ``frequency_shift_slices`` exactly
    as in the direct code, so FFT changes only the periodic 2D momentum sums.
    """
    X = _x_field(G, Gamma)
    zero = np.zeros_like(Gamma)
    gh = _hartree_from_x(X, Vq0, grid) if include_hartree else zero.copy()
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()

    if not include_mt and not include_al:
        return gh, gmt, gal1, gal2

    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xhat = np.fft.fftn(Xsrc, axes=(1, 2))
        Wm = W[im]

        if include_mt:
            WT = np.swapaxes(Wm, -1, -2)[None, :, :, :, :]
            WThat_minus = _reverse_fft_spectrum(WT, axes=(1, 2))
            mt_conv = np.fft.ifftn(Xhat * WThat_minus, axes=(1, 2))
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

            # L1(Q)_ef = int_k X_ef(k+Q) G_fe(k)
            L1_product = np.sum(Xhat * Gbase_hat_minus, axis=0)
            L1 = pref * np.fft.ifftn(L1_product, axes=(0, 1))

            # L2(Q)_ef = int_k G_ef(k+Q) X_fe(k)
            L2_product = np.sum(Ghat * Xbase_hat_minus, axis=0)
            L2 = pref * np.fft.ifftn(L2_product, axes=(0, 1))

            # Batch matrix products over all q.
            M1 = np.matmul(np.matmul(Wm, L1), Wm)
            M2 = np.matmul(np.matmul(Wm, L2), Wm)

            M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
            M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
            M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
            M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))

            al1_conv = np.fft.ifftn(Ghat * M1hat_minus, axes=(1, 2))
            al2_conv = np.fft.ifftn(Ghat * M2hat_minus, axes=(1, 2))
            gal1[dst] -= pref * al1_conv
            gal2[dst] -= pref * al2_conv

    return gh, gmt, gal1, gal2


def _vertex_corrections_q0(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
    include_hartree: bool,
    include_mt: bool,
    include_al: bool,
    backend: str,
):
    backend = _check_backend(backend)
    fn = _vertex_corrections_q0_fft if backend == "fft" else _vertex_corrections_q0_direct
    return fn(
        G, W, Gamma, Vq0, grid,
        include_hartree=include_hartree,
        include_mt=include_mt,
        include_al=include_al,
    )


def gamma_h_q0(G: np.ndarray, Gamma: np.ndarray, Vq0: np.ndarray,
               grid: MatsubaraGrid) -> np.ndarray:
    """Hartree vertex at external q=(0,0)."""
    return _hartree_from_x(_x_field(G, Gamma), Vq0, grid)


def gamma_mt_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid, backend: str = "fft") -> np.ndarray:
    """MT correction using either the FFT or direct momentum backend."""
    zero_v = np.zeros((NSUB, NSUB), dtype=complex)
    _, gmt, _, _ = _vertex_corrections_q0(
        G, W, Gamma, zero_v, grid,
        include_hartree=False,
        include_mt=True,
        include_al=False,
        backend=backend,
    )
    return gmt


def gamma_al_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid, backend: str = "fft"):
    """Return AL1 and AL2 using either the FFT or direct momentum backend."""
    zero_v = np.zeros((NSUB, NSUB), dtype=complex)
    _, _, al1, al2 = _vertex_corrections_q0(
        G, W, Gamma, zero_v, grid,
        include_hartree=False,
        include_mt=False,
        include_al=True,
        backend=backend,
    )
    return al1, al2


def _initial_gamma_field(initial_gamma: np.ndarray | None,
                         Kfield: np.ndarray) -> np.ndarray:
    if initial_gamma is None:
        return Kfield.copy()
    arr = np.asarray(initial_gamma)
    if arr.shape == Kfield.shape:
        return np.array(arr, copy=True)
    if arr.shape == Kfield.shape[-2:]:
        return np.broadcast_to(arr, Kfield.shape).copy()
    return Kfield.copy()


def solve_vertex_q0(G: np.ndarray, W: np.ndarray, Vq0: np.ndarray,
                    K: np.ndarray, grid: MatsubaraGrid,
                    opts: VertexOptions = VertexOptions(),
                    initial_gamma: np.ndarray | None = None) -> VertexResult:
    """Solve the q=(0,0) cGW eta vertex by fixed-point iteration.

    ``initial_gamma`` can be a previous converged vertex with the same
    fermionic grid shape.  It is particularly useful for continuation scans and
    for starting the full MT+AL solve from the already converged MT-only vertex.
    """
    backend = _check_backend(opts.momentum_backend)
    Kfield = np.broadcast_to(K, G.shape).copy()
    Gamma = _initial_gamma_field(initial_gamma, Kfield)
    gh = np.zeros_like(Gamma)
    gmt = np.zeros_like(Gamma)
    gal1 = np.zeros_like(Gamma)
    gal2 = np.zeros_like(Gamma)
    converged = False

    for it in range(1, opts.max_iter + 1):
        gh, gmt, gal1, gal2 = _vertex_corrections_q0(
            G, W, Gamma, Vq0, grid,
            include_hartree=opts.include_hartree,
            include_mt=opts.include_mt,
            include_al=opts.include_al,
            backend=backend,
        )
        rhs = Kfield + gh + gmt + gal1 + gal2
        Gnew = (1.0 - opts.mixing) * Gamma + opts.mixing * rhs
        err = float(np.max(np.abs(Gnew - Gamma)))
        Gamma = Gnew
        if opts.verbose:
            print(f"cGW vertex iter {it:4d}: err={err:.3e}, backend={backend}")
        if err < opts.tol:
            converged = True
            break

    return VertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        converged=converged,
        iterations=it,
    )
