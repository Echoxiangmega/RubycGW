"""Covariant-GW current response on the 18-site period-three Ruby supercell.

The supercell SC-GW self-energy is evaluated as

    Sigma = Sigma_H + Sigma_F + Sigma_c,
    Sigma_c = - G * (W - V),

with the instantaneous bare interaction treated as a static Fock term.  The
functional derivative used here follows the same decomposition:

    Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2.

``Gamma_MT,c`` contains ``W-V``.  ``Gamma_F`` is the equal-time bare-V
exchange derivative.  The AL terms continue to contain the full screened
interaction W because delta(W-V)=delta W=W (delta P) W.

All external response momenta in this module are q_sc=0.  Primitive-cell
q=0,+Q,-Q current harmonics are nevertheless all available because
Q=(1/3,1/3) folds to q_sc=0 in the index-three supercell.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import _check_backend
from .model import NSUB, eta_vertices
from .supercell import NSECTOR, NSUP
from .supercell_gw import _reverse_fft_spectrum
from .supercell_gw_split import compute_static_fock_matrix


@dataclass(frozen=True)
class SupercellVertexOptions:
    max_iter: int = 150
    tol: float = 1e-8
    mixing: float = 0.25
    include_hartree: bool = True
    include_fock: bool = True
    include_mt: bool = True
    include_al: bool = True
    verbose: bool = True
    momentum_backend: str = "fft"


@dataclass
class SupercellVertexResult:
    Gamma: np.ndarray
    Gamma_H: np.ndarray
    Gamma_F: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    converged: bool
    iterations: int
    final_error: float


def _x_field(G: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return np.einsum("...ab,...bc,...cd->...ad", G, Gamma, G, optimize=True)


def _hartree_from_x(
    X: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    norb = int(X.shape[-1])
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    diag = Vq0 @ response_density
    mat = np.zeros((norb, norb), dtype=complex)
    mat[np.diag_indices(norb)] = diag
    return np.broadcast_to(mat, X.shape).copy()


def _fock_from_x(
    X: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str,
) -> np.ndarray:
    """Derivative of the static Fock self-energy.

    Since X=G Gamma G decays as 1/omega^2, its equal-time response is
    absolutely convergent and can be summed directly over the fermionic box:

        delta rho(k) = T sum_n X(k,iw_n),
        Gamma_F(k) = -(1/Nk) sum_q delta rho(k+q) o V(q)^T.
    """
    delta_rho = grid.T * np.sum(X, axis=0)
    gf_static = compute_static_fock_matrix(delta_rho, Vq, grid, backend=backend)
    return np.broadcast_to(gf_static[None, ...], X.shape).copy()


def _dynamic_corrections_direct(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
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
                if include_mt:
                    WcQ = Wc[im, iq1, iq2]
                    gmt[dst] -= (
                        pref * Xq * WcQ.T[None, None, None, :, :]
                    )

                if include_al:
                    Gq = roll_spatial(Gsrc, iq1, iq2)
                    WQ = W[im, iq1, iq2]
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

    return gmt, gal1, gal2


def _dynamic_corrections_fft(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
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

            L1_product = np.sum(Xhat * Gbase_hat_minus, axis=0)
            L1 = pref * np.fft.ifftn(L1_product, axes=(0, 1))
            L2_product = np.sum(Ghat * Xbase_hat_minus, axis=0)
            L2 = pref * np.fft.ifftn(L2_product, axes=(0, 1))

            Wm = W[im]
            M1 = np.matmul(np.matmul(Wm, L1), Wm)
            M2 = np.matmul(np.matmul(Wm, L2), Wm)

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


def vertex_corrections_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    backend = _check_backend(opts.momentum_backend)
    X = _x_field(G, Gamma)
    zero = np.zeros_like(Gamma)
    gh = _hartree_from_x(X, Vq[0, 0], grid) if opts.include_hartree else zero.copy()
    gf = _fock_from_x(X, Vq, grid, backend) if opts.include_fock else zero.copy()
    Wc = W - Vq[None, :, :, :, :]
    dyn = _dynamic_corrections_fft if backend == "fft" else _dynamic_corrections_direct
    gmt, gal1, gal2 = dyn(
        G,
        W,
        Wc,
        X,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )
    return gh, gf, gmt, gal1, gal2


def _initial_gamma_field(initial_gamma: np.ndarray | None, Kfield: np.ndarray) -> np.ndarray:
    if initial_gamma is None:
        return Kfield.copy()
    arr = np.asarray(initial_gamma, dtype=complex)
    if arr.shape == Kfield.shape:
        return np.array(arr, copy=True)
    if arr.shape == Kfield.shape[-2:]:
        return np.broadcast_to(arr, Kfield.shape).copy()
    return Kfield.copy()


def solve_vertex_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> SupercellVertexResult:
    """Solve one q_sc=0 current vertex on an arbitrary matrix dimension."""
    norb = int(G.shape[-1])
    if K.shape != (norb, norb):
        raise ValueError(f"K shape {K.shape} != {(norb, norb)}")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    Kfield = np.broadcast_to(K, G.shape).copy()
    Gamma = _initial_gamma_field(initial_gamma, Kfield)
    gh = np.zeros_like(Gamma)
    gf = np.zeros_like(Gamma)
    gmt = np.zeros_like(Gamma)
    gal1 = np.zeros_like(Gamma)
    gal2 = np.zeros_like(Gamma)
    converged = False
    err = float("inf")

    for it in range(1, int(opts.max_iter) + 1):
        gh, gf, gmt, gal1, gal2 = vertex_corrections_q0(
            G, W, Vq, Gamma, grid, opts
        )
        rhs = Kfield + gh + gf + gmt + gal1 + gal2
        Gnew = (1.0 - float(opts.mixing)) * Gamma + float(opts.mixing) * rhs
        err = float(np.max(np.abs(Gnew - Gamma)))
        Gamma = Gnew
        if opts.verbose:
            print(
                f"supercell cGW vertex iter {it:4d}: err={err:.3e}, "
                f"backend={opts.momentum_backend}"
            )
        if err < float(opts.tol):
            converged = True
            break

    return SupercellVertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_F=gf,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        converged=converged,
        iterations=it,
        final_error=float(err),
    )


def supercell_current_vertices() -> tuple[np.ndarray, list[str]]:
    """Return the six local current vertices [+,s=0..2; -,s=0..2].

    ``+`` retains the project convention PHYSICAL OPPOSITE circulation and
    ``-`` retains PHYSICAL SAME circulation.
    """
    _, _, kp, km = eta_vertices()
    vertices = []
    labels = []
    for label, k6 in (("opposite", kp), ("same", km)):
        for s in range(NSECTOR):
            mat = np.zeros((NSUP, NSUP), dtype=complex)
            sl = slice(NSUB * s, NSUB * (s + 1))
            mat[sl, sl] = k6
            vertices.append(mat)
            labels.append(f"{label}_s{s}")
    return np.stack(vertices, axis=0), labels


def sector_harmonic_matrix() -> np.ndarray:
    """Orthogonal real transform (s0,s1,s2) -> (q0,Qc,Qs)."""
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=float,
    ) / np.array([[np.sqrt(3.0)], [np.sqrt(6.0)], [np.sqrt(2.0)]])


def current_harmonic_transform() -> tuple[np.ndarray, list[str]]:
    """6x6 transform local currents -> q0/Qc/Qs in each physical channel."""
    u = sector_harmonic_matrix()
    T = np.zeros((2 * NSECTOR, 2 * NSECTOR), dtype=float)
    T[:NSECTOR, :NSECTOR] = u
    T[NSECTOR:, NSECTOR:] = u
    labels = [
        "opposite_q0", "opposite_Qc", "opposite_Qs",
        "same_q0", "same_Qc", "same_Qs",
    ]
    return T, labels


def susceptibility_matrix_q0(
    G: np.ndarray,
    left_vertices: np.ndarray,
    gammas: list[np.ndarray] | np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return chi_ab=-int_k Tr[K_a G Gamma_b G]."""
    K = np.asarray(left_vertices, dtype=complex)
    nchan = int(K.shape[0])
    if len(gammas) != nchan:
        raise ValueError("number of right vertices must match left vertices")
    chi = np.zeros((nchan, nchan), dtype=complex)
    pref = -(grid.T / grid.nk)
    for b in range(nchan):
        Gamma = np.asarray(gammas[b], dtype=complex)
        chi[:, b] = pref * np.einsum(
            "iab,nxybc,nxycd,nxyda->i",
            K,
            G,
            Gamma,
            G,
            optimize=True,
        )
    return chi


def physical_symmetric_susceptibility(chi: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the real symmetric static response and discarded imaginary scale."""
    chi = np.asarray(chi, dtype=complex)
    herm = 0.5 * (chi + chi.conj().T)
    imag_max = float(np.max(np.abs(herm.imag)))
    return np.asarray(herm.real, dtype=float), imag_max


def curvature_from_susceptibility(
    chi: np.ndarray,
    rcond: float = 1e-12,
) -> dict:
    """Analyze R=chi^{-1} in the six local-current basis."""
    chi_sym, imag_max = physical_symmetric_susceptibility(chi)
    evals_chi, evecs = np.linalg.eigh(chi_sym)
    scale = max(float(np.max(np.abs(evals_chi))), 1.0)
    cutoff = float(rcond) * scale
    if np.any(np.abs(evals_chi) <= cutoff):
        raise np.linalg.LinAlgError(
            "current susceptibility is singular within requested rcond"
        )

    R = np.linalg.inv(chi_sym)
    R = 0.5 * (R + R.T)
    evals_r, evecs_r = np.linalg.eigh(R)
    soft = evecs_r[:, 0]
    soft = soft / np.linalg.norm(soft)

    T, harmonic_labels = current_harmonic_transform()
    chi_harm = T @ chi_sym @ T.T
    R_harm = T @ R @ T.T
    soft_harm = T @ soft

    uniform_idx = np.array([0, 3], dtype=int)
    chi_uniform = chi_harm[np.ix_(uniform_idx, uniform_idx)]
    R_uniform_relaxed = np.linalg.inv(chi_uniform)
    R_uniform_constrained = R_harm[np.ix_(uniform_idx, uniform_idx)]

    w_opposite = float(np.sum(np.abs(soft[:NSECTOR]) ** 2))
    w_same = float(np.sum(np.abs(soft[NSECTOR:]) ** 2))
    w_q0 = float(abs(soft_harm[0]) ** 2 + abs(soft_harm[3]) ** 2)
    w_Q = float(np.sum(np.abs(soft_harm[[1, 2, 4, 5]]) ** 2))

    return {
        "chi_symmetric": chi_sym,
        "chi_imag_max": imag_max,
        "chi_eigenvalues": evals_chi,
        "chi_eigenvectors": evecs,
        "R": R,
        "R_eigenvalues": evals_r,
        "R_eigenvectors": evecs_r,
        "soft_vector_local": soft,
        "harmonic_transform": T,
        "harmonic_labels": harmonic_labels,
        "chi_harmonic": chi_harm,
        "R_harmonic": R_harm,
        "soft_vector_harmonic": soft_harm,
        "chi_uniform": chi_uniform,
        "R_uniform_relaxed": R_uniform_relaxed,
        "R_uniform_constrained": R_uniform_constrained,
        "soft_weight_opposite": w_opposite,
        "soft_weight_same": w_same,
        "soft_weight_q0": w_q0,
        "soft_weight_Q": w_Q,
    }
