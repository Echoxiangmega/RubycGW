"""Self-consistent GW for the 18-site period-three Ruby supercell.

This module keeps Bloch momentum in the reduced supercell Brillouin zone and
allows the three primitive cells inside the supercell to acquire inequivalent
densities.  The matrix dimension is therefore 18 rather than 6, while the GW
equations and Matsubara/momentum convolutions are unchanged.

The implementation intentionally reuses the existing ``GWOptions`` / ``GWResult``
containers and Pulay mixer, but all matrix operations here infer the orbital
dimension from the supplied Hamiltonian instead of assuming six sublattices.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .model import RubyParameters
from .supercell import NSUP, build_supercell_h0, build_supercell_interaction


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    high = x > 40.0
    low = x < -40.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def _reverse_fft_spectrum(field: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return np.conj(np.fft.fftn(np.conj(field), axes=axes))


def build_g0_inverse_matrix(h0: np.ndarray, grid: MatsubaraGrid, mu: float) -> np.ndarray:
    norb = int(h0.shape[-1])
    if h0.shape[-2] != norb:
        raise ValueError("h0 must be square on its last two axes")
    eye = np.eye(norb, dtype=complex)
    return (
        (1j * grid.omega[:, None, None, None, None] + float(mu))
        * eye[None, None, None, :, :]
        - h0[None, :, :, :, :]
    )


def dyson_from_sigma_matrix(
    h0: np.ndarray,
    grid: MatsubaraGrid,
    mu: float,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
) -> np.ndarray:
    invg = build_g0_inverse_matrix(h0, grid, mu)
    invg -= sigma_h[None, None, None, :, :]
    invg -= sigma_gw
    return np.linalg.inv(invg)


def density_from_G_matrix(
    G: np.ndarray,
    grid: MatsubaraGrid,
    h0: np.ndarray | None = None,
    mu: float | None = None,
    sigma_h: np.ndarray | None = None,
) -> np.ndarray:
    """Orbital density with the same analytic tail subtraction as 6-site GW."""
    norb = int(G.shape[-1])
    diag = np.diagonal(G, axis1=-2, axis2=-1)

    if h0 is None or mu is None:
        return 0.5 + (grid.T / grid.nk) * np.sum(diag, axis=(0, 1, 2)).real

    if sigma_h is None:
        sigma_h = np.zeros((norb, norb), dtype=complex)

    href = h0 + sigma_h[None, None, :, :]
    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)
    weights = np.abs(evecs) ** 2

    occ = _fermi(evals - float(mu), grid.T)
    n_ref = np.sum(weights * occ[..., None, :], axis=(0, 1, 3)) / grid.nk

    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    gref_diag = np.einsum(
        "xyaj,nxyj->nxya", weights, 1.0 / denom, optimize=True
    )
    correction = (
        (grid.T / grid.nk)
        * np.sum(diag - gref_diag, axis=(0, 1, 2)).real
    )
    return n_ref + correction


def hartree_self_energy_matrix(density: np.ndarray, Vq0: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=float)
    norb = int(density.size)
    sigma = np.zeros((norb, norb), dtype=complex)
    sigma[np.diag_indices(norb)] = Vq0 @ density
    return sigma


def compute_polarization_matrix_direct(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    norb = int(G.shape[-1])
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc = G[src]
        Gbase = G[dst]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq = roll_spatial(Gsrc, iq1, iq2)
                P[im, iq1, iq2] = pref * np.einsum(
                    "nxyab,nxyba->ab", Gq, Gbase, optimize=True
                )
    return P


def compute_polarization_matrix_fft(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    norb = int(G.shape[-1])
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        A = G[src]
        B = np.swapaxes(G[dst], -1, -2)
        Ahat = np.fft.fftn(A, axes=(1, 2))
        Bhat_minus = _reverse_fft_spectrum(B, axes=(1, 2))
        product = np.sum(Ahat * Bhat_minus, axis=0)
        P[im] = pref * np.fft.ifftn(product, axes=(0, 1))
    return P


def compute_polarization_matrix(
    G: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    backend = _check_backend(backend)
    if backend == "fft":
        return compute_polarization_matrix_fft(G, grid)
    return compute_polarization_matrix_direct(G, grid)


def _screening_lhs_matrix(P: np.ndarray, Vq: np.ndarray) -> np.ndarray:
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    Vbatch = Vq[None, :, :, :, :]
    return eye[None, None, None, :, :] - np.matmul(Vbatch, P)


def compute_screened_interaction_matrix(P: np.ndarray, Vq: np.ndarray) -> np.ndarray:
    lhs = _screening_lhs_matrix(P, Vq)
    rhs = np.broadcast_to(Vq[None, :, :, :, :], P.shape)
    return np.linalg.solve(lhs, rhs)


def _canonicalize_mode(vector: np.ndarray) -> np.ndarray:
    mode = np.asarray(vector, dtype=complex).reshape(-1).copy()
    norm = float(np.linalg.norm(mode))
    if not np.isfinite(norm) or norm < 1e-14:
        return np.zeros_like(mode)
    mode /= norm
    pivot = int(np.argmax(np.abs(mode)))
    mode *= np.exp(-1j * np.angle(mode[pivot]))
    if mode[pivot].real < 0.0:
        mode *= -1.0
    if abs(mode[pivot].imag) < 1e-14:
        mode[pivot] = complex(mode[pivot].real, 0.0)
    return mode


def screening_soft_modes_matrix(
    P: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[float, int, float, float, float, np.ndarray, np.ndarray, float]:
    lhs = _screening_lhs_matrix(P, Vq)
    svals = np.linalg.svd(lhs, compute_uv=False)
    smin_grid = svals[..., -1]
    flat = int(np.argmin(smin_grid))
    im, iq1, iq2 = np.unravel_index(flat, smin_grid.shape)

    _, _, vh = np.linalg.svd(lhs[im, iq1, iq2], full_matrices=False)
    screening_mode = _canonicalize_mode(vh[-1].conj())

    psel = P[im, iq1, iq2]
    vsel = Vq[iq1, iq2]
    norb = int(psel.shape[-1])
    density_raw = psel @ screening_mode
    if np.linalg.norm(density_raw) < 1e-14:
        density_lhs = np.eye(norb, dtype=complex) - psel @ vsel
        _, _, vh_density = np.linalg.svd(density_lhs, full_matrices=False)
        density_raw = vh_density[-1].conj()
    density_mode = _canonicalize_mode(density_raw)
    density_lhs = np.eye(norb, dtype=complex) - psel @ vsel
    density_residual = float(np.max(np.abs(density_lhs @ density_mode)))

    q = grid.qmesh()[iq1, iq2]
    return (
        float(smin_grid[im, iq1, iq2]),
        int(grid.m_values[im]),
        float(grid.Omega[im]),
        float(q[0]),
        float(q[1]),
        screening_mode,
        density_mode,
        density_residual,
    )


def compute_sigma_gw_matrix_direct(
    G: np.ndarray,
    W: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    sigma = np.zeros_like(G)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc = G[src]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq = roll_spatial(Gsrc, iq1, iq2)
                sigma[dst] -= pref * Gq * W[im, iq1, iq2].T[None, None, None, :, :]
    return sigma


def compute_sigma_gw_matrix_fft(
    G: np.ndarray,
    W: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    sigma = np.zeros_like(G)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        A = G[src]
        B = np.swapaxes(W[im], -1, -2)[None, :, :, :, :]
        Ahat = np.fft.fftn(A, axes=(1, 2))
        Bhat_minus = _reverse_fft_spectrum(B, axes=(1, 2))
        conv = np.fft.ifftn(Ahat * Bhat_minus, axes=(1, 2))
        sigma[dst] -= pref * conv
    return sigma


def compute_sigma_gw_matrix(
    G: np.ndarray,
    W: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    backend = _check_backend(backend)
    if backend == "fft":
        return compute_sigma_gw_matrix_fft(G, W, grid)
    return compute_sigma_gw_matrix_direct(G, W, grid)


def _solve_mu_matrix(
    h0: np.ndarray,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    grid: MatsubaraGrid,
    target: float,
    mu0: float,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray]:
    def f(mu):
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        density = density_from_G_matrix(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
        return float(np.sum(density) - target), G

    width = 2.0
    lo, hi = float(mu0) - width, float(mu0) + width
    flo, _ = f(lo)
    fhi, _ = f(hi)
    for _ in range(30):
        if flo <= 0 <= fhi:
            break
        width *= 2.0
        lo, hi = float(mu0) - width, float(mu0) + width
        flo, _ = f(lo)
        fhi, _ = f(hi)
    else:
        raise RuntimeError(
            "Could not bracket supercell chemical potential; "
            f"target={target}, f(lo)={flo}, f(hi)={fhi}"
        )

    Gmid = None
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid, Gmid = f(mid)
        if abs(fmid) < tol:
            return float(mid), Gmid
        if fmid > 0:
            hi = mid
        else:
            lo = mid
    return float(mid), Gmid


def _compatible_initial(initial: GWResult | None, grid: MatsubaraGrid, norb: int) -> bool:
    if initial is None:
        return False
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    return initial.Sigma_GW.shape == expected and initial.Sigma_H.shape == (norb, norb)


def solve_matrix_gw(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
) -> GWResult:
    """Solve matrix-valued periodic GW for an arbitrary orbital dimension."""
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    if opts.pulay_history < 2:
        raise ValueError("pulay_history must be at least 2")
    if opts.pulay_start < 1:
        raise ValueError("pulay_start must be at least 1")

    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    expected_v = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h:
        raise ValueError(f"h0 shape {h0.shape} != expected {expected_h}")
    if Vq.shape != expected_v:
        raise ValueError(f"Vq shape {Vq.shape} != expected {expected_v}")
    Vq0 = Vq[0, 0]

    if _compatible_initial(initial, grid, norb):
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
        mu = float(opts.mu)

    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    if opts.target_filling is not None:
        mu, G = _solve_mu_matrix(
            h0, sigma_h, sigma_gw, grid, float(opts.target_filling),
            mu, opts.mu_tol, opts.mu_max_iter,
        )

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    err = float("inf")
    converged = False
    it = 0
    for it in range(1, opts.max_iter + 1):
        density = density_from_G_matrix(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)

        if opts.verbose:
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, method={method}, backend={backend}"
            )
        if err < opts.tol:
            converged = True
            break

        sigma_h_next, sigma_gw_next = _mixed_self_energies(
            sigma_h, sigma_gw, sigma_h_out, sigma_gw_out,
            opts, it, history,
        )

        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(h0, grid, mu, sigma_h_next, sigma_gw_next)
        else:
            mu, Gnext = _solve_mu_matrix(
                h0, sigma_h_next, sigma_gw_next, grid, float(opts.target_filling),
                mu, opts.mu_tol, opts.mu_max_iter,
            )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext

    # Re-evaluate the fixed-point map on exactly the returned iterate.
    density = density_from_G_matrix(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < opts.tol)

    (
        smin, mmin, omin, q1min, q2min,
        screening_mode, density_mode, density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWResult(
        G=G,
        W=W,
        P=P,
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        mu=mu,
        density=density,
        converged=converged,
        iterations=it,
        final_error=float(err),
        mixing_method=method,
        min_screening_singular_value=smin,
        min_screening_m=mmin,
        min_screening_Omega=omin,
        min_screening_q1=q1min,
        min_screening_q2=q2min,
        min_screening_mode=screening_mode,
        min_density_mode=density_mode,
        min_density_mode_residual=density_mode_residual,
    )


def solve_supercell_gw(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    source_strength: float = 0.0,
    initial: GWResult | None = None,
) -> GWResult:
    """Solve GW in the 18-site Q=(1/3,1/3)-compatible supercell."""
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=source_strength)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw(h0, Vq, grid, opts=opts, initial=initial)
