"""Covariant bare-SOX derivative at a general bosonic transfer Q.

The equilibrium SOX skeleton is local in the two endpoint times but its
functional derivative at finite external momentum/frequency is not obtained by
simply reusing the static q=0 X(tau).  This module keeps track of the phase of
the endpoint on which each of the three internal Green functions is varied.

For

    X(k;Q) = G(k+Q) Gamma(k;Q) G(k)

we factor the external phase at the first endpoint of the *output* self-energy.
The first SOX line then has no extra phase.  The reversed middle line carries
exp(+i Omega tau) and a spatial phase from its first endpoint, while the third
line carries only the corresponding spatial endpoint phase.  At Q=0 the
implementation reduces exactly to :func:`sox_covariant.compute_sox_vertex_periodic`.

The X(tau) transform uses an analytic static-reference tail for arbitrary
(q,iOmega), so the transfer kernel differentiates the same high-frequency
completion used by the production SOX path rather than reverting to a bare
finite fermionic box.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .sox_covariant import (
    SOXOptions,
    full_periodic_to_kfield,
    kfield_to_full_periodic,
    reconstruct_tau_pair,
    reference_green_iomega,
)
from .transfer_cgw import (
    estimate_transfer_static_vertex,
    normalize_q_index,
    transfer_x_field,
)


def _fermi(xi: np.ndarray, T: float) -> np.ndarray:
    x = np.clip(np.asarray(xi, dtype=float) / float(T), -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def _static_kfield(field: np.ndarray, grid: MatsubaraGrid, norb: int) -> np.ndarray:
    arr = np.asarray(field, dtype=complex)
    if arr.shape == (norb, norb):
        return np.broadcast_to(arr, (grid.nk1, grid.nk2, norb, norb)).copy()
    expected = (grid.nk1, grid.nk2, norb, norb)
    if arr.shape != expected:
        raise ValueError(f"static field shape {arr.shape} != {expected}")
    return np.array(arr, copy=True)


def _reference_transfer_tau(
    h_ref: np.ndarray,
    mu: float,
    source_static: np.ndarray,
    q_index: tuple[int, int],
    m_ext: int,
    tau: float,
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-Matsubara reference X(+tau), X(-tau) for one transfer."""
    h = np.asarray(h_ref, dtype=complex)
    norb = int(h.shape[-1])
    S = _static_kfield(source_static, grid, norb)
    p1, p2 = normalize_q_index(q_index, grid)
    t = float(tau)
    beta = 1.0 / float(grid.T)
    if not 0.0 < t < beta:
        raise ValueError("tau must lie in (0,beta)")

    href = 0.5 * (h + np.swapaxes(h.conj(), -1, -2))
    eval0, U0 = np.linalg.eigh(href)
    evalq = np.roll(eval0, shift=(-p1, -p2), axis=(0, 1))
    Uq = np.roll(U0, shift=(-p1, -p2), axis=(0, 1))
    xi0 = eval0 - float(mu)
    xiq = evalq - float(mu)
    f0 = _fermi(xi0, grid.T)
    fq = _fermi(xiq, grid.T)

    gp0 = -np.exp(-xi0 * t) * (1.0 - f0)
    gpq = -np.exp(-xiq * t) * (1.0 - fq)
    gm0 = np.exp(xi0 * t) * f0
    gmq = np.exp(xiq * t) * fq
    dgp0 = np.exp(-xi0 * t) * (1.0 - f0) * (t - beta * f0)
    dgpq = np.exp(-xiq * t) * (1.0 - fq) * (t - beta * fq)
    dgm0 = np.exp(xi0 * t) * f0 * (t - beta * (1.0 - f0))
    dgmq = np.exp(xiq * t) * fq * (t - beta * (1.0 - fq))

    Omega = 2.0 * np.pi * float(grid.T) * float(m_ext)
    denom = 1j * Omega + eval0[..., None, :] - evalq[..., :, None]
    coeff_p = np.empty_like(denom, dtype=complex)
    coeff_m = np.empty_like(denom, dtype=complex)
    num_p = gp0[..., None, :] - np.exp(1j * Omega * t) * gpq[..., :, None]
    num_m = gm0[..., None, :] - np.exp(-1j * Omega * t) * gmq[..., :, None]
    small = np.abs(denom) < 1.0e-12
    coeff_p[~small] = num_p[~small] / denom[~small]
    coeff_m[~small] = num_m[~small] / denom[~small]
    if np.any(small):
        # A true zero denominator is possible only for m_ext=0.  The divided
        # difference then tends to d g / d energy.
        dp = 0.5 * (
            np.broadcast_to(dgp0[..., None, :], denom.shape)
            + np.broadcast_to(dgpq[..., :, None], denom.shape)
        )
        dm = 0.5 * (
            np.broadcast_to(dgm0[..., None, :], denom.shape)
            + np.broadcast_to(dgmq[..., :, None], denom.shape)
        )
        coeff_p[small] = dp[small]
        coeff_m[small] = dm[small]

    Stilde = np.einsum(
        "xyia,xyij,xyjb->xyab", Uq.conj(), S, U0, optimize=True
    )
    Xpe = coeff_p * Stilde
    Xme = coeff_m * Stilde
    Xp = np.einsum(
        "xyia,xyab,xyjb->xyij", Uq, Xpe, U0.conj(), optimize=True
    )
    Xm = np.einsum(
        "xyia,xyab,xyjb->xyij", Uq, Xme, U0.conj(), optimize=True
    )
    return Xp, Xm


def reconstruct_transfer_tau_direction(
    G: np.ndarray,
    Gamma: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
    tau: float,
    *,
    edge_points: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tail-completed equilibrium G(+-tau) and transfer X(+-tau)."""
    G = np.asarray(G, dtype=complex)
    Gamma = np.asarray(Gamma, dtype=complex)
    norb = int(G.shape[-1])
    href = _static_kfield(h_ref, grid, norb)
    p = normalize_q_index(q_index, grid)
    S = estimate_transfer_static_vertex(
        Gamma, int(m_ext), grid, edge_points=edge_points
    )
    Gp, Gm = reconstruct_tau_pair(G, href, float(mu), grid, float(tau))
    Xiw = transfer_x_field(G, Gamma, p, int(m_ext), grid)

    Gref = reference_green_iomega(href, float(mu), grid)
    Xref = np.zeros_like(Xiw)
    src, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if src.stop != src.start:
        Grefq = roll_spatial(Gref[src], p[0], p[1])
        Xref[dst] = np.einsum(
            "nxyab,xybc,nxycd->nxyad",
            Grefq,
            S,
            Gref[dst],
            optimize=True,
        )
    Xp_ref, Xm_ref = _reference_transfer_tau(
        href, float(mu), S, p, int(m_ext), float(tau), grid
    )
    dX = Xiw - Xref
    omega = np.asarray(grid.omega, dtype=float)
    phase_p = np.exp(-1j * omega * float(tau))
    phase_m = np.exp(+1j * omega * float(tau))
    Xp = Xp_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_p, dX, optimize=True
    )
    Xm = Xm_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_m, dX, optimize=True
    )
    return Gp, Gm, Xp, Xm


def _cell_phase_vector(
    nk1: int, nk2: int, norb: int, q_index: tuple[int, int]
) -> np.ndarray:
    p1, p2 = q_index
    values = []
    for r1 in range(int(nk1)):
        for r2 in range(int(nk2)):
            phase = np.exp(
                2j
                * np.pi
                * (float(p1) * r1 / float(nk1) + float(p2) * r2 / float(nk2))
            )
            values.extend([phase] * int(norb))
    return np.asarray(values, dtype=complex)


def sox_vertex_transfer_full(
    Gp: np.ndarray,
    Gm: np.ndarray,
    Xp: np.ndarray,
    Xm: np.ndarray,
    v: np.ndarray,
    cell_phase: np.ndarray,
    omega_ext: float,
    tau: float,
    *,
    interaction_tol: float = 1.0e-13,
) -> np.ndarray:
    """D Sigma_SOX at one tau after factoring the output external phase."""
    Gp = np.asarray(Gp, dtype=complex)
    Gm = np.asarray(Gm, dtype=complex)
    Xp = np.asarray(Xp, dtype=complex)
    Xm = np.asarray(Xm, dtype=complex)
    v = np.asarray(v, dtype=complex)
    d = np.asarray(cell_phase, dtype=complex)
    if not (Gp.shape == Gm.shape == Xp.shape == Xm.shape == v.shape):
        raise ValueError("all SOX transfer matrices must have the same shape")
    n = int(v.shape[0])
    if d.shape != (n,):
        raise ValueError("cell_phase length mismatch")

    out = np.zeros_like(v)
    rows = [np.flatnonzero(np.abs(v[i]) > interaction_tol) for i in range(n)]
    cols = [np.flatnonzero(np.abs(v[:, j]) > interaction_tol) for j in range(n)]
    time_middle = np.exp(1j * float(omega_ext) * float(tau))
    for i in range(n):
        ls = rows[i]
        if ls.size == 0:
            continue
        di_inv = np.conj(d[i])
        for j in range(n):
            ks = cols[j]
            if ks.size == 0:
                continue
            gleft = Gp[i, ks] * v[ks, j]
            xleft = Xp[i, ks] * v[ks, j]
            gright = v[i, ls] * Gp[ls, j]
            sub_gm = Gm[np.ix_(ks, ls)]

            term_a = xleft @ sub_gm @ gright
            sub_xm = d[ks, None] * Xm[np.ix_(ks, ls)]
            term_b = (
                di_inv
                * time_middle
                * (gleft @ sub_xm @ gright)
            )
            xright = v[i, ls] * Xp[ls, j] * d[ls]
            term_c = di_inv * (gleft @ sub_gm @ xright)
            out[i, j] = term_a + term_b + term_c
    return out


def compute_sox_vertex_transfer_periodic(
    G: np.ndarray,
    Gamma: np.ndarray,
    Vq: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    q_index,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SOXOptions = SOXOptions(),
) -> np.ndarray:
    """Evaluate D Sigma_SOX[G][G Gamma G] at a general discrete transfer."""
    opts.validate()
    G = np.asarray(G, dtype=complex)
    Gamma = np.asarray(Gamma, dtype=complex)
    if G.shape != Gamma.shape:
        raise ValueError("G and Gamma shapes must match")
    norb = int(G.shape[-1])
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if G.shape != expected:
        raise ValueError(f"G shape {G.shape} != {expected}")
    Vfield = _static_kfield(Vq, grid, norb)
    href = _static_kfield(h_ref, grid, norb)
    p = normalize_q_index(q_index, grid)
    vfull = kfield_to_full_periodic(Vfield)
    d = _cell_phase_vector(grid.nk1, grid.nk2, norb, p)

    x, w = leggauss(int(opts.n_quad))
    beta = 1.0 / float(grid.T)
    tau_values = 0.5 * beta * (x + 1.0)
    weights = 0.5 * beta * w
    omega = np.asarray(grid.omega, dtype=float)
    Omega = 2.0 * np.pi * float(grid.T) * float(m_ext)
    out = np.zeros_like(Gamma)

    for tau, wt in zip(tau_values, weights):
        if opts.tail_complete:
            Gp, Gm, Xp, Xm = reconstruct_transfer_tau_direction(
                G,
                Gamma,
                href,
                float(mu),
                p,
                int(m_ext),
                grid,
                float(tau),
                edge_points=int(opts.tail_edge_points),
            )
        else:
            phase_p = np.exp(-1j * omega * float(tau))
            phase_m = np.exp(+1j * omega * float(tau))
            Gp = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_p, G, optimize=True
            )
            Gm = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_m, G, optimize=True
            )
            Xiw = transfer_x_field(G, Gamma, p, int(m_ext), grid)
            Xp = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_p, Xiw, optimize=True
            )
            Xm = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_m, Xiw, optimize=True
            )

        vert_full = sox_vertex_transfer_full(
            kfield_to_full_periodic(Gp),
            kfield_to_full_periodic(Gm),
            kfield_to_full_periodic(Xp),
            kfield_to_full_periodic(Xm),
            vfull,
            d,
            Omega,
            float(tau),
            interaction_tol=float(opts.interaction_tol),
        )
        vert_k = full_periodic_to_kfield(
            vert_full, grid.nk1, grid.nk2, norb
        )
        phase = np.exp(1j * omega * float(tau))
        out += float(wt) * phase[:, None, None, None, None] * vert_k[None, ...]

    # Gamma(k;Q) only exists where both base and shifted fermionic frequencies
    # lie in the represented box.
    masked = np.zeros_like(out)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        masked[dst] = out[dst]
    return masked


__all__ = [
    "reconstruct_transfer_tau_direction",
    "sox_vertex_transfer_full",
    "compute_sox_vertex_transfer_periodic",
]
