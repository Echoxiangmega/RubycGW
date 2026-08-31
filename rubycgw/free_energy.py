"""Finite-temperature Luttinger-Ward free energy for the 18-site split-GW solver.

The active supercell solver uses

    G^{-1} = G0^{-1} - Sigma_H - Sigma_GW,
    Sigma_GW = Sigma_F + Sigma_c,
    Sigma_c = - G * (W - V),
    P = + G G,
    W = (1 - V P)^{-1} V.

With exactly these sign conventions, a stationary GW Luttinger-Ward functional
can be written relative to the noninteracting grand potential as

    Omega = Omega0
          + Tr_f[ ln(G0^{-1} G) - Sigma G ]
          + Phi_H + Phi_F + Phi_c,

where

    Phi_H = 1/2 n V(q=0) n,
    Phi_F = 1/2 Tr_k[ Sigma_F rho ],
    Phi_c = 1/2 Tr_b[ ln(1 - V P) + V P ].

The ``+ V P`` term removes the first-order exchange diagram from the ring
functional because bare exchange is already included explicitly as ``Phi_F``.
Differentiating ``Phi_c`` gives ``Sigma_c=-G(W-V)``, matching the split-GW
implementation.

``Tr_f`` means ``T/Nk sum_{k,n} tr`` and ``Tr_b`` means
``T/Nk sum_{q,m} tr``.  The logarithmic fermion term and ``-Sigma G`` are kept
inside the same finite Matsubara sum so their 1/omega tails cancel before the
sum is taken.  The noninteracting ``Omega0`` and the equal-time density matrix
are evaluated analytically, so the remaining cutoff dependence is only that of
the interacting finite-box LW correction used by the numerical GW equations.

For the fixed-filling calculations used in this project, compare the Helmholtz
free energy

    F = Omega + mu N,

not ``Omega`` itself.  ``target_particles`` may be supplied so that the exact
requested filling, rather than the tiny numerically imperfect reconstructed
filling, is used in the Legendre transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid
from .supercell_gw import build_g0_inverse_matrix
from .supercell_gw_split import (
    compute_static_fock_matrix,
    one_body_density_matrix_tail,
)


@dataclass(frozen=True)
class GWFreeEnergyResult:
    """Thermodynamic functional components, all extensive per supercell."""

    omega0: float
    fermionic_lw: float
    phi_hartree: float
    phi_fock: float
    phi_correlation: float
    phi_total: float
    grand_potential: float
    mu_times_N: float
    helmholtz_free_energy: float
    particle_number_actual: float
    particle_number_legendre: float
    density_mismatch_max: float
    free_energy_per_primitive_cell: float
    grand_potential_per_primitive_cell: float


def _validate_shapes(gw, h0: np.ndarray, Vq: np.ndarray, grid: MatsubaraGrid) -> int:
    G = np.asarray(gw.G, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape in free-energy evaluation")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape in free-energy evaluation")
    if G.shape != (grid.nf, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected G shape in free-energy evaluation")
    if np.asarray(gw.Sigma_H).shape != (norb, norb):
        raise ValueError("unexpected Sigma_H shape in free-energy evaluation")
    if np.asarray(gw.Sigma_GW).shape != G.shape:
        raise ValueError("unexpected Sigma_GW shape in free-energy evaluation")
    if np.asarray(gw.P).shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected P shape in free-energy evaluation")
    return norb


def noninteracting_grand_potential(
    h0: np.ndarray,
    mu: float,
    T: float,
    nk: int,
) -> float:
    """Exact finite-T noninteracting grand potential per supercell."""
    h0 = np.asarray(h0, dtype=complex)
    herm = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))
    evals = np.linalg.eigvalsh(herm)
    x = -(evals - float(mu)) / float(T)
    return float(-(float(T) / int(nk)) * np.sum(np.logaddexp(0.0, x)))


def _fermionic_lw_term(
    G: np.ndarray,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    h0: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
) -> float:
    """Return Tr_f[ln(G0^{-1}G)-Sigma G] with cancellation before summation."""
    g0inv = build_g0_inverse_matrix(h0, grid, float(mu))
    sigma = np.asarray(sigma_gw, dtype=complex) + np.asarray(sigma_h, dtype=complex)[
        None, None, None, :, :
    ]

    # G0^{-1} G tends to I at large |omega|.  slogdet gives log|det|, which is
    # exactly the real part of the matrix logarithm needed by the real
    # thermodynamic functional and avoids branch-cut noise in conjugate
    # Matsubara pairs.
    ratio = np.matmul(g0inv, G)
    _, logabsdet = np.linalg.slogdet(ratio)
    tr_sigma_g = np.einsum("nxyab,nxyba->nxy", sigma, G, optimize=True)
    integrand = logabsdet - tr_sigma_g.real
    return float((grid.T / grid.nk) * np.sum(integrand))


def _phi_ring(P: np.ndarray, Vq: np.ndarray, grid: MatsubaraGrid) -> float:
    """Return 1/2 Tr_b[ln(1-VP)+VP] in the code's P=+GG convention."""
    P = np.asarray(P, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    vp = np.matmul(Vq[None, :, :, :, :], P)
    lhs = eye[None, None, None, :, :] - vp
    _, logabsdet = np.linalg.slogdet(lhs)
    tr_vp = np.trace(vp, axis1=-2, axis2=-1).real
    return float((grid.T / (2.0 * grid.nk)) * np.sum(logabsdet + tr_vp))


def evaluate_gw_free_energy(
    gw,
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    *,
    target_particles: float | None = None,
    primitive_cells_per_supercell: int = 3,
    momentum_backend: str = "fft",
) -> GWFreeEnergyResult:
    """Evaluate the stationary split-GW LW functional for one converged state.

    Parameters
    ----------
    gw
        A ``GWResult``-like object containing G, P, Sigma_H, Sigma_GW, mu and
        density.  The function is intended for converged states.
    h0, Vq, grid
        The same one-body Hamiltonian, bare interaction and Matsubara grid used
        to obtain ``gw``.  If a temporary source is present, ``h0`` must include
        that source.  Zero-source values are the physical free energies used to
        compare competing basins.
    target_particles
        Fixed particle number for the Legendre transform.  If omitted, the
        particle number reconstructed from G is used.
    primitive_cells_per_supercell
        Used only for the convenience per-primitive-cell output.
    """
    norb = _validate_shapes(gw, h0, Vq, grid)
    if int(primitive_cells_per_supercell) < 1:
        raise ValueError("primitive_cells_per_supercell must be positive")

    G = np.asarray(gw.G, dtype=complex)
    P = np.asarray(gw.P, dtype=complex)
    sigma_h = np.asarray(gw.Sigma_H, dtype=complex)
    sigma_gw = np.asarray(gw.Sigma_GW, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    mu = float(gw.mu)

    # Equal-time rho is tail-corrected in exactly the same way as the active
    # split-GW self-energy.  Use rho itself to reconstruct n so Phi_H, Phi_F and
    # the Legendre transform all refer to the same G.
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.mean(np.diagonal(rho, axis1=-2, axis2=-1), axis=(0, 1)).real
    N_actual = float(np.sum(density))
    N_legendre = N_actual if target_particles is None else float(target_particles)

    stored_density = np.asarray(getattr(gw, "density", density), dtype=float).reshape(-1)
    density_mismatch = float(np.max(np.abs(stored_density - density)))

    omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    fermionic = _fermionic_lw_term(G, sigma_h, sigma_gw, h0, mu, grid)

    V0 = Vq[0, 0]
    phi_h = float(0.5 * np.real(np.vdot(density, V0 @ density)))

    sigma_f = compute_static_fock_matrix(rho, Vq, grid, backend=momentum_backend)
    tr_f_rho = np.einsum("xyab,xyba->xy", sigma_f, rho, optimize=True)
    phi_f = float(0.5 * np.sum(tr_f_rho).real / grid.nk)

    phi_c = _phi_ring(P, Vq, grid)
    phi_total = float(phi_h + phi_f + phi_c)
    omega = float(omega0 + fermionic + phi_total)
    muN = float(mu * N_legendre)
    F = float(omega + muN)
    npc = float(primitive_cells_per_supercell)

    return GWFreeEnergyResult(
        omega0=omega0,
        fermionic_lw=fermionic,
        phi_hartree=phi_h,
        phi_fock=phi_f,
        phi_correlation=phi_c,
        phi_total=phi_total,
        grand_potential=omega,
        mu_times_N=muN,
        helmholtz_free_energy=F,
        particle_number_actual=N_actual,
        particle_number_legendre=N_legendre,
        density_mismatch_max=density_mismatch,
        free_energy_per_primitive_cell=float(F / npc),
        grand_potential_per_primitive_cell=float(omega / npc),
    )


__all__ = [
    "GWFreeEnergyResult",
    "evaluate_gw_free_energy",
    "noninteracting_grand_potential",
]
