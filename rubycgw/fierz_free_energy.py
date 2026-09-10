"""Finite-temperature Luttinger-Ward free energy for Fierz-channel GW.

The same-torus channel solver uses Hermitian bilinear vertices K_a, a bare
bosonic coupling matrix g_ab, and

    P_ab = + T sum_n Tr[K_a G_{n+m} K_b G_n],
    W = (1 - g P)^(-1) g,
    Sigma_c = - G * (W - g).

Its static first-order self-energy is the sum of tadpole and exchange pieces
constructed by :func:`channel_static_self_energy`.  With those conventions a
stationary real GW functional is

    Omega = Omega0
          + Tr_f[ln(G0^(-1) G) - Sigma G]
          + Phi_tad + Phi_x + Phi_c,

    Phi_tad = 1/2 O_a g_ab O_b,
    O_a = Tr[K_a rho],

    Phi_x = 1/2 Tr[Sigma_x rho],

    Phi_c = 1/2 Tr_b[ln(1 - g P) + g P].

The +gP term removes the first-order contribution already represented by the
static Hartree-Fock functional.  ``slogdet`` is used for the logarithm, i.e. we
retain the real part log|det| of the LW functional.  This matches the convention
used by ``rubycgw.free_energy`` for the density split-GW solver.

At fixed filling the thermodynamic quantity to compare between roots is the
Helmholtz free energy F = Omega + mu N_target.  Absolute GW-vs-ED comparisons
retain the finite Matsubara-box error of the approximate LW correction; Omega0
and the equal-time density are tail/analytic completed exactly as in the active
solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fierz_channel_gw import ChannelDefinition, channel_static_self_energy
from .free_energy import noninteracting_grand_potential
from .grids import MatsubaraGrid
from .supercell_gw import build_g0_inverse_matrix
from .supercell_gw_split import one_body_density_matrix_tail


@dataclass(frozen=True)
class ChannelGWFreeEnergyResult:
    """Thermodynamic functional components, extensive for the finite torus."""

    omega0: float
    fermionic_lw: float
    phi_tadpole: float
    phi_exchange: float
    phi_correlation: float
    phi_total: float
    grand_potential: float
    mu_times_N: float
    helmholtz_free_energy: float
    particle_number_actual: float
    particle_number_legendre: float
    free_energy_per_primitive_cell: float
    grand_potential_per_primitive_cell: float


@dataclass(frozen=True)
class ExactThermalFreeEnergyResult:
    """Exact grand-canonical thermodynamics at a fixed average particle number."""

    grand_potential: float
    mu_times_N: float
    helmholtz_free_energy: float
    particle_number_actual: float
    particle_number_legendre: float
    free_energy_per_primitive_cell: float
    grand_potential_per_primitive_cell: float


def _fermionic_lw_term(
    G: np.ndarray,
    sigma_static: np.ndarray,
    sigma_c: np.ndarray,
    h0: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
) -> float:
    """Tr_f[ln(G0^-1 G)-Sigma G], cancelling the 1/omega tail in-box."""
    G = np.asarray(G, dtype=complex)
    sigma_static = np.asarray(sigma_static, dtype=complex)
    sigma_c = np.asarray(sigma_c, dtype=complex)
    g0inv = build_g0_inverse_matrix(np.asarray(h0, dtype=complex), grid, float(mu))
    sigma = sigma_c + sigma_static[None, None, None, :, :]
    ratio = np.matmul(g0inv, G)
    _, logabsdet = np.linalg.slogdet(ratio)
    tr_sigma_g = np.einsum("nxyab,nxyba->nxy", sigma, G, optimize=True)
    integrand = logabsdet - tr_sigma_g.real
    return float((grid.T / grid.nk) * np.sum(integrand))


def _phi_static(
    rho: np.ndarray,
    definition: ChannelDefinition,
) -> tuple[float, float]:
    """Return the tadpole and exchange skeleton functionals."""
    rho = np.asarray(rho, dtype=complex)
    K = np.asarray(definition.vertices, dtype=complex)
    g = np.asarray(definition.coupling, dtype=complex)
    obs = np.einsum("aij,ji->a", K, rho, optimize=True)
    phi_tad = 0.5 * np.einsum("a,ab,b->", obs, g, obs, optimize=True)
    _, _, sigma_x = channel_static_self_energy(rho, definition)
    phi_x = 0.5 * np.einsum("ij,ji->", sigma_x, rho, optimize=True)
    return float(np.real(phi_tad)), float(np.real(phi_x))


def _phi_ring(
    P: np.ndarray,
    definition: ChannelDefinition,
    grid: MatsubaraGrid,
) -> float:
    """Return 1/2 Tr_b[ln(1-gP)+gP] in the P=+GG convention."""
    P = np.asarray(P, dtype=complex)
    g = np.asarray(definition.coupling, dtype=complex)
    nch = int(g.shape[0])
    if P.shape != (grid.nb, nch, nch):
        raise ValueError("unexpected channel polarization shape in free-energy evaluation")
    gp = np.matmul(g[None, :, :], P)
    lhs = np.eye(nch, dtype=complex)[None, :, :] - gp
    _, logabsdet = np.linalg.slogdet(lhs)
    tr_gp = np.trace(gp, axis1=-2, axis2=-1).real
    return float((grid.T / (2.0 * grid.nk)) * np.sum(logabsdet + tr_gp))


def evaluate_channel_gw_free_energy(
    gw,
    definition: ChannelDefinition,
    h0: np.ndarray,
    grid: MatsubaraGrid,
    *,
    target_particles: float | None = None,
    primitive_cells_per_supercell: int = 1,
) -> ChannelGWFreeEnergyResult:
    """Evaluate the stationary LW functional for a converged channel-GW root."""
    if int(grid.nk1) != 1 or int(grid.nk2) != 1:
        raise ValueError("channel-GW free energy currently requires the same-torus nk=1 formulation")
    npc = int(primitive_cells_per_supercell)
    if npc < 1:
        raise ValueError("primitive_cells_per_supercell must be positive")

    G = np.asarray(gw.G, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    sigma_static = np.asarray(gw.Sigma_static, dtype=complex)
    sigma_c = np.asarray(gw.Sigma_c, dtype=complex)
    P = np.asarray(gw.P, dtype=complex)
    norb = int(h0.shape[-1])
    expected_G = (grid.nf, 1, 1, norb, norb)
    if h0.shape != (1, 1, norb, norb):
        raise ValueError("unexpected h0 shape in channel-GW free-energy evaluation")
    if G.shape != expected_G or sigma_c.shape != expected_G:
        raise ValueError("unexpected G/Sigma_c shape in channel-GW free-energy evaluation")
    if sigma_static.shape != (norb, norb):
        raise ValueError("unexpected Sigma_static shape in channel-GW free-energy evaluation")
    if len(definition.labels) != P.shape[-1]:
        raise ValueError("channel definition and polarization dimensions disagree")

    mu = float(gw.mu)
    rho_full = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)
    rho = np.asarray(rho_full[0, 0], dtype=complex)
    N_actual = float(np.trace(rho).real)
    N_legendre = N_actual if target_particles is None else float(target_particles)

    omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    fermionic = _fermionic_lw_term(G, sigma_static, sigma_c, h0, mu, grid)
    phi_tad, phi_x = _phi_static(rho, definition)
    phi_c = _phi_ring(P, definition, grid)
    phi_total = float(phi_tad + phi_x + phi_c)
    omega = float(omega0 + fermionic + phi_total)
    muN = float(mu * N_legendre)
    F = float(omega + muN)

    return ChannelGWFreeEnergyResult(
        omega0=omega0,
        fermionic_lw=fermionic,
        phi_tadpole=phi_tad,
        phi_exchange=phi_x,
        phi_correlation=phi_c,
        phi_total=phi_total,
        grand_potential=omega,
        mu_times_N=muN,
        helmholtz_free_energy=F,
        particle_number_actual=N_actual,
        particle_number_legendre=N_legendre,
        free_energy_per_primitive_cell=float(F / npc),
        grand_potential_per_primitive_cell=float(omega / npc),
    )


def evaluate_exact_thermal_free_energy(
    exact,
    mu: float,
    T: float,
    *,
    target_particles: float | None = None,
    primitive_cells_per_supercell: int = 1,
) -> ExactThermalFreeEnergyResult:
    """Evaluate exact Omega and fixed-filling F from already diagonalized sectors."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    npc = int(primitive_cells_per_supercell)
    if npc < 1:
        raise ValueError("primitive_cells_per_supercell must be positive")

    beta = 1.0 / float(T)
    logs = [
        -beta * (np.asarray(sec.energies, dtype=float) - float(mu) * sec.n_particles)
        for sec in exact.sectors
    ]
    shift = max(float(np.max(x)) for x in logs)
    weights = [np.exp(x - shift) for x in logs]
    zscaled = float(sum(np.sum(w) for w in weights))
    logZ = float(shift + np.log(zscaled))
    omega = float(-float(T) * logZ)
    N_actual = float(
        sum(sec.n_particles * np.sum(w) for sec, w in zip(exact.sectors, weights)) / zscaled
    )
    N_legendre = N_actual if target_particles is None else float(target_particles)
    muN = float(mu * N_legendre)
    F = float(omega + muN)
    return ExactThermalFreeEnergyResult(
        grand_potential=omega,
        mu_times_N=muN,
        helmholtz_free_energy=F,
        particle_number_actual=N_actual,
        particle_number_legendre=N_legendre,
        free_energy_per_primitive_cell=float(F / npc),
        grand_potential_per_primitive_cell=float(omega / npc),
    )


__all__ = [
    "ChannelGWFreeEnergyResult",
    "ExactThermalFreeEnergyResult",
    "evaluate_channel_gw_free_energy",
    "evaluate_exact_thermal_free_energy",
]
