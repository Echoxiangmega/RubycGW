"""Luttinger-Ward thermodynamics for physical-pair cluster ED+GW.

The production embedding uses

    Sigma_tot = Sigma_H^lat
              + Sigma_GW^lat
              - Sigma_GW^C
              + Sigma_ED^C.

The corresponding SEET-like Phi functional is evaluated as

    Phi_emb[G] = Phi_GW^lat[G]
               + Phi_ED^C[G_C]
               - Phi_GW^C[G_C].

The lattice grand potential then follows from the same sign conventions used by
rubycgw.free_energy:

    Omega = Omega0
          + Tr_f[ln(G0^{-1} G) - Sigma_tot G]
          + Phi_emb.

Phi_ED^C is reconstructed from the exact finite-bath impurity partition
function and its full impurity+bath Dyson equation.  The auxiliary-bath grand
potential is therefore only an intermediate ingredient; it is never identified
with the lattice grand potential.

Because the self-consistency only matches G_imp and G_C approximately for a
finite bath, the returned mismatch diagnostics must accompany any branch free-
energy comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_interaction_matrix,
)
from .free_energy import (
    _fermionic_lw_term,
    _phi_ring,
    noninteracting_grand_potential,
)
from .grids import MatsubaraGrid
from .impurity_ed import FiniteBathImpurityED
from .supercell_gw import compute_polarization_matrix
from .supercell_gw_split import (
    compute_static_fock_matrix,
    one_body_density_matrix_tail,
)


@dataclass(frozen=True)
class ClusterEDGWFreeEnergyResult:
    omega0_lattice: float
    fermionic_lw_lattice: float
    phi_gw_lattice: float
    phi_gw_cluster: float
    phi_ed_cluster: float
    phi_cluster_correction: float
    phi_embedded_total: float
    grand_potential: float
    mu_times_N: float
    helmholtz_free_energy: float
    particle_number_actual: float
    particle_number_legendre: float
    impurity_grand_potential: float
    impurity_internal_energy: float
    impurity_entropy: float
    impurity_particle_number: float
    impurity_phi_fermionic_term: float
    impurity_noninteracting_grand_potential: float
    gimp_gc_relative_mismatch: float
    gimp_reconstruction_mismatch: float
    bath_fit_error: float


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _gw_phi_from_G(
    G: np.ndarray,
    P: np.ndarray,
    rho: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
) -> float:
    """Return Phi_H + Phi_F + Phi_c for the code's split-GW convention."""
    rho = np.asarray(rho, dtype=complex)
    density = np.mean(
        np.diagonal(rho, axis1=-2, axis2=-1), axis=(0, 1)
    ).real
    V0 = np.asarray(Vq, dtype=complex)[0, 0]
    phi_h = float(0.5 * np.real(np.vdot(density, V0 @ density)))

    sigma_f = compute_static_fock_matrix(
        rho, np.asarray(Vq, dtype=complex), grid, backend="fft"
    )
    tr_f_rho = np.einsum("xyab,xyba->xy", sigma_f, rho, optimize=True)
    phi_f = float(0.5 * np.sum(tr_f_rho).real / grid.nk)
    phi_c = _phi_ring(P, Vq, grid)
    return float(phi_h + phi_f + phi_c)


def _impurity_phi(
    h_cluster: np.ndarray,
    bath: BathParameters,
    interaction_terms,
    omega: np.ndarray,
    mu: float,
    T: float,
    *,
    correlated_orbitals: tuple[int, ...],
    discard_weight_tol: float,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, float]:
    """Reconstruct the exact cluster Phi from the finite-bath reference system."""
    himp = build_impurity_one_body(h_cluster, bath)
    impurity = FiniteBathImpurityED(
        himp,
        interaction_terms,
        correlated_orbitals=correlated_orbitals,
    )
    impurity.diagonalize()
    thermo = impurity.thermodynamics(mu, T)
    Gimp, _ = impurity.green_iomega(
        1j * np.asarray(omega, dtype=float),
        mu,
        T,
        orbitals=correlated_orbitals,
        discard_weight_tol=discard_weight_tol,
    )

    nc = len(correlated_orbitals)
    eye_c = np.eye(nc, dtype=complex)
    delta = bath_hybridization(
        np.asarray(omega, dtype=float),
        mu,
        bath.energies,
        bath.couplings,
    )
    g0c_inv = (
        (1j * np.asarray(omega)[:, None, None] + float(mu))
        * eye_c[None, :, :]
        - np.asarray(h_cluster, dtype=complex)[None, :, :]
        - delta
    )
    sigma_c = g0c_inv - np.linalg.inv(Gimp)

    norb = himp.shape[0]
    eye = np.eye(norb, dtype=complex)
    g0_inv = (
        (1j * np.asarray(omega)[:, None, None] + float(mu))
        * eye[None, :, :]
        - himp[None, :, :]
    )
    sigma_full = np.zeros_like(g0_inv)
    sigma_full[:, :nc, :nc] = sigma_c
    Gfull = np.linalg.inv(g0_inv - sigma_full)

    ratio = np.matmul(g0_inv, Gfull)
    _, logabsdet = np.linalg.slogdet(ratio)
    tr_sigma_g = np.einsum("nab,nba->n", sigma_full, Gfull, optimize=True)
    fermionic = float(
        float(T) * np.sum(logabsdet - tr_sigma_g.real)
    )
    omega0 = noninteracting_grand_potential(himp, mu, T, 1)
    phi = float(
        thermo["grand_potential"] - omega0 - fermionic
    )
    block = Gfull[:, :nc, :nc]
    reconstruct_mismatch = _relative_error(block, Gimp)
    return phi, thermo, Gimp, sigma_c, reconstruct_mismatch


def evaluate_cluster_ed_gw_free_energy(
    state,
    h0: np.ndarray,
    Vq: np.ndarray,
    cluster_interactions,
    grid: MatsubaraGrid,
    *,
    target_particles: float | None = None,
    discard_weight_tol: float = 1e-11,
) -> ClusterEDGWFreeEnergyResult:
    """Evaluate the stationary SEET-like LW functional of a converged state."""
    G = np.asarray(state.G, dtype=complex)
    sigma_h = np.asarray(state.Sigma_H, dtype=complex)
    sigma_emb = np.asarray(state.Sigma_emb, dtype=complex)
    P = np.asarray(state.P, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    mu = float(state.mu)

    norb = h0.shape[-1]
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if G.shape != expected or sigma_emb.shape != expected:
        raise ValueError("state/h0/grid shape mismatch")
    if sigma_h.shape != (norb, norb):
        raise ValueError("unexpected Hartree shape")

    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    rho_c = np.mean(rho, axis=(0, 1))
    density = np.mean(
        np.diagonal(rho, axis1=-2, axis2=-1), axis=(0, 1)
    ).real
    N_actual = float(np.sum(density))
    N_legendre = (
        N_actual if target_particles is None else float(target_particles)
    )

    omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    fermionic = _fermionic_lw_term(
        G, sigma_h, sigma_emb, h0, mu, grid
    )
    phi_gw_lat = _gw_phi_from_G(G, P, rho, Vq, grid)

    Vc = cluster_interaction_matrix(cluster_interactions, norb)
    cgrid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=grid.nw,
        nOmega=grid.nOmega,
        T=grid.T,
    )
    Gc = np.asarray(state.G_cluster, dtype=complex)
    if Gc.shape != (grid.nf, norb, norb):
        Gc = np.mean(G, axis=(1, 2))
    Gc5 = Gc[:, None, None, :, :]
    Pc = compute_polarization_matrix(Gc5, cgrid, backend="fft")
    phi_gw_c = _gw_phi_from_G(
        Gc5,
        Pc,
        rho_c[None, None, :, :],
        Vc[None, None, :, :],
        cgrid,
    )

    bath = state.bath
    if not isinstance(bath, BathParameters):
        bath = BathParameters(
            np.asarray(bath.energies, dtype=float),
            np.asarray(bath.couplings, dtype=complex),
            float(getattr(bath, "fit_error", np.nan)),
            int(getattr(bath, "nfev", 0)),
        )
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    h_cluster = h_cluster + np.asarray(
        state.impurity_static_shift, dtype=complex
    )
    phi_ed_c, thermo, Gimp, _, reconstruction_mismatch = _impurity_phi(
        h_cluster,
        bath,
        cluster_interactions,
        grid.omega,
        mu,
        grid.T,
        correlated_orbitals=tuple(range(norb)),
        discard_weight_tol=float(discard_weight_tol),
    )

    correction = float(phi_ed_c - phi_gw_c)
    phi_emb = float(phi_gw_lat + correction)
    omega = float(omega0 + fermionic + phi_emb)
    muN = float(mu * N_legendre)
    F = float(omega + muN)

    return ClusterEDGWFreeEnergyResult(
        omega0_lattice=float(omega0),
        fermionic_lw_lattice=float(fermionic),
        phi_gw_lattice=float(phi_gw_lat),
        phi_gw_cluster=float(phi_gw_c),
        phi_ed_cluster=float(phi_ed_c),
        phi_cluster_correction=correction,
        phi_embedded_total=phi_emb,
        grand_potential=omega,
        mu_times_N=muN,
        helmholtz_free_energy=F,
        particle_number_actual=N_actual,
        particle_number_legendre=N_legendre,
        impurity_grand_potential=float(thermo["grand_potential"]),
        impurity_internal_energy=float(thermo["internal_energy"]),
        impurity_entropy=float(thermo["entropy"]),
        impurity_particle_number=float(thermo["average_particles"]),
        impurity_phi_fermionic_term=float(
            thermo["grand_potential"]
            - noninteracting_grand_potential(
                build_impurity_one_body(h_cluster, bath), mu, grid.T, 1
            )
            - phi_ed_c
        ),
        impurity_noninteracting_grand_potential=float(
            noninteracting_grand_potential(
                build_impurity_one_body(h_cluster, bath), mu, grid.T, 1
            )
        ),
        gimp_gc_relative_mismatch=_relative_error(Gimp, Gc),
        gimp_reconstruction_mismatch=float(reconstruction_mismatch),
        bath_fit_error=float(bath.fit_error),
    )


__all__ = [
    "ClusterEDGWFreeEnergyResult",
    "evaluate_cluster_ed_gw_free_energy",
]
