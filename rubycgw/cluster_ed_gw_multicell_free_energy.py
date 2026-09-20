"""Luttinger-Ward thermodynamics for multi-impurity finite-q cluster ED+GW."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster_ed_gw import BathParameters, cluster_interaction_matrix
from .cluster_ed_gw_free_energy import _gw_phi_from_G, _impurity_phi
from .free_energy import _fermionic_lw_term, noninteracting_grand_potential
from .grids import MatsubaraGrid
from .supercell_gw import compute_polarization_matrix
from .supercell_gw_split import one_body_density_matrix_tail


NSUB = 6


@dataclass(frozen=True)
class MultiCellFreeEnergyResult:
    omega0_lattice: float
    fermionic_lw_lattice: float
    phi_gw_lattice: float
    phi_gw_cluster_sum: float
    phi_ed_cluster_sum: float
    phi_cluster_correction_sum: float
    phi_embedded_total: float
    grand_potential_supercell: float
    helmholtz_free_energy_supercell: float
    helmholtz_free_energy_per_primitive_cell: float
    particle_number_actual: float
    particle_number_legendre: float
    impurity_grand_potential: np.ndarray
    impurity_internal_energy: np.ndarray
    impurity_entropy: np.ndarray
    impurity_particle_number: np.ndarray
    gimp_gc_relative_mismatch: np.ndarray
    gimp_reconstruction_mismatch: np.ndarray


def _relative_error(a, b):
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def evaluate_multicell_free_energy(
    state,
    h0: np.ndarray,
    Vq: np.ndarray,
    cluster_interactions,
    grid: MatsubaraGrid,
    *,
    target_particles: float | None = None,
    discard_weight_tol: float = 1e-11,
) -> MultiCellFreeEnergyResult:
    G = np.asarray(state.G, dtype=complex)
    P = np.asarray(state.P, dtype=complex)
    sigma_h = np.asarray(state.Sigma_H, dtype=complex)
    sigma_emb = np.asarray(state.Sigma_emb, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    mu = float(state.mu)
    norb = h0.shape[-1]
    if norb % NSUB:
        raise ValueError("supercell dimension must be a multiple of six")
    ncell = norb // NSUB

    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.diagonal(rho[0, 0], axis1=-2, axis2=-1).real
    N_actual = float(np.sum(density))
    N_legendre = N_actual if target_particles is None else float(target_particles)

    omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    fermionic = _fermionic_lw_term(
        G, sigma_h, sigma_emb, h0, mu, grid
    )
    phi_gw_lat = _gw_phi_from_G(G, P, rho, Vq, grid)

    Vc = cluster_interaction_matrix(cluster_interactions, NSUB)
    cgrid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=grid.nw,
        nOmega=grid.nOmega,
        T=grid.T,
    )
    phi_gw_cells = []
    phi_ed_cells = []
    imp_omega = []
    imp_energy = []
    imp_entropy = []
    imp_particles = []
    g_mismatch = []
    reconstruct_mismatch = []

    for c in range(ncell):
        sl = slice(c * NSUB, (c + 1) * NSUB)
        Gc = np.asarray(state.G_cluster[c], dtype=complex)
        rhoc = np.asarray(rho[0, 0, sl, sl], dtype=complex)
        Gc5 = Gc[:, None, None, :, :]
        Pc = compute_polarization_matrix(Gc5, cgrid, backend="fft")
        phi_gw_c = _gw_phi_from_G(
            Gc5,
            Pc,
            rhoc[None, None, :, :],
            Vc[None, None, :, :],
            cgrid,
        )
        phi_gw_cells.append(float(phi_gw_c))

        bath = BathParameters(
            np.asarray(state.bath_energies[c], dtype=float),
            np.asarray(state.bath_couplings[c], dtype=complex),
            float(np.asarray(state.bath_fit_error)[c]),
            0,
        )
        hloc = np.asarray(h0[0, 0, sl, sl], dtype=complex)
        hcorr = hloc + np.asarray(state.impurity_static_shift[c], dtype=complex)
        phi_ed, thermo, Gimp, _, rec = _impurity_phi(
            hcorr,
            bath,
            cluster_interactions,
            grid.omega,
            mu,
            grid.T,
            correlated_orbitals=tuple(range(NSUB)),
            discard_weight_tol=float(discard_weight_tol),
        )
        phi_ed_cells.append(float(phi_ed))
        imp_omega.append(float(thermo["grand_potential"]))
        imp_energy.append(float(thermo["internal_energy"]))
        imp_entropy.append(float(thermo["entropy"]))
        imp_particles.append(float(thermo["average_particles"]))
        g_mismatch.append(_relative_error(Gimp, Gc))
        reconstruct_mismatch.append(float(rec))

    phi_gw_csum = float(np.sum(phi_gw_cells))
    phi_ed_csum = float(np.sum(phi_ed_cells))
    correction = float(phi_ed_csum - phi_gw_csum)
    phi_emb = float(phi_gw_lat + correction)
    omega = float(omega0 + fermionic + phi_emb)
    F = float(omega + mu * N_legendre)

    return MultiCellFreeEnergyResult(
        omega0_lattice=float(omega0),
        fermionic_lw_lattice=float(fermionic),
        phi_gw_lattice=float(phi_gw_lat),
        phi_gw_cluster_sum=phi_gw_csum,
        phi_ed_cluster_sum=phi_ed_csum,
        phi_cluster_correction_sum=correction,
        phi_embedded_total=phi_emb,
        grand_potential_supercell=omega,
        helmholtz_free_energy_supercell=F,
        helmholtz_free_energy_per_primitive_cell=float(F / ncell),
        particle_number_actual=N_actual,
        particle_number_legendre=N_legendre,
        impurity_grand_potential=np.asarray(imp_omega),
        impurity_internal_energy=np.asarray(imp_energy),
        impurity_entropy=np.asarray(imp_entropy),
        impurity_particle_number=np.asarray(imp_particles),
        gimp_gc_relative_mismatch=np.asarray(g_mismatch),
        gimp_reconstruction_mismatch=np.asarray(reconstruct_mismatch),
    )


__all__ = ["MultiCellFreeEnergyResult", "evaluate_multicell_free_energy"]
