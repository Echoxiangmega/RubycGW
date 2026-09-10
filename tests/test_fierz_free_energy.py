from types import SimpleNamespace

import numpy as np

from rubycgw.fierz_channel_gw import (
    channel_static_self_energy,
    compute_channel_polarization,
)
from rubycgw.fierz_free_energy import (
    evaluate_channel_gw_free_energy,
    evaluate_exact_thermal_free_energy,
)
from rubycgw.fierz_mixed import FierzWeights, build_weighted_nbj_definition
from rubycgw.free_energy import evaluate_gw_free_energy
from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


def test_density_channel_free_energy_matches_existing_split_gw():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=6, nOmega=3, T=0.17)
    h0 = np.array([[[[0.15, -0.22], [-0.22, -0.08]]]], dtype=complex)
    mu = 0.04
    sigma_static_guess = np.array([[0.11, 0.015], [0.015, -0.03]], dtype=complex)
    sigma_c = np.zeros((grid.nf, 1, 1, 2, 2), dtype=complex)
    for k in range(grid.nw):
        z = 0.012 / (1.0 + k)
        pos = np.array([[z, 0.002j * z], [-0.002j * z, -0.5 * z]], dtype=complex)
        sigma_c[grid.nw + k, 0, 0] = pos
        sigma_c[grid.nw - 1 - k, 0, 0] = pos.conj().T

    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_static_guess, sigma_c)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static_guess)[0, 0]
    definition = build_weighted_nbj_definition([(0, 1)], 2, 0.7, FierzWeights(1.0, 0.0, 0.0))
    sigma_static, tad, exchange = channel_static_self_energy(rho, definition)

    # Rebuild G with the static self-energy used by both functionals.
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_static, sigma_c)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)[0, 0]
    sigma_static, tad, exchange = channel_static_self_energy(rho, definition)
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_static, sigma_c)
    Pch = compute_channel_polarization(G, definition, grid)

    bg = SimpleNamespace(
        G=G, P=Pch, Sigma_static=sigma_static, Sigma_c=sigma_c,
        Sigma_tadpole=tad, Sigma_exchange=exchange, mu=mu, rho=rho,
        density=np.real(np.diag(rho)),
    )
    new = evaluate_channel_gw_free_energy(
        bg, definition, h0, grid, target_particles=1.0, primitive_cells_per_supercell=1
    )

    old_state = SimpleNamespace(
        G=G,
        P=Pch[:, None, None],
        W=np.zeros_like(Pch[:, None, None]),
        Sigma_H=tad,
        Sigma_GW=exchange[None, None, None] + sigma_c,
        mu=mu,
        density=np.real(np.diag(rho)),
    )
    Vq = definition.coupling[None, None]
    old = evaluate_gw_free_energy(
        old_state, h0, Vq, grid, target_particles=1.0,
        primitive_cells_per_supercell=1, momentum_backend="direct"
    )

    np.testing.assert_allclose(new.omega0, old.omega0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(new.fermionic_lw, old.fermionic_lw, rtol=0, atol=2e-12)
    np.testing.assert_allclose(new.phi_tadpole, old.phi_hartree, rtol=0, atol=2e-12)
    np.testing.assert_allclose(new.phi_exchange, old.phi_fock, rtol=0, atol=2e-12)
    np.testing.assert_allclose(new.phi_correlation, old.phi_correlation, rtol=0, atol=2e-12)
    np.testing.assert_allclose(new.helmholtz_free_energy, old.helmholtz_free_energy, rtol=0, atol=2e-12)


def test_exact_thermal_free_energy_two_level_toy():
    class Sector:
        def __init__(self, n, e):
            self.n_particles = n
            self.energies = np.asarray(e, dtype=float)

    exact = SimpleNamespace(sectors=(Sector(0, [0.0]), Sector(1, [0.3, 0.8])))
    mu = 0.2
    T = 0.25
    out = evaluate_exact_thermal_free_energy(
        exact, mu, T, target_particles=0.7, primitive_cells_per_supercell=1
    )
    weights = np.array([1.0, np.exp(-(0.3 - mu) / T), np.exp(-(0.8 - mu) / T)])
    Z = np.sum(weights)
    omega = -T * np.log(Z)
    nactual = (weights[1] + weights[2]) / Z
    np.testing.assert_allclose(out.grand_potential, omega, rtol=0, atol=1e-14)
    np.testing.assert_allclose(out.particle_number_actual, nactual, rtol=0, atol=1e-14)
    np.testing.assert_allclose(out.helmholtz_free_energy, omega + mu * 0.7, rtol=0, atol=1e-14)
