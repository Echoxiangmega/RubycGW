from types import SimpleNamespace

import numpy as np

from rubycgw.free_energy import evaluate_gw_free_energy, noninteracting_grand_potential
from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


def _noninteracting_state(h0, grid, mu):
    norb = h0.shape[-1]
    sigma_h = np.zeros((norb, norb), dtype=complex)
    sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.mean(np.diagonal(rho, axis1=-2, axis2=-1), axis=(0, 1)).real
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    return SimpleNamespace(
        G=G,
        P=P,
        W=np.zeros_like(P),
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        mu=float(mu),
        density=density,
    )


def test_noninteracting_limit_is_exact_helmholtz_transform():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=6, nOmega=2, T=0.2)
    h0 = np.zeros((1, 1, 2, 2), dtype=complex)
    h0[0, 0] = np.array([[-0.4, 0.15], [0.15, 0.7]], dtype=complex)
    mu = 0.13
    gw = _noninteracting_state(h0, grid, mu)
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)

    result = evaluate_gw_free_energy(
        gw,
        h0,
        Vq,
        grid,
        primitive_cells_per_supercell=1,
    )
    expected_omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    expected_F = expected_omega0 + mu * result.particle_number_actual

    assert np.isclose(result.omega0, expected_omega0, atol=1e-12)
    assert abs(result.fermionic_lw) < 1e-12
    assert abs(result.phi_hartree) < 1e-12
    assert abs(result.phi_fock) < 1e-12
    assert abs(result.phi_correlation) < 1e-12
    assert np.isclose(result.grand_potential, expected_omega0, atol=1e-12)
    assert np.isclose(result.helmholtz_free_energy, expected_F, atol=1e-12)


def test_fixed_filling_legendre_transform_uses_requested_N():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=1, T=0.25)
    h0 = np.zeros((1, 1, 1, 1), dtype=complex)
    h0[0, 0, 0, 0] = -0.2
    mu = 0.07
    gw = _noninteracting_state(h0, grid, mu)
    Vq = np.zeros((1, 1, 1, 1), dtype=complex)
    target = 0.4321

    result = evaluate_gw_free_energy(
        gw,
        h0,
        Vq,
        grid,
        target_particles=target,
        primitive_cells_per_supercell=1,
    )

    assert result.particle_number_legendre == target
    assert np.isclose(
        result.helmholtz_free_energy,
        result.grand_potential + mu * target,
        atol=1e-13,
    )


def test_time_reversed_noninteracting_current_sources_have_equal_free_energy():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=8, nOmega=1, T=0.15)
    hp = np.zeros((1, 1, 2, 2), dtype=complex)
    hp[0, 0] = np.array([[0.2, 0.35j], [-0.35j, -0.2]], dtype=complex)
    hm = hp.conj()
    mu = 0.0
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)

    gp = _noninteracting_state(hp, grid, mu)
    gm = _noninteracting_state(hm, grid, mu)
    fp = evaluate_gw_free_energy(gp, hp, Vq, grid, primitive_cells_per_supercell=1)
    fm = evaluate_gw_free_energy(gm, hm, Vq, grid, primitive_cells_per_supercell=1)

    assert np.isclose(fp.helmholtz_free_energy, fm.helmholtz_free_energy, atol=1e-12)
