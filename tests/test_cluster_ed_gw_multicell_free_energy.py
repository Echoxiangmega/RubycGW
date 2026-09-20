from types import SimpleNamespace

import numpy as np

from rubycgw.cluster_ed_gw_multicell_free_energy import evaluate_multicell_free_energy
from rubycgw.free_energy import noninteracting_grand_potential
from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


def test_multicell_free_energy_noninteracting_limit():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=6, nOmega=1, T=0.2)
    norb = 12
    h0 = np.zeros((1, 1, norb, norb), dtype=complex)
    onsite = np.linspace(-0.4, 0.5, norb)
    h0[0, 0] = np.diag(onsite)
    mu = 0.03
    sigma_h = np.zeros((norb, norb), dtype=complex)
    sigma = np.zeros((grid.nf, 1, 1, norb, norb), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.diagonal(rho[0, 0]).real

    Gc = np.stack([
        G[:, 0, 0, :6, :6],
        G[:, 0, 0, 6:12, 6:12],
    ])
    state = SimpleNamespace(
        G=G,
        P=np.zeros((grid.nb, 1, 1, norb, norb), dtype=complex),
        Sigma_H=sigma_h,
        Sigma_emb=sigma,
        mu=mu,
        density=density,
        G_cluster=Gc,
        bath_energies=np.array([[0.8], [0.8]], dtype=float),
        bath_couplings=np.zeros((2, 6, 1), dtype=complex),
        bath_fit_error=np.zeros(2),
        impurity_static_shift=np.zeros((2, 6, 6), dtype=complex),
    )
    Vq = np.zeros((1, 1, norb, norb), dtype=complex)
    result = evaluate_multicell_free_energy(
        state,
        h0,
        Vq,
        (),
        grid,
    )
    omega0 = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    F = omega0 + mu * result.particle_number_actual
    assert abs(result.phi_gw_lattice) < 1e-12
    assert abs(result.phi_gw_cluster_sum) < 1e-12
    assert abs(result.phi_ed_cluster_sum) < 1e-10
    assert np.isclose(result.grand_potential_supercell, omega0, atol=1e-10)
    assert np.isclose(result.helmholtz_free_energy_supercell, F, atol=1e-10)
    assert np.isclose(
        result.helmholtz_free_energy_per_primitive_cell,
        F / 2.0,
        atol=1e-10,
    )
