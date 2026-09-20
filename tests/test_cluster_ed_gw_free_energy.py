from types import SimpleNamespace

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_free_energy import evaluate_cluster_ed_gw_free_energy
from rubycgw.free_energy import noninteracting_grand_potential
from rubycgw.grids import MatsubaraGrid
from rubycgw.impurity_ed import FiniteBathImpurityED
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


def test_impurity_thermodynamics_matches_direct_noninteracting_partition():
    h = np.array([[-0.3, 0.12], [0.12, 0.5]], dtype=complex)
    imp = FiniteBathImpurityED(h, (), correlated_orbitals=(0,))
    imp.diagonalize()
    mu = 0.07
    T = 0.2
    th = imp.thermodynamics(mu, T)

    eps = np.linalg.eigvalsh(h)
    omega = -T * np.sum(np.logaddexp(0.0, -(eps - mu) / T))
    occ = 1.0 / (np.exp((eps - mu) / T) + 1.0)
    energy = float(np.dot(eps, occ))
    n = float(np.sum(occ))
    entropy = float((energy - mu * n - omega) / T)

    assert np.isclose(th["grand_potential"], omega, atol=1e-12)
    assert np.isclose(th["internal_energy"], energy, atol=1e-12)
    assert np.isclose(th["average_particles"], n, atol=1e-12)
    assert np.isclose(th["entropy"], entropy, atol=1e-12)


def test_cluster_ed_gw_free_energy_noninteracting_limit():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=8, nOmega=2, T=0.2)
    h0 = np.zeros((1, 1, 2, 2), dtype=complex)
    h0[0, 0] = np.array([[-0.25, 0.11], [0.11, 0.45]], dtype=complex)
    mu = 0.08
    sigma_h = np.zeros((2, 2), dtype=complex)
    sigma = np.zeros((grid.nf, 1, 1, 2, 2), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.mean(np.diagonal(rho, axis1=-2, axis2=-1), axis=(0, 1)).real

    bath = BathParameters(
        energies=np.array([0.6]),
        couplings=np.zeros((2, 1), dtype=complex),
        fit_error=0.0,
        nfev=0,
    )
    state = SimpleNamespace(
        G=G,
        P=np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex),
        Sigma_H=sigma_h,
        Sigma_emb=sigma,
        mu=mu,
        density=density,
        G_cluster=G[:, 0, 0],
        bath=bath,
        impurity_static_shift=np.zeros((2, 2), dtype=complex),
    )
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)
    result = evaluate_cluster_ed_gw_free_energy(
        state,
        h0,
        Vq,
        (),
        grid,
    )

    expected_omega = noninteracting_grand_potential(h0, mu, grid.T, grid.nk)
    expected_F = expected_omega + mu * result.particle_number_actual
    assert abs(result.phi_gw_lattice) < 1e-12
    assert abs(result.phi_gw_cluster) < 1e-12
    assert abs(result.phi_ed_cluster) < 1e-10
    assert abs(result.phi_cluster_correction) < 1e-10
    assert np.isclose(result.grand_potential, expected_omega, atol=1e-10)
    assert np.isclose(result.helmholtz_free_energy, expected_F, atol=1e-10)
