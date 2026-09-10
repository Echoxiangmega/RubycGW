import numpy as np

from rubycgw.fierz_channel_gw import (
    build_channel_definition,
    channel_static_self_energy,
    channel_vertex_kernel_parts,
    compute_channel_correlation_self_energy,
    compute_channel_polarization,
    compute_channel_screened_interaction,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
)


PAIRS = ((0, 1), (1, 2), (2, 0))


def _fixture():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=7, nOmega=2, T=0.17)
    h = np.array(
        [[0.1, 0.23, -0.11], [0.23, -0.07, 0.19], [-0.11, 0.19, 0.04]],
        dtype=complex,
    )
    mu = -0.03
    eye = np.eye(3, dtype=complex)
    G = np.linalg.inv(
        (1j * grid.omega[:, None, None] + mu) * eye[None] - h[None]
    )[:, None, None]
    return grid, h, mu, G


def test_fierz_channels_have_identical_hartree_fock_self_energy():
    rng = np.random.default_rng(17)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = 0.5 * (A + A.conj().T)
    V = 0.61
    Vmat = np.zeros((3, 3), dtype=complex)
    for i, j in PAIRS:
        Vmat[i, j] += V
        Vmat[j, i] += V
    ref = np.diag(Vmat @ np.diag(rho)) - rho * Vmat.T

    for mode in ("density", "B", "J", "BJ"):
        definition = build_channel_definition(PAIRS, 3, V, mode)
        total, _, _ = channel_static_self_energy(rho, definition)
        np.testing.assert_allclose(total, ref, rtol=2e-13, atol=2e-13)


def test_density_channel_reproduces_matrix_gw_contractions():
    grid, _, _, G = _fixture()
    definition = build_channel_definition(PAIRS, 3, 0.37, "density")
    Vq = definition.coupling[None, None]

    P_generic = compute_channel_polarization(G, definition, grid)
    P_matrix = compute_polarization_matrix(G, grid, backend="direct")
    np.testing.assert_allclose(
        P_generic, P_matrix[:, 0, 0], rtol=2e-13, atol=2e-13
    )

    W_generic = compute_channel_screened_interaction(P_generic, definition)
    W_matrix = compute_screened_interaction_matrix(P_matrix, Vq)
    np.testing.assert_allclose(
        W_generic, W_matrix[:, 0, 0], rtol=2e-13, atol=2e-13
    )

    sigma_generic = compute_channel_correlation_self_energy(
        G, W_generic, definition, grid
    )
    Wc_matrix = (
        W_generic - definition.coupling[None, :, :]
    )[:, None, None]
    sigma_matrix = compute_sigma_gw_matrix(
        G, Wc_matrix, grid, backend="direct"
    )
    np.testing.assert_allclose(
        sigma_generic, sigma_matrix, rtol=3e-13, atol=3e-13
    )


def test_analytic_channel_kernel_matches_directional_derivative():
    grid, _, _, G = _fixture()
    definition = build_channel_definition(PAIRS, 3, 0.29, "J")
    P = compute_channel_polarization(G, definition, grid)
    W = compute_channel_screened_interaction(P, definition)

    rng = np.random.default_rng(31)
    X0 = 0.03 * (
        rng.normal(size=(grid.nf, 3, 3))
        + 1j * rng.normal(size=(grid.nf, 3, 3))
    )
    Gamma0 = np.empty_like(X0)
    for n in range(grid.nf):
        Gi = np.linalg.inv(G[n, 0, 0])
        Gamma0[n] = Gi @ X0[n] @ Gi
    Gamma = Gamma0[:, None, None]

    gs, gm, ga = channel_vertex_kernel_parts(
        G, W, definition, Gamma, grid
    )

    eps = 2e-7

    def dyn_sigma(Gtrial):
        Ptr = compute_channel_polarization(Gtrial, definition, grid)
        Wtr = compute_channel_screened_interaction(Ptr, definition)
        return compute_channel_correlation_self_energy(
            Gtrial, Wtr, definition, grid
        )

    Gplus = G + eps * X0[:, None, None]
    Gminus = G - eps * X0[:, None, None]
    fd_dyn = (dyn_sigma(Gplus) - dyn_sigma(Gminus)) / (2 * eps)
    np.testing.assert_allclose(
        gm + ga, fd_dyn, rtol=2e-6, atol=2e-8
    )

    rho0 = np.array(
        [[0.4, 0.07+0.03j, -0.02],
         [0.07-0.03j, 0.35, 0.04j],
         [-0.02, -0.04j, 0.25]],
        dtype=complex,
    )
    drho = grid.T * np.sum(X0, axis=0)
    sp = channel_static_self_energy(rho0 + eps*drho, definition)[0]
    sm = channel_static_self_energy(rho0 - eps*drho, definition)[0]
    fd_static = (sp - sm) / (2*eps)
    np.testing.assert_allclose(
        gs[0, 0, 0], fd_static, rtol=2e-7, atol=2e-9
    )
