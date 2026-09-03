import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.magnetic_self_energy import (
    geometric_uniform_B_green_source,
    solve_uniform_B_self_energy_derivative,
)
from rubycgw.supercell_cgw import SupercellVertexOptions


def test_geometric_uniform_B_source_matches_a13_rewrite():
    rng = np.random.default_rng(1234)
    shape = (3, 1, 1, 2, 2)
    G = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    Jx = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    Jy = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    got = geometric_uniform_B_green_source(G, Jx, Jy)

    # A13: (i/2) G[(D_x G^-1)(D_y G)-(D_y G^-1)(D_x G)]
    # with D G^-1=-J and D G=G J G.
    Dginv_x = -Jx
    Dginv_y = -Jy
    Dg_x = np.matmul(np.matmul(G, Jx), G)
    Dg_y = np.matmul(np.matmul(G, Jy), G)
    expected = 0.5j * (
        np.matmul(np.matmul(G, Dginv_x), Dg_y)
        - np.matmul(np.matmul(G, Dginv_y), Dg_x)
    )
    assert np.allclose(got, expected, rtol=1e-13, atol=1e-13)


def test_noninteracting_uniform_B_sigma_derivative_is_exactly_zero():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.25)
    rng = np.random.default_rng(5)
    G = rng.normal(size=(grid.nf, 1, 1, 2, 2)) + 1j * rng.normal(
        size=(grid.nf, 1, 1, 2, 2)
    )
    Jx = rng.normal(size=G.shape) + 1j * rng.normal(size=G.shape)
    Jy = rng.normal(size=G.shape) + 1j * rng.normal(size=G.shape)
    W = np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex)
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)

    opts = SupercellVertexOptions(
        max_iter=10,
        tol=1e-12,
        solver="gmres",
        gmres_restart=4,
        verbose=False,
        momentum_backend="direct",
    )
    result = solve_uniform_B_self_energy_derivative(
        G, W, Vq, Jx, Jy, grid, opts=opts
    )

    expected_Y = geometric_uniform_B_green_source(G, Jx, Jy)
    assert result.converged
    assert result.iterations == 0
    assert result.final_error == 0.0
    assert np.max(np.abs(result.Sigma_B_code)) == 0.0
    assert np.allclose(result.G_B_code, expected_Y, rtol=0.0, atol=0.0)


def test_hartree_only_uniform_B_sigma_solver_satisfies_linearized_gw_equation():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.35)
    rng = np.random.default_rng(91)

    # Keep G small enough that the Hartree-only kernel is safely contractive.
    G = 0.08 * (
        rng.normal(size=(grid.nf, 1, 1, 2, 2))
        + 1j * rng.normal(size=(grid.nf, 1, 1, 2, 2))
    )
    Jx = rng.normal(size=G.shape) + 1j * rng.normal(size=G.shape)
    Jy = rng.normal(size=G.shape) + 1j * rng.normal(size=G.shape)
    W = np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex)
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)
    Vq[0, 0, 0, 1] = 0.2
    Vq[0, 0, 1, 0] = 0.2

    opts = SupercellVertexOptions(
        max_iter=30,
        tol=1e-11,
        solver="gmres",
        gmres_restart=6,
        include_hartree=True,
        include_fock=False,
        include_mt=False,
        include_al=False,
        verbose=False,
        momentum_backend="direct",
    )
    result = solve_uniform_B_self_energy_derivative(
        G, W, Vq, Jx, Jy, grid, opts=opts
    )

    assert result.converged
    assert result.final_error < 1e-11
    # Hartree-only response must remain diagonal and frequency/momentum independent.
    sigma = result.Sigma_B_code
    assert np.max(np.abs(sigma[..., 0, 1])) < 1e-12
    assert np.max(np.abs(sigma[..., 1, 0])) < 1e-12
    ref = sigma[0, 0, 0]
    assert np.max(np.abs(sigma - ref[None, None, None, :, :])) < 1e-12
