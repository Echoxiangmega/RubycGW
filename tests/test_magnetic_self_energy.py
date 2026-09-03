import numpy as np

from rubycgw.bulk_orbital_magnetization import (
    bulk_orbital_magnetization_from_arrays,
    spectral_cartesian_covariant_derivatives,
    supercell_h0_cartesian_derivatives,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.magnetic_self_energy import (
    geometric_uniform_B_green_source,
    solve_uniform_B_self_energy_derivative,
)
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast


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


def test_interacting_time_reversal_symmetric_ruby_has_zero_complete_bulk_M():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=2, T=0.30)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.15, V=0.01)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    gw_opts = GWOptions(
        mu=0.08,
        target_filling=None,
        max_iter=100,
        tol=3e-10,
        mixing=0.35,
        mixing_method="pulay",
        pulay_history=6,
        pulay_start=3,
        verbose=False,
        momentum_backend="direct",
    )
    gw = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    assert gw.converged

    sigma_total = gw.Sigma_GW + gw.Sigma_H[None, None, None, :, :]
    Dx_H0, Dy_H0 = supercell_h0_cartesian_derivatives(grid.kmesh(), params)
    Dx_Sigma, Dy_Sigma = spectral_cartesian_covariant_derivatives(sigma_total)
    Jx = Dx_H0[None, :, :, :, :] + Dx_Sigma
    Jy = Dy_H0[None, :, :, :, :] + Dy_Sigma

    vopts = SupercellVertexOptions(
        max_iter=60,
        tol=2e-10,
        solver="gmres",
        gmres_restart=8,
        verbose=False,
        momentum_backend="direct",
    )
    field = solve_uniform_B_self_energy_derivative(
        gw.G, gw.W, Vq, Jx, Jy, grid, opts=vopts
    )
    assert field.converged
    assert field.final_error < 2e-10

    bulk = bulk_orbital_magnetization_from_arrays(
        h0,
        gw.G,
        sigma_total,
        gw.mu,
        grid,
        params,
        sigma_b=field.Sigma_B_code,
    )
    assert bulk.complete
    assert abs(bulk.main_term_code) < 2e-10
    assert abs(bulk.field_self_energy_term_code) < 2e-10
    assert abs(bulk.total_code) < 2e-10
