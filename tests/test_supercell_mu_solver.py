import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_fast import (
    _build_tail_cache,
    _effective_mu_tol,
    _solve_mu_matrix_fast,
    _total_filling_slope,
    density_from_G_cached,
)


def _small_problem():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=2, T=0.05)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    h0 = build_supercell_h0(grid.kmesh(), params)
    sigma_h = np.zeros((18, 18), dtype=complex)
    sigma_gw = np.zeros((grid.nf, 1, 1, 18, 18), dtype=complex)
    return grid, h0, sigma_h, sigma_gw


def test_total_filling_slope_matches_finite_difference():
    grid, h0, sigma_h, sigma_gw = _small_problem()
    mu = 0.13
    cache = _build_tail_cache(h0, sigma_h)
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    analytic = _total_filling_slope(G, grid, mu, cache)

    dmu = 2.0e-6
    Gp = dyson_from_sigma_matrix(h0, grid, mu + dmu, sigma_h, sigma_gw)
    Gm = dyson_from_sigma_matrix(h0, grid, mu - dmu, sigma_h, sigma_gw)
    np_ = np.sum(density_from_G_cached(Gp, grid, mu + dmu, cache))
    nm_ = np.sum(density_from_G_cached(Gm, grid, mu - dmu, cache))
    finite_difference = (np_ - nm_) / (2.0 * dmu)

    assert analytic > 0.0
    assert np.isclose(analytic, finite_difference, rtol=2e-5, atol=2e-6)


def test_newton_mu_solver_reaches_half_filling_from_offset_guess():
    grid, h0, sigma_h, sigma_gw = _small_problem()
    mu, G, cache, neval = _solve_mu_matrix_fast(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        target=9.0,
        mu0=-0.8,
        tol=1e-9,
        max_iter=30,
    )
    filling = float(np.sum(density_from_G_cached(G, grid, mu, cache)))
    assert abs(filling - 9.0) < 1e-9
    assert neval <= 12


def test_effective_mu_tolerance_is_loose_far_and_strict_near_fixed_point():
    strict = 1e-8
    assert np.isclose(_effective_mu_tol(strict, 0.3), 1e-4)
    assert np.isclose(_effective_mu_tol(strict, 1e-3), 1e-5)
    assert np.isclose(_effective_mu_tol(strict, 1e-6), 1e-8)
