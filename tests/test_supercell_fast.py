import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_gw import solve_matrix_gw
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast, solve_supercell_gw_fast


def test_fast_supercell_solver_matches_reference_at_v_zero():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.1)
    opts = GWOptions(
        target_filling=8.3,
        max_iter=4,
        tol=1e-10,
        mixing=0.5,
        mu_tol=1e-9,
        mu_max_iter=40,
        verbose=False,
    )
    h0 = build_supercell_h0(grid.kmesh(), params)
    vq = build_supercell_interaction(grid.qmesh(), params)

    ref = solve_matrix_gw(h0, vq, grid, opts)
    fast = solve_matrix_gw_fast(h0, vq, grid, opts)

    assert fast.converged
    assert abs(np.sum(fast.density) - 8.3) < 2e-8
    assert abs(fast.mu - ref.mu) < 2e-7
    assert np.max(np.abs(fast.density - ref.density)) < 2e-7


def test_fast_supercell_solver_has_expected_shapes():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    opts = GWOptions(
        target_filling=9.0,
        max_iter=3,
        tol=1e-9,
        mixing=0.5,
        mu_tol=1e-8,
        verbose=False,
    )
    result = solve_supercell_gw_fast(params, grid, opts)
    assert result.converged
    assert result.G.shape[-2:] == (NSUP, NSUP)
    assert result.Sigma_H.shape == (NSUP, NSUP)
    assert result.min_density_mode.shape == (NSUP,)
