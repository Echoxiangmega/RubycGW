import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, build_supercell_interaction
from rubycgw.supercell_gw_anderson import (
    AndersonOptions,
    solve_supercell_gw_anderson,
)
from rubycgw.supercell_gw_block import _uniform_hartree_seed


def test_uniform_half_filled_hartree_seed_is_2v_identity():
    V = 0.7
    params = RubyParameters(V=V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    seed = _uniform_hartree_seed(Vq[0, 0], NSUP, 9.0)
    assert np.max(np.abs(seed - 2.0 * V * np.eye(NSUP))) < 1e-12


def test_lightweight_solver_keeps_v_zero_exact():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.1)
    opts = GWOptions(
        target_filling=9.0,
        max_iter=4,
        tol=1e-10,
        mixing=0.5,
        verbose=False,
    )
    result = solve_supercell_gw_anderson(
        params,
        grid,
        opts,
        anderson=AndersonOptions(),
    )
    assert result.converged
    assert abs(np.sum(result.density) - 9.0) < 1e-8
    assert result.final_error < opts.tol
    assert result.mixing_method == "adaptive-block-late-anderson"
