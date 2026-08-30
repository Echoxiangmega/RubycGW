import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, build_supercell_interaction
from rubycgw.supercell_gw_anderson import (
    AndersonOptions,
    solve_supercell_gw_anderson,
)
from rubycgw.supercell_gw_periodic_pulay import (
    _gw_only_pulay_step,
    _uniform_hartree_seed,
)


def test_uniform_half_filled_hartree_seed_is_2v_identity():
    V = 0.7
    params = RubyParameters(V=V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    seed = _uniform_hartree_seed(Vq[0, 0], NSUP, 9.0)
    assert np.max(np.abs(seed - 2.0 * V * np.eye(NSUP))) < 1e-12


def test_gw_only_pulay_solves_scalar_unstable_linear_fixed_point():
    # F(x)=1.2*x+1 has fixed point x=-5.  Positive linear mixing cannot make
    # the fixed-point eigenvalue contract, but two exact residuals determine
    # the scalar DIIS extrapolation.
    x0 = np.array([0.0 + 0.0j])
    out0 = 1.2 * x0 + 1.0
    r0 = out0 - x0

    x1 = np.array([0.1 + 0.0j])
    out1 = 1.2 * x1 + 1.0
    r1 = out1 - x1

    xnext, valid = _gw_only_pulay_step(
        x1,
        r1,
        [(out0, r0), (out1, r1)],
        damping=1.0,
        regularization=0.0,
        step_cap=1000.0,
        linear_beta=0.1,
        scale_floor=1e-4,
    )
    assert valid
    assert np.allclose(xnext, -5.0, atol=1e-10)


def test_periodic_pulay_solver_keeps_v_zero_exact():
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
    assert result.mixing_method == "gw-periodic-pulay"
